"""Fit and run local neural image prototypes for the fixed menu and table objects."""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .types import TaskError
from .yolo import load_local_yolo

GROUPS = {"mushroom": "蘑菇", "tofu": "豆腐", "chicken": "鸡胸肉", "vegetable": "新鲜蔬菜"}
TABLE_CLASSES = {
    "clear_water_bottle": "透明水瓶",
    "pink_water_bottle": "粉色水壶",
    "bowl": "碗",
    "cup": "杯",
    "slotted_spoon": "带孔勺",
    "container": "盒子",
}
PROPOSALS = {
    "bottle": {"clear_water_bottle", "pink_water_bottle"},
    "bowl": {"bowl", "cup"},
    "cup": {"bowl", "cup"},
    "spoon": {"slotted_spoon"},
}
KINDS = ("recipe", "ingredient", "tabletop")
IMAGE_SIZE = 320
MIN_ONSITE_CAPTURE_GROUPS = 2


@dataclass(frozen=True)
class Reference:
    kind: str
    class_id: str
    name: str
    image: Path
    crop: tuple[int, int, int, int] | None = None
    capture_id: str | None = None


def _photo_map(source: str, variable: str) -> dict:
    declaration = re.search(rf"^const {variable}\s*=\s*", source, re.M)
    if declaration is None:
        raise TaskError(f"Frontend {variable} is missing")
    result, _ = json.JSONDecoder().raw_decode(source[declaration.end() :])
    return result


def _asset(frontend: Path, row: dict) -> Path:
    url = row["src"]
    if not re.fullmatch(r"/images/[a-z0-9.-]+\.webp", url):
        raise TaskError("Frontend reference photo must be a local WebP asset")
    path = frontend / url.lstrip("/")
    if not path.is_file() or not path.resolve().is_relative_to(frontend.resolve()):
        raise TaskError(f"Missing local reference photo: {path.name}")
    return path


def frontend_references(frontend: Path) -> list[Reference]:
    """Read the fixed frontend catalog without executing browser JavaScript."""
    source = (frontend / "app.js").read_text(encoding="utf-8")
    catalog = re.search(r"^const recipeCatalog\s*=\s*\[(.*?)^\];", source, re.M | re.S)
    if catalog is None:
        raise TaskError("Frontend recipeCatalog is missing")
    entries = re.findall(
        r'\{\s*"?id"?\s*:\s*"([a-z0-9-]+)"\s*,\s*"?name"?\s*:\s*"([^"]+)"',
        catalog.group(1),
    )
    photos = _photo_map(source, "foodPhotos")
    ingredients = _photo_map(source, "ingredientPhotos")
    if not entries or len(entries) != len(photos) or {k for k, _ in entries} != photos.keys():
        raise TaskError("Frontend recipe IDs and dish photos disagree")
    if ingredients.keys() != {f"raw-{group}" for group in GROUPS}:
        raise TaskError("Frontend ingredient photos and supported groups disagree")
    return [
        *(Reference("recipe", key, name, _asset(frontend, photos[key])) for key, name in entries),
        *(
            Reference("ingredient", key, name, _asset(frontend, ingredients[f"raw-{key}"]))
            for key, name in GROUPS.items()
        ),
    ]


def capture_references(manifest: Path, names: dict) -> list[Reference]:
    rows = json.loads(manifest.read_text(encoding="utf-8"))["references"]
    references = []
    for row in rows:
        kind = row.get("kind", "tabletop")
        class_id = row["class_id"]
        capture_id = row["capture_id"]
        if not isinstance(capture_id, str) or not capture_id.strip():
            raise TaskError("Every onsite crop needs a capture ID")
        references.append(
            Reference(
                kind,
                class_id,
                names[(kind, class_id)],
                (manifest.parent / row["image"]).resolve(),
                tuple(row["xyxy_px"]),
                capture_id,
            )
        )
    return references


def _pixels(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise TaskError(f"Unreadable image: {path.name}")
    return image


def _crop(image: np.ndarray, box) -> np.ndarray:
    x1, y1, x2, y2 = box
    h, w = image.shape[:2]
    if any(type(x) is not int for x in box) or not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
        raise TaskError("Crop must be an in-image integer x1 y1 x2 y2 rectangle")
    return image[y1:y2, x1:x2]


def _unit(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if not np.isfinite(values).all() or (norms <= 0).any():
        raise TaskError("Neural features must be finite and nonzero")
    return values / norms


def _embed(model, images, device) -> np.ndarray:
    vectors = model.embed(images, imgsz=IMAGE_SIZE, rect=False, device=device, verbose=False)
    return _unit(np.asarray([item.detach().cpu().numpy() for item in vectors], dtype=np.float32))


def fit_references(weights: Path, frontend: Path, manifest: Path | None, output: Path, device):
    if output.exists():
        raise TaskError("Choose a new output path for the prototype bank")
    references = frontend_references(frontend)
    if manifest is not None:
        names = {(ref.kind, ref.class_id): ref.name for ref in references}
        names.update({("tabletop", key): name for key, name in TABLE_CLASSES.items()})
        references += capture_references(manifest, names)
    images = []
    for sample in references:
        image = _pixels(sample.image)
        images.append(image if sample.crop is None else _crop(image, sample.crop))
    features = _embed(load_local_yolo(weights), images, device)
    groups = {}
    onsite = {}
    for sample, feature in zip(references, features, strict=True):
        key = (sample.kind, sample.class_id, sample.name)
        groups.setdefault(key, []).append(feature)
        if sample.capture_id:
            onsite.setdefault(key, set()).add(sample.capture_id)
    centers = _unit(np.stack([np.mean(rows, axis=0) for rows in groups.values()]))
    metadata = {
        "schema_version": 2,
        "backbone_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "reference_source": "frontend_photos_and_manually_labeled_capture_crops",
        "classes": [
            {
                "kind": kind,
                "class_id": key,
                "name": name,
                "references": len(rows),
                "onsite_capture_groups": len(onsite.get((kind, key, name), ())),
            }
            for (kind, key, name), rows in groups.items()
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        np.savez_compressed(stream, prototypes=centers, metadata=json.dumps(metadata))
    return metadata


def recognize_references(
    weights: Path,
    bank: Path,
    image: Path,
    kind: str,
    *,
    roi=None,
    propose=False,
    min_similarity=0.8,
    min_margin=0.04,
    device="cpu",
) -> dict:
    if not (0 <= min_similarity <= 1 and 0 <= min_margin <= 2):
        raise TaskError("Invalid similarity or margin gate")
    if propose and kind != "tabletop":
        raise TaskError("Box proposals apply only to tabletop classes")
    with np.load(bank, allow_pickle=False) as packed:
        metadata = json.loads(str(packed["metadata"]))
        centers = packed["prototypes"]
    if metadata.get("schema_version") != 2:
        raise TaskError("Refit the prototype bank to record onsite capture groups")
    if metadata["backbone_sha256"] != hashlib.sha256(weights.read_bytes()).hexdigest():
        raise TaskError("Feature backbone differs from the fitted prototype bank")
    classes = metadata["classes"]
    if centers.ndim != 2 or len(centers) != len(classes):
        raise TaskError("Invalid prototype bank")
    indices = [i for i, row in enumerate(classes) if row["kind"] == kind]
    if not indices:
        raise TaskError(f"No reference classes are fitted for {kind}")

    pixels = _pixels(image)
    area = tuple(roi) if roi is not None else (0, 0, pixels.shape[1], pixels.shape[0])
    region = _crop(pixels, area)
    model = load_local_yolo(weights)
    boxes = [(area, None, None)]
    if propose:
        proposals = model.predict(region, imgsz=640, device=device, verbose=False)[0]
        boxes = []
        for box in proposals.boxes:
            label = proposals.names[int(box.cls.item())]
            if label not in PROPOSALS:
                continue
            x1, y1, x2, y2 = (round(v) for v in box.xyxy[0].cpu().tolist())
            x, y = area[:2]
            if x2 > x1 and y2 > y1:
                boxes.append(((x + x1, y + y1, x + x2, y + y2), label, float(box.conf.item())))

    features = _embed(model, [_crop(pixels, box) for box, _, _ in boxes], device) if boxes else []
    results = []
    for (box, label, detector_confidence), feature in zip(boxes, features, strict=True):
        possible = [
            i for i in indices if label is None or classes[i]["class_id"] in PROPOSALS[label]
        ]
        if not possible:
            continue
        ranked = sorted(
            ((float(np.clip(centers[i] @ feature, -1, 1)), i) for i in possible), reverse=True
        )
        score, top = ranked[0]
        margin = score - ranked[1][0] if len(ranked) > 1 else None
        onsite_groups = classes[top]["onsite_capture_groups"]
        if onsite_groups < MIN_ONSITE_CAPTURE_GROUPS:
            status = "onsite_references_insufficient"
        elif score < min_similarity or (margin is not None and margin < min_margin):
            status = "ambiguous_reference_match"
        else:
            status = "candidate"
        results.append(
            {
                "xyxy_px": list(box),
                "detector_label": label,
                "detector_confidence": detector_confidence,
                "class_id": classes[top]["class_id"] if status == "candidate" else "unknown",
                "nearest_reference": classes[top]["class_id"],
                "similarity": score,
                "margin": margin,
                "status": status,
                "onsite_capture_groups": onsite_groups,
            }
        )
    return {
        "kind": kind,
        "image": str(image),
        "roi_xyxy_px": list(area),
        "detections": results,
        "evidence": "reference_similarity_only",
        "dispatch_allowed": False,
    }
