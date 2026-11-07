"""Download pinned public model and sample assets and record their hashes."""

import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASSETS = {
    "models/yolo11n.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
    "models/bus.jpg": "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg",
    "models/tableware-cc0.jpg": "https://upload.wikimedia.org/wikipedia/commons/4/40/Fiber_One_in_a_bowl_with_milk_and_a_spoon.jpg",
}
EXPECTED_SHA256 = {
    "models/yolo11n.pt": "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1",
    "models/bus.jpg": "c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63",
    "models/tableware-cc0.jpg": "ccabfc06372f048a09d5eba24fe068720bf214e9cba8423b81016c646d845928",
}


def main():
    manifest = []
    for relative, url in ASSETS.items():
        target = ROOT / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read()
            target.write_bytes(data)
        data = target.read_bytes()
        if len(data) < 1000:
            raise RuntimeError(f"Unexpectedly small asset: {relative}")
        digest = hashlib.sha256(data).hexdigest()
        if digest != EXPECTED_SHA256[relative]:
            raise RuntimeError(f"Asset differs from locked SHA256: {relative}")
        manifest.append(
            {
                "path": relative,
                "source": url,
                "size_bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    (ROOT / "models/assets.lock.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
