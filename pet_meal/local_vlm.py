"""Optional local Qwen3-VL inference for scene action selection."""

import io
import time
from pathlib import Path

from .bailian import prepare_image
from .recognition import SceneObject
from .scene_choice import grasp_prompt, parse_grasp_choice
from .types import TaskError


def _infer(model_dir: Path, jpeg: bytes, prompt: str) -> tuple[str, str]:
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    processor = AutoProcessor.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=False
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_dir,
        dtype=dtype,
        local_files_only=True,
        use_safetensors=True,
        attn_implementation="sdpa",
    ).to(device)
    model.eval()
    with Image.open(io.BytesIO(jpeg)) as image:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image.convert("RGB")},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(device)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=96, do_sample=False)
    content = processor.batch_decode(
        output[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
    )[0]
    return content, device


def choose_grasp_local(
    image: Path, intent: str, candidates: list[SceneObject], model_dir: Path
) -> dict:
    prompt = grasp_prompt(intent, candidates)
    report = {
        "source": "local_vlm",
        "model": model_dir.name,
        "status": "unavailable",
        "selection": None,
        "dispatch_allowed": False,
    }
    if not any(
        (model_dir / name).is_file()
        for name in ("model.safetensors", "model.safetensors.index.json")
    ):
        return {**report, "error": "Local Qwen3-VL safetensors checkpoint is missing"}
    started = time.perf_counter()
    try:
        jpeg, report["image"] = prepare_image(image)
        content, report["device"] = _infer(model_dir, jpeg, prompt)
        selection = parse_grasp_choice(content, {obj.object_id for obj in candidates})
        report.update(status="completed", selection=selection)
    except TaskError as exc:
        report["error"] = str(exc)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        report["error"] = f"Local Qwen3-VL inference failed ({type(exc).__name__})"
    report["elapsed_ms"] = round((time.perf_counter() - started) * 1000)
    return report
