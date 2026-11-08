"""Run the take01 ACT checkpoint on saved G1 captures without connecting to a robot."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from statistics import median
from time import perf_counter
from xml.etree import ElementTree

import cv2
import numpy as np
import torch
from galbot_mcap2lerobot.joint_constants import GRIPPER_VALUE_SCALE, JOINT_NAMES, JOINT_STRUCTURE
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACT
from torch import nn
from torchvision.transforms.functional import normalize

ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "742c753c0d4a5d87076c8f69e5628c79a8cc5488"
REFERENCE_URDF = ROOT / "external/galbot_one_golf_description/urdf/galbot_one_golf_fixed_base.urdf"
REFERENCE_MJCF = ROOT / "external/galbot_one_golf_description/mjcf/galbot_one_golf_fixed_base.xml"
CAPTURE_RGB = {
    "left_arm_camera_color": "left-wrist-rgb",
    "front_head_camera_left_color": "head-left-rgb",
    "front_head_camera_right_color": "head-right-rgb",
    "right_arm_camera_color": "right-wrist-rgb",
}
STATE_DIM = len(JOINT_NAMES)
GRIPPER_INDICES = [i for i, name in enumerate(JOINT_NAMES) if "gripper" in name]
KEY_PREFIXES = {
    "backbones.0.0.body.": "backbone.",
    "transformer.encoder.": "encoder.",
    "transformer.decoder.": "decoder.",
    "encoder.": "vae_encoder.",
    "pos_table": "vae_encoder_pos_enc",
    "input_proj.": "encoder_img_feat_input_proj.",
    "input_proj_robot_state.": "encoder_robot_state_input_proj.",
    "cls_embed.": "vae_encoder_cls_embed.",
    "encoder_action_proj.": "vae_encoder_action_input_proj.",
    "encoder_joint_proj.": "vae_encoder_robot_state_input_proj.",
    "latent_proj.": "vae_encoder_latent_output_proj.",
    "latent_out_proj.": "encoder_latent_input_proj.",
    "additional_pos_embed.": "encoder_1d_feature_pos_embed.",
    "query_embed.": "decoder_pos_embed.",
}


class Take01ACT(ACT):
    def __init__(self, config):
        super().__init__(config)
        self.is_pad_head = nn.Linear(config.dim_model, 1)

    def forward(self, state, rgb):
        # Match the pinned ACT forward pass: zero latent, camera features joined on
        # the width axis, and the first normalized decoder output used by DETRVAE.
        features = [self.backbone(view.contiguous())["feature_map"] for view in rgb.unbind(1)]
        images = torch.cat([self.encoder_img_feat_input_proj(f) for f in features], dim=3)
        positions = torch.cat([self.encoder_cam_feat_pos_embed(f) for f in features], dim=3)
        state_token = self.encoder_robot_state_input_proj(state)
        latent = self.encoder_latent_input_proj(state.new_zeros(len(state), self.config.latent_dim))
        tokens = torch.cat((torch.stack((latent, state_token)), images.flatten(2).permute(2, 0, 1)))
        pos = torch.cat(
            (
                self.encoder_1d_feature_pos_embed.weight.unsqueeze(1),
                positions.flatten(2).permute(2, 0, 1),
            )
        )
        encoded = self.encoder(tokens, pos_embed=pos)
        queries = self.decoder_pos_embed.weight.unsqueeze(1)
        first = self.decoder.layers[0](
            torch.zeros_like(queries).expand(-1, len(state), -1),
            encoded,
            decoder_pos_embed=queries,
            encoder_pos_embed=pos,
        )
        return self.action_head(self.decoder.norm(first).transpose(0, 1))


def load_policy(checkpoint):
    info, weights = checkpoint["config"], checkpoint["model"]
    spec = info["model"]
    cameras = spec["camera_names"]
    if (
        info["source_commit"] != SOURCE_COMMIT
        or spec["state_dim"] != STATE_DIM
        or cameras != info["camera_names"]
        or len(cameras) != len(CAPTURE_RGB)
        or set(cameras) != CAPTURE_RGB.keys()
        or spec["position_embedding"] != "sine"
        or spec["masks"]
    ):
        raise ValueError("Unsupported checkpoint for the take01 ACT adapter")
    height, width = info["image_height"], info["image_width"]
    input_features = {"observation.state": PolicyFeature(FeatureType.STATE, (STATE_DIM,))}
    input_features.update(
        {
            f"observation.images.{name}": PolicyFeature(FeatureType.VISUAL, (3, height, width))
            for name in cameras
        }
    )
    config = ACTConfig(
        device="cpu",
        input_features=input_features,
        output_features={"action": PolicyFeature(FeatureType.ACTION, (STATE_DIM,))},
        vision_backbone=spec["backbone"],
        pretrained_backbone_weights=None,
        replace_final_stride_with_dilation=spec["dilation"],
        chunk_size=spec["num_queries"],
        n_action_steps=spec["num_queries"],
        dim_model=spec["hidden_dim"],
        dim_feedforward=spec["dim_feedforward"],
        n_heads=spec["nheads"],
        pre_norm=spec["pre_norm"],
        dropout=spec["dropout"],
        n_encoder_layers=spec["enc_layers"],
        n_decoder_layers=spec["dec_layers"],
        n_vae_encoder_layers=spec["enc_layers"],
        latent_dim=weights["latent_out_proj.weight"].shape[1],
    )
    model = Take01ACT(config)
    mapped = {}
    for key, value in weights.items():
        new_key = next(
            (new + key[len(old) :] for old, new in KEY_PREFIXES.items() if key.startswith(old)), key
        )
        if new_key in mapped:
            raise ValueError(f"Duplicate mapped weight: {new_key}")
        mapped[new_key] = value
    model.load_state_dict(mapped, strict=True)
    stats = {
        name: torch.as_tensor(checkpoint["stats"][name], dtype=torch.float32)
        for name in ("qpos_mean", "qpos_std", "action_mean", "action_std")
    }
    for name, values in stats.items():
        if (
            values.shape != (STATE_DIM,)
            or not values.isfinite().all()
            or (name.endswith("std") and (values <= 0).any())
        ):
            raise ValueError(f"Invalid normalization statistic: {name}")
    return model.eval(), stats


def read_capture(directory, cameras, height, width):
    capture = json.loads((directory / "capture.json").read_text(encoding="utf-8"))
    joint_state = capture["joint_state"]
    joints = joint_state["positions_rad"]
    # The SDK capture's raw gripper channels are millimeters, as in the MCAP converter.
    state = torch.tensor([joints[name] for name in JOINT_NAMES], dtype=torch.float32)
    state[GRIPPER_INDICES] *= GRIPPER_VALUE_SCALE
    if not state.isfinite().all():
        raise ValueError(f"Non-finite joint state: {directory}")

    images, stamps = [], []
    for camera in cameras:
        label = CAPTURE_RGB[camera]
        frame = capture["frames"][label]
        w, h = frame["width"], frame["height"]
        size = w * h * 3 // 2
        if frame["format"].lower() != "nv12" or w <= 0 or h <= 0 or w % 2 or h % 2:
            raise ValueError(f"Expected a raw NV12 frame for {label}")
        with gzip.open(directory / f"{label}.bin.gz", "rb") as stream:
            data = stream.read(size + 1)
        if len(data) != size:
            raise ValueError(f"Unexpected NV12 payload size for {label}")
        nv12 = np.frombuffer(data, dtype=np.uint8).reshape(h * 3 // 2, w)
        rgb = cv2.cvtColor(nv12, cv2.COLOR_YUV2RGB_NV12)
        images.append(cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA))
        stamps.append(frame["timestamp_ns"])
    camera_skew = max(stamps) - min(stamps)
    joint_skew = max(abs(joint_state["timestamp_ns"] - stamp) for stamp in stamps)
    if min(*stamps, joint_state["timestamp_ns"]) <= 0 or max(camera_skew, joint_skew) > 50_000_000:
        raise ValueError(f"Camera or joint state is outside the 50 ms capture window: {directory}")
    images = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).float().div_(255)
    return state, images, {"camera_skew_ms": camera_skew / 1e6, "joint_skew_ms": joint_skew / 1e6}


def reference_limits(path):
    limits = {}
    joints = {j.get("name"): j for j in ElementTree.parse(path).getroot().findall("joint")}
    for index, name in enumerate(JOINT_NAMES):
        if index in GRIPPER_INDICES:
            continue  # The URDF uses gripper angles; the checkpoint uses opening widths.
        bound = joints[name].find("limit")
        limits[index] = (float(bound.attrib["lower"]), float(bound.attrib["upper"]))
    return limits


def summarize_targets(targets, state, limits):
    groups = {}
    for name, span in JOINT_STRUCTURE.items():
        start, end = span["start"], span["end"]
        chunk = targets[:, start:end]
        first_delta = (chunk[0] - state[start:end]).abs()
        groups[name] = {
            "unit": "m" if "gripper" in name else "rad",
            "max_first_step_delta": first_delta.max().item(),
            "max_first_step_delta_joint": JOINT_NAMES[start + first_delta.argmax().item()],
            "max_adjacent_step_delta": chunk.diff(dim=0).abs().max().item(),
            "min": chunk.min(dim=0).values.tolist(),
            "max": chunk.max(dim=0).values.tolist(),
        }
        if "gripper" in name:
            groups[name]["negative_width_steps"] = (chunk[:, 0] < 0).sum().item()
    violations = {}
    for index, (lower, upper) in limits.items():
        values = targets[:, index]
        excess = torch.maximum(lower - values, values - upper)
        if (excess > 0).any():
            violations[JOINT_NAMES[index]] = {
                "steps_outside": (excess > 0).sum().item(),
                "max_excess_rad": excess.max().item(),
                "reference_rad": [lower, upper],
            }
    return groups, violations


def make_kinematic_probe(path):
    import mujoco

    # The installed MuJoCo can evaluate the joint hierarchy without the newer mesh schema.
    source = ElementTree.parse(path).getroot()
    skeleton = ElementTree.Element("mujoco")
    skeleton.append(source.find("compiler"))
    robot = source.find("./worldbody/body[@name='base_link']")
    ElementTree.SubElement(skeleton, "worldbody").append(robot)
    for body in robot.iter("body"):
        body.attrib.pop("childclass", None)
        for child in list(body):
            if child.tag not in ("body", "joint", "inertial"):
                body.remove(child)
            elif child.tag == "joint":
                child.attrib.pop("class", None)
                if "range" in child.attrib:
                    child.set("limited", "true")
    model = mujoco.MjModel.from_xml_string(ElementTree.tostring(skeleton, encoding="unicode"))
    data = mujoco.MjData(model)
    indices = [i for i in range(STATE_DIM) if i not in GRIPPER_INDICES]
    addresses = [model.jnt_qposadr[model.joint(JOINT_NAMES[i]).id] for i in indices]
    base = model.body("base_link").id
    sides = ("left", "right")
    flanges = [model.body(f"{side}_arm_end_effector_mount_link").id for side in sides]

    def probe(directory, state, targets):
        capture = json.loads((directory / "capture.json").read_text(encoding="utf-8"))
        measured = [capture["flanges"][side]["pose_in_base_link"][:3] for side in sides]
        poses = []
        for row in torch.vstack((state, targets)).numpy():
            data.qpos[addresses] = row[indices]
            mujoco.mj_kinematics(model, data)
            poses.append((data.xpos[flanges] - data.xpos[base]) @ data.xmat[base].reshape(3, 3))
        poses = np.stack(poses)
        displacement = np.linalg.norm(poses[1:] - poses[0], axis=2)
        return {
            "scope": "flange_position_only_no_dynamics_or_collision",
            "model": str(path.relative_to(ROOT)),
            "input_pose_error_m": dict(zip(sides, np.linalg.norm(poses[0] - measured, axis=1))),
            "first_target_displacement_m": dict(zip(sides, displacement[0])),
            "max_target_displacement_m": dict(zip(sides, displacement.max(axis=0))),
        }

    return probe


@torch.inference_mode()
def evaluate(model, stats, state, rgb, device, repeats):
    stats = {key: value.to(device) for key, value in stats.items()}
    qpos = ((state.to(device) - stats["qpos_mean"]) / stats["qpos_std"]).unsqueeze(0)
    rgb = rgb.to(device).unsqueeze(0)
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    image = normalize(rgb, mean, std)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    first = model(qpos, image)
    sync()
    expected = (1, model.config.chunk_size, STATE_DIM)
    timings, repeat_delta = [], 0.0
    for _ in range(repeats):
        start = perf_counter()
        repeated = model(qpos, image)
        sync()
        timings.append((perf_counter() - start) * 1000)
        repeat_delta = max(repeat_delta, (first - repeated).abs().max().item())
        torch.testing.assert_close(first, repeated, rtol=0, atol=1e-6)
    targets = (first * stats["action_std"] + stats["action_mean"])[0]
    black = model(qpos, normalize(torch.zeros_like(rgb), mean, std))
    mean_state = model(torch.zeros_like(qpos), image)
    if (
        first.shape != expected
        or not targets.isfinite().all()
        or not black.isfinite().all()
        or not mean_state.isfinite().all()
    ):
        raise RuntimeError("ACT produced an invalid prediction")
    rgb_delta = (first - black).abs().mean().item()
    state_delta = (first - mean_state).abs().mean().item()
    if min(rgb_delta, state_delta) <= 1e-6:
        raise RuntimeError("ACT output did not respond to image or joint state changes")
    return targets.cpu(), {
        "shape": list(first.shape),
        "all_outputs_finite": True,
        "repeats": repeats,
        "median_forward_ms": median(timings),
        "max_repeat_delta_normalized": repeat_delta,
        "black_images_mae_normalized": rgb_delta,
        "mean_joint_state_mae_normalized": state_delta,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--capture", type=Path, action="append", required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cpu")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--kinematics", action="store_true", help="Project targets onto G1 flanges")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or args.output.exists():
        parser.error("Use a positive repeat count and a new output file")
    if args.device == "cpu":
        torch.set_num_threads(4)
    elif not torch.cuda.is_available():
        parser.error("CUDA is not available")
    else:
        # Compare against CPU FP32 without TF32 truncation in the convolution backbone.
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    info = checkpoint["config"]
    model, stats = load_policy(checkpoint)
    model.to(device)
    limits = reference_limits(REFERENCE_URDF)
    kinematics = make_kinematic_probe(REFERENCE_MJCF) if args.kinematics else None
    with args.checkpoint.open("rb") as stream:
        sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    report = {
        "scope": "offline_open_loop_capture_inference",
        "checkpoint": str(args.checkpoint.resolve()),
        "sha256": sha256,
        "step": checkpoint["step"],
        "task": info["task"],
        "source_commit": SOURCE_COMMIT,
        "strict_weight_load": True,
        "weight_tensors": len(checkpoint["model"]),
        "decoder_output": "first_normalized_layer_as_in_pinned_upstream",
        "device": args.device,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch": torch.__version__,
        "motion_commands_sent": 0,
        "camera_order": info["camera_names"],
        "joint_order": JOINT_NAMES,
        "image_preprocessing": "NV12 to RGB; area resize; RGB/255; upstream ImageNet mean/std",
        "image_shape": [4, 3, info["image_height"], info["image_width"]],
        "position_limit_reference": str(REFERENCE_URDF.relative_to(ROOT)),
        "checkpoint_training_metrics": checkpoint["metrics"],
        "placement_success": "not_measured_no_labeled_trajectory_or_rollout",
        "captures": [],
    }
    for directory in args.capture:
        state, rgb, skew = read_capture(
            directory,
            info["camera_names"],
            info["image_height"],
            info["image_width"],
        )
        targets, checks = evaluate(model, stats, state, rgb, device, args.repeats)
        groups, violations = summarize_targets(targets, state, limits)
        capture_result = {
            "source": str(directory.resolve()),
            **skew,
            **checks,
            "input_state": state.tolist(),
            "groups": groups,
            "reference_position_limit_violations": violations,
            "predicted_position_targets": targets.tolist(),
        }
        if kinematics is not None:
            capture_result["kinematics"] = kinematics(directory, state, targets)
        report["captures"].append(capture_result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Offline ACT inference passed on {len(report['captures'])} capture(s): {args.output}")


if __name__ == "__main__":
    main()
