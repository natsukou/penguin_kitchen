"""Official SDK inspection and opt-in sensor capture; no robot motion calls."""

import importlib
import platform
import time
from pathlib import Path

import cv2
import numpy as np

from .types import HardwareDisabled, TaskError


class GalbotBackend:
    evidence_level = "hardware_not_validated"

    def execute(self, action) -> None:
        raise HardwareDisabled("Real robot motion is not implemented in V0.1")

    def observe(self):
        raise HardwareDisabled("Use capture-readonly for independent sensor validation")

    def stop(self):
        raise HardwareDisabled("No hardware controller has been acquired")


def inspect_sdk(workspace: Path) -> dict:
    sdk = workspace / "external" / "GalbotSDK"
    version = sdk / "VERSION"
    linux = platform.system() == "Linux"
    return {
        "host_os": platform.system(),
        "host_arch": platform.machine(),
        "sdk_downloaded": version.is_file(),
        "downloaded_sdk_version": version.read_text().strip() if version.is_file() else None,
        "linux_library_count": sum(1 for _ in sdk.glob("galbot_sdk/**/*.so")),
        "windows_extension_count": sum(1 for _ in sdk.glob("galbot_sdk/**/*.pyd")),
        "sdk_runtime": "needs_linux_installation_and_probe" if linux else "unsupported_native_os",
        "hardware_motion": "disabled_not_implemented",
        "robot_connected": False,
        "gbs_version": None,
        "hardware_acceptance": "not_tested",
    }


def capture_readonly(output: Path, camera_name: str, depth_name: str) -> dict:
    """Opt-in Linux capture using explicit SDK sensor enum names, not hardware models."""
    if platform.system() != "Linux":
        raise TaskError("Galbot SDK capture requires a configured Linux SDK environment")
    sdk = importlib.import_module("galbot_sdk.g1")
    camera = getattr(sdk.SensorType, camera_name)
    depth_camera = getattr(sdk.SensorType, depth_name)
    robot = sdk.GalbotRobot()
    output.mkdir(parents=True, exist_ok=False)
    try:
        if not robot.init({camera, depth_camera}, True):
            raise TaskError("SDK sensor initialization failed")
        deadline = time.monotonic() + 5
        while True:
            obs = robot.get_synced_observation([camera, depth_camera], True)
            if obs and obs.rgb_data_map.get(camera) and obs.depth_data_map.get(depth_camera):
                break
            if time.monotonic() >= deadline:
                raise TaskError("No RGB/depth pair received within capture deadline")
            time.sleep(0.1)
        rgb = obs.rgb_data_map[camera]
        depth = obs.depth_data_map[depth_camera]
        raw = np.frombuffer(bytes(rgb.data), dtype=np.uint8)
        if "nv12" not in rgb.format.lower() or raw.size != rgb.width * rgb.height * 3 // 2:
            raise TaskError(f"Unsupported synchronized RGB layout: {rgb.format}")
        bgr = cv2.cvtColor(raw.reshape(rgb.height * 3 // 2, rgb.width), cv2.COLOR_YUV2BGR_NV12)
        # ROS compressedDepth can include a 12-byte header; locate the PNG signature.
        payload = bytes(depth.data)
        signature = payload.find(b"\x89PNG\r\n\x1a\n")
        if signature < 0 or "16uc1" not in depth.format.lower():
            raise TaskError(f"Unsupported depth encoding: {depth.format}")
        decoded = cv2.imdecode(np.frombuffer(payload[signature:], np.uint8), cv2.IMREAD_UNCHANGED)
        if decoded is None or decoded.dtype != np.uint16 or depth.depth_scale <= 0:
            raise TaskError("Invalid metric depth data")
        if decoded.shape != (depth.height, depth.width):
            raise TaskError("Depth metadata dimensions differ from decoded image")
        skew_ns = abs(rgb.header.timestamp_ns - depth.header.timestamp_ns)
        if skew_ns > 50_000_000:
            raise TaskError("Captured RGB/depth timestamps differ by more than 50 ms")
        if not cv2.imwrite(str(output / "rgb.png"), bgr):
            raise TaskError("Failed to save camera image")
        np.save(output / "depth-m.npy", decoded.astype(np.float32) / depth.depth_scale)
        return {
            "evidence_level": "hardware_sensor_capture_only",
            "camera_enum": camera_name,
            "depth_enum": depth_name,
            "rgb_timestamp_ns": rgb.header.timestamp_ns,
            "depth_timestamp_ns": depth.header.timestamp_ns,
            "sensor_skew_ns": skew_ns,
            "device": robot.get_device_information(),
            "rgb_intrinsics": robot.get_camera_intrinsic(camera),
            "rgb_depth_registration": "not_verified_do_not_use_for_grasping",
            "hardware_motion": "disabled_not_implemented",
        }
    finally:
        try:
            robot.request_shutdown()
        finally:
            try:
                robot.wait_for_shutdown()
            finally:
                robot.destroy()
