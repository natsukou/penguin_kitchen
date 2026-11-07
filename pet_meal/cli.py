import argparse
import json
import os
import sys
from datetime import datetime, timezone
from getpass import getpass
from pathlib import Path

from .decision import TaskRunner
from .recording import Recorder
from .relevance import PHASES, fixture_test, save_report
from .sdk import capture_readonly, inspect_sdk
from .simulator import FAULTS, Simulator
from .types import TaskError

WORKSPACE = Path(__file__).resolve().parent.parent


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Pet Meal Robotics: explicit simulation baseline")
    commands = parser.add_subparsers(dest="command", required=True)
    simulate = commands.add_parser("simulate")
    simulate.add_argument("--config", type=Path, default=WORKSPACE / "configs/simulation.json")
    simulate.add_argument("--fault", choices=FAULTS, default="none")
    simulate.add_argument("--output", type=Path)
    commands.add_parser("doctor")
    capture = commands.add_parser(
        "capture-readonly", help="Linux only; never acquires motion control"
    )
    capture.add_argument("--camera", required=True, help="Verified RGB SensorType enum name")
    capture.add_argument("--depth", required=True, help="Verified depth SensorType enum name")
    capture.add_argument("--output", type=Path, required=True)
    vision = commands.add_parser(
        "detect-image", help="Offline YOLO detection without robot control"
    )
    vision.add_argument("--weights", type=Path, required=True)
    vision.add_argument("--image", type=Path, required=True)
    vision.add_argument("--output", type=Path, required=True)
    vision.add_argument("--device", default="cpu")
    monitor = commands.add_parser("monitor", help="Local control hub and live monitoring interface")
    monitor.add_argument("--port", type=int, default=8765)
    monitor.add_argument("--config", type=Path, default=WORKSPACE / "configs/simulation.json")
    fixtures = commands.add_parser(
        "relevance-test", help="Test relevance rules with offline fixtures"
    )
    fixtures.add_argument(
        "--output", type=Path, default=WORKSPACE / "artifacts/relevance/fixtures.json"
    )
    bailian = commands.add_parser("bailian-vision", help="Analyze a selected image through Bailian")
    bailian.add_argument(
        "--live", required=True, action="store_true", help="Send the selected image to Bailian"
    )
    bailian.add_argument("--image", required=True, type=Path)
    bailian.add_argument("--phase", choices=tuple(PHASES), action="append")
    bailian.add_argument("--base-url", help="Official Bailian base URL for the API key region")
    bailian.add_argument("--model", help="Bailian visual model ID")
    bailian.add_argument(
        "--prompt-key", action="store_true", help="Enter the key in this terminal without echo"
    )
    bailian.add_argument(
        "--output", type=Path, default=WORKSPACE / "artifacts/relevance/bailian.json"
    )
    args = parser.parse_args(argv)
    try:
        if args.command == "relevance-test":
            report = fixture_test()
            save_report(args.output, report)
            print(
                json.dumps(
                    {
                        "passed": report["passed"],
                        "total": report["total"],
                        "report": str(args.output),
                    },
                    ensure_ascii=False,
                )
            )
            return 0 if report["passed"] == report["total"] else 1
        if args.command == "bailian-vision":
            from .bailian import DEFAULT_BASE_URL, DEFAULT_MODEL, DEFAULT_PHASES, analyze_image

            if args.prompt_key and not sys.stdin.isatty():
                raise TaskError("Run --prompt-key in an interactive local terminal")
            key = (
                getpass("Bailian API Key: ")
                if args.prompt_key
                else os.getenv("DASHSCOPE_API_KEY", "")
            )
            report = analyze_image(
                args.image,
                key.strip(),
                args.base_url or os.getenv("DASHSCOPE_BASE_URL") or DEFAULT_BASE_URL,
                args.model or os.getenv("DASHSCOPE_MODEL") or DEFAULT_MODEL,
                tuple(args.phase or DEFAULT_PHASES),
            )
            save_report(args.output, report)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0 if report["status"] == "completed" else 2
        if args.command == "monitor":
            from .server import serve

            serve(WORKSPACE, json.loads(args.config.read_text(encoding="utf-8")), args.port)
            return 0
        if args.command == "doctor":
            result = inspect_sdk(WORKSPACE)
            print(json.dumps(result, indent=2))
            return 0 if result["sdk_downloaded"] else 1
        if args.command == "capture-readonly":
            result = capture_readonly(args.output, args.camera, args.depth)
            (args.output / "metadata.json").write_text(
                json.dumps(result, indent=2, default=str), encoding="utf-8"
            )
            print(json.dumps(result, indent=2, default=str))
            return 0
        if args.command == "detect-image":
            from .yolo import predict_image

            result = predict_image(args.weights, args.image, args.output, args.device)
            print(json.dumps(result, indent=2))
            return 0
        config = json.loads(args.config.read_text(encoding="utf-8"))
        output = args.output or WORKSPACE / "artifacts/runs" / datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%S.%fZ"
        )
        backend = Simulator(config, args.fault, output / "frames")
        recorder = Recorder(output, config, backend.evidence_level)
        try:
            recorder.event("scenario", {"fault": args.fault})
            result = TaskRunner(backend, config, recorder).run()
        finally:
            recorder.close()
        print(json.dumps({**result, "artifact_directory": str(output)}, indent=2))
        return 0 if result["success"] else 1
    except (TaskError, OSError, ValueError, KeyError, AttributeError, ImportError) as exc:
        print(json.dumps({"success": False, "error": f"{type(exc).__name__}: {exc}"}))
        return 2
