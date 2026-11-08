import argparse
import json
import os
import sys
from datetime import datetime, timezone
from getpass import getpass
from pathlib import Path

from .decision import TaskRunner
from .g1_task import SKILLS, TaskRequest
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
    scene = commands.add_parser("scene-action", help="Select a grasp step from scene evidence")
    scene.add_argument("--observation", type=Path, required=True)
    target = scene.add_mutually_exclusive_group(required=True)
    target.add_argument("--object-id")
    target.add_argument("--category")
    scene.add_argument("--capture", type=Path, help="G1 capture directory for matched landmarks")
    vision_source = scene.add_mutually_exclusive_group()
    vision_source.add_argument(
        "--bailian-image", type=Path, help="Send this image to Bailian for selection"
    )
    vision_source.add_argument("--local-image", type=Path, help="Run Qwen3-VL on this local image")
    vision_source.add_argument(
        "--vision-choice", type=Path, help="Import a choice JSON from an interactive vision tool"
    )
    scene.add_argument("--local-model", type=Path, help="Local Qwen3-VL checkpoint directory")
    scene.add_argument("--bailian-config", type=Path, help="Read the sk entry from a local note")
    scene.add_argument("--intent", help="Grasp request to resolve against the selected image")
    scene.add_argument("--output", type=Path, help="Optional JSON report path")
    grasp = commands.add_parser("g1-task-plan", help="Run the staged G1 task through ControlHub")
    grasp.add_argument("--capture", type=Path, required=True)
    grasp.add_argument("--observation", type=Path, required=True)
    grasp.add_argument("--vision-choice", type=Path)
    grasp.add_argument("--skill", choices=SKILLS, default="shake_then_place")
    grasp.add_argument("--repeat-count", type=int)
    grasp.add_argument("--planner", choices=("preview", "sdk"), default="preview")
    task_target = grasp.add_mutually_exclusive_group(required=True)
    task_target.add_argument("--object-id")
    task_target.add_argument("--category")
    task_target.add_argument("--text")
    grasp.add_argument("--nlp-model", type=Path, help="Locally trained intent and slot checkpoint")
    nlp = commands.add_parser("g1-task-nlp-fit", help="Train local BERT intent and slot heads")
    nlp.add_argument("--train", required=True, type=Path)
    nlp.add_argument("--validation", required=True, type=Path)
    nlp.add_argument("--output", required=True, type=Path)
    nlp.add_argument("--encoder", default="bert-base-chinese")
    nlp.add_argument("--epochs", type=int, default=20)
    fit = commands.add_parser("tabletop-fit", help="Fit local neural reference prototypes")
    fit.add_argument("--frontend", type=Path, default=WORKSPACE / "working/frontend")
    fit.add_argument("--references", type=Path, help="Optional manually labeled capture crops")
    fit.add_argument("--weights", type=Path, default=WORKSPACE / "models/yolo11n.pt")
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--device", default="cpu")
    tabletop = commands.add_parser("tabletop-recognize", help="Match a local fixed class set")
    tabletop.add_argument("--image", type=Path, required=True)
    tabletop.add_argument("--bank", type=Path, required=True)
    tabletop.add_argument("--weights", type=Path, default=WORKSPACE / "models/yolo11n.pt")
    tabletop.add_argument("--kind", choices=("recipe", "ingredient", "tabletop"), required=True)
    tabletop.add_argument("--roi", type=int, nargs=4, metavar=("X1", "Y1", "X2", "Y2"))
    tabletop.add_argument("--propose", action="store_true", help="Detect boxes inside the ROI")
    tabletop.add_argument("--min-similarity", type=float, default=0.8)
    tabletop.add_argument("--min-margin", type=float, default=0.04)
    tabletop.add_argument("--device", default="cpu")
    tabletop.add_argument("--output", type=Path)
    spatial = commands.add_parser("g1-tabletop-range", help="Read-only G1 surface range check")
    spatial.add_argument("--capture", type=Path, required=True)
    spatial.add_argument("--wrist", choices=("left", "right"), required=True)
    spatial.add_argument("--pixel", type=int, nargs=2, required=True, metavar=("U", "V"))
    spatial.add_argument(
        "--table-pixels", type=int, nargs=6, metavar=("U0", "V0", "UX", "VX", "UY", "VY")
    )
    spatial.add_argument("--tool-profile", type=Path)
    spatial.add_argument(
        "--grip", type=float, nargs=3, metavar=("WIDTH_M", "SPEED_MPS", "EFFORT_NM")
    )
    spatial.add_argument("--output", type=Path)
    monitor = commands.add_parser("monitor", help="Local control hub and live monitoring interface")
    monitor.add_argument("--port", type=int, default=8765)
    monitor.add_argument("--config", type=Path, default=WORKSPACE / "configs/simulation.json")
    monitor.add_argument("--hpu-host", help="Pinned SSH host or user@host for the deployed worker")
    monitor.add_argument("--hpu-profile", help="Absolute path to the HPU profile")
    monitor.add_argument(
        "--hpu-passwords", action="store_true",
        help="Prompt for HPU and XCU SSH passwords and keep them only in memory",
    )
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
        if args.command == "g1-task-nlp-fit":
            from .task_nlp import fit_heads

            metrics = fit_heads(args.train, args.validation, args.output, args.encoder, args.epochs)
            print(json.dumps(metrics, ensure_ascii=False, indent=2))
            return 0
        if args.command == "g1-task-plan":
            from .hub import ControlHub

            if (args.text is None) != (args.nlp_model is None):
                raise TaskError("Supply --text and --nlp-model together")
            if args.text is not None:
                from .task_nlp import resolve_text

                if args.repeat_count is not None or args.skill != "shake_then_place":
                    raise TaskError("The text request supplies the skill and repeat count")
                request = resolve_text(args.text, args.nlp_model, args.observation)
            else:
                request = TaskRequest(args.skill, args.object_id, args.category, args.repeat_count)
            planner = None
            if args.planner == "sdk":
                from .g1_planner import GalbotStagePlanner

                planner = GalbotStagePlanner()
            config = json.loads((WORKSPACE / "configs/simulation.json").read_text(encoding="utf-8"))
            hub = ControlHub(WORKSPACE, config)
            try:
                hub.start_tabletop(
                    request, args.capture, args.observation,
                    vision_choice=args.vision_choice, planner=planner,
                )
                hub.wait(timeout=180)
                snapshot = hub.snapshot()
            finally:
                hub.close()
            report = {**snapshot["summary"], "artifact_directory": snapshot["artifact_directory"]}
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0 if report["success"] else 1
        if args.command in ("tabletop-fit", "tabletop-recognize"):
            from .tabletop_model import fit_references, recognize_references

            if args.command == "tabletop-fit":
                report = fit_references(
                    args.weights, args.frontend, args.references, args.output, args.device
                )
            else:
                report = recognize_references(
                    args.weights,
                    args.bank,
                    args.image,
                    args.kind,
                    roi=args.roi,
                    propose=args.propose,
                    min_similarity=args.min_similarity,
                    min_margin=args.min_margin,
                    device=args.device,
                )
                if args.output:
                    save_report(args.output, report)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0
        if args.command == "g1-tabletop-range":
            from .g1_tabletop import measure_tabletop

            report = measure_tabletop(
                args.capture,
                args.wrist,
                tuple(args.pixel),
                table_pixels=list(zip(args.table_pixels[::2], args.table_pixels[1::2], strict=True))
                if args.table_pixels
                else None,
                profile_path=args.tool_profile,
                command=tuple(args.grip) if args.grip else None,
            )
            if args.output:
                save_report(args.output, report)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0
        if args.command == "scene-action":
            from .policy import recommend_scene_grasp
            from .recognition import load_scene

            vision_image = args.bailian_image or args.local_image
            if (vision_image is None) != (args.intent is None):
                raise TaskError("Supply a vision image and --intent together")
            if args.local_model is not None and args.local_image is None:
                raise TaskError("--local-model requires --local-image")
            if args.bailian_config is not None and args.bailian_image is None:
                raise TaskError("--bailian-config requires --bailian-image")
            inputs = {
                path.resolve()
                for path in (
                    args.observation,
                    vision_image,
                    args.vision_choice,
                    args.bailian_config,
                )
                if path is not None
            }
            if args.output and args.output.resolve() in inputs:
                raise TaskError("Choose a report path different from the input files")
            observation = load_scene(args.observation)
            has_visual_choice = vision_image is not None or args.vision_choice is not None
            report = recommend_scene_grasp(
                observation,
                object_id=args.object_id,
                category=args.category,
                capture=None if has_visual_choice else args.capture,
            )
            exit_code = 0
            if has_visual_choice and report["candidates"]:
                candidates = [
                    observation.detections[row["object_id"]] for row in report["candidates"]
                ]
                if args.vision_choice is not None:
                    from .scene_choice import parse_grasp_choice

                    provider = "imported_vision"
                    proposal = {"status": "unavailable", "selection": None}
                    try:
                        with args.vision_choice.open(encoding="utf-8-sig") as stream:
                            selection = parse_grasp_choice(
                                stream.read(2049), {obj.object_id for obj in candidates}
                            )
                        proposal.update(status="completed", selection=selection)
                    except (OSError, TaskError, UnicodeError):
                        proposal["error"] = "Could not import a valid choice for these candidates"
                elif args.local_image is not None:
                    from .local_vlm import choose_grasp_local

                    provider = "local_vlm"
                    proposal = choose_grasp_local(
                        vision_image,
                        args.intent,
                        candidates,
                        args.local_model or WORKSPACE / "models/qwen3-vl-2b-instruct",
                    )
                else:
                    from .bailian import DEFAULT_BASE_URL, DEFAULT_MODEL, choose_grasp, read_api_key

                    provider = "bailian"
                    api_key = (
                        read_api_key(args.bailian_config)
                        if args.bailian_config is not None
                        else os.getenv("DASHSCOPE_API_KEY", "").strip()
                    )
                    proposal = choose_grasp(
                        vision_image,
                        args.intent,
                        candidates,
                        api_key,
                        os.getenv("DASHSCOPE_BASE_URL") or DEFAULT_BASE_URL,
                        os.getenv("DASHSCOPE_MODEL") or DEFAULT_MODEL,
                    )
                choice = proposal["selection"]
                if choice is None:
                    report.update(next_action="hold", reason=f"{provider}_unavailable", target=None)
                    exit_code = 2
                elif choice["action"] == "grasp":
                    selected = recommend_scene_grasp(
                        observation, object_id=choice["object_id"], capture=args.capture
                    )
                    report.update(
                        target=selected["target"],
                        geometry=selected["geometry"],
                        next_action=selected["next_action"],
                        reason=selected["reason"],
                    )
                    if "geometry_error" in selected:
                        report["geometry_error"] = selected["geometry_error"]
                else:
                    report.update(
                        next_action=choice["action"],
                        reason=f"{provider}_visual_evidence",
                        target=None,
                    )
                report[provider] = proposal
            if args.output:
                save_report(args.output, report)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return exit_code
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

            hpu = None
            if bool(args.hpu_host) != bool(args.hpu_profile) or (
                args.hpu_passwords and not args.hpu_host
            ):
                raise TaskError("Supply --hpu-host and --hpu-profile together")
            if args.hpu_host:
                from .g1_link import HpuClient

                hpu = HpuClient(
                    args.hpu_host, args.hpu_profile,
                    password=getpass("HPU SSH password: ") if args.hpu_passwords else None,
                    xcu_password=getpass("XCU SSH password: ") if args.hpu_passwords else None,
                )
            config = json.loads(args.config.read_text(encoding="utf-8"))
            serve(WORKSPACE, config, args.port, hpu=hpu)
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
