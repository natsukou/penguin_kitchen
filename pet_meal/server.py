"""Loopback-only monitoring API and buildless web interface."""

import hmac
import json
import secrets
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .g1_show import PCM_RATE, PRESET_CLIP, SHOW_SECONDS, show_manifest
from .g1_task import TaskRequest
from .hub import ControlHub
from .types import TaskError
from .voice import interpret_voice

STATIC = Path(__file__).parent / "ui"
STATIC_ROUTES = {
    "/": ("index.html", "text/html"),
    "/app.js": ("app.js", "text/javascript"),
    "/style.css": ("style.css", "text/css"),
    "/operator": ("operator.html", "text/html"),
    "/operator.js": ("operator.js", "text/javascript"),
    "/operator.css": ("operator.css", "text/css"),
    "/show.js": ("show.js", "text/javascript"),
}


class HubServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, hub: ControlHub, port: int = 8765):
        self.hub = hub
        self.token = secrets.token_urlsafe(32)
        super().__init__(("127.0.0.1", port), HubHandler)

    def allowed_hosts(self):
        port = self.server_address[1]
        return {f"127.0.0.1:{port}", f"localhost:{port}"}


class HubHandler(BaseHTTPRequestHandler):
    server: HubServer

    def log_message(self, format, *args):
        pass

    def reply(self, data, status=200, content_type="application/json; charset=utf-8"):
        if isinstance(data, (dict, list)):
            data = json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            (
                "default-src 'self'; script-src 'self'; style-src 'self'; "
                "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'"
            ),
        )
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def valid_host(self):
        if self.headers.get("Host") not in self.server.allowed_hosts():
            self.reply({"error": "Invalid local host"}, 403)
            return False
        return True

    def report(self, path: Path):
        try:
            self.reply(json.loads(path.read_text(encoding="utf-8")))
        except FileNotFoundError:
            self.reply({"available": False})
        except (OSError, ValueError):
            self.reply({"error": "Report is temporarily unavailable"}, 503)

    def do_GET(self):
        if not self.valid_host():
            return
        parsed = urlparse(self.path)
        hub = self.server.hub
        if parsed.path == "/api/status":
            self.reply({**hub.snapshot(), "control_token": self.server.token})
        elif parsed.path == "/api/events":
            try:
                cursor = int(parse_qs(parsed.query).get("after", ["0"])[0])
                if cursor < 0:
                    raise ValueError()
            except ValueError:
                self.reply({"error": "Invalid event cursor"}, 400)
                return
            self.reply(hub.bus.after(cursor))
        elif parsed.path == "/api/runs":
            self.reply(hub.run_history())
        elif parsed.path.startswith("/api/runs/"):
            run = hub.run_status(parsed.path.removeprefix("/api/runs/"))
            if run is None:
                self.reply({"error": "Run not found"}, 404)
            else:
                self.reply(run)
        elif parsed.path == "/api/validation":
            self.report(hub.workspace / "artifacts/validation.json")
        elif parsed.path == "/api/vision-sample":
            self.report(hub.workspace / "artifacts/vision-tableware/predictions.json")
        elif parsed.path == "/api/relevance-test":
            self.report(hub.workspace / "artifacts/relevance/fixtures.json")
        elif parsed.path == "/api/bailian-vision":
            self.report(hub.workspace / "artifacts/relevance/bailian.json")
        elif parsed.path == "/api/vision-sample.jpg":
            image = hub.workspace / "artifacts/vision-tableware/annotated.jpg"
            if image.is_file():
                self.reply(image.read_bytes(), content_type="image/jpeg")
            else:
                self.reply({"error": "No vision sample"}, 404)
        elif parsed.path == "/api/frame":
            frame = hub.latest_frame()
            if frame:
                self.reply(frame, content_type="image/png")
            else:
                self.reply({"error": "No recorded frame yet"}, 404)
        elif parsed.path == "/api/summary":
            self.reply(hub.snapshot().get("summary") or {"available": False})
        elif parsed.path == "/api/show":
            audio_available = (hub.workspace / PRESET_CLIP).is_file()
            self.reply({**show_manifest(), "audio_available": audio_available})
        elif parsed.path == "/api/show/audio":
            clip = hub.workspace / PRESET_CLIP
            limit = PCM_RATE * SHOW_SECONDS * 2 + 4096
            try:
                with clip.open("rb") as stream:
                    audio = stream.read(limit + 1)
                if not (
                    audio.startswith(b"RIFF") and audio[8:12] == b"WAVE" and len(audio) <= limit
                ):
                    raise ValueError("Invalid show WAV")
            except (OSError, ValueError):
                self.reply({"error": "演出音源不可用"}, 404)
            else:
                self.reply(audio, content_type="audio/wav")
        elif parsed.path in STATIC_ROUTES:
            name, mime = STATIC_ROUTES[parsed.path]
            self.reply((STATIC / name).read_bytes(), content_type=f"{mime}; charset=utf-8")
        else:
            self.reply({"error": "Not found"}, 404)

    def do_POST(self):
        if not self.valid_host():
            return
        origin = self.headers.get("Origin")
        if origin and origin not in {f"http://{h}" for h in self.server.allowed_hosts()}:
            self.reply({"error": "Cross-origin control is disabled"}, 403)
            return
        supplied = self.headers.get("X-Control-Token", "")
        if not hmac.compare_digest(supplied, self.server.token):
            self.reply({"error": "Missing or invalid control token"}, 403)
            return
        if self.path not in (
            "/api/commands",
            "/api/tabletop/plan",
            "/api/tabletop/interpret",
            "/api/tabletop/simulate",
            "/api/tabletop/execute",
            "/api/hardware/probe",
        ):
            self.reply({"error": "Not found"}, 404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 4096:
                raise ValueError("Invalid request size")
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                raise ValueError("JSON content type required")
            request = json.loads(self.rfile.read(length))
            if self.path == "/api/hardware/probe":
                if request != {}:
                    raise ValueError("Hardware probe accepts an empty JSON object")
                self.reply(self.server.hub.probe_hardware())
                return
            if self.path == "/api/tabletop/execute":
                if not isinstance(request, dict) or set(request) != {
                    "plan_id", "plan_sha256", "robot_sn", "request_id",
                }:
                    raise ValueError(
                        "Execution requires plan ID, SHA-256, robot serial and request ID"
                    )
                self.reply(self.server.hub.start_hardware(**request), 202)
                return
            if self.path == "/api/tabletop/interpret":
                if not isinstance(request, dict) or set(request) != {"text"}:
                    raise ValueError("语音解析只接受 text 字段")
                task = interpret_voice(request["text"])
                self.reply({"task": asdict(task), "dispatch_allowed": False})
                return
            if self.path in ("/api/tabletop/plan", "/api/tabletop/simulate"):
                simulation = self.path == "/api/tabletop/simulate"
                self.reply(self._tabletop_plan(request, simulation=simulation), 202)
                return
            if not isinstance(request, dict) or set(request) - {
                "command",
                "fault",
                "step_interval_s",
            }:
                raise ValueError("Invalid command fields")
            name = request.get("command")
            if not isinstance(name, str):
                raise ValueError("Command name required")
            interval = request.get("step_interval_s", 0.8)
            if isinstance(interval, bool) or not isinstance(interval, (int, float)):
                raise ValueError("Step interval must be numeric")
            fault = request.get("fault", "none")
            if not isinstance(fault, str):
                raise ValueError("Scenario must be a string")
            result = self.server.hub.command(name, fault=fault, step_interval_s=interval)
            self.reply(result)
        except TaskError as exc:
            self.reply({"error": str(exc)}, 409)
        except (ValueError, TypeError) as exc:
            self.reply({"error": str(exc)}, 400)

    def _tabletop_plan(self, request, *, simulation=False):
        required = {"task", "capture", "observation"}
        allowed = required | {"vision_choice", "request_id"}
        if simulation:
            allowed.add("fault")
        if not isinstance(request, dict) or not required <= request.keys() <= allowed:
            raise ValueError("Invalid tabletop plan fields")
        fault = request.get("fault", "none") if simulation else None
        if simulation and not isinstance(fault, str):
            raise ValueError("Tabletop simulation fault must be a string")
        task = request["task"]
        if (
            not isinstance(task, dict)
            or "skill" not in task
            or set(task)
            - {
                "skill",
                "object_id",
                "category",
                "repeat_count",
            }
        ):
            raise ValueError("Invalid task fields")
        hub = self.server.hub
        root = (hub.workspace / "artifacts").resolve()

        def artifact_path(name, *, directory=False):
            value = request[name]
            if not isinstance(value, str) or not value.strip() or Path(value).anchor:
                raise ValueError(f"{name} must be relative to artifacts")
            path = (root / value).resolve()
            if not path.is_relative_to(root) or not (
                path.is_dir() if directory else path.suffix == ".json" and path.is_file()
            ):
                raise ValueError(f"Invalid tabletop {name} artifact")
            return path

        return hub.start_tabletop(
            TaskRequest(**task),
            artifact_path("capture", directory=True),
            artifact_path("observation"),
            vision_choice=artifact_path("vision_choice") if "vision_choice" in request else None,
            simulation_fault=fault,
            request_id=request.get("request_id"),
        )


def serve(workspace: Path, config: dict, port: int, *, hpu=None):
    hub = ControlHub(workspace, config, hpu=hpu)
    server = HubServer(hub, port)
    print(f"Pet Meal Control Center: http://127.0.0.1:{server.server_address[1]}", flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        hub.close()
        server.server_close()
