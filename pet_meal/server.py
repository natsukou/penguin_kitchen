"""Loopback-only monitoring API and buildless web interface."""

import hmac
import json
import re
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .hub import ControlHub
from .types import TaskError

STATIC = Path(__file__).parent / "ui"
STATIC_ROUTES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
    "/nutrition_ui/index.html": ("nutrition_ui/index.html", "text/html; charset=utf-8"),
    "/nutrition_ui/app.js": ("nutrition_ui/app.js", "text/javascript; charset=utf-8"),
    "/nutrition_ui/style.css": ("nutrition_ui/style.css", "text/css; charset=utf-8"),
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

    def static_file(self, name: str, content_type: str):
        try:
            data = (STATIC / name).read_bytes()
        except FileNotFoundError:
            self.reply({"error": "Not found"}, 404)
        else:
            self.reply(data, content_type=content_type)

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
        elif parsed.path in STATIC_ROUTES:
            self.static_file(*STATIC_ROUTES[parsed.path])
        elif re.fullmatch(r"/images/[a-z0-9][a-z0-9.-]*\.webp", parsed.path):
            self.static_file(parsed.path[1:], "image/webp")
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
        if self.path != "/api/commands":
            self.reply({"error": "Not found"}, 404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 4096:
                raise ValueError("Invalid request size")
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                raise ValueError("JSON content type required")
            request = json.loads(self.rfile.read(length))
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


def serve(workspace: Path, config: dict, port: int):
    hub = ControlHub(workspace, config)
    server = HubServer(hub, port)
    print(f"Pet Meal Control Center: http://127.0.0.1:{server.server_address[1]}", flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        hub.close()
        server.server_close()
