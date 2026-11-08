"""Hub transport for the G1 HPU worker. SSH authenticates the deployed worker."""

import json
import posixpath
import queue
import re
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from .types import TaskError

WIRE = b"PETMEAL-G1/1 "
MAX_FRAME = 65536
ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
SHA = re.compile(r"[0-9a-f]{64}")
RUN_ID = re.compile(r"[0-9a-f]{32}")


def encode_frame(value):
    return WIRE + json.dumps(value, allow_nan=False, separators=(",", ":")).encode() + b"\n"


def validate_execution(request):
    fields = {"run_id": RUN_ID, "plan_id": ID, "plan_sha256": SHA, "robot_sn": ID}
    if (
        not isinstance(request, dict)
        or request.keys() != fields.keys()
        or any(
            not isinstance(request[key], str) or not pattern.fullmatch(request[key])
            for key, pattern in fields.items()
        )
    ):
        raise TaskError("Expected a run ID, HPU plan ID, SHA-256 and robot serial")


class HpuClient:
    def __init__(self, host, profile, *, password=None, xcu_password=None):
        if not re.fullmatch(
            r"(?:[A-Za-z0-9][A-Za-z0-9_.-]*@)?[A-Za-z0-9][A-Za-z0-9_.-]{0,127}",
            host,
        ):
            raise TaskError("Invalid HPU SSH target")
        if not re.fullmatch(r"/[A-Za-z0-9_./-]+", profile):
            raise TaskError("HPU profile must be an absolute Linux path without spaces")
        self.host = host
        self.password = password
        self.xcu_password = xcu_password
        self.remote = ["sh", posixpath.join(posixpath.dirname(profile), "launch.sh"), profile]
        self.command = [
            "ssh",
            "-T",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=5",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            "ServerAliveInterval=2",
            "-o",
            "ServerAliveCountMax=2",
            host,
            *self.remote,
        ]

    @contextmanager
    def _connection(self):
        if self.password is not None:
            import paramiko

            user, _, host = self.host.rpartition("@")
            with paramiko.SSHClient() as client:
                client.load_host_keys(str(Path.home() / ".ssh" / "known_hosts"))
                client.connect(
                    host,
                    username=user or None,
                    password=self.password,
                    timeout=5,
                    auth_timeout=5,
                    allow_agent=False,
                    look_for_keys=False,
                )
                channel = client.get_transport().open_session(timeout=5)
                try:
                    channel.set_combine_stderr(True)
                    channel.exec_command(" ".join(self.remote))
                    with channel.makefile_stdin("wb") as writer, channel.makefile("rb") as reader:
                        yield writer, reader
                finally:
                    channel.close()
            return
        process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        try:
            yield process.stdin, process.stdout
        finally:
            try:
                process.stdin.close()
            except OSError:
                pass
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            process.stdout.close()

    def probe(self):
        return self._exchange({"operation": "probe"}, lambda *_: None, lambda: False, 20)

    def execute(self, request, emit, stopped):
        validate_execution(request)
        return self._exchange({"operation": "execute", **request}, emit, stopped, 300)

    def _exchange(self, request, emit, stopped, timeout_s):
        with self._connection() as (writer, reader):
            return self._session(request, emit, stopped, timeout_s, writer, reader)

    def _session(self, request, emit, stopped, timeout_s, writer, reader):
        incoming = queue.Queue()

        def receive():
            try:
                while line := reader.readline(MAX_FRAME + 1):
                    if len(line) > MAX_FRAME:
                        incoming.put(TaskError("HPU returned an oversized frame"))
                        return
                    if line.startswith(WIRE):
                        incoming.put(json.loads(line[len(WIRE) :]))
            except (OSError, ValueError) as exc:
                incoming.put(TaskError(f"Invalid HPU reply: {exc}"))
            finally:
                incoming.put(None)

        threading.Thread(target=receive, daemon=True).start()
        deadline = time.monotonic() + timeout_s
        stop_sent = False

        def send(row):
            try:
                writer.write(json.dumps(row, allow_nan=False).encode() + b"\n")
                writer.flush()
            except OSError:
                # A closing worker may still have its terminal receipt in stdout.
                pass

        if self.xcu_password is not None:
            send({**request, "xcu_password": self.xcu_password})
        else:
            send(request)
        next_ping = time.monotonic()
        while time.monotonic() < deadline:
            now = time.monotonic()
            if stopped() and not stop_sent:
                send({"operation": "cancel"})
                stop_sent = True
                deadline = min(deadline, now + 12)
            if request["operation"] == "execute" and not stop_sent and now >= next_ping:
                send({"operation": "heartbeat"})
                next_ping = now + 0.25
            try:
                row = incoming.get(timeout=0.25)
            except queue.Empty:
                continue
            if isinstance(row, TaskError):
                raise row
            if row is None:
                break
            if not isinstance(row, dict):
                raise TaskError("HPU frame must be a JSON object")
            if request["operation"] == "execute" and row.get("run_id") != request["run_id"]:
                raise TaskError("HPU reply belongs to a different run")
            if row.get("type") == "result" and isinstance(row.get("result"), dict):
                return row["result"]
            if (
                row.get("type") == "event"
                and isinstance(row.get("event"), str)
                and row["event"].startswith("hpu_")
                and isinstance(row.get("payload"), dict)
            ):
                emit(row["event"], row["payload"])
            else:
                raise TaskError("Unexpected HPU response frame")
        raise TaskError("HPU session ended without a terminal receipt")
