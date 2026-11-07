"""Reproduce software validation; no network, SDK import, or robot connection."""

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from pet_meal.decision import TaskRunner
from pet_meal.recording import Recorder
from pet_meal.simulator import FAULTS, Simulator

ROOT = Path(__file__).resolve().parents[1]


def main():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    config = json.loads((ROOT / "configs/simulation.json").read_text())
    suite_dir = ROOT / "artifacts/runs" / f"validation-{stamp}"
    results = []
    for fault in FAULTS:
        output = suite_dir / fault
        backend = Simulator(config, fault, output / "frames")
        recorder = Recorder(output, config, backend.evidence_level)
        try:
            recorder.event("scenario", {"fault": fault})
            summary = TaskRunner(backend, config, recorder).run()
        finally:
            recorder.close()
        passed = summary["success"] == (fault == "none") and backend.stopped
        results.append(
            {"scenario": fault, "passed": passed, "summary": summary, "artifacts": str(output)}
        )
    checks = {}
    for name, args in (
        ("lint", ["-m", "ruff", "check", "pet_meal", "scripts", "tests"]),
        ("format", ["-m", "ruff", "format", "--check", "pet_meal", "scripts", "tests"]),
        ("tests", ["-m", "pytest", "--junitxml=artifacts/pytest-results.xml"]),
    ):
        result = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True)
        checks[name] = {"passed": result.returncode == 0, "output": result.stdout + result.stderr}
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_level": "software_and_synthetic_scene_only",
        "all_passed": all(r["passed"] for r in results)
        and all(c["passed"] for c in checks.values()),
        "hardware_acceptance": "not_tested",
        "scenarios": results,
        "checks": checks,
    }
    (ROOT / "artifacts/validation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
