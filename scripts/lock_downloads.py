"""Record official repository revisions without importing or executing their code."""

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPOS = {
    "GalbotSDK": "https://github.com/GalaxyGeneralRobotics/GalbotSDK.git",
    "galbot-mcap2lerobot": "https://github.com/GalaxyGeneralRobotics/galbot-mcap2lerobot.git",
    "galbot_one_golf_description": "https://github.com/GalaxyGeneralRobotics/galbot_one_golf_description.git",
    "lerobot": "https://github.com/huggingface/lerobot.git",
}


def git(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def main():
    repos = []
    for name, url in REPOS.items():
        path = ROOT / "external" / name
        revision = git(path, "rev-parse", "HEAD")
        files = list(path.rglob("*"))
        size = sum(p.stat().st_size for p in files if p.is_file())
        license_path = path / "LICENSE"
        repos.append(
            {
                "name": name,
                "url": url,
                "commit": revision,
                "checkout": path.relative_to(ROOT).as_posix(),
                "total_bytes_including_git": size,
                "tracked_files": len(git(path, "ls-files").splitlines()),
                "license_sha256": hashlib.sha256(license_path.read_bytes()).hexdigest()
                if license_path.is_file()
                else None,
                "working_tree_clean": not bool(git(path, "status", "--porcelain")),
            }
        )
    (ROOT / "external.lock.json").write_text(
        json.dumps(
            {"downloaded_at_utc": datetime.now(timezone.utc).isoformat(), "repositories": repos},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    packages = {}
    for name in ("numpy", "opencv-python", "torch", "pytest", "ruff", "ultralytics"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    (ROOT / "artifacts/environment.json").write_text(
        json.dumps(
            {
                "python": sys.version,
                "platform": platform.platform(),
                "packages": packages,
                "venv_reuses_system_packages": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(repos, indent=2))


if __name__ == "__main__":
    main()
