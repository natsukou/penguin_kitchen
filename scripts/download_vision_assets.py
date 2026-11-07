"""Restore public model and sample assets from the tracked lock file."""

import hashlib
import json
import urllib.request
from pathlib import Path
from tempfile import NamedTemporaryFile

ROOT = Path(__file__).resolve().parent.parent
LOCK = ROOT / "models/assets.lock.json"


def main():
    manifest = json.loads(LOCK.read_text(encoding="utf-8"))
    for asset in manifest:
        target = ROOT / asset["path"]
        already_present = target.exists()
        if already_present:
            data = target.read_bytes()
        else:
            with urllib.request.urlopen(asset["source"], timeout=60) as response:
                data = response.read()
        if len(data) != asset["size_bytes"] or hashlib.sha256(data).hexdigest() != asset["sha256"]:
            raise RuntimeError(f"Asset differs from locked size or SHA256: {target}")
        if not already_present:
            with NamedTemporaryFile(dir=target.parent, delete=False) as download:
                temporary = Path(download.name)
            try:
                temporary.write_bytes(data)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
