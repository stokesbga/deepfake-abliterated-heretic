"""Download only the needed weights, pinned to upstream revisions and SHA-256."""

import hashlib
import json
import shutil
import tempfile
import urllib.request
from importlib.resources import files
from pathlib import Path


def manifest() -> list[dict]:
    return json.loads(files("deepfakeheretic").joinpath("assets.json").read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_assets(directory: Path, *, hashes: bool = False) -> list[str]:
    problems = []
    for item in manifest():
        path = directory / item["name"]
        if not path.is_file():
            problems.append(f"Missing: {path}")
        elif path.stat().st_size != item["size"]:
            problems.append(f"Wrong size: {path}")
        elif hashes and sha256(path) != item["sha256"]:
            problems.append(f"Checksum mismatch: {path}")
    return problems


def download(directory: Path) -> None:
    from huggingface_hub import hf_hub_download

    directory.mkdir(parents=True, exist_ok=True)
    for item in manifest():
        target = directory / item["name"]
        if target.is_file() and sha256(target) == item["sha256"]:
            print(f"Verified {target.name}")
            continue
        print(f"Downloading {target.name} ({item['size'] / 1e9:.2f} GB)", flush=True)
        # Work on the destination filesystem so publication is atomic.
        with tempfile.TemporaryDirectory(prefix=".download-", dir=directory) as temp:
            staging = Path(temp) / target.name
            if "repo" in item:
                cached = hf_hub_download(
                    repo_id=item["repo"], filename=item["name"], revision=item["revision"]
                )
                shutil.copyfile(cached, staging)
            else:
                with (
                    urllib.request.urlopen(item["url"], timeout=120) as response,
                    staging.open("wb") as stream,
                ):
                    shutil.copyfileobj(response, stream)
            if staging.stat().st_size != item["size"] or sha256(staging) != item["sha256"]:
                raise RuntimeError(f"Integrity check failed for {target.name}; not installed.")
            staging.replace(target)
    print("All model assets verified.")
