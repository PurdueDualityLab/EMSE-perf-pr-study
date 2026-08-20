"""Download and verify the exact PerfMiner model bundle from its Figshare archive."""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

from perfminer_reproduction import PERFMINER_MODEL_FILE_SHA256, verify_figshare_model


ARCHIVE_URL = "https://ndownloader.figshare.com/files/64155001"


def download(url: str, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with urllib.request.urlopen(url) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)
    os.replace(temporary, destination)


def extract_model(archive: Path, output_dir: Path) -> None:
    expected = set(PERFMINER_MODEL_FILE_SHA256)
    with zipfile.ZipFile(archive) as package:
        grouped: dict[PurePosixPath, set[str]] = {}
        for info in package.infolist():
            path = PurePosixPath(info.filename)
            if not info.is_dir() and path.name in expected:
                grouped.setdefault(path.parent, set()).add(path.name)
        roots = [root for root, names in grouped.items() if names == expected]
        if len(roots) != 1:
            raise ValueError(f"Expected exactly one complete model directory in {archive}, found {roots}")
        with tempfile.TemporaryDirectory(dir=output_dir.parent) as temporary:
            staging = Path(temporary) / "model"
            staging.mkdir()
            for filename in expected:
                source = str(roots[0] / filename)
                with package.open(source) as input_file, (staging / filename).open("wb") as output_file:
                    shutil.copyfileobj(input_file, output_file, length=1024 * 1024)
            verify_figshare_model(staging)
            if output_dir.exists():
                shutil.rmtree(output_dir)
            os.replace(staging, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the verified PerfMiner Figshare model.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    try:
        verify_figshare_model(args.output_dir)
    except (FileNotFoundError, ValueError):
        args.output_dir.parent.mkdir(parents=True, exist_ok=True)
        archive = args.archive or args.output_dir.parent / "replication_package.zip"
        if not archive.is_file():
            download(ARCHIVE_URL, archive)
        extract_model(archive, args.output_dir)
        verify_figshare_model(args.output_dir)
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()
