#!/usr/bin/env python3
"""Fail closed when a public release contains private paths, secrets, or data."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAX_SOURCE_BYTES = 5 * 1024 * 1024
DISALLOWED_SUFFIXES = {
    ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".gpkg", ".shp", ".dbf",
    ".parquet", ".pt", ".pth", ".onnx", ".pkl", ".npy", ".npz", ".pyc",
}
TEXT_SUFFIXES = {
    ".py", ".md", ".txt", ".yaml", ".yml", ".json", ".toml", ".cff",
    ".csv", ".example", "",
}
DISALLOWED_TEXT = (
    re.compile("/" + "Users/", re.IGNORECASE),
    re.compile("/" + "Volumes/", re.IGNORECASE),
    re.compile("/" + "volume1/", re.IGNORECASE),
    re.compile("Google" + "Drive-", re.IGNORECASE),
    re.compile("sur" + "vey", re.IGNORECASE),
    re.compile("question" + "naire", re.IGNORECASE),
    re.compile("respond" + "ent", re.IGNORECASE),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"AIza[0-9A-Za-z_-]{30,}"),
    re.compile(r"gh[pousr]_[0-9A-Za-z]{30,}"),
)


def candidate_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, check=False, capture_output=True, text=True
    )
    if result.returncode == 0 and result.stdout.strip():
        return [ROOT / line for line in result.stdout.splitlines() if line.strip()]
    excluded = {".git", "data", "outputs", "__pycache__", ".pytest_cache", ".ruff_cache"}
    return sorted(
        path for path in ROOT.rglob("*")
        if path.is_file()
        and path.name != ".DS_Store"
        and not path.name.startswith("._")
        and not excluded.intersection(path.relative_to(ROOT).parts)
    )


def main() -> int:
    failures: list[str] = []
    for path in candidate_files():
        relative = path.relative_to(ROOT)
        if not path.is_file():
            failures.append(f"tracked path is missing or not a file: {relative}")
            continue
        if path.suffix.lower() in DISALLOWED_SUFFIXES:
            failures.append(f"data/model/generated file is not allowed: {relative}")
        if path.stat().st_size > MAX_SOURCE_BYTES:
            failures.append(f"source file exceeds 5 MiB: {relative}")
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {"LICENSE", "Makefile"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            failures.append(f"text-like file is not UTF-8: {relative}")
            continue
        for pattern in DISALLOWED_TEXT:
            if pattern.search(text):
                failures.append(f"disallowed public-release text in {relative}: {pattern.pattern}")
    license_path = ROOT / "LICENSE"
    if not license_path.is_file() or "GNU AFFERO GENERAL PUBLIC LICENSE" not in license_path.read_text(
        encoding="utf-8"
    ):
        failures.append("LICENSE is not the GNU Affero General Public License text")
    if failures:
        print("Public-release audit failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    print(f"Public-release audit passed ({len(candidate_files())} files checked).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
