#!/usr/bin/env python3
"""Build a portable, checksum-bound manifest for a camera archive.

This is the recommended entry point for a new study. The exact historical
Payette inventory-construction code did not survive, so manuscript replay must
start from its frozen daylight manifest instead of using this command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
DEFAULT_TIMESTAMP_REGEX = r"(?P<timestamp>20\d{6}[_-]\d{6})"
DEFAULT_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def image_is_readable(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.load()
        return True
    except Exception:
        return False


def infer_site(relative_path: Path) -> tuple[str, str]:
    """Return camera type and site from ``type/site/.../image`` layout."""

    parts = relative_path.parts
    lower = [part.lower() for part in parts]
    for camera_type in ("parking", "river"):
        if camera_type in lower:
            index = lower.index(camera_type)
            site = parts[index + 1] if index + 1 < len(parts) - 1 else "unknown"
            return camera_type, site
    return "unknown", relative_path.parent.name or "unknown"


def parse_timestamp(name: str, pattern: re.Pattern[str], timestamp_format: str) -> pd.Timestamp | None:
    match = pattern.search(name)
    if not match:
        return None
    token = match.groupdict().get("timestamp") or match.group(0)
    if timestamp_format == DEFAULT_TIMESTAMP_FORMAT and len(token) == 15:
        token = token[:8] + "_" + token[9:]
    try:
        return pd.Timestamp(datetime.strptime(token, timestamp_format))
    except ValueError:
        return None


def write_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        frame.to_csv(path, index=False)
    elif path.suffix.lower() in {".parquet", ".pq"}:
        frame.to_parquet(path, index=False)
    else:
        raise ValueError("--output must end in .csv or .parquet")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Archive root; paths are stored relative to it.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timestamp-regex", default=DEFAULT_TIMESTAMP_REGEX)
    parser.add_argument("--timestamp-format", default=DEFAULT_TIMESTAMP_FORMAT)
    parser.add_argument("--timezone", default="America/Denver", help="IANA timezone for parsed local timestamps.")
    parser.add_argument(
        "--daylight-hours",
        default="",
        metavar="START:END",
        help="Optional local-hour screen such as 06:21; omit to retain every readable image.",
    )
    parser.add_argument(
        "--hash-content",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Calculate SHA-256 for each readable image (default: enabled).",
    )
    return parser


def _daylight_bounds(value: str) -> tuple[int, int] | None:
    if not value:
        return None
    try:
        start, end = (int(item) for item in value.split(":"))
    except (TypeError, ValueError) as exc:
        raise ValueError("--daylight-hours must be START:END using integer hours") from exc
    if not (0 <= start <= 23 and 1 <= end <= 24 and start < end):
        raise ValueError("--daylight-hours must satisfy 0 <= START < END <= 24")
    return start, end


def run(args: argparse.Namespace) -> int:
    root = args.root.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve(strict=False)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    pattern = re.compile(args.timestamp_regex)
    bounds = _daylight_bounds(args.daylight_hours)

    rows: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name.startswith(".") or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        relative = path.relative_to(root)
        camera_type, site = infer_site(relative)
        timestamp = parse_timestamp(path.name, pattern, args.timestamp_format)
        if timestamp is not None:
            timestamp = timestamp.tz_localize(args.timezone, ambiguous="NaT", nonexistent="NaT")
        readable = image_is_readable(path)
        daylight = bool(
            readable
            and timestamp is not None
            and not pd.isna(timestamp)
            and (bounds is None or bounds[0] <= int(timestamp.hour) < bounds[1])
        )
        rows.append(
            {
                "image_id": hashlib.sha256(relative.as_posix().encode("utf-8")).hexdigest()[:20],
                "path": relative.as_posix(),
                "camera_type": camera_type,
                "site_id": site,
                "captured_at": timestamp.isoformat() if timestamp is not None and not pd.isna(timestamp) else None,
                "acquired": True,
                "readable": readable,
                "daylight": daylight,
                "file_size": path.stat().st_size,
                "content_sha256": sha256_file(path) if readable and args.hash_content else None,
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError(f"No supported image files found beneath {root}")
    if frame["image_id"].duplicated().any():
        raise RuntimeError("Generated image_id values are not unique")
    write_table(frame, output)
    provenance = {
        "stage": "06_build_camera_manifest",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "archive_root_recorded": False,
        "rows": int(len(frame)),
        "readable_rows": int(frame["readable"].sum()),
        "daylight_rows": int(frame["daylight"].sum()),
        "timestamp_regex": args.timestamp_regex,
        "timestamp_format": args.timestamp_format,
        "timezone": args.timezone,
        "daylight_hours": args.daylight_hours or None,
        "hash_content": bool(args.hash_content),
        "output_sha256": sha256_file(output),
    }
    output.with_suffix(output.suffix + ".run.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), "rows": len(frame)}, indent=2))
    return 0


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
