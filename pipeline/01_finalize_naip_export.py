#!/usr/bin/env python3
"""Finalize an existing Earth Engine export from its immutable submission record.

This command never submits a new export. It polls the exact Earth Engine task
ID captured by ``01_export_naip_earth_engine.py`` and writes the final outcome
manifest only after the task reaches a terminal state. This makes ``--no-wait``
and local polling timeouts resumable without mutating provenance records.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


WORKING_ROOT = Path(__file__).resolve().parents[1]
if str(WORKING_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKING_ROOT))

from project_paths import (  # noqa: E402
    PARKING_EE_RUN_MANIFEST,
    PARKING_EE_SUBMISSION_MANIFEST,
    WORKFLOW_ROOT,
)
from pipeline.parking_common import (  # noqa: E402
    ContractError,
    load_run_manifest,
    refuse_existing_output,
    resolve_manifest_artifact_path,
    verify_manifest_artifact,
    write_run_manifest,
)


TERMINAL_TASK_STATES = frozenset({"COMPLETED", "FAILED", "CANCELLED", "CANCEL_REQUESTED"})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--submission-manifest",
        type=Path,
        default=PARKING_EE_SUBMISSION_MANIFEST,
        help="Immutable manifest produced immediately after task submission.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PARKING_EE_RUN_MANIFEST,
        help="New terminal-outcome manifest consumed by the mosaic stage.",
    )
    parser.add_argument("--project", default=os.environ.get("GEE_PROJECT_ID"))
    parser.add_argument(
        "--local-export-dir",
        type=Path,
        default=None,
        help="Optional relocated synced-Drive directory; the recorded filename prefix remains immutable.",
    )
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--timeout-seconds", type=float, default=86_400.0, help="Use 0 for no timeout.")
    return parser


def _task_status(ee: Any, task_id: str) -> dict[str, Any]:
    response = ee.data.getTaskStatus(task_id)
    if isinstance(response, list):
        if len(response) != 1 or not isinstance(response[0], Mapping):
            raise ContractError(f"Earth Engine returned no unique status for task {task_id!r}.")
        return dict(response[0])
    if not isinstance(response, Mapping):
        raise ContractError(f"Earth Engine returned an unreadable status for task {task_id!r}.")
    return dict(response)


def _wait_for_task(
    ee: Any,
    task_id: str,
    *,
    poll_seconds: float,
    timeout_seconds: float,
) -> dict[str, Any]:
    if poll_seconds <= 0:
        raise ValueError("--poll-seconds must be positive.")
    if timeout_seconds < 0:
        raise ValueError("--timeout-seconds cannot be negative.")
    started = time.monotonic()
    while True:
        status = _task_status(ee, task_id)
        state = str(status.get("state", "UNKNOWN")).upper()
        print(f"Earth Engine task {task_id}: {state}", flush=True)
        if state in TERMINAL_TASK_STATES:
            return status
        if timeout_seconds and time.monotonic() - started >= timeout_seconds:
            raise TimeoutError(
                "Earth Engine task remains nonterminal; the submission manifest is intact. "
                "Run this finalizer again later."
            )
        time.sleep(poll_seconds)


def _recorded_prefix(payload: Mapping[str, Any]) -> Path:
    raw = payload.get("extra", {}).get("expected_local_prefix")
    if not isinstance(raw, str) or not raw.strip():
        raise ContractError("Submission manifest has no recorded local export prefix.")
    prefix = Path(raw)
    if prefix.is_absolute():
        return prefix.resolve(strict=False)
    manifest_path = Path(str(payload.get("_manifest_path", "")))
    if not manifest_path.is_absolute():
        raise ContractError("Submission manifest runtime location is unavailable.")
    return (manifest_path.parent / prefix).resolve(strict=False)


def _export_family(directory: Path, prefix: str) -> list[Path]:
    exact = directory / f"{prefix}.tif"
    shards = sorted(path for path in directory.glob(f"{prefix}-*.tif") if path.is_file())
    return ([exact] if exact.is_file() else []) + shards


def run(args: argparse.Namespace) -> int:
    submission_manifest = args.submission_manifest.expanduser().resolve(strict=False)
    final_manifest = args.manifest.expanduser().resolve(strict=False)
    if submission_manifest == final_manifest:
        raise ValueError("--submission-manifest and --manifest must differ.")
    refuse_existing_output(final_manifest)
    payload = load_run_manifest(
        submission_manifest,
        expected_stage="01_export_naip_earth_engine_submission",
    )
    contract = str(payload.get("contract", ""))
    if contract not in {"naip-unit-scaled-rgb", "naip-unit-scaled-rgbnir"}:
        raise ContractError(f"Unsupported Earth Engine submission contract: {contract!r}.")
    if contract == "naip-unit-scaled-rgb" and final_manifest == PARKING_EE_RUN_MANIFEST.resolve(strict=False):
        raise ContractError(
            "Historical RGB finalization requires an explicit isolated --manifest; the corrected RGB+NIR "
            "terminal-manifest default is reserved."
        )
    submission_status = payload.get("extra", {}).get("task_status_at_submission")
    if not isinstance(submission_status, Mapping) or not submission_status.get("id"):
        raise ContractError("Submission manifest does not preserve an Earth Engine task ID.")
    task_id = str(submission_status["id"])
    input_records = payload.get("inputs")
    if not isinstance(input_records, list) or len(input_records) != 1:
        raise ContractError("Submission manifest must identify exactly one ROI input.")
    roi_path = resolve_manifest_artifact_path(payload, input_records[0])
    verify_manifest_artifact(payload, roi_path, section="inputs")

    recorded_prefix = _recorded_prefix(payload)
    local_export_dir = (
        args.local_export_dir.expanduser().resolve(strict=False)
        if args.local_export_dir is not None
        else recorded_prefix.parent
    )
    prefix = recorded_prefix.name

    try:
        import ee
    except ModuleNotFoundError as exc:
        raise RuntimeError("Finalizing an Earth Engine task requires earthengine-api.") from exc
    try:
        if args.project:
            ee.Initialize(project=args.project)
        else:
            ee.Initialize()
    except Exception as exc:
        raise RuntimeError(
            "Earth Engine initialization failed. Authenticate with the CLI and pass --project if needed."
        ) from exc

    status = _wait_for_task(
        ee,
        task_id,
        poll_seconds=args.poll_seconds,
        timeout_seconds=args.timeout_seconds,
    )
    state = str(status.get("state", "")).upper()
    exported_files = _export_family(local_export_dir, prefix)
    submission_extra = dict(payload.get("extra", {}))
    submission_extra.pop("task_status_at_submission", None)
    write_run_manifest(
        final_manifest,
        workflow_root=WORKFLOW_ROOT,
        stage="01_export_naip_earth_engine",
        contract=contract,
        arguments=payload.get("arguments", {}),
        inputs=(roi_path, submission_manifest),
        # Drive may expose only part of a shard family when the remote task
        # becomes terminal. The mosaic stage therefore owns the first stable
        # local inventory and checksum binding.
        outputs=(),
        extra={
            **submission_extra,
            "task_status": status,
            "run_outcome": "completed" if state == "COMPLETED" else "failed",
            "local_export_count": len(exported_files),
            "shard_inventory_policy": "deferred-to-mosaic-after-drive-sync",
            "submission_manifest": submission_manifest,
            "finalized_from_task_id": task_id,
            "finalizer_arguments": vars(args),
            "arguments_source": submission_manifest,
        },
    )
    print(f"Wrote terminal Earth Engine manifest: {final_manifest}")
    if state != "COMPLETED":
        raise ContractError(f"Earth Engine export ended in {state}: {status}")
    if not exported_files:
        print("Task completed, but synced Drive shards are not yet locally visible.")
    else:
        print(
            f"Observed {len(exported_files)} local TIFF(s); their authoritative inventory is deferred "
            "to the mosaic stage after Drive synchronization."
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run(build_parser().parse_args(argv))
    except (ContractError, FileExistsError, FileNotFoundError, RuntimeError, TimeoutError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
