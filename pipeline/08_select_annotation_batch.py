

"""Active-learning batch selection for YOLO detections.

Reads a detections parquet produced by `yolo1.py` (sampled images only) and selects a
labeling batch using a mix of "most informative" heuristics:

A) Uncertainty sampling
   - many detections near decision boundary (conf in [low, high])
   - low max confidence but nonzero detections
   - high entropy over predicted class labels (proxy; true class-prob entropy not available)

B) Diversity sampling
   - stratify by site, hour-of-day, and detection density

C) Hard negatives
   - images likely to contain false positives (many low/borderline conf detections)
   - suspicious context rules (e.g., boat predicted in Parking)

D) Zero-case coverage
   - include diverse true-zero predictions to stabilize false positives

Outputs
-------
- <out_dir>/batch_manifest.parquet : selected rows with scores + detections
- <out_dir>/paths.txt             : selected absolute paths (one per line)
- <out_dir>/summary.json          : selection summary and parameters

Example
-------
python labeling_sample.py \
  --detections "../data_intermediate/Manifest/image_manifest_yolo_daytime_yolo26_sample_detections.parquet" \
  --batch-size 500 \
  --out-dir "./al_batches/batch_001" \
  --verbose

Optional dedup (opens images):
python labeling_sample.py --dedup-hash --hash-threshold 6 --verbose
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import time
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from project_paths import AL_BATCHES_ROOT, BASELINE_DETECTIONS_MANIFEST, TRAINING_ROOT

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore

# Optional, only used when --dedup-hash is enabled
try:
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover
    Image = None  # type: ignore


# -----------------------------
# Helpers
# -----------------------------

def _fmt_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    m, s = divmod(int(seconds + 0.5), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:d}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"{m:02d}m {s:02d}s"
    return f"{s:d}s"


def _iter(items: Sequence, desc: str, enabled: bool):
    if not enabled:
        for x in items:
            yield x
        return
    if tqdm is not None:
        for x in tqdm(items, total=len(items), desc=desc, unit="item"):
            yield x
    else:
        n = len(items)
        step = max(200, n // 50) if n else 1
        t0 = time.perf_counter()
        for i, x in enumerate(items, start=1):
            if i == 1 or i % step == 0 or i == n:
                elapsed = time.perf_counter() - t0
                rate = i / elapsed if elapsed > 0 else 0.0
                remain = (n - i) / rate if rate > 0 else 0.0
                print(f"{desc}: {i}/{n} ({rate:.1f}/s, ETA {_fmt_seconds(remain)})")
            yield x


def safe_list_of_dicts(x: object) -> List[Dict[str, object]]:
    """Normalize detections from parquet, Python, or serialized values."""
    if x is None or (isinstance(x, (float, np.floating)) and math.isnan(float(x))):
        return []
    # PyArrow-backed parquet columns commonly deserialize nested records as an
    # object ndarray rather than the list written by the inference stage.
    if isinstance(x, np.ndarray):
        x = x.tolist()
    if isinstance(x, list):
        out: List[Dict[str, object]] = []
        for item in x:
            if isinstance(item, dict):
                out.append(item)
        return out
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "[]"}:
        return []
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, list):
            out = [d for d in obj if isinstance(d, dict)]
            return out
    except Exception:
        return []
    return []


def extract_site_from_path(path: str) -> str:
    """Best-effort site extraction from a filesystem path."""
    p = Path(str(path))
    parts = list(p.parts)
    for anchor in ("Parking", "River"):
        if anchor in parts:
            i = parts.index(anchor)
            if i + 1 < len(parts):
                return f"{anchor}/{parts[i+1]}"
            return anchor
    parent = p.parent
    if parent.name and parent.parent.name:
        return f"{parent.parent.name}/{parent.name}"
    if parent.name:
        return parent.name
    return "unknown"


def shannon_entropy_from_counts(counts: Dict[str, int], eps: float = 1e-12) -> float:
    """Entropy over discrete class-count distribution (proxy for class uncertainty)."""
    if not counts:
        return 0.0
    vals = np.asarray([max(0, int(v)) for v in counts.values()], dtype=float)
    s = float(vals.sum())
    if s <= 0:
        return 0.0
    p = vals / (s + eps)
    p = np.clip(p, eps, 1.0)
    return float(-(p * np.log(p)).sum())


def compute_image_hash_ahash(path: str, size: int = 8) -> Optional[int]:
    """Compute a simple average-hash (aHash) as an integer.

    Requires Pillow. Opens image, converts to grayscale, resizes to (size,size), and compares to mean.
    """
    if Image is None:
        return None
    try:
        with Image.open(path) as im:
            im = im.convert("L")
            im = im.resize((size, size))
            arr = np.asarray(im, dtype=np.float32)
            mean = float(arr.mean())
            bits = (arr > mean).astype(np.uint8).flatten()
            # pack bits into int
            h = 0
            for b in bits:
                h = (h << 1) | int(b)
            return int(h)
    except Exception:
        return None


def hamming_distance_int(a: int, b: int) -> int:
    return int((a ^ b).bit_count())

def build_copy_destination(copy_root: Path, copy_structure: str, site: str, date_obj: object, filename: str) -> Path:
    """Build a deterministic destination path for a copied image."""
    if copy_structure == "flat":
        return copy_root / filename

    parts = str(site).split("/")
    if len(parts) >= 2:
        anchor, subsite = parts[0], parts[1]
    else:
        anchor, subsite = (parts[0] if parts else "unknown"), "unknown"

    if copy_structure == "site":
        return copy_root / anchor / subsite / filename

    # site_date
    day = "unknown-date"
    try:
        if date_obj is not None and not pd.isna(date_obj):
            day = str(date_obj)
    except Exception:
        day = "unknown-date"

    return copy_root / anchor / subsite / day / filename


def copy_batch_images(batch: pd.DataFrame, copy_to: str, copy_structure: str, verbose: bool) -> pd.DataFrame:
    """Copy selected images to a destination folder for labeling.

    Adds columns:
    - copied_path: destination path
    - copy_ok: 1/0
    - copy_error: error message if any
    """
    copy_root = Path(copy_to).expanduser().resolve()
    copy_root.mkdir(parents=True, exist_ok=True)

    copied_path: List[str] = []
    copy_ok: List[int] = []
    copy_error: List[str] = []

    if "site" not in batch.columns:
        batch = batch.copy()
        batch["site"] = [extract_site_from_path(p) for p in batch["path"].astype(str).tolist()]

    if "filename" not in batch.columns:
        batch = batch.copy()
        batch["filename"] = [Path(p).name for p in batch["path"].astype(str).tolist()]

    date_col = "date" if "date" in batch.columns else None

    paths = batch["path"].astype(str).tolist()
    sites = batch["site"].astype(str).tolist()
    fns = batch["filename"].astype(str).tolist()
    dates = batch[date_col].tolist() if date_col else [None] * len(batch)

    for p, site, d, fn in zip(_iter(paths, "Copying images", enabled=verbose), sites, dates, fns):
        try:
            src = Path(p)
            dst = build_copy_destination(copy_root, copy_structure, site=site, date_obj=d, filename=fn)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied_path.append(str(dst))
            copy_ok.append(1)
            copy_error.append("")
        except Exception as e:
            copied_path.append("")
            copy_ok.append(0)
            copy_error.append(str(e))

    out = batch.copy()
    out["copied_path"] = copied_path
    out["copy_ok"] = copy_ok
    out["copy_error"] = copy_error
    return out

# -----------------------------
# Scoring
# -----------------------------

@dataclass
class SelectionConfig:
    detections_path: str
    batch_size: int
    conf_low: float
    conf_high: float
    quota_zero: int
    quota_uncertainty: int
    quota_hardneg: int
    quota_diversity: int
    per_stratum_cap: int
    seed: int
    dedup_hash: bool
    hash_threshold: int


def add_scores(df: pd.DataFrame, conf_low: float, conf_high: float, verbose: bool) -> pd.DataFrame:
    """Add per-image heuristics-based scores."""

    # Ensure essential columns
    if "site" not in df.columns:
        df["site"] = [extract_site_from_path(p) for p in df["path"].astype(str).tolist()]

    if "datetime_from_filename" in df.columns:
        dt = pd.to_datetime(df["datetime_from_filename"], errors="coerce")
        df["hour"] = dt.dt.hour
        df["month"] = dt.dt.month
        df["date"] = dt.dt.date
    else:
        df["hour"] = np.nan
        df["month"] = np.nan
        df["date"] = np.nan

    dets_col = "detections"
    if dets_col not in df.columns:
        raise ValueError("Input parquet must contain a 'detections' column.")

    # Compute metrics row-wise (sample size is typically manageable, but keep it efficient)
    n_det: List[int] = []
    max_conf: List[float] = []
    mean_conf: List[float] = []
    n_border: List[int] = []
    border_frac: List[float] = []
    class_entropy: List[float] = []
    boat_in_parking: List[int] = []

    for x, site in zip(_iter(df[dets_col].tolist(), "Scoring images", enabled=verbose), df["site"].astype(str).tolist()):
        dets = safe_list_of_dicts(x)
        n = len(dets)
        n_det.append(int(n))

        if n == 0:
            max_conf.append(float("nan"))
            mean_conf.append(float("nan"))
            n_border.append(0)
            border_frac.append(0.0)
            class_entropy.append(0.0)
            boat_in_parking.append(0)
            continue

        confs = []
        counts: Dict[str, int] = {}
        boat_flag = 0
        for d in dets:
            try:
                cf = float(d.get("conf", float("nan")))
            except Exception:
                cf = float("nan")
            if math.isfinite(cf):
                confs.append(cf)

            cls_name = str(d.get("class_name", ""))
            if cls_name:
                counts[cls_name] = counts.get(cls_name, 0) + 1

            if cls_name == "boat" and site.startswith("Parking"):
                boat_flag = 1

        if len(confs) == 0:
            max_conf.append(float("nan"))
            mean_conf.append(float("nan"))
            nb = 0
        else:
            max_conf.append(float(np.max(confs)))
            mean_conf.append(float(np.mean(confs)))
            nb = int(np.sum((np.asarray(confs) >= conf_low) & (np.asarray(confs) <= conf_high)))

        n_border.append(nb)
        border_frac.append(float(nb) / float(max(n, 1)))
        class_entropy.append(shannon_entropy_from_counts(counts))
        boat_in_parking.append(int(boat_flag))

    df = df.copy()
    df["n_det"] = n_det
    df["max_conf"] = max_conf
    df["mean_conf"] = mean_conf
    df["n_borderline"] = n_border
    df["borderline_frac"] = border_frac
    df["class_entropy"] = class_entropy
    df["boat_in_parking"] = boat_in_parking

    # Uncertainty score (higher => more informative)
    # - emphasize borderline density
    # - penalize high max confidence
    # - incorporate class-entropy proxy
    # Note: no per-class probability vectors, so entropy is over predicted labels.
    mc = pd.to_numeric(df["max_conf"], errors="coerce").fillna(0.0)
    mnc = pd.to_numeric(df["mean_conf"], errors="coerce").fillna(0.0)
    nb = pd.to_numeric(df["n_borderline"], errors="coerce").fillna(0.0)
    bf = pd.to_numeric(df["borderline_frac"], errors="coerce").fillna(0.0)
    ce = pd.to_numeric(df["class_entropy"], errors="coerce").fillna(0.0)

    # Normalize components roughly
    nb_norm = nb / (nb.quantile(0.95) + 1e-9)
    ce_norm = ce / (ce.quantile(0.95) + 1e-9)

    df["uncertainty_score"] = (
        1.2 * nb_norm
        + 1.0 * bf
        + 0.8 * (1.0 - mc)
        + 0.5 * (1.0 - mnc)
        + 0.6 * ce_norm
    )

    # Hard negative score: heuristic flags
    # - many borderline detections
    # - low max confidence but nonzero detections
    # - high density with low average confidence
    # - suspicious boat in Parking
    df["hardneg_score"] = 0
    df.loc[(df["n_det"] > 0) & (df["max_conf"] < max(0.35, conf_high)), "hardneg_score"] += 1
    df.loc[df["n_borderline"] >= 3, "hardneg_score"] += 1
    df.loc[(df["borderline_frac"] >= 0.6) & (df["n_det"] >= 3), "hardneg_score"] += 1
    df.loc[(df["n_det"] >= 10) & (df["mean_conf"] < 0.4), "hardneg_score"] += 1
    df.loc[df["boat_in_parking"] == 1, "hardneg_score"] += 2  # strong indicator

    # Detection density bin for stratification
    def _density_bin(n: int) -> str:
        if n <= 0:
            return "0"
        if n <= 2:
            return "1-2"
        if n <= 5:
            return "3-5"
        return "6+"

    df["density_bin"] = [ _density_bin(int(x)) for x in df["n_det"].tolist() ]

    # Hour bin (more stable than raw hour if you want bigger strata)
    def _hour_bin(h: object) -> str:
        try:
            if pd.isna(h):
                return "unknown"
            hh = int(h)
        except Exception:
            return "unknown"
        # daylight-focused bins (adjust as needed)
        if 5 <= hh <= 8:
            return "dawn"
        if 9 <= hh <= 12:
            return "morning"
        if 13 <= hh <= 16:
            return "afternoon"
        if 17 <= hh <= 20:
            return "dusk"
        return "night"

    df["hour_bin"] = [_hour_bin(h) for h in df["hour"].tolist()]

    # Primary stratum key
    df["stratum"] = df["site"].astype(str) + "|" + df["hour_bin"].astype(str) + "|" + df["density_bin"].astype(str)

    return df


# -----------------------------
# Selection
# -----------------------------

def _stratified_sample(df: pd.DataFrame, n: int, seed: int, per_stratum_cap: int) -> pd.DataFrame:
    """Sample up to n rows, roughly balanced across df['stratum'] with a cap."""
    if n <= 0 or len(df) == 0:
        return df.iloc[0:0].copy()

    rng = np.random.default_rng(seed)

    groups = list(df.groupby("stratum"))
    rng.shuffle(groups)

    picked_idx: List[int] = []
    # round-robin across strata
    while len(picked_idx) < n and len(groups) > 0:
        made_progress = False
        for _, g in groups:
            if len(picked_idx) >= n:
                break
            # cap per stratum
            already = 0
            if picked_idx:
                # count already selected in this stratum
                # (fast approximation: compute once per round using membership)
                pass
            # sample one from remaining within stratum
            candidates = g.index.difference(picked_idx)
            if len(candidates) == 0:
                continue

            # enforce cap
            if per_stratum_cap > 0:
                current = df.loc[picked_idx, "stratum"].value_counts().to_dict() if picked_idx else {}
                if current.get(g["stratum"].iloc[0], 0) >= per_stratum_cap:
                    continue

            chosen = int(rng.choice(candidates, size=1)[0])
            picked_idx.append(chosen)
            made_progress = True

        if not made_progress:
            break

    return df.loc[picked_idx].copy()


def select_batch(
    df: pd.DataFrame,
    cfg: SelectionConfig,
    verbose: bool,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Select a labeling batch using a mixture of heuristics."""

    # Ensure quotas sum to batch size (adjust if needed)
    quotas = {
        "zero": int(cfg.quota_zero),
        "uncertainty": int(cfg.quota_uncertainty),
        "hardneg": int(cfg.quota_hardneg),
        "diversity": int(cfg.quota_diversity),
    }
    q_sum = sum(quotas.values())
    if q_sum != cfg.batch_size:
        # normalize: keep relative proportions
        if q_sum <= 0:
            quotas = {k: 0 for k in quotas}
            quotas["diversity"] = cfg.batch_size
        else:
            scale = cfg.batch_size / float(q_sum)
            quotas = {k: int(round(v * scale)) for k, v in quotas.items()}
            # correct rounding drift
            drift = cfg.batch_size - sum(quotas.values())
            # put drift into uncertainty (most useful by default)
            quotas["uncertainty"] = quotas.get("uncertainty", 0) + drift

    if verbose:
        print(f"Quotas: {quotas} (sum={sum(quotas.values())})")

    # Pools
    df0 = df[df["n_det"] == 0].copy()
    df_pos = df[df["n_det"] > 0].copy()

    # 1) Zero-case coverage: stratified across site/hour
    zero_pick = _stratified_sample(df0, n=quotas["zero"], seed=cfg.seed, per_stratum_cap=cfg.per_stratum_cap)

    selected_idx = set(zero_pick.index.tolist())

    # 2) Uncertainty sampling: rank + per-stratum cap
    # Candidate filter: either borderline evidence or low max conf
    cand_u = df_pos[
        (df_pos["n_borderline"] >= 1)
        | (df_pos["max_conf"].fillna(0.0) < max(0.40, cfg.conf_high))
    ].copy()
    cand_u = cand_u.sort_values(["uncertainty_score", "n_borderline", "class_entropy"], ascending=[False, False, False])
    # Apply per-stratum cap via iterative selection
    u_pick_rows: List[int] = []
    stratum_counts: Dict[str, int] = {}
    for idx, row in cand_u.iterrows():
        if idx in selected_idx:
            continue
        if len(u_pick_rows) >= quotas["uncertainty"]:
            break
        st = str(row["stratum"])
        if cfg.per_stratum_cap > 0 and stratum_counts.get(st, 0) >= cfg.per_stratum_cap:
            continue
        u_pick_rows.append(int(idx))
        selected_idx.add(int(idx))
        stratum_counts[st] = stratum_counts.get(st, 0) + 1
    uncertainty_pick = df.loc[u_pick_rows].copy()

    # 3) Hard negatives: rank by hardneg_score, then uncertainty_score
    cand_h = df_pos[df_pos["hardneg_score"] >= 2].copy()
    cand_h = cand_h.sort_values(["hardneg_score", "uncertainty_score", "n_det"], ascending=[False, False, False])
    h_pick_rows: List[int] = []
    stratum_counts_h: Dict[str, int] = {}
    for idx, row in cand_h.iterrows():
        if idx in selected_idx:
            continue
        if len(h_pick_rows) >= quotas["hardneg"]:
            break
        st = str(row["stratum"])
        if cfg.per_stratum_cap > 0 and stratum_counts_h.get(st, 0) >= cfg.per_stratum_cap:
            continue
        h_pick_rows.append(int(idx))
        selected_idx.add(int(idx))
        stratum_counts_h[st] = stratum_counts_h.get(st, 0) + 1
    hardneg_pick = df.loc[h_pick_rows].copy()

    # 4) Diversity fill: stratified sample from remaining pool
    remaining = df.loc[df.index.difference(list(selected_idx))].copy()
    diversity_pick = _stratified_sample(remaining, n=quotas["diversity"], seed=cfg.seed + 7, per_stratum_cap=cfg.per_stratum_cap)

    batch = pd.concat([zero_pick, uncertainty_pick, hardneg_pick, diversity_pick], axis=0)

    # If quotas were infeasible, top up from remaining by uncertainty (keeps informativeness)
    if len(batch) < cfg.batch_size:
        need = cfg.batch_size - len(batch)
        rem2 = df.loc[df.index.difference(batch.index)].copy()
        rem2 = rem2.sort_values(["uncertainty_score", "hardneg_score"], ascending=[False, False])
        topup = rem2.head(need)
        batch = pd.concat([batch, topup], axis=0)

    # Finally enforce uniqueness
    batch = batch.drop_duplicates(subset=["path"]).copy()

    summary: Dict[str, object] = {
        "n_total": int(len(df)),
        "n_zero": int(len(df0)),
        "n_nonzero": int(len(df_pos)),
        "selected": int(len(batch)),
        "selected_zero": int((batch["n_det"] == 0).sum()),
        "selected_nonzero": int((batch["n_det"] > 0).sum()),
        "selected_by_density": batch["density_bin"].value_counts().to_dict(),
        "selected_by_hour_bin": batch["hour_bin"].value_counts().to_dict(),
        "selected_by_site_top10": batch["site"].value_counts().head(10).to_dict(),
        "quotas": quotas,
    }

    return batch, summary


def dedup_by_ahash(batch: pd.DataFrame, threshold: int, verbose: bool) -> pd.DataFrame:
    """Remove near-duplicates within the batch using aHash Hamming distance.

    This opens images, so it is optional. Keeps the first occurrence and drops later images
    whose hash is within `threshold` bits of any already-kept image.
    """
    if Image is None:
        raise ImportError("Pillow is required for --dedup-hash, but it is not available.")

    kept_rows: List[int] = []
    kept_hashes: List[int] = []

    paths = batch["path"].astype(str).tolist()
    idxs = batch.index.tolist()

    for idx, p in zip(_iter(idxs, "Deduplicating (aHash)", enabled=verbose), paths):
        h = compute_image_hash_ahash(p)
        if h is None:
            kept_rows.append(int(idx))
            continue
        is_dup = False
        for kh in kept_hashes:
            if hamming_distance_int(h, kh) <= threshold:
                is_dup = True
                break
        if not is_dup:
            kept_rows.append(int(idx))
            kept_hashes.append(int(h))

    out = batch.loc[kept_rows].copy()
    if verbose:
        print(f"Dedup kept {len(out)}/{len(batch)} images (threshold={threshold})")
    return out


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Select a model-assisted annotation batch from nested detections")
    p.add_argument(
        "--detections",
        type=str,
        default=str(BASELINE_DETECTIONS_MANIFEST),
        help="Path to the nested detections Parquet produced by stage 09",
    )
    p.add_argument("--batch-size", type=int, default=500, help="Number of images to select")

    # Uncertainty window
    p.add_argument("--conf-low", type=float, default=0.20, help="Lower bound for borderline confidence")
    p.add_argument("--conf-high", type=float, default=0.35, help="Upper bound for borderline confidence")

    # Quotas
    p.add_argument("--quota-zero", type=int, default=150, help="Number of predicted-zero images to include")
    p.add_argument("--quota-uncertainty", type=int, default=200, help="Number of uncertainty-sampled images")
    p.add_argument("--quota-hardneg", type=int, default=100, help="Number of hard-negative images")
    p.add_argument("--quota-diversity", type=int, default=50, help="Number of diversity-fill images")

    p.add_argument(
        "--per-stratum-cap",
        type=int,
        default=15,
        help="Maximum images per (site|hour_bin|density_bin) stratum in each ranked selection stage (0 disables)",
    )

    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--out-dir", type=str, default=str(AL_BATCHES_ROOT / "batch_001"), help="Output directory")

    # Optional dedup
    p.add_argument(
        "--dedup-hash",
        action="store_true",
        help="Optional: open images and remove near-duplicates using a simple perceptual hash.",
    )
    p.add_argument(
        "--hash-threshold",
        type=int,
        default=6,
        help="Hamming distance threshold for aHash near-duplicate filtering (smaller=more strict)",
    )
    p.add_argument(
        "--copy-to",
        type=str,
        default=str(TRAINING_ROOT),
        help="Destination root to copy selected images for labeling.",
    )
    p.add_argument(
        "--no-copy",
        action="store_true",
        help="Do not copy selected images to --copy-to (still writes batch outputs).",
    )
    p.add_argument(
        "--copy-structure",
        choices=["site", "site_date", "flat"],
        default="site",
        help=(
            "How to structure copied images under --copy-to. "
            "'site' => Training/<Parking|River>/<Site>/<filename>. "
            "'site_date' => Training/<Parking|River>/<Site>/<YYYY-MM-DD>/<filename>. "
            "'flat' => Training/<filename> (may overwrite if duplicates)."
        ),
    )

    p.add_argument("--verbose", action="store_true", help="Print progress")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    t0 = time.perf_counter()

    det_path = Path(args.detections).expanduser().resolve()
    if not det_path.exists():
        raise FileNotFoundError(f"Detections parquet not found: {det_path}")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.verbose:
        print(f"Reading: {det_path}")

    df = pd.read_parquet(det_path)

    required = {"path", "detections"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    cfg = SelectionConfig(
        detections_path=str(det_path),
        batch_size=int(args.batch_size),
        conf_low=float(args.conf_low),
        conf_high=float(args.conf_high),
        quota_zero=int(args.quota_zero),
        quota_uncertainty=int(args.quota_uncertainty),
        quota_hardneg=int(args.quota_hardneg),
        quota_diversity=int(args.quota_diversity),
        per_stratum_cap=int(args.per_stratum_cap),
        seed=int(args.seed),
        dedup_hash=bool(args.dedup_hash),
        hash_threshold=int(args.hash_threshold),
    )

    # Add scores
    df_s = add_scores(df, conf_low=cfg.conf_low, conf_high=cfg.conf_high, verbose=bool(args.verbose))

    # Select batch
    batch, summary = select_batch(df_s, cfg=cfg, verbose=bool(args.verbose))

    # Optional dedup
    if cfg.dedup_hash:
        batch = dedup_by_ahash(batch, threshold=cfg.hash_threshold, verbose=bool(args.verbose))
        # If dedup shrinks too much, top up by uncertainty
        if len(batch) < cfg.batch_size:
            need = cfg.batch_size - len(batch)
            rem = df_s.loc[df_s.index.difference(batch.index)].copy()
            rem = rem.sort_values(["uncertainty_score", "hardneg_score"], ascending=[False, False])
            topup = rem.head(need)
            batch = pd.concat([batch, topup], axis=0).drop_duplicates(subset=["path"]).copy()

    # Trim to batch-size deterministically
    batch = batch.sort_values(["hardneg_score", "uncertainty_score"], ascending=[False, False]).head(cfg.batch_size).copy()

    # Copy images for labeling (optional)
    if not bool(args.no_copy) and bool(args.copy_to):
        batch = copy_batch_images(
            batch,
            copy_to=str(args.copy_to),
            copy_structure=str(args.copy_structure),
            verbose=bool(args.verbose),
        )

    # Write outputs
    out_parquet = out_dir / "batch_manifest.parquet"
    out_txt = out_dir / "paths.txt"
    out_json = out_dir / "summary.json"

    batch.to_parquet(out_parquet, index=False)
    out_txt.write_text("\n".join(batch["path"].astype(str).tolist()) + "\n", encoding="utf-8")

    payload = {
        "config": asdict(cfg),
        "summary": summary,
        "written": {
            "batch_manifest": str(out_parquet),
            "paths_txt": str(out_txt),
        },
        "copy": {
            "enabled": (not bool(args.no_copy)) and bool(args.copy_to),
            "copy_to": str(args.copy_to),
            "copy_structure": str(args.copy_structure),
            "copied_ok": int(batch["copy_ok"].sum()) if "copy_ok" in batch.columns else None,
            "copied_failed": int((batch["copy_ok"] == 0).sum()) if "copy_ok" in batch.columns else None,
        },
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.verbose:
        elapsed = time.perf_counter() - t0
        print(f"Selected {len(batch)}/{cfg.batch_size} images")
        print(f"Wrote: {out_parquet}")
        print(f"Wrote: {out_txt}")
        print(f"Wrote: {out_json}")
        print(f"Elapsed: {_fmt_seconds(elapsed)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
