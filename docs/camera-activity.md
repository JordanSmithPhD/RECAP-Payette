# Camera-activity workflow

This workflow converts fixed-camera images into two relative indicators: vehicle activity at parking viewpoints and watercraft activity at river viewpoints. Both are counts of distinct detector tracks observed within monitored site-hours. They remain conditional on camera placement, image availability, image quality, detector behavior, thresholds, and tracking rules.

## Workflow map

```text
governed camera archive
        │
        ├── 06 portable image manifest
        ├── 07–08 annotation, duplicate-safe splits, optional batch selection,
        │       detector training, and threshold checks
        ├── 09 checksum-verified inference + schema normalization
        └── 10 tracking → hourly relative activity → optional calibration
```

## Recommended contract

### 1. Build a portable manifest

Organize the archive as described in [Data access](data-access.md). The default timestamp expression expects a filename token such as `20240715_143000`; override both timestamp arguments when needed.

```bash
python pipeline/06_build_camera_manifest.py \
  --root data/external_camera_archive \
  --output data/intermediate/camera/image_manifest.parquet \
  --timezone America/Denver \
  --daylight-hours 06:21
```

The manifest stores archive-relative paths, a stable image ID, a timestamp with offset, readability, a `daylight` flag, file size, and—by default—SHA-256. Inspect parsing failures and `unknown` site types before proceeding. A hash proves file identity, not that the image is scientifically usable.

The original Payette archive-inventory code was not recovered. This command is the recommended replacement for a new run; it is not a reconstruction of the original inventory step.

### 2. Annotate and split

Create a persistent [FiftyOne](https://docs.voxel51.com/) dataset whose samples point to the governed archive. Use a `ground_truth` detection field and the two-class label contract. Review predicted-zero images as well as positive images.

Assign byte-identical files to the same deterministic partition and reserve an independent test set:

```bash
python pipeline/08_assign_detection_splits.py \
  --dataset YOUR_FIFTYONE_DATASET \
  --summary outputs/runs/camera/split_summary.json
```

Export the labeled partitions to YOLO format. Test export is enabled when a `test` partition is present; keep it untouched until the final evaluation:

```bash
python pipeline/07_export_yolo.py \
  --dataset YOUR_FIFTYONE_DATASET \
  --out data/external_training/yolo_v1
```

Do not move near-duplicate frames between partitions. A stronger design also groups temporally adjacent frames from one camera so that the test set reflects transfer beyond near-identical observations.

### 3. Optional model-assisted annotation rounds

After fitting a provisional detector, stage 08 can select a review batch using predicted zeros, borderline confidence, likely hard negatives, and site/time/density diversity:

```bash
python pipeline/08_select_annotation_batch.py \
  --detections data/intermediate/camera/annotation_pool_detections.parquet \
  --batch-size 500 \
  --out-dir outputs/annotation_batches/round_001 \
  --no-copy \
  --verbose
```

Treat every selected image as untrusted until a person reviews it. Freeze each batch manifest before annotation, record additions and corrections, and rebuild the training export without changing the validation or test inventory.

The article's reconstructed annotation stages changed image volume, class balance, and selection at the same time. Stage-to-stage score changes are descriptive; they do not show that the selection strategy caused an improvement.

### 4. Train and check thresholds

```bash
python pipeline/08_train_detector.py \
  --data data/external_training/yolo_v1/data.yaml \
  --model yolo26n.pt \
  --run-dir outputs/runs/camera/detector_v1 \
  --imgsz 960 \
  --device cpu
```

The command writes a checksum-bound training manifest beside `weights/best.pt`. Use validation data—not the held-out test set—to choose class-specific confidence thresholds. The threshold utility evaluates one class at a time:

```bash
python pipeline/08_tune_thresholds.py \
  --data-root data/external_training/yolo_v1 \
  --weights outputs/runs/camera/detector_v1/weights/best.pt \
  --output-dir outputs/runs/camera/thresholds_vehicle \
  --cls 0 \
  --imgsz 960 \
  --device cpu

python pipeline/08_tune_thresholds.py \
  --data-root data/external_training/yolo_v1 \
  --weights outputs/runs/camera/detector_v1/weights/best.pt \
  --output-dir outputs/runs/camera/thresholds_watercraft \
  --cls 1 \
  --imgsz 960 \
  --device cpu
```

Record the threshold grid, matching IoU, selected rule, validation inventory hash, and output in the run record. Threshold choice is part of the estimator.

### 5. Run resumable inference

Use the stage-06 manifest, the selected checkpoint, and a new run directory. Stage 09 filters `daylight == true` and `readable == true` by default when those columns exist; disabling either filter changes the analysis contract and is recorded:

```bash
python pipeline/09_run_camera_inference.py \
  --manifest data/intermediate/camera/image_manifest.parquet \
  --root data/external_camera_archive \
  --weights outputs/runs/camera/detector_v1/weights/best.pt \
  --run-dir outputs/runs/camera/inference_v1 \
  --imgsz 960 \
  --vehicle-conf YOUR_VALIDATION_THRESHOLD \
  --watercraft-conf YOUR_VALIDATION_THRESHOLD \
  --device cpu \
  --verbose
```

Stage 09 verifies source identities, writes atomic shards, resumes only verified shards, distinguishes decode failures from inference failures, and commits `detections.parquet` only after row-count checks pass. For a durable run, supply expected input and final row counts after auditing the frozen manifest.

Normalize nested output into portable, long-form tables:

```bash
python pipeline/09_normalize_camera_outputs.py \
  --image-manifest data/intermediate/camera/image_manifest.parquet \
  --inference outputs/runs/camera/inference_v1/detections.parquet \
  --camera-root data/external_camera_archive \
  --output-dir outputs/runs/camera/normalized_v1
```

The adapter removes the archive root from output paths, joins detections to stable image IDs, and marks images without a successful inference row as unreadable rather than as observed zeros.

### 6. Track and aggregate

```bash
python pipeline/10_summarize_camera_activity.py \
  --image-manifest outputs/runs/camera/normalized_v1/image_manifest.parquet \
  --detections outputs/runs/camera/normalized_v1/detections.parquet \
  --output-dir outputs/results/camera/recommended_v1 \
  --mode recommended \
  --iou-threshold 0.5 \
  --max-gap-seconds 900 \
  --output-format parquet
```

Recommended mode greedily matches boxes by intersection-over-union within site and class, expires a track after the configured gap, retains singleton tracks, and builds a complete site-hour-by-class grid. A zero is reported only when at least one image in the hour was readable; an hour with no readable image receives a missing activity value.

### 7. Optional calibration check

```bash
python pipeline/10_calibrate_activity.py \
  --hourly-activity outputs/results/camera/recommended_v1/hourly_activity.parquet \
  --manual-counts data/external_training/manual_reference_counts.csv \
  --output-dir outputs/results/camera/recommended_v1/calibration \
  --grouping class \
  --output-format parquet
```

Calibration fits a transparent linear association with independent manual references. It does not remove camera or detector dependence, and its generated value is named `calibrated_reference_estimate` rather than an absolute visitor estimate.

## Historical replay

The article's operational estimator used a frozen daylight inventory, a 960-pixel YOLO26n checkpoint, vehicle confidence `0.25`, watercraft confidence `0.477`, and nonmaximum-suppression IoU `0.70`. Exact replay starts from that frozen manifest and checkpoint because their complete creation history is unavailable.

After stage 09 inference and normalization, replay the original aggregation semantics with:

```bash
python pipeline/10_summarize_camera_activity.py \
  --image-manifest outputs/replay/camera/normalized/image_manifest.parquet \
  --detections outputs/replay/camera/normalized/detections.parquet \
  --output-dir outputs/replay/camera/activity \
  --mode historical_replay \
  --iou-threshold 0.5 \
  --output-format parquet
```

Historical replay never expires inactive tracks, omits singleton tracks from its track-summary table while retaining their observations, and omits monitored hours with no detections. These behaviors can suppress true zeros and can link temporally distant boxes. They are preserved only to reconstruct the article-era computation; `recommended` is the default for new work.

## Quality checks

Before interpreting camera outputs, confirm that:

- every manifest timestamp has the intended offset and `daylight` classification;
- archive-relative paths and content hashes match;
- unreadable or missing images are not treated as observed zeros;
- class IDs and thresholds match the checkpoint;
- train, validation, and test identities are frozen and duplicate-safe;
- threshold selection did not use the test set;
- the tracking gap is plausible for the capture interval;
- activity tables retain coverage columns; and
- any distributed result is aggregated and stripped of archive paths and sensitive site detail.

See [Outputs and interpretation](outputs.md) for the indicator guardrails.
