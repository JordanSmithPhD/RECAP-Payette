# Parking-opportunity workflow

This workflow maps surface area that the model classifies as paved or dirt/gravel parking within a road-and-river corridor and a terrain constraint. It does not, by itself, establish marked-space counts, safe operating limits, regulatory limits, or a legal carrying capacity.

## Workflow map

```text
source vectors + NAIP + elevation
            │
            ├── 01 corridor, imagery, and slope manifests
            │
            ├── 02 SAM proposals → human curation → lineage validation
            │
            ├── 03 chips + spatial train/validation/test manifest
            │
            ├── 04 DeepLabV3+ training and held-out evaluation
            │
            └── 05 full-corridor inference → area by named reach
```

Run each durable output into a new directory. The scripts refuse many existing outputs so that a rerun cannot silently replace its evidence.

## Recommended contract

The corrected workflow uses four real NAIP bands in R, G, B, N order, source-validity masks, slope `<= 5%`, spatially grouped partitions, negative chips, fixed-grid inference with context, and boundary-clipped reach accounting.

The examples below assume the default run ID `recommended_v1` from `config.yaml`.

### 1. Acquire and bind imagery

Prepare the source vectors described in [Data access](data-access.md), authenticate the Earth Engine CLI, and provide a Google Cloud project. An export ROI is an analyst-supplied polygon and is not included.

```bash
python pipeline/01_export_naip_earth_engine.py \
  --roi data/raw/study_areas/naip_export_roi.gpkg \
  --project YOUR_EARTH_ENGINE_PROJECT \
  --band-contract rgbnir \
  --no-wait
```

The submission manifest records the immutable task ID. If the task is still running, finalize that same task rather than submitting another:

```bash
python pipeline/01_finalize_naip_export.py \
  --project YOUR_EARTH_ENGINE_PROJECT
```

After the exported TIFF family is available in the configured local export directory:

```bash
python pipeline/01_mosaic_naip_exports.py \
  --export-manifest data/intermediate/parking_runs/recommended_v1/imagery/naip.earth_engine.run.json
```

The recovered acquisition window is encoded in the exporter defaults, but the original source-asset list was not retained. Historical descending-date sorting followed by Earth Engine mosaicking gives oldest-valid-pixel priority. A new run must preserve the generated manifests, inspect contributing assets, and choose dates appropriate to its study area.

### 2. Build the spatial analysis domain

Supply a TIGER/Line roads file, a reviewed river system-of-interest layer, and the selected road `LINEARID` values. The built-in IDs are a recovered Payette selection; replace them explicitly for another corridor.

```bash
python pipeline/01_build_corridor.py \
  --roads data/raw/roads/tiger_primary_roads.gpkg \
  --river data/raw/hydrography/river_system_of_interest.gpkg \
  --band-contract rgbnir

python pipeline/01_build_dem_slope.py \
  --contract corrected
```

Stage 01 buffers roads and the river by 500 m, intersects the buffers, clips NAIP, derives percent slope in a projected CRS, retains slope `<= 5%`, and checksum-binds all products. Confirm the CRS is appropriate for metre-based distance and area in a new region.

### 3. Generate and curate SAM proposals

```bash
python pipeline/02_generate_sam_candidates.py \
  --contract corrected \
  --model-type vit_h \
  --checkpoint data/external_models/sam_vit_h.pth \
  --checkpoint-sha256 EXPECTED_SHA256
```

The generated proposal GeoPackage is immutable. Copy proposals into a separate curated layer, classify them with the six-class vocabulary, edit geometry where necessary, and retain the lineage edge table described in [Data access](data-access.md). SAM provides proposals; a human remains responsible for the reference labels.

The recovered proposal settings are ViT-H, 1,024-pixel tiles, 32 points per side, predicted-IoU threshold `0.86`, stability threshold `0.92`, no crop layers, and minimum mask-region area 100 pixels. These are the stage-02 defaults and are recorded in its manifest. The original checkpoint hash is missing; a new run must record the supplied checkpoint SHA-256.

Validate before creating chips:

```bash
python pipeline/02_validate_candidate_labels.py \
  --labels data/intermediate/parking_runs/recommended_v1/labels/curated_candidates.gpkg \
  --raw-proposals data/intermediate/parking_runs/recommended_v1/labels/sam_raw_proposals.gpkg \
  --lineage-mode adjudicated \
  --lineage-table data/intermediate/parking_runs/recommended_v1/labels/curation_edges.parquet \
  --count-contract schema-only
```

Use `exact` only when curated geometry and proposal IDs are unchanged, or `subset` when review only removed proposals. `legacy` is an audit mode for the surviving paper-era artifact and is not the recommended contract.

### 4. Build spatial partitions and train

```bash
python pipeline/03_build_segmentation_dataset.py \
  --contract corrected

python pipeline/04_train_segmentation.py \
  --contract corrected \
  --variant v3_augmented \
  --device auto
```

Stage 03 creates 512 × 512 NumPy chips, samples negative chips, and assigns 4,096-pixel spatial groups to train, validation, and test partitions. Both parking classes must occur in every partition. Stage 04 uses validation mean intersection-over-union (IoU) for model selection and stores its selected checkpoint with input hashes.

The recovered `v3_augmented` training specification is DeepLabV3+ with a ResNet-34 ImageNet encoder, six output classes, weighted cross-entropy with weights `1,50,50,10,10,10`, Adam at `1e-4`, batch size 8, and 15 epochs. It applies horizontal and vertical flips, 90-degree rotation, limited shift/scale/rotation, and brightness/contrast augmentation; chips containing either parking class are repeated 20 times. The corrected contract preserves this model specification while changing the input, split, validity, and evaluation contracts.

Evaluate the held-out test partition once after model selection:

```bash
python pipeline/04_evaluate_segmentation.py \
  --contract corrected \
  --partition test \
  --evaluation-mask slope \
  --device auto
```

Do not tune labels, preprocessing, hyperparameters, or checkpoints after reading the test result.

### 5. Infer and summarize area

```bash
python pipeline/05_infer_segmentation.py \
  --contract corrected \
  --device auto

python pipeline/05_summarize_parking_capacity.py \
  --contract corrected \
  --study-areas data/raw/study_areas/study_areas.gpkg \
  --area-only
```

The canonical output is parking-opportunity area by reach and surface. If an independently produced reference polygon layer is available, pass it to create the model-versus-reference comparison.

An estimated number of spaces is optional and must never be implicit:

```bash
python pipeline/05_summarize_parking_capacity.py \
  --contract corrected \
  --study-areas data/raw/study_areas/study_areas.gpkg \
  --spaces-per-acre YOUR_DOCUMENTED_RATE \
  --capacity-rate-source "DOI, URL, or documented local derivation"
```

The conversion is a scenario based on the supplied rate, not a count of marked stalls or an official limit. Never reuse the paper-era exploratory rate without an applicable source.

The optional road-proximity diagnostic uses checksum-bound stage-01 roads and boundary-accounted stage-05 polygons:

```bash
python pipeline/05_test_road_proximity.py --contract corrected
```

## Historical replay

Historical mode deliberately preserves these recovered behaviors:

- RGB values exported on a 0–1 scale, followed by an unintended second division by 255 during model input;
- a fourth GDAL alpha band rather than NIR;
- only chips containing a labeled polygon;
- no original train/validation/test split, so reported chip IoUs are resubstitution evidence;
- slope `< 5%` in the notebook implementation;
- GeoTIFF internal-block inference; and
- an optional 3,000 ft² connected-component filter plus full-polygon intersection assignment.

Keep all replay paths isolated and explicit:

```bash
python pipeline/03_build_segmentation_dataset.py \
  --contract historical \
  --imagery data/intermediate/historical/naip_corridor_rgba.tif \
  --labels data/intermediate/historical/curated_candidates.gpkg \
  --output-root outputs/replay/parking/chips

python pipeline/04_train_segmentation.py \
  --contract historical \
  --variant v3_augmented \
  --chips-root outputs/replay/parking/chips \
  --run-dir outputs/replay/parking/model

python pipeline/04_evaluate_segmentation.py \
  --contract historical \
  --partition resubstitution \
  --chips-root outputs/replay/parking/chips \
  --weights outputs/replay/parking/model/weights/final.pth \
  --output-dir outputs/replay/parking/evaluation
```

Stages 05 can then be run with `--contract historical` and explicit replay inputs and outputs. A replay can match article-era computation only when the corresponding source raster, edited labels, study-area polygons, and checkpoint are available. It should remain labeled `historical`, even if its arithmetic matches a reported value.

## Quality checks

Before interpreting a corrected run, verify that:

- every input manifest resolves and its checksum matches;
- the NAIP fourth band is genuine NIR, not alpha;
- the slope and corridor masks use the intended projected CRS;
- raw SAM proposals and curated labels are different files;
- curation lineage validation has no errors;
- spatial groups do not cross partitions;
- both parking classes occur in all partitions;
- background class `0` is distinct from nodata value `255` in inference; and
- reach polygons are unique, valid, and nonoverlapping.

See [Outputs and interpretation](outputs.md) before using the resulting area table.
