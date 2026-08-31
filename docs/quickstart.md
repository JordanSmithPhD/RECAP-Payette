# Quick start

Run commands from the repository root. The repository never needs primary inputs to run its lightweight tests.

## 1. Clone and test the code

```bash
git clone https://github.com/JordanSmithPhD/RECAP-Payette.git
cd RECAP-Payette
conda env create -f environment.yml
conda activate recap-payette
pytest -q
```

The lightweight environment supports tests and tabular schema utilities. It is not the full geospatial or deep-learning stack.

## 2. Create local configuration

```bash
cp config/example.yaml config.yaml
```

Edit `config.yaml` to point at local data and output roots. The file is ignored by Git. The same values can be set with:

- `RECAP_PAYETTE_DATA_ROOT`
- `RECAP_PAYETTE_OUTPUT_ROOT`
- `RECAP_PAYETTE_CAMERA_ROOT`
- `RECAP_PAYETTE_TRAINING_ROOT`
- `RECAP_PAYETTE_PARKING_RUN_ID`

Environment variables take precedence over `config.yaml`; the repository-relative defaults come last. Inspect the resolved layout without reading data:

```bash
python -c "import json; from project_paths import describe_layout; print(json.dumps(describe_layout(), indent=2))"
```

Keep large inputs outside the clone if desired. Camera-manifest paths are stored relative to the configured camera archive.

## 3. Install one workflow environment

For parking opportunity:

```bash
conda env create -f environment-parking.yml
conda activate recap-payette-parking
python pipeline/01_build_corridor.py --help
```

For camera activity:

```bash
conda env create -f environment-camera.yml
conda activate recap-payette-camera
python pipeline/06_build_camera_manifest.py --help
```

Large model initializers and checkpoints are acquired or supplied separately; they are never committed.

## 4. Select a contract

- Use the recommended contract for a new corridor or a new analysis of Payette inputs.
- Use historical replay only when the corresponding frozen inputs and checkpoints are available and the goal is to reconstruct article-era behavior.
- Run benchmarks separately, after the source corpus has passed its checksum and split audits.

Continue with [Parking opportunity](parking-opportunity.md) or [Camera activity](camera-activity.md). Read [Reproducibility status](reproducibility.md) before comparing a new run with the article.

## 5. Before sharing a run

Retain the stage manifests and checksums with the results, but do not publish restricted inputs, archive paths, site coordinates, model weights, or row-level camera outputs. Review [Privacy and responsible use](privacy.md) before distributing artifacts.
