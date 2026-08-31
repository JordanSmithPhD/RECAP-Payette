# Post-hoc architecture benchmarks

The benchmarks were added to answer a narrow question: whether the chosen architectures behaved differently from standard alternatives when trained and evaluated on the same available labels. They are isolated from the operational workflows so that a fresh comparison fit cannot silently replace the checkpoint used for article inference.

## What the benchmarks establish

They provide controlled, same-data architecture checks. They do **not** establish:

- independent performance on a new corridor or camera network;
- superiority of one architecture in general;
- that any benchmark checkpoint reproduces the article indicators; or
- that model-assisted annotation caused a performance change.

## Segmentation benchmark

The benchmark uses the 561 surviving historical positive-only chips, discards the alpha channel, forms 4,096-pixel spatial groups, and freezes a 399/75/87 train/validation/test split. DeepLabV3+, U-Net, and FPN use the same ResNet-34 encoder specification, weighted loss, deterministic augmentation, optimizer, batch size, and 15-epoch limit. Validation parking-class mean IoU selects a checkpoint; each test partition is evaluated once.

The chips are not distributed. With an authorized checksum-identical corpus:

```bash
python pipeline/12_benchmark_segmentation.py \
  --chips-root data/external_training/segmentation_chips \
  --output-dir outputs/benchmarks/segmentation \
  --models deeplabv3plus_resnet34 unet_resnet34 fpn_resnet34 \
  --epochs 15 \
  --device auto
```

The reported parking-class mean IoUs were:

| Architecture | Test parking mean IoU |
|---|---:|
| DeepLabV3+ | 0.071 |
| U-Net | 0.035 |
| FPN | 0.045 |

Only 10 of 87 test chips contained any parking pixel, and none of the models correctly segmented paved parking in the three paved-positive chips. The result provisionally favors DeepLabV3+ on this split, but more strongly shows that sparse, foreground-selected labels limit the comparison.

## Detection benchmark

The detection comparison starts YOLO26n and FCOS ResNet-50-FPN from public COCO initializers and uses one frozen two-class inventory, corrected train-only hard examples, identical 15-epoch limits, an optimizer-update batch size of eight, 640-pixel inputs, and the same validation images. The validation partition was frozen before these fits and was not used for early stopping or checkpoint selection. It is still a dependent validation comparison, not an independent test.

Audit inputs without loading a deep-learning framework:

```bash
python pipeline/12_benchmark_detection.py \
  --dataset-root data/external_training/yolo_benchmark \
  --historical-hard-list data/external_training/hard_training_names.txt \
  --models yolo26n fcos_resnet50_fpn \
  --output-dir outputs/benchmarks/detection \
  --dry-run
```

Then run the full comparison:

```bash
python pipeline/12_benchmark_detection.py \
  --dataset-root data/external_training/yolo_benchmark \
  --historical-hard-list data/external_training/hard_training_names.txt \
  --models yolo26n fcos_resnet50_fpn \
  --output-dir outputs/benchmarks/detection \
  --epochs 15 \
  --imgsz 640 \
  --batch-size 8 \
  --device auto
```

Reported point estimates were:

| Class | Model | AP50 | AP50:95 |
|---|---|---:|---:|
| Vehicle | YOLO26n | 0.539 | 0.278 |
| Vehicle | FCOS | 0.579 | 0.245 |
| Watercraft | YOLO26n | 0.002 | 0.001 |
| Watercraft | FCOS | 0.061 | 0.016 |

AP50 summarizes the precision–recall curve when a prediction overlaps a reference box by at least 0.50 IoU. AP50:95 averages AP over increasingly strict IoU thresholds from 0.50 through 0.95. FCOS had a slightly higher vehicle AP50, while YOLO26n had a slightly higher vehicle AP50:95. FCOS used about 32.1 million parameters versus about 2.5 million for YOLO26n.

Only 18 watercraft boxes were available for training and 13 for validation. Bootstrap intervals were wide and overlapping. The comparison therefore shows architecture-sensitive point estimates but does not establish that either detector is superior.

## Optional annotation-stage runs

`--yolo-cumulative-stages` reconstructs fresh YOLO26n fits over cumulative annotation stages when the archived batch manifests are supplied. Annotation volume, class composition, and selection changed together. Report these as descriptive stage comparisons, never as an isolated effect of the selection loop.

## Output and reporting rules

Each benchmark writes configuration, split bytes or hashes, initializer provenance, package versions, histories, predictions, metrics, and bootstrap summaries. `--resume` is permitted only when the immutable configuration and input identities match. `--smoke` performs a one-epoch wiring check on small deterministic subsets; its outputs are labeled smoke-only and must not be reported as scientific results.

Neither benchmark reran full-corpus detection, tracking, calibration, or parking-area inference. The article's operational outputs remain tied to their historical checkpoints.
