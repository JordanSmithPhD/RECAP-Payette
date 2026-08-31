# Reproducibility status

## Plain-language answer

A public clone does **not** contain enough material to regenerate every numerical result in the article. It contains executable workflow logic, explicit historical-replay behavior, corrected alternatives, schemas, and tests. Exact numerical replay additionally requires source editions, curated labels, frozen manifests, camera media, and operational checkpoints that are not publicly distributed.

This distinction is intentional: reproducible code does not require unrestricted publication of governed media or large primary datasets, but claims of exact numerical replication must state which artifacts were available.

## Three reproducibility targets

| Target | What can be reproduced | Required inputs |
|---|---|---|
| Workflow reproduction | The sequence of acquisition, proposal generation, curation checks, training, inference, tracking, aggregation, and output interpretation | Comparable analyst-supplied inputs satisfying the public schemas |
| Historical numerical replay | The paper-era arithmetic and model calls, including known defects | Exact frozen inputs, labels, manifests, and checkpoints listed below |
| Corrected reanalysis | A more defensible new estimate using provenance binding, held-out evaluation, explicit missingness, and corrected tracking | Newly acquired or authorized inputs with complete provenance |

A corrected reanalysis is not expected to match the article exactly because it changes the scientific contract on purpose.

## Parking opportunity

### Publicly reproducible

- the 500 m road/river buffer intersection;
- percent-slope derivation and explicit `< 5%` or `<= 5%` contract;
- SAM proposal generation with recorded settings and checkpoint hash;
- six-class curation schema and lineage validation;
- historical positive-only chips or corrected spatial partitions with negative chips;
- DeepLabV3+ training, resubstitution or held-out evaluation, tiled inference, polygonization, and area accounting; and
- U-Net/FPN post-hoc comparisons when the benchmark chips are supplied.

### Needed for exact article values

- the exact NAIP source assets and original export artifact;
- the checksum-identical 2025 TIGER/Line county All Roads file used with the six recovered IDs;
- the archived response to the recovered NLDI NHDPlus V2 request from COMID `24166124`, upstream tributaries to 1,000 km;
- the exact bytes returned by the recovered dynamic 1 m 3DEP request, including missing vertical-reference metadata;
- the original edited surface-label GeoPackage;
- the 561 surviving historical chips or enough exact inputs to rebuild them;
- the operational DeepLabV3+ checkpoint or an exactly reproducible training environment and initialization; and
- the exact reach and independent reference polygons used for reported summaries.

The historical edited layer has 18,217 records, including 17,400 background, 16 paved-parking, 50 dirt/gravel-parking, 321 roof, 227 water, and 203 road labels. The recovered SAM run logged 12,942 proposals, but a complete proposal-to-edit graph was not retained and the checkpoint hash is missing. Matching those totals therefore checks an artifact; it does not reconstruct the missing curation history.

The article reports 13.8 mapped parking acres—2.8 paved and 11.0 dirt/gravel—against 20.8 acres in the independent reference layer. The reported chip IoUs were calculated on the training corpus and must remain labeled resubstitution evidence, not held-out performance.

The recovered notebook cells were not executed in pipeline order: the SAM cell is dated February 9, before the surviving NAIP cell dated February 13 and elevation cell dated February 26. The notebook therefore documents component computations, but its execution record does not prove that the reported analysis ran end to end from a single frozen set of upstream sources.

## Camera activity

### Publicly reproducible

- portable manifest construction for a newly governed archive;
- two-class annotation export and duplicate-safe partition assignment;
- deterministic batch-selection heuristics;
- detector training and validation-based threshold checks;
- checksum-verified, resumable full-archive inference;
- nested-to-long detection normalization;
- historical or corrected greedy IoU tracking;
- complete site-hour coverage and explicit missingness in recommended mode; and
- transparent calibration against an analyst-supplied manual-reference table.

### Needed for exact article values

- the frozen Payette daylight manifest;
- the underlying camera photographs and their exact bytes;
- the final training labels and partition inventory;
- the operational 960-pixel YOLO26n checkpoint;
- the detector class mapping and exact software environment;
- the manual reference observations used for calibration; and
- the preserved intermediate tables underlying each reported plot.

The reported full run began with 406,896 inputs, excluded 27 unreadable files, and produced 406,869 image outputs containing 233,139 vehicle and 2,036 watercraft boxes. These are useful replay checks only after the input-manifest checksum and checkpoint checksum match. The originally plotted calibrated intermediates were overwritten, so the corresponding panels cannot be byte-for-byte reconstructed from the public clone.

The exact code that first assembled the historical daylight inventory was not recovered. The earliest defensible historical boundary is therefore the frozen manifest, not raw-directory scanning.

## Post-hoc benchmarks

The benchmark scripts and protocols are public. Their training corpora are not. Reproducing the reported numbers requires the exact 561-chip segmentation corpus and the exact two-class detection inventory, hard-example list, split bytes, public initializer versions, and package environment. A smoke run tests wiring only and must not be reported as a benchmark result.

Fresh benchmark checkpoints were never substituted into full-corpus inference, tracking, calibration, or article indicators. See [Benchmarks](benchmarks.md).

## What a reproducibility report should include

For every attempted replay, publish a small run ledger containing:

1. repository commit;
2. scientific contract;
3. input manifest and file checksums;
4. source editions and acquisition dates;
5. checkpoint and initializer checksums;
6. package lock or environment export;
7. random seeds and device;
8. completed stage manifests;
9. row-count and class-count checks; and
10. every deviation from the article or public defaults.

Use “replicated” only when the target, inputs, estimator, aggregation, and acceptance tolerances were declared in advance. Otherwise describe the run as a workflow reproduction, partial replay, or corrected reanalysis.
