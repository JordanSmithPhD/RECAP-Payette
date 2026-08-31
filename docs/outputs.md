# Outputs and interpretation

The workflows produce auditable indicators, not self-interpreting management limits. Preserve run manifests with every result and report the data coverage, model contract, and uncertainty alongside the headline value.

## Parking-opportunity outputs

### Segmentation raster

`segmentation_predictions.tif` stores one class per valid pixel:

| Value | Meaning |
|---:|---|
| 0 | valid background |
| 1 | paved parking prediction |
| 2 | dirt/gravel parking prediction |
| 3 | roof prediction |
| 4 | water prediction |
| 5 | road prediction |
| 255 | corrected-mode nodata: invalid source pixel or outside the slope contract |

Background and nodata are not interchangeable. The adjacent class-pixel CSV and run manifest record counts, tiling, masks, checkpoint metadata, and source-manifest hashes.

### Reach summary

The primary table contains one row for each declared reach and each parking surface in corrected mode:

| Field | Interpretation |
|---|---|
| `reach_name` | Unique analyst-declared reach |
| `class_id`, `surface` | Paved or dirt/gravel class |
| `acres` | Boundary-clipped predicted area |
| `spaces_per_acre` | User-supplied scenario rate, otherwise missing |
| `capacity_rate_source` | Source or derivation for that rate |
| `estimated_spaces` | `acres × spaces_per_acre`, otherwise missing |
| `contract` | `corrected` or `historical` |
| `assignment_contract` | Boundary clipping or historical full-polygon intersection |
| `minimum_area_sqft` | Connected-component filter used |

Use `acres` as the supported mapped-opportunity output. `estimated_spaces` is meaningful only when the rate is locally applicable and documented. Neither value establishes striped-stall geometry, accessible-space requirements, circulation needs, safety constraints, land tenure, allowed use, or an official site limit.

`parking_by_reach.gpkg` retains the assigned prediction geometry. The optional area-comparison table contrasts model and independently interpreted parking area on a common grid/domain. Agreement with that layer is a reference check, not a guarantee that either layer captures all operational constraints.

The road-proximity output is a geometric diagnostic: paved predicted area within the specified road-centerline distance. It does not prove vehicle access or suitability.

## Camera-inference outputs

Stage 09 writes:

- `plan.json`: immutable run contract and shard inventory;
- `shards/`: atomic per-shard Parquet and sidecar records;
- `detections.parquet`: one row per successfully decoded image, with counts and nested boxes;
- `bad_images.txt`: verified source files that failed image decoding; and
- `run_manifest.json`: input, output, checkpoint, package, class, threshold, and row-count evidence.

Per-image counts are detector outputs. Repeated frames can contain the same object, so summing them is not the hourly activity indicator.

The normalization stage writes an archive-relative image manifest and a long detection table. These are the stable inputs to tracking.

## Tracking outputs

### `track_observations`

One row per detection with its image, site, time, class, box, confidence, assigned `track_id`, ordinal, time since the previous observation, and whether the track appears in the summary table.

### `track_summary`

One row per retained track with start/end time, elapsed seconds, detection observations, image observations, first/last image IDs, and the summary rule. A track is an algorithmic association of boxes, not a verified unique vehicle, vessel, group, or person across the full study.

### `hourly_activity`

The primary fields are:

| Field | Interpretation |
|---|---|
| `site_id`, `class_name`, `hour_start` | Site-hour-class key |
| `manifest_images` | Scheduled or inventoried image opportunities |
| `images_acquired` | Images present |
| `images_readable` | Images usable after decoding/inference normalization |
| `images_with_detections` | Readable images with at least one target box |
| `readable_coverage_fraction` | Readable divided by acquired images |
| `acquisition_coverage_fraction` | Acquired divided by manifest images |
| `detected_object_observations` | Detection rows before track deduplication |
| `relative_activity_index` | Distinct track IDs observed in that monitored site-hour |
| `monitoring_status` | `observed_detection`, `observed_zero`, or `unobserved_no_readable_image` in recommended mode |
| `aggregation_mode` | `recommended` or `historical_replay` |

Recommended zeros mean “the detector found no target track in at least one readable image during this monitored hour.” They do not mean that no target passed between captures. Missing values mean the hour lacked a readable observation and must not be converted to zero.

The vehicle index is not a parking-space occupancy percentage or an absolute visitor count. The watercraft index is not river throughput, a census of launches, or physical on-river capacity. Both are relative, camera- and detector-conditioned indicators suitable for comparing patterns only under a stable monitoring contract.

Historical replay contains only site-hours with detections and never expires tracks. Compare its values only with outputs generated under the same replay rules.

## Calibration outputs

`calibration_models` reports grouping, number of matched site-hours, intercept, slope, R², and model status. `calibration_matches` preserves each paired observation. `calibrated_hourly_activity` adds `calibrated_reference_estimate` where a model can be fit.

Wide residuals, few matches, limited index variation, or weak R² should be reported directly. Calibration does not turn the relative index into a universally transferable count; it remains conditional on the reference protocol and the monitored cameras.

## Minimum reporting checklist

Alongside any figure or table, report:

- contract and repository commit;
- source and checkpoint manifest hashes;
- analysis period and aggregation unit;
- class thresholds and tracking parameters;
- valid/readable coverage and missing-hour handling;
- held-out or resubstitution status of model metrics;
- surface-area conversion rule, if any; and
- limitations that affect the intended management inference.
