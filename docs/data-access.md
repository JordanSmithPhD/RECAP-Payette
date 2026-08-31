# Data access and input contracts

This is a code-only repository. The tables below identify where an input originates, what the workflow expects, and what is required for an exact replay. A current download is appropriate for a new application, but it is not automatically equivalent to the edition used in the article.

## Parking-opportunity inputs

| Input | Authoritative source or origin | Minimum contract | Exact Payette replay status |
|---|---|---|---|
| Aerial imagery | [USDA NAIP in the Earth Engine Data Catalog](https://developers.google.com/earth-engine/datasets/catalog/USDA_NAIP_DOQQ) | Ordered R, G, B, N bands for the corrected workflow; export query, source asset IDs, scale, CRS, nodata rule, and checksum recorded | The recovered query and export settings are known, but the contributing asset IDs and acquisition dates are not. |
| Road centerlines | [2025 U.S. Census Bureau TIGER/Line All Roads files](https://www.census.gov/cgi-bin/geo/shapefiles/index.php?layergroup=Roads&year=2025) | County-based line geometry with a unique `LINEARID`; selected segment IDs recorded | The 2025 county-file naming pattern and six selected IDs are recovered. Retain the downloaded ZIP checksum. |
| River geometry | [USGS Network Linked Data Index](https://api.water.usgs.gov/nldi/docs) and its NHDPlus V2 network | Valid line geometry with a declared CRS; the navigation request recorded | The recovered request starts from COMID `24166124` and requests upstream tributaries to 1,000 km. Retain the returned bytes because the service is dynamic. |
| Elevation | [USGS 3D Elevation Program](https://www.usgs.gov/3d-elevation-program) | Elevation grid and acquisition manifest sufficient to derive percent slope in a projected metre CRS | A dynamic 1 m `py3dep` request is recovered, but fixed tile IDs and vertical-reference metadata are not. |
| NAIP export ROI | Prepared by the analyst from the study corridor | Valid polygon geometry with a CRS | The exact original ROI must be supplied. |
| Curated surface labels | Human review of SAM proposals | GeoPackage geometry with integer `class_id`; corrected runs also preserve curation lineage | Not distributed. The historical edited layer lacks complete proposal-to-edit lineage. |
| Named reaches | Analyst-defined management units | Nonoverlapping polygons; unique, nonempty `area_name` by default | Not distributed. |
| Independent parking reference polygons | Analyst-created reference interpretation | Polygon geometry and a documented surface-class field | Not distributed. Used only when a model-versus-reference comparison is requested. |

The corrected parking class vocabulary is fixed:

| `class_id` | Class |
|---:|---|
| 0 | background |
| 1 | paved parking |
| 2 | dirt/gravel parking |
| 3 | roof |
| 4 | water |
| 5 | road |

SAM weights are not stored here. Stage 02 supports the official [Segment Anything repository](https://github.com/facebookresearch/segment-anything) and records the model type and checkpoint checksum. Supply an expected SHA-256 whenever acquiring a checkpoint for a durable run.

### Recovered Payette acquisition recipe

The historical NAIP query used `USDA/NAIP/DOQQ`, an inclusive start of `2022-01-01`, an exclusive end of `2025-12-31`, RGB output at 0.6 m, and EPSG:4326. It sorted images in descending time order and then called Earth Engine's mosaic operation; because later collection elements take pixel priority, that combination yields the **oldest valid pixel**, not the newest. The contributing NAIP asset IDs and acquisition dates were not written to the surviving record, so the query can be reconstructed but its exact raster cannot be independently identified from the catalog alone.

The road source was a county-based 2025 All Roads ZIP following `tl_2025_<state-county-FIPS>_roads.zip`, filtered to the six `LINEARID` values encoded in stage 01. The river request was equivalent to:

```text
feature source: comid
feature ID: 24166124
navigation: upstream tributaries (UT)
distance: 1000 km
```

These identifiers establish request provenance, but a dynamic service response can change. Archive the raw response and its checksum for any durable run.

### Corrected curation lineage

Keep raw proposals immutable. Curated geometry belongs in a separate GeoPackage with unique `curated_id`, integer `class_id`, valid geometry, and a declared CRS. When a proposal is changed, supply a CSV or Parquet edge table with:

| Field | Meaning |
|---|---|
| `curated_id` | ID of an included curated geometry; blank for a rejected proposal |
| `parent_proposal_id` | Source `proposal_id`; blank only for a manual addition |
| `edit_type` | `unchanged`, `relabeled`, `geometry_edit`, `split`, `merge`, `manual_addition`, or `rejected` |
| `reviewer` | Stable reviewer identifier |
| `decision` | `include` or `exclude` |

## Camera-workflow inputs

There is no public download route for the Payette photographs, row-level annotations, frozen daylight manifest, or operational checkpoint. They are excluded to protect governed media and site information. Apply the workflow to an archive collected under appropriate authority, or obtain separately authorized access to the study artifacts.

The recommended archive layout is:

```text
camera-archive/
├── Parking/<site-id>/.../*.jpg
└── River/<site-id>/.../*.jpg
```

Site identifiers should be pseudonymous. Filenames must contain a parseable local timestamp, or the manifest must be prepared by another documented method.

### Image-manifest schema

Stage 06 writes CSV or Parquet with the following durable fields:

| Field | Type | Meaning |
|---|---|---|
| `image_id` | string | Stable hash of the archive-relative path |
| `path` | string | Path relative to the configured archive root |
| `camera_type` | string | `parking`, `river`, or `unknown` |
| `site_id` | string | Pseudonymous monitoring location |
| `captured_at` | timestamp string | Timestamp with an explicit UTC offset |
| `acquired` | boolean | Whether the scheduled image exists |
| `readable` | boolean | Whether the image decoded successfully |
| `daylight` | boolean | Whether the image passed the configured local-hour screen |
| `file_size` | integer | Source bytes |
| `content_sha256` | string | Content checksum when hashing is enabled |

For downstream hourly accounting, `image_id`, `site_id`, `captured_at`, `acquired`, and `readable` are required. Naive timestamps are treated as UTC by the tabular utilities; attach the correct offset before use.

### Detection-label contract

Use two foreground classes in a stable order:

```text
0 vehicle
1 watercraft
```

YOLO label rows are `class_id x_center y_center width height`, normalized to `[0, 1]`. Create empty label files for reviewed images with no target object. Draw a box around the visible portion of each target and omit ambiguous objects, reflections, and glare. Preserve a data dictionary and an adjudication log with the labels.

### Normalized detections

Stage 09 produces one row per image with nested detections. The normalization adapter writes the long form consumed by stage 10:

| Field | Type |
|---|---|
| `detection_id` | unique string |
| `image_id` | manifest foreign key |
| `class_name` | `vehicle` or `watercraft` |
| `confidence` | number in `[0, 1]` |
| `x_min`, `y_min`, `x_max`, `y_max` | finite pixel coordinates with positive width and height |

### Optional manual-reference table

Calibration accepts CSV or Parquet with one row per site, class, and hour:

| Field | Type |
|---|---|
| `site_id` | string |
| `class_name` | `vehicle` or `watercraft` |
| `hour_start` | timestamp with offset |
| `manual_reference_value` | finite, nonnegative numeric reference |

The source, temporal alignment rule, observer protocol, and units for the reference values must be documented outside the table.

## Record the source edition, not just the provider

For every downloaded input, retain:

- provider and product name;
- release, vintage, or access date;
- query bounds and temporal filters;
- source asset, tile, task, or feature identifiers;
- original CRS, resolution, bands, and nodata definition;
- any selection or editing rule;
- file SHA-256; and
- the license or terms in force at acquisition.

A provider URL alone is not enough to establish that two runs used the same input.
