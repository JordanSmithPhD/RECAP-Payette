# Curated parking-label contract

The stage-02 GeoPackage must contain valid polygon or multipolygon geometry in
a declared CRS and these attributes:

| Field | Type | Meaning |
|---|---|---|
| `proposal_id` | string | Stable identifier from the raw SAM proposal table. |
| `class_id` | integer | `0` background, `1` paved parking, `2` dirt/gravel parking, `3` roof, `4` water, `5` road. |
| `review_status` | string | Explicit review/adjudication status. |

For a recommended run, keep a separate lineage CSV with one disposition for
every raw proposal. A historical Payette replay cannot reconstruct this table
because proposal-to-edit lineage was not retained.
