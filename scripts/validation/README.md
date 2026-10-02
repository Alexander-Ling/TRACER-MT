# Validation tools for tiled processing

These compare a whole-slide TRACER run with a tiled run (or two whole-slide
runs). All work on transcript ids and treat entity names as labels, so renamed
entities are not mismatches unless the names themselves are compared.

To dump per-stage labels, set `TRACER_DUMP_DIR=<dir>` for a whole-slide run, or
pass `--dump-stages <dir>` to `run_tracer_tiled.py` (one sub-directory per tile).

| Script | Compares |
|---|---|
| `compare_strict.py REF_STAGES TEST_STAGES` | two whole-slide stage dumps, label strings and entity types |
| `compare_stages.py --ref --test --core-ids` | partitions per stage, optionally restricted to a set of transcript ids |
| `compare_tiled_stages.py --ref-stages --tiled-stages --phase-a` | per-tile stage dumps vs a whole-slide dump, summed over tiles; names the first divergent stage |
| `compare_runs.py --ref --test [--core-ids]` | two `transcripts_tracer_refined.parquet` files as partitions |
| `compare_final.py --ref OUTPUTS --test OUTPUTS` | final transcripts (all label fields), the cell-by-gene counts, and per-cell scores |

A tiled run passes when `compare_tiled_stages.py` reports every stage exact with
no label-string differences, and `compare_final.py` reports zero mismatches,
`counts_equal: true` and `scores_max_abs_diff: 0.0`.
