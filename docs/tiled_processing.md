# Tiled (core + halo) processing

`scripts/run_tracer_tiled.py` runs TRACER on a slide that is cut into many
rectangular patches processed in parallel processes, and assembles the same
result as `scripts/run_tracer.py` on the whole slide.

## Why it is exact
TRACER's decisions about a transcript depend on its surroundings only. A patch
= a *core* plus a *halo* of surrounding transcripts. Only core transcripts are
kept from a patch, so every transcript is decided by exactly one patch, and the
halo guarantees that patch saw everything the decision depends on. This holds
only if the pipeline contains no slide-wide input, so three things are handled
explicitly:

1. **Group-cascade thresholds** are derived from the slide's residual pool.
   Phase A (input -> Rescue) returns each tile's *core-only* per-bin counts of
   that pool; the runner sums them (counts are additive, bins are absolute) and
   derives the thresholds once (`density_cascade.merge_bin_counts`,
   `cascade_thresholds_from_bin_counts`). Phase B then uses them for every tile.
2. **Entity names must not depend on what else is in the input.** Rescue breaks
   ties by label order, and cascade partials used to be numbered by a slide-wide
   counter. They are now named from their own threshold and hot bin.
3. **Row order**: a few tie-breaks depend on relative row order. Patches keep the
   source order, and equal-density cascade bins are visited in order of their
   earliest transcript (`density_cascade_phase1`).

`stitch.g_z_um` must be explicit and `overlaps_nucleus` present (the runner checks):
the only other slide-derived value, the within-cell dz estimate, is then unused.

## Using it
    python scripts/run_tracer_tiled.py --transcripts T.parquet --npmi prior.csv.gz \
      --platform xenium --outdir out --sample-name S --halo-um 200 \
      --target-core-tx 8000000 --workers 8 --scratch /fast/scratch

* `--target-core-tx`: max transcripts per core (the patch is ~1.3-1.5x larger).
  Peak memory is about 0.8 GB per million *patch* transcripts per worker.
* `--workers` / `--threads-per-worker`: processes and OpenMP/BLAS/numba threads
  each (default cores // workers). Workers are capped by `--memory-gb`.
* Restartable: finished tiles (`done.json`) are skipped; nothing is overwritten.
* Scratch holds patches and the spilled state between phases (roughly the size of
  the transcript table times the halo overhead, plus the phase-A state).

## Outputs
Same files as `run_tracer.py` (`transcripts_tracer_refined.parquet`,
`cell_by_gene_tracer.h5ad`, `cell_scores.tsv.gz`) plus `tile_plan.json`,
`global_params.json`, `tile_stats.tsv`, `config_receipt.json`. Differences: rows
are grouped by tile (source order within a tile) and categorical columns are plain
strings. Only `--score-mode count` is supported.

## Limits
* The halo must exceed the farthest dependency (200 um held on the slides tested;
  see `scripts/validation`). An entity larger than the halo would break exactness.
* Not exact for the NOSEG pipeline or when the cascade is disabled.

## Validation so far
Exactness is checked against a whole-slide run made with the same code and seed
(`scripts/validation`): every pipeline stage, per tile core, as partitions and as
label strings; then the final transcripts, cell-by-gene counts and scores.

| Slide | Transcripts | Layout | Result vs whole slide |
|---|---|---|---|
| patient_1 | 3.4M | 13 tiles, 200 um halo | all 15 stages exact; final outputs identical |
| 0059267 | 24.2M | 32 tiles, 100 um halo | all 15 stages exact; final outputs identical |
| 0059267 | 24.2M | 32 tiles, 200 um halo | all 15 stages exact; final outputs identical |

The 100 um and 200 um runs of 0059267 are also identical to each other. scRNA
prior, seed 1, `stitch.g_z_um = 1.0`.

0059667 (134M transcripts, which was OOM-killed as a whole-slide run) ran tiled
(128 tiles, 200 um halo, 23 workers) in 38 min with a container peak of 84 GB. It
has no whole-slide reference, so only structural checks were possible (all rows
present once, cell count equal to the earlier partial whole-slide run).
