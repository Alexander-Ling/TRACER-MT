#!/usr/bin/env python3
"""Tiled TRACER runner: many core+halo patches in parallel processes.

Produces the same per-transcript result as ``scripts/run_tracer.py`` on the
whole slide (validated stage by stage by the harness in ``patch_test``), but
never holds the whole slide in memory and uses many cores at once.

Pipeline
--------
 0. plan      one streaming pass -> 50 um density histogram -> balanced k-d
              split into CORES (every transcript in exactly one core).
 1. cut       one streaming pass -> one parquet per tile with core + HALO
              transcripts in source order and an ``_is_core`` flag.
 2. phase A   (parallel) input -> Prune -> Phase 1 -> Rescue on each patch;
              spill state to scratch; return the CORE-only histogram of the
              Group-cascade residual pool.
 3. reduce    sum the histograms -> the slide-wide cascade thresholds, exactly
              as a whole-slide run would derive them.
 4. phase B   (parallel) Group -> ... -> Finalize with the global thresholds;
              keep CORE rows only; write a part parquet + (entity, gene, count).
 5. assemble  concatenate parts; sum counts; compute per-cell scores once.

Everything is restartable: a tile's step is skipped when its ``done.json``
exists; nothing is ever overwritten or deleted except scratch after success
(and only with ``--clean-scratch``).

Example
-------
    python scripts/run_tracer_tiled.py \\
      --transcripts transcripts_standardized.parquet --npmi prior.csv.gz \\
      --platform xenium --outdir out/slide --sample-name slide \\
      --halo-um 200 --target-core-tx 8000000 --workers 8
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import multiprocessing as mp
import os
import pickle
import sys
import time
import traceback
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT / "src"), str(_REPO_ROOT), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np          # noqa: E402
import pandas as pd         # noqa: E402

# Peak resident memory of a pipeline run, per million transcripts in the patch
# Used only to cap concurrency. Measured on 27 completed arms plus one OOM-killed
# arm (14 slides, 0.9-4.4M-transcript patches, 19-23 workers): the container total
# was at most (GB_FIXED + 0.92 * M_largest_patch) GB per worker, reached by the
# arm that was OOM-killed at 20 workers on 4.25M-transcript patches. 1.2 leaves
# about 25% headroom over that.
GB_PER_MTX = 1.2
GB_FIXED = 1.3      # per-worker overhead (interpreter, torch/numba, prior); patient_1 workers peaked at ~1.5 GB on ~0.6M-transcript patches
THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--transcripts", required=True, type=Path)
    p.add_argument("--npmi", required=True, type=Path)
    p.add_argument("--outdir", required=True, type=Path)
    p.add_argument("--sample-name", required=True)
    p.add_argument("--platform", default="xenium")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--pmi-threshold", type=float, default=None)
    p.add_argument("--user-config", type=Path, default=None)
    p.add_argument("--g-z-um", default=None,
                   help="Override stitch.g_z_um (positive float).")
    p.add_argument("--min-tx-per-cell-for-scores", type=int, default=5)
    p.add_argument("--tau", type=float, default=None)
    g = p.add_argument_group("tiling")
    g.add_argument("--halo-um", type=float, default=200.0)
    g.add_argument("--target-core-tx", type=int, default=8_000_000,
                   help="Max transcripts per core (patch is ~1.3-1.5x larger).")
    g.add_argument("--bin-um", type=float, default=50.0,
                   help="Histogram bin for planning; core edges are multiples of it.")
    g.add_argument("--max-tiles", type=int, default=None)
    h = p.add_argument_group("parallelism")
    h.add_argument("--workers", type=int, default=8)
    h.add_argument("--threads-per-worker", type=int, default=None,
                   help="OpenMP/BLAS/numba threads per worker (default: cores // workers).")
    h.add_argument("--memory-gb", type=float, default=96.0,
                   help="RAM budget used to cap concurrent workers.")
    h.add_argument("--scratch", type=Path, default=None,
                   help="Scratch dir for patches and spilled state (default: OUTDIR/scratch).")
    p.add_argument("--dump-stages", type=Path, default=None,
                   help="Write per-stage labels per tile (validation only; heavy).")
    p.add_argument("--stop-after", choices=["plan", "cut", "a", "reduce", "b"],
                   default=None, help="Stop after a step (for debugging).")
    p.add_argument("--clean-scratch", action="store_true",
                   help="Delete scratch after a successful assembly.")
    return p


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _peak_rss_gb() -> float:
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    except Exception:
        return float("nan")


def _json_default(o):
    """Numpy scalars/arrays -> plain Python (never silently stringify numbers)."""
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _write_json(path: Path, obj) -> None:
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=_json_default))
    tmp.replace(path)


def _read_json(path: Path):
    return json.loads(Path(path).read_text())


def tile_dir(root: Path, tile_id: int) -> Path:
    return root / f"tile_{tile_id:04d}"


# ---------------------------------------------------------------------------
# worker side
# ---------------------------------------------------------------------------
_W: dict = {}


def _init_worker(args_dict: dict) -> None:
    """Runs once per worker process: load the prior and build the config."""
    import logging
    import torch
    import run_tracer as rt
    import tracer.pipeline as pipeline

    try:
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    except Exception:
        pass
    log = logging.getLogger("tile_worker")
    log.setLevel(logging.WARNING)
    g_z = args_dict["g_z_um"]
    _W["rt"] = rt
    _W["pipeline"] = pipeline
    _W["args"] = args_dict
    _W["panel"] = rt.load_npmi_panel(Path(args_dict["npmi"]), logging.getLogger("quiet"))
    _W["cfg"] = rt.build_config(
        platform_name=args_dict["platform"],
        user_config=Path(args_dict["user_config"]) if args_dict["user_config"] else None,
        pmi_threshold_override=args_dict["pmi_threshold"],
        g_z_um_override=(float(g_z) if g_z not in (None, "auto") else g_z),
        log=log)


def _set_dump(tid: int) -> None:
    d = _W["args"].get("dump_stages")
    if d:
        os.environ["TRACER_DUMP_DIR"] = str(Path(d) / f"tile_{tid:04d}")
    else:
        os.environ.pop("TRACER_DUMP_DIR", None)


def _phase_a_task(tid: int) -> dict:
    import logging
    from tracer import tiling
    from tracer.pipeline import GlobalParams, segmented_phase_a
    from tracer.density_cascade import residual_pool_bin_counts

    a = _W["args"]
    out = tile_dir(Path(a["scratch"]) / "phase_a", tid)
    if (out / "done.json").exists():
        return _read_json(out / "done.json")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    df, core_ids = tiling.read_patch(tiling.patch_path(Path(a["scratch"]) / "patches", tid))
    df = _W["rt"].normalize_transcripts(df, logging.getLogger("quiet"))
    n_patch = len(df)
    _set_dump(tid)
    state = segmented_phase_a(df, _W["panel"], _W["cfg"],
                              GlobalParams(auto_Gz=float(a["auto_gz"])))
    del df
    dfr = state["df_rescued"]
    if not (isinstance(dfr.index, pd.RangeIndex)
            or np.array_equal(dfr.index.to_numpy(), np.arange(len(dfr)))):
        raise RuntimeError("df_rescued does not have a positional index; "
                           "cannot spill/restore it faithfully")
    keep = np.isin(dfr["transcript_id"].to_numpy(), core_ids)
    bx, by, cnt = residual_pool_bin_counts(dfr, state["aux"], G=2.0, keep=keep)
    np.savez(out / "pool_counts.npz", bx=bx, by=by, cnt=cnt)
    np.save(out / "core_ids.npy", core_ids)
    dfr.reset_index(drop=True).to_parquet(out / "state.parquet", index=False,
                                          compression="snappy")
    with open(out / "aux.pkl", "wb") as f:
        pickle.dump(state["aux"], f, protocol=pickle.HIGHEST_PROTOCOL)
    state["input_cell_id"].rename("cell_id").rename_axis("transcript_id").reset_index() \
        .to_parquet(out / "input_cell_id.parquet", index=False)
    _write_json(out / "progression.json", state["progression"])
    meta = {"tile_id": tid, "n_patch": n_patch, "n_core": int(len(core_ids)),
            "n_pool_core": int(cnt.sum()), "seconds": time.perf_counter() - t0,
            "peak_rss_gb": _peak_rss_gb()}
    _write_json(out / "done.json", meta)
    return meta


def _phase_b_task(tid: int) -> dict:
    from tracer import tiled_outputs
    from tracer.pipeline import GlobalParams, segmented_phase_b, _resolve_pipeline_cfg

    a = _W["args"]
    ain = tile_dir(Path(a["scratch"]) / "phase_a", tid)
    out = tile_dir(Path(a["scratch"]) / "phase_b", tid)
    if (out / "done.json").exists():
        return _read_json(out / "done.json")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    gp = _read_json(Path(a["outdir"]) / "global_params.json")
    core_ids = np.load(ain / "core_ids.npy")
    with open(ain / "aux.pkl", "rb") as f:
        aux = pickle.load(f)
    icid = pd.read_parquet(ain / "input_cell_id.parquet")
    input_cell_id = pd.Series(icid["cell_id"].to_numpy(),
                              index=icid["transcript_id"].to_numpy())
    state = {
        "df_rescued": pd.read_parquet(ain / "state.parquet"),
        "aux": aux, "cfg": _resolve_pipeline_cfg(_W["cfg"]),
        "progression": _read_json(ain / "progression.json"),
        "input_cell_id": input_cell_id, "auto_Gz": float(a["auto_gz"]),
    }
    _set_dump(tid)
    df_out, _prog = segmented_phase_b(
        state, GlobalParams(cascade_thresholds=tuple(gp["thresholds"])))
    core = np.isin(df_out["transcript_id"].to_numpy(), core_ids)
    core_df = df_out.loc[core].copy()
    del df_out
    labels = pd.unique(core_df["tracer_id"].astype(str))
    bad = [l for l in labels if l.startswith("UNASSIGNED") or l.startswith("tile")]
    if bad:
        raise RuntimeError(
            f"tile {tid}: labels {bad[:3]} are not patch-invariant (UNASSIGNED_*/tile*). "
            "Tiled processing requires the cascade to be enabled.")
    for c in core_df.columns:                      # stable schema across tiles
        if str(core_df[c].dtype) == "category":
            core_df[c] = core_df[c].astype(str)
    core_df.to_parquet(out / "core.parquet", index=False, compression="snappy")
    tiled_outputs.counts_from_transcripts(core_df).to_parquet(
        out / "counts.parquet", index=False)
    meta = {"tile_id": tid, "n_core": int(len(core_df)),
            "seconds": time.perf_counter() - t0, "peak_rss_gb": _peak_rss_gb()}
    _write_json(out / "done.json", meta)
    return meta


# ---------------------------------------------------------------------------
# driver side
# ---------------------------------------------------------------------------
def _run_pool(task, tile_ids: list[int], n_workers: int, args_dict: dict,
              label: str) -> dict[int, dict]:
    """Run ``task(tid)`` for each tile in a spawn pool; return metas. Raises
    after ALL tiles were attempted if any failed (completed ones stay done)."""
    results: dict[int, dict] = {}
    failures: dict[int, str] = {}
    t0 = time.perf_counter()
    ctx = mp.get_context("spawn")
    with cf.ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx,
                                initializer=_init_worker,
                                initargs=(args_dict,)) as ex:
        futs = {ex.submit(task, tid): tid for tid in tile_ids}
        for i, fu in enumerate(cf.as_completed(futs), 1):
            tid = futs[fu]
            try:
                results[tid] = fu.result()
                _log(f"{label}: tile {tid:04d} done "
                     f"({i}/{len(tile_ids)}, {results[tid].get('seconds', 0):.0f}s, "
                     f"peak {results[tid].get('peak_rss_gb', float('nan')):.1f} GB)")
            except Exception as exc:                       # keep going
                failures[tid] = "".join(traceback.format_exception_only(type(exc), exc)).strip()
                _log(f"{label}: tile {tid:04d} FAILED: {failures[tid]}")
    _log(f"{label}: finished in {time.perf_counter() - t0:.0f}s "
         f"({len(results)} ok, {len(failures)} failed)")
    if failures:
        raise SystemExit(f"{label}: {len(failures)} tile(s) failed: "
                         f"{sorted(failures)}; re-run to resume. First error: "
                         f"{next(iter(failures.values()))}")
    return results


def main(argv=None) -> int:
    import logging
    args = build_argparser().parse_args(argv)
    from tracer import tiling, tiled_outputs
    from tracer.density_cascade import merge_bin_counts, cascade_thresholds_from_bin_counts
    import tracer.pipeline as pipeline
    import run_tracer as rt
    import pyarrow as pa
    import pyarrow.parquet as pq

    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    scratch = (args.scratch or (outdir / "scratch")).resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    (outdir / "outputs").mkdir(exist_ok=True)

    log = rt.setup_logging(outdir)
    t_start = time.perf_counter()

    # ---- preflight: things a patch cannot re-derive from its own data ----
    log.info("Preflight")
    g_z = None if args.g_z_um in (None, "auto") else float(args.g_z_um)
    cfg0 = rt.build_config(platform_name=args.platform, user_config=args.user_config,
                           pmi_threshold_override=args.pmi_threshold,
                           g_z_um_override=g_z if args.g_z_um is not None else None,
                           log=log)
    gz_cfg = cfg0.stitch.g_z_um
    if not isinstance(gz_cfg, (int, float)):
        raise SystemExit(f"tiled mode needs an explicit stitch.g_z_um (got {gz_cfg!r}); "
                         "the auto value is estimated from the whole slide")
    import pyarrow.parquet as _pq
    cols = set(_pq.ParquetFile(str(args.transcripts)).schema_arrow.names)
    need = {"x", "y", "z", "feature_name", "cell_id", "transcript_id", "overlaps_nucleus"}
    if need - cols:
        raise SystemExit(f"transcripts parquet lacks columns {sorted(need - cols)}; "
                         "tiled mode requires the nuclear-seed path and transcript_id")

    # ---- thread budget ----
    ncpu = os.cpu_count() or 8
    threads = args.threads_per_worker or max(1, ncpu // max(1, args.workers))
    for k in THREAD_ENV:
        os.environ[k] = str(threads)            # inherited by spawned workers
    args_dict = {
        "npmi": str(args.npmi), "platform": args.platform,
        "user_config": str(args.user_config) if args.user_config else None,
        "pmi_threshold": args.pmi_threshold,
        "g_z_um": args.g_z_um if args.g_z_um is not None else None,
        "scratch": str(scratch), "outdir": str(outdir),
        "dump_stages": str(args.dump_stages) if args.dump_stages else None,
        "auto_gz": float(gz_cfg),
    }

    # ---- 0. plan ----
    plan_path = outdir / "tile_plan.json"
    src_stat = Path(args.transcripts).stat()
    meta_now = {"source": str(args.transcripts), "size": src_stat.st_size,
                "halo_um": args.halo_um, "bin_um": args.bin_um,
                "target_core_tx": args.target_core_tx}
    if plan_path.exists():
        tiles, meta = tiling.load_plan(plan_path)
        if {k: meta.get(k) for k in meta_now} != meta_now:
            raise SystemExit(f"{plan_path} was made with different settings "
                             f"({meta}); use a new --outdir")
        log.info("Plan: reusing %s (%d tiles)", plan_path, len(tiles))
    else:
        log.info("Plan: scanning density")
        dens = tiling.scan_density(args.transcripts, bin_um=args.bin_um)
        tiles = tiling.plan_tiles(dens, target_core_tx=args.target_core_tx,
                                  max_tiles=args.max_tiles)
        tiling.save_plan(tiles, plan_path, n_total=dens.n_total, **meta_now)
        log.info("Plan: %d transcripts -> %d tiles (largest core %d)",
                 dens.n_total, len(tiles), max(t.n_core_est for t in tiles))
    if args.stop_after == "plan":
        return 0

    # ---- 1. cut ----
    patches = scratch / "patches"
    t0 = time.perf_counter()
    counts = tiling.cut_patches(args.transcripts, tiles, args.halo_um, patches)
    if counts:
        _write_json(scratch / f"patch_counts_{int(time.time())}.json", counts)
        log.info("Cut %d patches in %.0fs (patch/core transcripts: %s)", len(counts),
                 time.perf_counter() - t0,
                 {k: (v["n_patch"], v["n_core"]) for k, v in list(counts.items())[:4]})
    if args.stop_after == "cut":
        return 0

    # ---- concurrency from the memory budget ----
    n_patch_est = {}
    for t in tiles:
        pf = _pq.ParquetFile(str(tiling.patch_path(patches, t.tile_id)))
        n_patch_est[t.tile_id] = pf.metadata.num_rows
    order = sorted(n_patch_est, key=lambda k: -n_patch_est[k])
    biggest_gb = GB_FIXED + GB_PER_MTX * max(n_patch_est.values()) / 1e6
    mem_cap = max(1, int(args.memory_gb // max(biggest_gb, 0.1)))
    workers = max(1, min(args.workers, mem_cap, len(tiles)))
    log.info("Workers: %d (requested %d; memory cap %d from largest patch %.1f GB "
             "est.; %d threads each)", workers, args.workers, mem_cap, biggest_gb, threads)

    # ---- 2. phase A ----
    meta_a = _run_pool(_phase_a_task, order, workers, args_dict, "phase A")
    if args.stop_after == "a":
        return 0

    # ---- 3. reduce -> slide-wide cascade thresholds ----
    parts = []
    for t in tiles:
        z = np.load(tile_dir(scratch / "phase_a", t.tile_id) / "pool_counts.npz")
        parts.append((z["bx"], z["by"], z["cnt"]))
    bx, by, cnt = merge_bin_counts(parts)
    ceiling, floor, _curve = cascade_thresholds_from_bin_counts(
        bx, by, cnt, territory_radius_bins=1,
        target_cov=pipeline.PHASE1_SEG_RESIDUAL_CASCADE_TARGET_COV,
        hard_min=pipeline.PHASE1_SEG_RESIDUAL_CASCADE_HARD_MIN)
    gp = {"ceiling": ceiling, "floor": floor, "n_pool": int(cnt.sum()),
          "thresholds": list(range(ceiling, floor - 1, -1))}
    _write_json(outdir / "global_params.json", gp)
    log.info("Global cascade thresholds: ceiling=%d floor=%d (pool %d tx)",
             ceiling, floor, gp["n_pool"])
    if args.stop_after == "reduce":
        return 0

    # ---- 4. phase B ----
    meta_b = _run_pool(_phase_b_task, order, workers, args_dict, "phase B")
    if args.stop_after == "b":
        return 0

    # ---- 5. assemble ----
    log.info("Assembling outputs")
    outputs = outdir / "outputs"
    final = outputs / "transcripts_tracer_refined.parquet"
    if final.exists():
        raise SystemExit(f"{final} exists; refusing to overwrite")
    writer = None
    n_rows = 0
    tmp = Path(str(final) + ".partial")
    try:
        for t in tiles:
            tbl = pq.read_table(tile_dir(scratch / "phase_b", t.tile_id) / "core.parquet")
            if writer is None:
                schema = tbl.schema
                writer = pq.ParquetWriter(str(tmp), schema, compression="snappy")
            elif not tbl.schema.equals(schema):
                tbl = tbl.cast(schema)
            writer.write_table(tbl)
            n_rows += tbl.num_rows
    finally:
        if writer is not None:
            writer.close()
    tmp.replace(final)
    log.info("Wrote %s (%d rows)", final, n_rows)

    counts_all = tiled_outputs.merge_counts(
        pd.read_parquet(tile_dir(scratch / "phase_b", t.tile_id) / "counts.parquet")
        for t in tiles)
    panel = rt.load_npmi_panel(args.npmi, log)
    scores, adata = tiled_outputs.scores_and_adata_from_counts(
        counts_all, npmi_panel=panel, log=log,
        min_tx=args.min_tx_per_cell_for_scores, tau=args.tau)
    adata.write_h5ad(outputs / "cell_by_gene_tracer.h5ad")
    scores.to_csv(outputs / "cell_scores.tsv.gz", sep="\t", index=False, compression="gzip")

    tile_rows = []
    for t in tiles:
        a, b = meta_a[t.tile_id], meta_b[t.tile_id]
        tile_rows.append({"tile_id": t.tile_id, "x0": t.x0, "x1": t.x1, "y0": t.y0,
                          "y1": t.y1, "n_patch": a["n_patch"], "n_core": a["n_core"],
                          "phase_a_s": a["seconds"], "phase_a_peak_gb": a["peak_rss_gb"],
                          "phase_b_s": b["seconds"], "phase_b_peak_gb": b["peak_rss_gb"]})
    pd.DataFrame(tile_rows).to_csv(outdir / "tile_stats.tsv", sep="\t", index=False)
    _write_json(outdir / "config_receipt.json", {
        "command": " ".join(sys.argv), "sample_name": args.sample_name,
        "args": {k: str(v) for k, v in vars(args).items()},
        "git_commit": rt.git_commit_hash(), "n_tiles": len(tiles),
        "global_params": gp, "workers": workers, "threads_per_worker": threads,
        "wall_seconds": time.perf_counter() - t_start,
        "note": "Rows in transcripts_tracer_refined.parquet are grouped by tile "
                "(source order within a tile); categorical columns are plain strings.",
    })
    log.info("Done in %.0fs. Outputs in %s", time.perf_counter() - t_start, outputs)
    if args.clean_scratch:
        import shutil
        shutil.rmtree(scratch)
        log.info("Removed scratch %s", scratch)
    return 0


if __name__ == "__main__":
    sys.exit(main())
