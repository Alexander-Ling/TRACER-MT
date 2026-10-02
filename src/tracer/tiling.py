"""Exact core + halo tiling of a transcript table.

A slide is cut into rectangular CORES that tile the plane (every transcript is
in exactly one core). Each core is processed together with a HALO of
surrounding transcripts (the "patch"), so that for transcripts inside the core
the result equals processing the whole slide, provided the halo is wider than
the longest chain of dependencies. Only core transcripts are kept from a patch.

This module has no TRACER-algorithm logic. It provides:

* ``scan_density``  - one streaming pass: 2D transcript-count histogram.
* ``plan_tiles``    - balanced k-d split of the histogram into cores.
* ``cut_patches``   - one streaming pass: write each patch (core + halo) to its
                      own parquet file, preserving the source row order, with
                      an ``_is_core`` flag.
* ``read_patch``    - load a patch and its sorted core transcript ids.

Core membership is half-open, ``x0 <= x < x1`` and ``y0 <= y < y1``, decided
in float64 on the stored float32 coordinates. Core edges are multiples of the
histogram bin (default 50 um), hence also multiples of the 2 um bins the
pipeline works in. The outermost edges are +-inf.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

IS_CORE = "_is_core"
_KEY_OFFSET = 1 << 20           # bin index offset so packed keys stay positive
_KEY_SPAN = 1 << 21


@dataclass(frozen=True)
class Tile:
    tile_id: int
    x0: float
    x1: float
    y0: float
    y1: float
    n_core_est: int = 0          # transcripts in the core (exact: from the histogram)

    def patch_bounds(self, halo_um: float) -> tuple[float, float, float, float]:
        return (self.x0 - halo_um, self.x1 + halo_um,
                self.y0 - halo_um, self.y1 + halo_um)


@dataclass
class Density:
    bin_um: float
    bx: np.ndarray               # occupied bin indices, floor(x / bin_um)
    by: np.ndarray
    cnt: np.ndarray
    n_total: int

    def extent_um(self) -> tuple[float, float, float, float]:
        b = self.bin_um
        return (float(self.bx.min()) * b, float(self.bx.max() + 1) * b,
                float(self.by.min()) * b, float(self.by.max() + 1) * b)


# --------------------------------------------------------------------------
# Pass 0: density histogram
# --------------------------------------------------------------------------
def scan_density(path: str | Path, *, bin_um: float = 50.0,
                 x_col: str = "x", y_col: str = "y",
                 batch_rows: int = 4_000_000) -> Density:
    """Stream the x/y columns and count transcripts per ``bin_um`` bin."""
    pf = pq.ParquetFile(str(path))
    keys: list[np.ndarray] = []
    cnts: list[np.ndarray] = []
    n_total = 0

    def _compact():
        nonlocal keys, cnts
        if len(keys) > 1:
            k = np.concatenate(keys)
            c = np.concatenate(cnts)
            uk, inv = np.unique(k, return_inverse=True)
            tot = np.zeros(len(uk), dtype=np.int64)
            np.add.at(tot, inv, c)
            keys, cnts = [uk], [tot]

    for batch in pf.iter_batches(batch_size=batch_rows, columns=[x_col, y_col]):
        x = batch.column(0).to_numpy(zero_copy_only=False).astype(np.float64)
        y = batch.column(1).to_numpy(zero_copy_only=False).astype(np.float64)
        bx = np.floor(x / bin_um).astype(np.int64) + _KEY_OFFSET
        by = np.floor(y / bin_um).astype(np.int64) + _KEY_OFFSET
        uk, c = np.unique(bx * _KEY_SPAN + by, return_counts=True)
        keys.append(uk)
        cnts.append(c.astype(np.int64))
        n_total += len(x)
        if len(keys) >= 16:
            _compact()
    _compact()
    if not keys:
        raise ValueError(f"no rows in {path}")
    k, c = keys[0], cnts[0]
    return Density(bin_um=float(bin_um),
                   bx=(k // _KEY_SPAN) - _KEY_OFFSET,
                   by=(k % _KEY_SPAN) - _KEY_OFFSET,
                   cnt=c, n_total=int(n_total))


# --------------------------------------------------------------------------
# Plan: balanced k-d split
# --------------------------------------------------------------------------
def plan_tiles(d: Density, *, target_core_tx: int,
               max_tiles: Optional[int] = None) -> list[Tile]:
    """Recursively bisect the occupied area so each core has at most
    ``target_core_tx`` transcripts (when the bins allow it). Splits are at bin
    boundaries, along the longer occupied side, at the weighted median.

    The outermost cores extend to +-inf so every transcript is owned.
    """
    b = d.bin_um
    leaves: list[tuple[int, int, int, int, int]] = []   # bx_lo,bx_hi,by_lo,by_hi,n

    def split(idx: np.ndarray, bx_lo: int, bx_hi: int, by_lo: int, by_hi: int):
        n = int(d.cnt[idx].sum())
        bxs, bys = d.bx[idx], d.by[idx]
        if n <= target_core_tx or len(idx) <= 1:
            leaves.append((bx_lo, bx_hi, by_lo, by_hi, n))
            return
        ext_x = int(bxs.max() - bxs.min())
        ext_y = int(bys.max() - bys.min())
        for axis in ((0, 1) if ext_x >= ext_y else (1, 0)):
            coord = bxs if axis == 0 else bys
            if coord.max() == coord.min():
                continue
            order = np.argsort(coord, kind="stable")
            cs = np.cumsum(d.cnt[idx][order])
            half = cs[-1] / 2.0
            k = int(np.searchsorted(cs, half))          # first position >= half
            cut = int(coord[order][k]) + 1              # boundary above that bin
            cut = min(max(cut, int(coord.min()) + 1), int(coord.max()))
            left = coord < cut
            right = ~left
            if not left.any() or not right.any():
                continue
            if axis == 0:
                split(idx[left], bx_lo, cut, by_lo, by_hi)
                split(idx[right], cut, bx_hi, by_lo, by_hi)
            else:
                split(idx[left], bx_lo, bx_hi, by_lo, cut)
                split(idx[right], bx_lo, bx_hi, cut, by_hi)
            return
        leaves.append((bx_lo, bx_hi, by_lo, by_hi, n))

    BIG = 1 << 40
    split(np.arange(len(d.cnt)), -BIG, BIG, -BIG, BIG)
    if max_tiles is not None and len(leaves) > max_tiles:
        raise ValueError(f"plan has {len(leaves)} tiles > max_tiles={max_tiles}; "
                         f"raise target_core_tx")
    inf = float("inf")
    tiles = []
    for i, (xl, xh, yl, yh, n) in enumerate(leaves):
        tiles.append(Tile(
            tile_id=i,
            x0=-inf if xl == -BIG else xl * b, x1=inf if xh == BIG else xh * b,
            y0=-inf if yl == -BIG else yl * b, y1=inf if yh == BIG else yh * b,
            n_core_est=int(n)))
    return tiles


def save_plan(tiles: Iterable[Tile], path: str | Path, **meta) -> None:
    rec = {"meta": meta, "tiles": [asdict(t) for t in tiles]}
    Path(path).write_text(json.dumps(rec, indent=1))


def load_plan(path: str | Path) -> tuple[list[Tile], dict]:
    rec = json.loads(Path(path).read_text())
    return [Tile(**t) for t in rec["tiles"]], rec["meta"]


# --------------------------------------------------------------------------
# Cut patches (one streaming pass)
# --------------------------------------------------------------------------
def tile_masks(x: np.ndarray, y: np.ndarray, t: Tile, halo_um: float):
    """``(in_patch, in_core)`` boolean arrays for one tile (float64 compares)."""
    px0, px1, py0, py1 = t.patch_bounds(halo_um)
    in_patch = (x >= px0) & (x < px1) & (y >= py0) & (y < py1)
    in_core = (x >= t.x0) & (x < t.x1) & (y >= t.y0) & (y < t.y1)
    return in_patch, in_core


def patch_path(out_dir: str | Path, tile_id: int) -> Path:
    return Path(out_dir) / f"patch_{tile_id:04d}.parquet"


def cut_patches(src: str | Path, tiles: list[Tile], halo_um: float,
                out_dir: str | Path, *, x_col: str = "x", y_col: str = "y",
                batch_rows: int = 2_000_000,
                skip_existing: bool = True) -> dict[int, dict]:
    """Write one parquet per tile containing core + halo transcripts of
    ``src`` in the source row order, plus a boolean ``_is_core`` column.

    Files are written under a ``.partial`` name and renamed on success, so a
    killed run never leaves a patch that looks finished. Returns per-tile
    ``{"n_patch", "n_core"}`` counts (for tiles written in this call).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [t for t in tiles
            if not (skip_existing and patch_path(out_dir, t.tile_id).exists())]
    if not todo:
        return {}
    pf = pq.ParquetFile(str(src))
    schema = pf.schema_arrow.append(pa.field(IS_CORE, pa.bool_()))
    writers = {t.tile_id: pq.ParquetWriter(
        str(patch_path(out_dir, t.tile_id)) + ".partial", schema,
        compression="snappy") for t in todo}
    counts = {t.tile_id: {"n_patch": 0, "n_core": 0} for t in todo}
    try:
        for batch in pf.iter_batches(batch_size=batch_rows):
            tbl = pa.Table.from_batches([batch])
            x = tbl.column(x_col).to_numpy().astype(np.float64)
            y = tbl.column(y_col).to_numpy().astype(np.float64)
            for t in todo:
                in_patch, in_core = tile_masks(x, y, t, halo_um)
                if not in_patch.any():
                    continue
                sub = tbl.filter(pa.array(in_patch))
                sub = sub.append_column(IS_CORE, pa.array(in_core[in_patch]))
                writers[t.tile_id].write_table(sub)
                counts[t.tile_id]["n_patch"] += int(in_patch.sum())
                counts[t.tile_id]["n_core"] += int(in_core.sum())
    except BaseException:
        for w in writers.values():
            w.close()
        raise
    for w in writers.values():
        w.close()
    for t in todo:
        p = patch_path(out_dir, t.tile_id)
        Path(str(p) + ".partial").replace(p)
    return counts


def read_patch(path: str | Path):
    """Load a patch parquet. Returns ``(df_without_is_core, core_ids)`` where
    ``core_ids`` is the sorted int64 array of core transcript ids."""
    import pandas as pd
    df = pd.read_parquet(path)
    core = df.pop(IS_CORE).to_numpy(dtype=bool)
    core_ids = np.sort(df["transcript_id"].to_numpy()[core].astype(np.int64))
    return df, core_ids
