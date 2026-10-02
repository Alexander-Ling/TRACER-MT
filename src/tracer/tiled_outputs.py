"""Cell-by-gene counts, per-cell scores and AnnData from an (entity, gene,
count) table instead of a transcript-level table.

Tiled processing never assembles the whole slide's transcript table. Each tile
reduces its core transcripts to ``(entity, gene, count)`` rows; these are
additive across tiles, so the parent sums them and everything
``scripts/run_tracer.py::build_outputs`` derives from the transcript table is
reproduced exactly from the summed counts:

* ``cell_by_gene_tracer.h5ad``  - counts matrix (sums of counts).
* ``cell_scores.tsv.gz``        - per-cell coherence, computed ONCE on the
                                  merged presence matrix (it is not additive).

``build_outputs`` in ``scripts/run_tracer.py`` is the reference; the two must
be kept in sync (``tests`` / the validation harness compare them on a real
slide).
"""
from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd
import scipy.sparse as sp

# Must match scripts/run_tracer.py::UNASSIGNED_TOKENS.
UNASSIGNED_TOKENS = frozenset({
    "UNASSIGNED", "Unassigned", "unassigned",
    "DROP", "nan", "None", "", "0", "-1", "NA",
})
KEEP_ETYPES = {"cell", "partial", "component"}
COUNT_COLS = ["cell_id", "feature_name", "count"]


def counts_from_transcripts(df_post: pd.DataFrame, label_col: str = "tracer_id"
                            ) -> pd.DataFrame:
    """Reduce a transcript table to ``(cell_id, feature_name, count)`` rows,
    keeping the same rows ``build_outputs`` keeps (``_etype`` in
    cell/partial/component, else not an unassigned token)."""
    if label_col not in df_post.columns:
        label_col = next((c for c in ("tracer_id", "stitched", "cell_id")
                          if c in df_post.columns), label_col)
    if "_etype" in df_post.columns:
        keep = df_post["_etype"].astype(str).isin(KEEP_ETYPES)
    else:
        keep = ~df_post[label_col].astype(str).isin(UNASSIGNED_TOKENS)
    work = df_post.loc[keep, [label_col, "feature_name"]]
    work = work.rename(columns={label_col: "cell_id"})
    cg = (work.groupby(["cell_id", "feature_name"], observed=True).size()
              .rename("count").reset_index())
    cg["cell_id"] = cg["cell_id"].astype(str)
    cg["feature_name"] = cg["feature_name"].astype(str)
    cg["count"] = cg["count"].astype(np.int64)
    return cg[COUNT_COLS]


def merge_counts(parts: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Sum ``(cell_id, feature_name, count)`` tables from several tiles."""
    parts = [p for p in parts if len(p)]
    if not parts:
        return pd.DataFrame({"cell_id": [], "feature_name": [],
                             "count": np.zeros(0, dtype=np.int64)})
    cat = pd.concat(parts, ignore_index=True)
    return (cat.groupby(["cell_id", "feature_name"], sort=True, observed=True)["count"]
               .sum().reset_index())


def cell_gene_matrix_from_counts(counts: pd.DataFrame, *, min_transcripts: int,
                                 genes_npm: pd.DataFrame, exclude_ids=None):
    """Same return as ``tracer.metrics.build_cell_gene_matrix`` (``cell_ids``,
    ``genes_cell``, binary ``M``, ``col_idx``), from per-(cell, gene) counts.

    ``min_transcripts`` is applied to a cell's TOTAL transcript count (all genes,
    before restricting to the NPMI vocabulary), as in the original.
    """
    if exclude_ids is None:
        exclude_ids = {"UNASSIGNED"}
    c = counts
    if exclude_ids:
        c = c[~c["cell_id"].astype(str).isin(exclude_ids)]
    totals = c.groupby("cell_id", observed=True)["count"].sum()
    good = totals.index[totals.to_numpy() >= min_transcripts]
    c = c[c["cell_id"].isin(good)]
    all_genes = np.union1d(genes_npm["gene_i"].unique(), genes_npm["gene_j"].unique())
    c = c[c["feature_name"].astype(str).isin(all_genes)]

    cell_cat = pd.Categorical(c["cell_id"].astype(str))
    gene_cat = pd.Categorical(c["feature_name"].astype(str), categories=all_genes)
    rows_i = cell_cat.codes.astype(np.int32)
    cols_i = gene_cat.codes.astype(np.int32)
    ok = cols_i >= 0
    if not ok.all():
        rows_i, cols_i = rows_i[ok], cols_i[ok]
    csr = sp.coo_matrix(
        (np.ones(len(rows_i), dtype=np.int8), (rows_i, cols_i)),
        shape=(len(cell_cat.categories), len(all_genes))).tocsr()
    csr.data = np.ones_like(csr.data, dtype=np.int8)
    cell_ids = cell_cat.categories.to_numpy().astype(str)
    col_mass = np.asarray(csr.sum(axis=0)).ravel() > 0
    csr = csr[:, col_mass]
    genes_cell = all_genes[col_mass]
    col_idx = np.flatnonzero(col_mass).astype(np.int32)
    M = np.asarray(csr.todense(), dtype=np.int8)
    return cell_ids, genes_cell, M, col_idx


def scores_and_adata_from_counts(counts: pd.DataFrame, *, npmi_panel: pd.DataFrame,
                                 log: logging.Logger, min_tx: int = 5,
                                 tau: float | None = None,
                                 score_mode: str = "count"):
    """Counterpart of ``scripts/run_tracer.py::build_outputs`` working from the
    merged ``(cell_id, feature_name, count)`` table. Returns ``(scores, adata)``."""
    import anndata as ad
    from tracer.metrics import build_pmi_matrix, compute_cell_coherence

    if score_mode != "count":
        raise NotImplementedError(
            "tiled outputs support score_mode='count' only (the default)")
    if tau is None:
        mcol = "PMI" if "PMI" in npmi_panel.columns else "NPMI"
        tau = 0.2 if mcol == "PMI" else 0.05
        log.info("Coherence threshold=%.3f (auto, %s-scale panel)", tau, mcol)

    cell_ids, _genes, M, col_idx = cell_gene_matrix_from_counts(
        counts, min_transcripts=min_tx, genes_npm=npmi_panel,
        exclude_ids=set(UNASSIGNED_TOKENS))
    npmi_mat, _gix = build_pmi_matrix(npmi_panel)
    _, _, _, scores = compute_cell_coherence(
        M=M, col_idx=col_idx, npmi_mat=npmi_mat, threshold=tau, cell_ids=cell_ids)
    log.info("Per-cell scores (%s): %d with purity, %d total in cell-by-gene",
             score_mode, int(scores["purity_score"].notna().sum()), len(cell_ids))

    cell_cat = pd.Categorical(counts["cell_id"].astype(str))
    gene_cat = pd.Categorical(counts["feature_name"].astype(str))
    X = sp.csr_matrix(
        (counts["count"].to_numpy(dtype=np.int32),
         (cell_cat.codes, gene_cat.codes)),
        shape=(len(cell_cat.categories), len(gene_cat.categories)))
    obs = scores.set_index("cell_id").reindex(cell_cat.categories.astype(str))
    var = pd.DataFrame(index=pd.Index(gene_cat.categories.astype(str),
                                      name="feature_name"))
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.layers["counts"] = X.copy()
    return scores, adata
