"""Compare final outputs of a whole-slide run vs a tiled run (transcripts, h5ad, scores)."""
import argparse, json, numpy as np, pandas as pd
p = argparse.ArgumentParser(); p.add_argument('--ref', required=True); p.add_argument('--test', required=True)
a = p.parse_args()
res = {}
cols = ['transcript_id', 'tracer_id', '_etype', 'cell_id', 'feature_name']
r = pd.read_parquet(f'{a.ref}/transcripts_tracer_refined.parquet', columns=cols)
t = pd.read_parquet(f'{a.test}/transcripts_tracer_refined.parquet', columns=cols)
res['rows_ref'], res['rows_test'] = len(r), len(t)
m = r.merge(t, on='transcript_id', how='outer', suffixes=('_r', '_t'), indicator=True)
res['only_ref'] = int((m._merge == 'left_only').sum()); res['only_test'] = int((m._merge == 'right_only').sum())
b = m[m._merge == 'both']
for c in ['tracer_id', '_etype', 'cell_id', 'feature_name']:
    res[f'{c}_mismatch'] = int((b[f'{c}_r'].astype(str) != b[f'{c}_t'].astype(str)).sum())
res['duplicate_ids_test'] = int(t.transcript_id.duplicated().sum())
import anndata as ad, scipy.sparse as sp
A = ad.read_h5ad(f'{a.ref}/cell_by_gene_tracer.h5ad'); B = ad.read_h5ad(f'{a.test}/cell_by_gene_tracer.h5ad')
res['adata_shape_ref'], res['adata_shape_test'] = list(A.shape), list(B.shape)
if A.shape == B.shape and (A.obs_names == B.obs_names).all() and (A.var_names == B.var_names).all():
    res['counts_equal'] = bool((A.X != B.X).nnz == 0)
    oa, ob = A.obs.sort_index(axis=1), B.obs.sort_index(axis=1)
    res['obs_columns_equal'] = list(oa.columns) == list(ob.columns)
    num = oa.select_dtypes('number').columns
    res['scores_max_abs_diff'] = float(np.nanmax(np.abs(oa[num].to_numpy(float) - ob[num].to_numpy(float)))) if len(num) else None
    res['scores_nan_pattern_equal'] = bool((oa[num].isna().to_numpy() == ob[num].isna().to_numpy()).all()) if len(num) else None
else:
    res['counts_equal'] = False; res['obs_var_names_equal'] = False
sa = pd.read_csv(f'{a.ref}/cell_scores.tsv.gz', sep='\t').sort_values('cell_id').reset_index(drop=True)
sb = pd.read_csv(f'{a.test}/cell_scores.tsv.gz', sep='\t').sort_values('cell_id').reset_index(drop=True)
res['scores_tsv_equal'] = bool(sa.equals(sb)) if sa.shape == sb.shape else False
res['scores_rows'] = [len(sa), len(sb)]
print(json.dumps(res, indent=1))
