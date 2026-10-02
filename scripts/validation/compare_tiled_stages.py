"""Per-stage comparison of a tiled run (per-tile stage dumps) against a whole-slide run.
For each tile, stage files are restricted to that tile's core transcripts and compared as
partitions (labels may be renamed); results are summed over tiles."""
import argparse, glob, json, os, numpy as np, pandas as pd
p = argparse.ArgumentParser()
p.add_argument('--ref-stages', required=True); p.add_argument('--tiled-stages', required=True)
p.add_argument('--phase-a', required=True, help='scratch/phase_a dir (has tile_XXXX/core_ids.npy)')
p.add_argument('--out', default=None)
a = p.parse_args()
tiles = sorted(d for d in os.listdir(a.tiled_stages) if d.startswith('tile_'))
ref_files = sorted(glob.glob(os.path.join(a.ref_stages, '*.parquet')))
rows = []
for rf in ref_files:
    name = os.path.basename(rf)
    ref = pd.read_parquet(rf).set_index('transcript_id')
    tot = dict(stage=name, n=0, status_mismatch=0, etype_mismatch=0, off_majority=0, ref_split=0, test_split=0, string_mismatch=0, tiles_missing=0)
    for t in tiles:
        tf = os.path.join(a.tiled_stages, t, name)
        if not os.path.exists(tf): tot['tiles_missing'] += 1; continue
        core = np.load(os.path.join(a.phase_a, t, 'core_ids.npy'))
        te = pd.read_parquet(tf).set_index('transcript_id')
        te = te.loc[te.index.isin(core)]
        r = ref.loc[te.index]
        rl, tl = r.label.astype(str), te.label.astype(str)
        ra, ta = rl.eq('-1'), tl.eq('-1')
        both = ~ra & ~ta
        pairs = pd.DataFrame({'r': rl[both], 't': tl[both]}).groupby(['r', 't']).size().reset_index(name='n')
        tot['n'] += len(te); tot['status_mismatch'] += int((ra != ta).sum())
        tot['etype_mismatch'] += int((r.etype.astype(str) != te.etype.astype(str)).sum()) if 'etype' in r else 0
        tot['ref_split'] += int((pairs.groupby('r').t.nunique() > 1).sum()); tot['test_split'] += int((pairs.groupby('t').r.nunique() > 1).sum())
        maj = pairs.sort_values('n', ascending=False).drop_duplicates('r').set_index('r').t
        tot['off_majority'] += int((rl[both].map(maj) != tl[both]).sum())
        tot['string_mismatch'] += int((rl != tl).sum())
    tot['exact_partition'] = bool(tot['status_mismatch'] == 0 and tot['ref_split'] == 0 and tot['test_split'] == 0 and tot['tiles_missing'] == 0)
    rows.append(tot)
df = pd.DataFrame(rows); print(df.to_string(index=False))
bad = df[~df.exact_partition]
print('\nFIRST DIVERGENT STAGE:', bad.stage.iloc[0] if len(bad) else 'none (all stages exact, summed over %d tiles)' % len(tiles))
print('stages with label-string differences:', int((df.string_mismatch > 0).sum()))
if a.out: df.to_csv(a.out, index=False)
