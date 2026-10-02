"""Per-stage partition comparison of two TRACER stage dumps, restricted to core transcripts.
Reports the FIRST stage whose core partition differs."""
import argparse, glob, json, os, numpy as np, pandas as pd
p = argparse.ArgumentParser()
p.add_argument('--ref', required=True); p.add_argument('--test', required=True)
p.add_argument('--core-ids', required=True); p.add_argument('--out', default=None)
a = p.parse_args()
ids = np.load(a.core_ids)
rows = []
for rf in sorted(glob.glob(os.path.join(a.ref, '*.parquet'))):
    name = os.path.basename(rf); tf = os.path.join(a.test, name)
    if not os.path.exists(tf): rows.append(dict(stage=name, note='missing in test')); continue
    r = pd.read_parquet(rf); r = r[r.transcript_id.isin(ids)]
    t = pd.read_parquet(tf); t = t[t.transcript_id.isin(ids)]
    m = r.merge(t, on='transcript_id', suffixes=('_r', '_t'))
    ra, ta = m.label_r.eq('-1'), m.label_t.eq('-1')
    both = m[~ra & ~ta]
    pairs = both.groupby(['label_r', 'label_t']).size().reset_index(name='n')
    rs = int((pairs.groupby('label_r').label_t.nunique() > 1).sum())
    ts = int((pairs.groupby('label_t').label_r.nunique() > 1).sum())
    maj = pairs.sort_values('n', ascending=False).drop_duplicates('label_r').set_index('label_r').label_t
    off = int((both.label_r.map(maj) != both.label_t).sum())
    et = int((m.etype_r != m.etype_t).sum()) if 'etype_r' in m else -1
    rows.append(dict(stage=name, n=len(m), status_mismatch=int((ra != ta).sum()), ref_split=rs, test_split=ts,
                     off_majority=off, etype_mismatch=et,
                     exact=bool((ra == ta).all() and rs == 0 and ts == 0)))
df = pd.DataFrame(rows); print(df.to_string(index=False))
bad = df[(df.get('exact') == False)] if 'exact' in df else df.iloc[0:0]
print('\nFIRST DIVERGENT STAGE:', bad.stage.iloc[0] if len(bad) else 'none (all stages exact)')
if a.out: df.to_csv(a.out, index=False)
