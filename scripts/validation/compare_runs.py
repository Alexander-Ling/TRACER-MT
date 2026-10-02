"""Compare TRACER refined assignments between a reference run and another run on the SAME transcript_ids.
Entity labels may differ (global numbering), so the test is on the partition, not the names."""
import argparse, json, numpy as np, pandas as pd
p = argparse.ArgumentParser()
p.add_argument('--ref', required=True); p.add_argument('--test', required=True)
p.add_argument('--core-ids', default=None, help='npy of transcript_ids to restrict to')
p.add_argument('--out', default=None)
a = p.parse_args()
cols = ['transcript_id', 'tracer_id', '_etype']
r = pd.read_parquet(a.ref, columns=cols).rename(columns={'tracer_id': 'r', '_etype': 'rt'})
t = pd.read_parquet(a.test, columns=cols).rename(columns={'tracer_id': 't', '_etype': 'tt'})
if a.core_ids:
    ids = np.load(a.core_ids); r = r[r.transcript_id.isin(ids)]; t = t[t.transcript_id.isin(ids)]
m = r.merge(t, on='transcript_id', how='outer', indicator=True)
res = dict(n_ref=len(r), n_test=len(t), n_only_ref=int((m._merge == 'left_only').sum()),
           n_only_test=int((m._merge == 'right_only').sum()))
m = m[m._merge == 'both'].copy()
m['r'] = m.r.astype(str); m['t'] = m.t.astype(str)
ra, ta = m.r.eq('-1'), m.t.eq('-1')
res['n_compared'] = len(m)
res['assigned_status_mismatch'] = int((ra != ta).sum())
res['etype_mismatch'] = int((m.rt.astype(str) != m.tt.astype(str)).sum())
both = m[~ra & ~ta]
# partition equivalence: every ref entity must map to exactly one test entity and vice versa
pairs = both.groupby(['r', 't']).size().reset_index(name='n')
r_multi = pairs.groupby('r').t.nunique(); t_multi = pairs.groupby('t').r.nunique()
res['ref_entities_split'] = int((r_multi > 1).sum()); res['test_entities_split'] = int((t_multi > 1).sum())
# transcripts not in the majority pairing of their ref entity
maj = pairs.sort_values('n', ascending=False).drop_duplicates('r').set_index('r').t
res['transcripts_off_majority_mapping'] = int((both.r.map(maj) != both.t).sum())
res['exact_partition_match'] = bool(res['assigned_status_mismatch'] == 0 and res['ref_entities_split'] == 0
                                   and res['test_entities_split'] == 0 and res['n_only_ref'] == 0 and res['n_only_test'] == 0)
print(json.dumps(res, indent=1))
if a.out: open(a.out, 'w').write(json.dumps(res, indent=1))
