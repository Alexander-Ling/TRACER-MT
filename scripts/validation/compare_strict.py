"""Strict (label-string) per-stage comparison of two whole-slide stage dumps."""
import glob, os, sys, numpy as np, pandas as pd
ref, test = sys.argv[1], sys.argv[2]
ok = True
for rf in sorted(glob.glob(os.path.join(ref, '*.parquet'))):
    n = os.path.basename(rf); tf = os.path.join(test, n)
    if not os.path.exists(tf): print(n, 'MISSING'); ok = False; continue
    a = pd.read_parquet(rf).set_index('transcript_id'); b = pd.read_parquet(tf).set_index('transcript_id').reindex(a.index)
    lab = int((a.label.astype(str) != b.label.astype(str)).sum())
    et = int((a.etype.astype(str) != b.etype.astype(str)).sum()) if 'etype' in a else -1
    print(f'{n:34s} rows={len(a):>9,} label_diff={lab:>8,} etype_diff={et:>8,}')
    ok &= (lab == 0 and et <= 0)
print('IDENTICAL' if ok else 'DIFFERENT')
