#!/usr/bin/env python3
import pandas as pd
from pathlib import Path
import sys

def process(project_root):
    base = Path(project_root) / 'evalEmbeddings' / 'results' / 'cnn_diff_cells_last_layer'
    if not base.exists():
        print(f"Missing path: {base}")
        return
    # find results csv
    csvs = list(base.glob('*_CNN_results.csv'))
    if not csvs:
        print(f"No results CSV found in {base}")
        return
    res_csv = csvs[0]
    print(f"Project {project_root}: using results CSV: {res_csv.name}")
    df = pd.read_csv(res_csv)
    finished = set((str(r['model_variant']), str(r['task_id'])) for _, r in df.iterrows())
    cache_dir = base / 'cache'
    if not cache_dir.exists():
        print(f"No cache dir: {cache_dir}")
        return
    candidates = []
    for variant_dir in sorted(cache_dir.iterdir()):
        if not variant_dir.is_dir():
            continue
        vname = variant_dir.name
        for split_dir in sorted(variant_dir.iterdir()):
            if not split_dir.is_dir():
                continue
            sname = split_dir.name
            task = sname.replace('__','/')
            key = (vname, task)
            if key in finished:
                # list npy files
                for f in split_dir.glob('*.npy'):
                    try:
                        size = f.stat().st_size
                    except Exception:
                        size = 0
                    candidates.append((size, str(f)))
    candidates.sort(reverse=True)
    out = Path('/tmp') / f'candidates_{project_root}.txt'
    with out.open('w') as fh:
        for size, path in candidates:
            fh.write(f"{size}\t{path}\n")
    print(f"Project {project_root}: candidate files: {len(candidates)} -> {out}")
    for size, path in candidates[:20]:
        print(f"  {size:12d} {path}")

if __name__ == '__main__':
    root = Path('/home/fr/fr_fr/fr_ml642/Thesis')
    for proj in ['DNABERT2','LAMAR']:
        process(proj)
    print('Done')
