import gzip, numpy as np, pandas as pd
from collections import Counter
from pathlib import Path

ID, RIF, INH = "UNIQUEID", "RIF_BINARY_PHENOTYPE", "INH_BINARY_PHENOTYPE"
meta = pd.read_csv("data/selected_samples.csv")

per_sample = {}
for sid in meta[ID]:
    f = Path("data/vcfs") / f"{sid}.vcf.gz"
    if not f.exists():
        continue
    s = set()
    with gzip.open(f, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            c = line.rstrip("\n").split("\t")
            pos, ref, alt, filt = c[1], c[3], c[4], c[6]
            gt = c[9].split(":")[0]
            if (filt in ("PASS", ".") and gt.startswith("1")
                    and len(ref) == 1 and len(alt) == 1 and alt != "."):
                s.add(f"{pos}_{ref}_{alt}")
    per_sample[sid] = s

n = len(per_sample)
counts = Counter(v for s in per_sample.values() for v in s)
keep = sorted(v for v, c in counts.items() if 15 <= c <= n - 15)
print(n, "isolates,", len(counts), "distinct SNPs,", len(keep), "kept after frequency filter")

col_idx = {v: j for j, v in enumerate(keep)}
M = np.zeros((n, len(keep)), dtype="uint8")
for r, s in enumerate(per_sample.values()):
    for v in s:
        j = col_idx.get(v)
        if j is not None:
            M[r, j] = 1

X = pd.DataFrame(M, index=list(per_sample), columns=keep)
m = meta.set_index(ID).loc[X.index]
X["y_rif"] = (m[RIF] == "R").astype(int).values
X["y_inh"] = (m[INH] == "R").astype(int).values
X.to_pickle("data/matrix.pkl")
print(X[["y_rif", "y_inh"]].mean())