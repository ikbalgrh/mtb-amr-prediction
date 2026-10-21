import pandas as pd, requests
from pathlib import Path
from urllib.parse import urljoin

BASE = "https://ftp.ebi.ac.uk/pub/databases/cryptic/release_june2022/reuse/"
N = 1500   # TEST first. Change to 1500 for the real run.

df = pd.read_csv("data/reuse_table.csv", low_memory=False)
df = df.dropna(subset=["VCF", "RIF_BINARY_PHENOTYPE", "INH_BINARY_PHENOTYPE"])
df = df[df["RIF_BINARY_PHENOTYPE"].isin(["R", "S"]) & df["INH_BINARY_PHENOTYPE"].isin(["R", "S"])]
good = ["HIGH", "MEDIUM"]
df = df[df["RIF_PHENOTYPE_QUALITY"].isin(good) & df["INH_PHENOTYPE_QUALITY"].isin(good)]
print(len(df), "isolates pass the filters")

df = df.sample(n=N, random_state=42)
df.to_csv("data/selected_samples.csv", index=False)

Path("data/vcfs").mkdir(parents=True, exist_ok=True)
failed = 0
for i, (_, r) in enumerate(df.iterrows(), 1):
    out = Path("data/vcfs") / f"{r['UNIQUEID']}.vcf.gz"
    if out.exists():
        continue
    url = urljoin(BASE, r["VCF"])
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        out.write_bytes(resp.content)
    except Exception as e:
        failed += 1
        print("FAILED", url, e)
    if i % 50 == 0:
        print(i, "of", len(df), "done")
print("finished, failed:", failed)