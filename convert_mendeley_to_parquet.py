"""Convert the Mendeley Data V1 onion price CSV release (Shelar, R., Hajare, A.,
and Mangarule, S., 2025, DOI: 10.17632/ds9jmxp9zy.1, CC BY 4.0) into the typed
parquet input (data/onion_2014_2024.parquet) used by the Paper 3 benchmark.

Usage:
    python convert_mendeley_to_parquet.py --csv-dir <path_to_mendeley_csvs> \
        --out data/onion_2014_2024.parquet
"""
import argparse, hashlib, json
from pathlib import Path
import pandas as pd


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def convert(csv_dir: Path, out_path: Path, hash_out: Path | None = None) -> None:
    csv_files = sorted(csv_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {csv_dir}")

    frames = []
    hashes = {}
    for p in csv_files:
        hashes[p.name] = sha256_file(p)
        df = pd.read_csv(p, low_memory=False)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        frames.append(df)

    full = pd.concat(frames, ignore_index=True)
    if "date" in full.columns:
        full["date"] = pd.to_datetime(full["date"], errors="coerce")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    full.to_parquet(out_path, index=False)
    hashes[out_path.name] = sha256_file(out_path)
    if hash_out is not None:
        hash_out.write_text(json.dumps(hashes, indent=2))
    print(f"Wrote {len(full):,} rows from {len(csv_files)} CSV file(s) to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/onion_2014_2024.parquet"))
    ap.add_argument("--hash-out", type=Path, default=Path("results/data_sha256.json"))
    args = ap.parse_args()
    convert(args.csv_dir, args.out, args.hash_out)
