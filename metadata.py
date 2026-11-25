import argparse
from pathlib import Path

import pandas as pd

FIELDS = ("SESSION ID", "AGE", "M/F", "HAND", "EDUC", "SES", "CDR", "MMSE", "eTIV", "ASF", "nWBV")
NUMERIC = ["Age", "SES", "CDR", "MMSE", "eTIV", "ASF", "nWBV"]
RENAME = {"SESSION ID": "ID", "MRI ID": "ID", "AGE": "Age", "M/F": "Sex", "HAND": "Hand", "EDUC": "Educ"}


def parse_txt_file(path):
    fields = {}
    with open(path, "r") as f:
        for line in f:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            if key in FIELDS:
                fields[key] = value.strip()
    return fields


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registered_dir", required=True, type=Path)
    parser.add_argument("--output_csv", required=True, type=Path)
    parser.add_argument("--txt_dir", type=Path)
    parser.add_argument("--xlsx", type=Path)
    args = parser.parse_args()

    if not args.txt_dir and not args.xlsx:
        raise SystemExit("Provide --txt_dir (OASIS-1) and/or --xlsx (OASIS-2)")

    ids = {f.name.replace("_registered.nii.gz", "") for f in args.registered_dir.glob("*_registered.nii.gz")}

    frames = []

    if args.txt_dir:
        rows = []
        for txt_file in sorted(args.txt_dir.rglob("*.txt")):
            fields = parse_txt_file(txt_file)
            if fields.get("SESSION ID") in ids:
                rows.append(fields)
        if rows:
            frames.append(pd.DataFrame(rows).rename(columns=RENAME))

    if args.xlsx:
        meta = pd.read_excel(args.xlsx).rename(columns=RENAME)
        frames.append(meta[meta["ID"].isin(ids)])

    if not frames:
        raise SystemExit("No matching metadata found for the registered files")

    df = pd.concat(frames, ignore_index=True)

    for col in NUMERIC:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    print(f"Registered files: {len(ids)}")
    print(f"Metadata rows kept: {len(df)}")
    print(f"Registered files with no metadata: {len(ids - set(df['ID']))}")


if __name__ == "__main__":
    main()