import argparse
from pathlib import Path

import pandas as pd


def load(path):
    return pd.read_excel(path) if path.suffix == ".xlsx" else pd.read_csv(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output_csv", required=True, type=Path)
    args = parser.parse_args()

    df = load(args.metadata)
    id_col = "Subject ID" if "Subject ID" in df.columns else "ID"
    is_longitudinal = "Visit" in df.columns

    print(f"Loaded {len(df)} row(s) from {args.metadata.name}")
    print(f"Columns: {df.columns.tolist()}\n")

    issues = {}

    issues["missing critical fields (MMSE/CDR/Age)"] = df[df[["MMSE", "CDR", "Age"]].isna().any(axis=1)]

    if is_longitudinal:
        dup_mask = df.duplicated(subset=[id_col, "Visit"], keep="first")
    else:
        dup_mask = df.duplicated(subset=[id_col], keep="first")
    issues["duplicate rows"] = df[dup_mask]

    valid_cdr = [0, 0.5, 1, 2, 3]
    issues["invalid CDR value"] = df[df["CDR"].notna() & ~df["CDR"].isin(valid_cdr)]

    issues["MMSE out of range (0-30)"] = df[df["MMSE"].notna() & ~df["MMSE"].between(0, 30)]

    issues["Age out of range (1-120)"] = df[df["Age"].notna() & ~df["Age"].between(1, 120)]

    for col in ["eTIV", "nWBV", "ASF"]:
        if col in df.columns:
            issues[f"{col} not positive"] = df[df[col].notna() & (df[col] <= 0)]

    if is_longitudinal and "MR Delay" in df.columns:
        bad_delay = df[(df["Visit"] == 1) & (df["MR Delay"] != 0)]
        issues["Visit 1 with nonzero MR Delay"] = bad_delay

    drop_index = set()
    for label, affected in issues.items():
        if len(affected) == 0:
            continue
        print(f"\n{len(affected)} row(s) flagged: {label}")
        print(affected[id_col].tolist())
        answer = input(f"Drop these {len(affected)} row(s)? (y/n): ").strip().lower()
        if answer == "y":
            drop_index.update(affected.index.tolist())
            print("Marked for removal.")
        else:
            print("Kept.")

    cleaned = df.drop(index=list(drop_index)).reset_index(drop=True)

    print(f"\nOriginal rows: {len(df)}")
    print(f"Removed: {len(drop_index)}")
    print(f"Remaining: {len(cleaned)}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(args.output_csv, index=False)
    print(f"\nSaved cleaned metadata to {args.output_csv}")


if __name__ == "__main__":
    main()