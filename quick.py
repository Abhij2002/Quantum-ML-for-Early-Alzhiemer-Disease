import argparse
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import zscore


def get_template(resolution):
    from nilearn.datasets import load_mni152_template
    path = Path("mni152_template_cache") / f"mni152_{resolution}mm.nii.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        nib.save(load_mni152_template(resolution=resolution), str(path))
    return nib.load(str(path)).get_fdata().astype(np.float32)


def template_correlation(data, template):
    a, b = data.ravel(), template.ravel()
    mask = (a > 0) | (b > 0)
    return float(np.corrcoef(a[mask], b[mask])[0, 1]) if mask.any() else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--registered_dir", required=True, type=Path)
    p.add_argument("--metadata", type=Path)
    p.add_argument("--resolution", type=int, default=2, choices=[1, 2])
    p.add_argument("--output_csv", type=Path)
    p.add_argument("--quarantine_dir", type=Path)
    p.add_argument("--zscore_threshold", type=float, default=3.0)
    args = p.parse_args()

    template = get_template(args.resolution)
    files = sorted(args.registered_dir.glob("*_registered.nii.gz"))

    rows = []
    for f in files:
        sid = f.name.replace("_registered.nii.gz", "")
        row = {"ID": sid, "Subject": re.sub(r"_MR\d+$", "", sid), "file": f.name, "corrupt": False}
        try:
            data = nib.load(str(f)).get_fdata().astype(np.float32)
            if data.shape != template.shape:
                row["corrupt"] = True
            else:
                row["brain_fraction"] = np.count_nonzero(data > 0) / data.size
                row["template_correlation"] = template_correlation(data, template)
        except Exception:
            row["corrupt"] = True
        rows.append(row)

    df = pd.DataFrame(rows)
    for col in ["brain_fraction", "template_correlation"]:
        if col not in df:
            df[col] = np.nan
        valid = df[col].notna()
        df[f"{col}_outlier"] = False
        df.loc[valid, f"{col}_outlier"] = np.abs(zscore(df.loc[valid, col])) > args.zscore_threshold

    if args.metadata:
        meta = pd.read_excel(args.metadata) if args.metadata.suffix == ".xlsx" else pd.read_csv(args.metadata)
        id_col = next(c for c in ["MRI ID", "ID", "Subject ID"] if c in meta.columns)
        keep = [id_col]
        for col in ["eTIV", "nWBV"]:
            if col in meta.columns:
                meta[f"{col}_outlier"] = np.abs(zscore(meta[col], nan_policy="omit")) > args.zscore_threshold
                keep.append(f"{col}_outlier")
        df = df.merge(meta[keep].rename(columns={id_col: "ID"}), on="ID", how="left")

    checks = ["corrupt"] + [c for c in df.columns if c.endswith("_outlier")]
    df[checks] = df[checks].fillna(False).astype(bool)
    df["flagged"] = df[checks].any(axis=1)
    df["subject_flagged"] = df.groupby("Subject")["flagged"].transform("any")

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output_csv, index=False)

    print(f"Files: {len(df)}, subjects: {df['Subject'].nunique()}")
    for c in checks:
        print(f"{c}: {df[c].sum()}")
    print(f"Subjects flagged: {df.loc[df['flagged'], 'Subject'].nunique()}")

    quarantine_dir = args.quarantine_dir or args.registered_dir.parent / f"{args.registered_dir.name}_quarantine"
    done = set()

    for c in checks:
        subjects = sorted(set(df.loc[df[c], "Subject"]) - done)
        if not subjects:
            continue
        print(f"\n{len(subjects)} subject(s) flagged for {c}: {subjects}")
        if input("Quarantine these subjects with all visits? (y/n): ").strip().lower() != "y":
            continue
        quarantine_dir.mkdir(parents=True, exist_ok=True)
        for fname in df.loc[df["Subject"].isin(subjects), "file"]:
            src = args.registered_dir / fname
            if src.exists():
                shutil.move(str(src), str(quarantine_dir / fname))
        done.update(subjects)


if __name__ == "__main__":
    main()