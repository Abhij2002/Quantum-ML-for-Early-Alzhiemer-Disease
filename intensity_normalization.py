import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import nibabel as nib

def zscore(data: np.ndarray, clip: float | None) -> np.ndarray:
    brain_mask = data > 0
    brain_voxels = data[brain_mask]
    mean = np.mean(brain_voxels)
    std = np.std(brain_voxels)


    zscored_data = np.zeros_like(data, dtype=np.float32)
    zscored_data[brain_mask] = (data[brain_mask] - mean) / std

    if clip is not None:
        zscored_data = np.clip(zscored_data, -clip, clip)

    return zscored_data

def main():
    parser = argparse.ArgumentParser(description="Z-score normalization of skull stripped MRI data.")
    parser.add_argument("-i", "--input_dir", required=True, type=Path,
                        help="Folder containing *_brain.nii.gz skull-stripped files.")
    parser.add_argument("-o", "--output_dir", required=True, type=Path,
                        help="Folder to write *_norm.nii.gz normalized outputs to.")
    parser.add_argument("--clip", type=float, default=None,
                        help="Optional clipping value for z-scored data (e.g., 3.0).")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only process the first N files (quick test run).")
    args = parser.parse_args()

    if not args.input_dir.exists():
        sys.exit(f"Input directory does not exist: {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(args.input_dir.glob("*_brain.nii.gz"))
    if not files:
        sys.exit(f"No files matching '{args.pattern}' found under {args.input_dir}")
    if args.limit:
        files = files[: args.limit]

    failures = []
    for idx, f in enumerate(files, start=1):
        out_name = f.name.replace("_brain.nii.gz", "_norm.nii.gz")
        out_path = args.output_dir / out_name
        print(f"[{idx}/{len(files)}] {f.name} -> {out_name}")
        try:
            img = nib.load(str(f))
            data = img.get_fdata().astype(np.float32)
 
            normalized = zscore(data, clip=args.clip)
 
            norm_img = nib.Nifti1Image(normalized, img.affine, img.header)
            nib.save(norm_img, str(out_path))
            print("  done.")
        except Exception as e:
            print(f"  !! Error on {f.name}: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            failures.append(f.name)
 
    print("\nDone.")
    print(f"  Succeeded: {len(files) - len(failures)}")
    print(f"  Failed:    {len(failures)}")
    if failures:
        print("  Failed files:", ", ".join(failures))

if __name__ == "__main__":
    main()