import argparse
import shutil
import subprocess
import sys
import sysconfig
import traceback
from pathlib import Path

import numpy as np

try:
    import nibabel as nib
except ImportError:
    sys.exit("Missing dependency. Run: pip install nibabel numpy")


def find_hdbet_executable() -> str:
    found = shutil.which("hd-bet")
    if found:
        return found

    scripts_dir = Path(sysconfig.get_path("scripts"))
    candidate = scripts_dir / ("hd-bet.exe" if sys.platform == "win32" else "hd-bet")
    if candidate.exists():
        return str(candidate)

    return None


def find_subject_dirs(input_dir: Path) -> list[Path]:
    all_hdr_dirs = {p.parent for p in input_dir.rglob("*.hdr")}
    subject_dirs = sorted(d for d in all_hdr_dirs if d.name.upper() == "RAW")

    if not subject_dirs:
        subject_dirs = sorted(all_hdr_dirs)

    if not subject_dirs:
        raise FileNotFoundError(f"No .hdr files found anywhere under {input_dir}")
    return subject_dirs


def find_mpr_pairs(subject_dir: Path) -> list[Path]:
    hdrs = sorted(subject_dir.glob("*mpr-*.hdr"))
    if not hdrs:
        hdrs = sorted(subject_dir.glob("*.hdr"))
    return hdrs


def analyze_to_nifti(hdr_path: Path) -> nib.Nifti1Image:
    img = nib.load(str(hdr_path))
    data = img.get_fdata().astype(np.float32)

    if isinstance(img, (nib.Nifti1Image, nib.Nifti1Pair)):
        affine = img.affine
    else:
        zooms = img.header.get_zooms()[:3]
        affine = np.diag(list(zooms) + [1.0])

    return nib.Nifti1Image(data, affine)


def subject_id_from_dir(subject_dir: Path) -> str:
    if subject_dir.name.upper() == "RAW":
        return subject_dir.parent.name
    return subject_dir.name


def convert_subject(subject_dir: Path, average_reps: bool) -> nib.Nifti1Image:
    hdrs = find_mpr_pairs(subject_dir)
    if not hdrs:
        raise FileNotFoundError(f"No scans found in {subject_dir}")

    if not average_reps:
        return analyze_to_nifti(hdrs[0])

    volumes = []
    affine = None
    for h in hdrs:
        nifti_img = analyze_to_nifti(h)
        volumes.append(nifti_img.get_fdata())
        if affine is None:
            affine = nifti_img.affine
    mean_vol = np.mean(np.stack(volumes, axis=0), axis=0).astype(np.float32)
    return nib.Nifti1Image(mean_vol, affine)


def run_hdbet(hdbet_exe: str, in_file: Path, out_file: Path, device: str, fast: bool) -> bool:
    cmd = [hdbet_exe, "-i", str(in_file), "-o", str(out_file), "-device", device]
    if fast:
        cmd.append("--disable_tta")
    print(f"[HD-BET] {in_file.name} -> {out_file.name}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  !! HD-BET failed:\n{result.stderr}", file=sys.stderr)
        return False
    print("  done.")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", required=True, type=Path)
    parser.add_argument("-o", "--output_dir", required=True, type=Path)
    parser.add_argument("--average_reps", action="store_true")
    parser.add_argument("--keep_intermediate", action="store_true")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"])
    parser.add_argument("--disable_tta", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if not args.input_dir.exists():
        sys.exit(f"Input directory does not exist: {args.input_dir}")

    hdbet_exe = find_hdbet_executable()
    if hdbet_exe is None:
        sys.exit("ERROR: could not find the 'hd-bet' executable.")
    print(f"Using HD-BET executable: {hdbet_exe}\n")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    intermediate_dir = args.output_dir / "_converted_nifti"
    if args.keep_intermediate:
        intermediate_dir.mkdir(parents=True, exist_ok=True)

    subject_dirs = find_subject_dirs(args.input_dir)
    if args.limit:
        subject_dirs = subject_dirs[: args.limit]

    print(f"Found {len(subject_dirs)} subject folder(s).\n")

    failures = []
    for idx, subj_dir in enumerate(subject_dirs, start=1):
        sid = subject_id_from_dir(subj_dir)
        print(f"[{idx}/{len(subject_dirs)}] {sid}")
        try:
            nifti_img = convert_subject(subj_dir, average_reps=args.average_reps)

            nifti_path = (intermediate_dir if args.keep_intermediate else args.output_dir / "_tmp") / f"{sid}_T1.nii.gz"
            nifti_path.parent.mkdir(parents=True, exist_ok=True)
            nib.save(nifti_img, str(nifti_path))

            out_file = args.output_dir / f"{sid}_brain.nii.gz"
            ok = run_hdbet(hdbet_exe, nifti_path, out_file, device=args.device, fast=args.disable_tta)
            if not ok:
                failures.append(sid)

            if not args.keep_intermediate:
                nifti_path.unlink(missing_ok=True)
        except Exception as e:
            print(f"  !! Error on {sid}: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            failures.append(sid)

    tmp_dir = args.output_dir / "_tmp"
    if tmp_dir.exists() and not any(tmp_dir.iterdir()):
        tmp_dir.rmdir()

    print("\nDone.")
    print(f"  Succeeded: {len(subject_dirs) - len(failures)}")
    print(f"  Failed:    {len(failures)}")
    if failures:
        print("  Failed subjects:", ", ".join(failures))


if __name__ == "__main__":
    main()