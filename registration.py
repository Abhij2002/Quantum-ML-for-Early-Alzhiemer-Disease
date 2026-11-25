import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import nibabel as nib


def get_mni_template_path(resolution):
    from nilearn.datasets import load_mni152_template

    template_path = Path("mni152_template_cache") / f"mni152_{resolution}mm.nii.gz"
    template_path.parent.mkdir(parents=True, exist_ok=True)

    if template_path.exists():
        return template_path

    template_img = load_mni152_template(resolution=resolution)
    nib.save(template_img, str(template_path))

    if not template_path.exists():
        sys.exit(f"Failed to save template to {template_path}")

    return template_path


def nifti_to_sitk(path):
    return sitk.ReadImage(str(path), sitk.sitkFloat32)


def physical_center(img):
    size = img.GetSize()
    index_center = [(s - 1) / 2.0 for s in size]
    return img.TransformContinuousIndexToPhysicalPoint(index_center)


def build_registration_method():
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.2)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsGradientDescent(
        learningRate=1.0, numberOfIterations=200,
        convergenceMinimumValue=1e-6, convergenceWindowSize=10,
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    return reg


def register_rigid(fixed, moving):
    fixed_center = physical_center(fixed)
    moving_center = physical_center(moving)

    initial_transform = sitk.Euler3DTransform()
    initial_transform.SetCenter(moving_center)
    initial_transform.SetTranslation([moving_center[i] - fixed_center[i] for i in range(3)])

    reg = build_registration_method()
    reg.SetInitialTransform(initial_transform, inPlace=True)
    return reg.Execute(fixed, moving)


def register_affine(fixed, moving, rigid_transform):
    affine_transform = sitk.AffineTransform(3)
    affine_transform.SetCenter(rigid_transform.GetCenter())
    affine_transform.SetMatrix(rigid_transform.GetMatrix())
    affine_transform.SetTranslation(rigid_transform.GetTranslation())

    reg = build_registration_method()
    reg.SetInitialTransform(affine_transform, inPlace=True)
    return reg.Execute(fixed, moving)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", required=True, type=Path)
    parser.add_argument("-o", "--output_dir", required=True, type=Path)
    parser.add_argument("--pattern", default="*_norm.nii.gz")
    parser.add_argument("--resolution", type=int, default=2, choices=[1, 2])
    parser.add_argument("--template", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--interpolator", default="bspline", choices=["linear", "bspline", "nearest"])
    args = parser.parse_args()

    if not args.input_dir.exists():
        sys.exit(f"Input directory does not exist: {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    interpolators = {
        "linear": sitk.sitkLinear,
        "bspline": sitk.sitkBSpline,
        "nearest": sitk.sitkNearestNeighbor,
    }
    interpolator = interpolators[args.interpolator]

    template_path = args.template if args.template else get_mni_template_path(args.resolution)
    print(f"Using template: {template_path}")
    fixed_img = nifti_to_sitk(template_path)

    files = sorted(args.input_dir.glob(args.pattern))
    if not files:
        sys.exit(f"No files matching '{args.pattern}' found under {args.input_dir}")
    if args.limit:
        files = files[: args.limit]

    print(f"Found {len(files)} file(s) to register.\n")

    failures = []
    for idx, f in enumerate(files, start=1):
        out_name = f.name.replace("_norm.nii.gz", "_registered.nii.gz")
        out_path = args.output_dir / out_name
        print(f"[{idx}/{len(files)}] {f.name} -> {out_name}")
        try:
            moving_img = nifti_to_sitk(f)

            rigid_transform = register_rigid(fixed_img, moving_img)
            transform = register_affine(fixed_img, moving_img, rigid_transform)

            registered_img = sitk.Resample(
                moving_img, fixed_img, transform,
                interpolator, 0.0, moving_img.GetPixelID(),
            )

            sitk.WriteImage(registered_img, str(out_path))
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