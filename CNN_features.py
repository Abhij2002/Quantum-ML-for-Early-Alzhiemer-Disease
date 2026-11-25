import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
import torch
from monai.networks.nets import resnet18


def load_volume(path):
    x = nib.load(str(path)).get_fdata().astype(np.float32)
    m = x > 0
    if m.any():
        x = np.where(m, (x - x[m].mean()) / (x[m].std() + 1e-8), 0).astype(np.float32)
    return torch.from_numpy(x)[None, None]


def build_model(weights, device):
    model = resnet18(spatial_dims=3, n_input_channels=1, feed_forward=False, shortcut_type="A", bias_downsample=False)
    s = torch.load(weights, map_location="cpu", weights_only=False)
    s = s.get("state_dict", s)
    s = {k.replace("module.", ""): v for k, v in s.items() if "fc" not in k}
    r = model.load_state_dict(s, strict=False)
    print("Missing:", r.missing_keys, "Unexpected:", r.unexpected_keys)
    return model.eval().to(device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--registered_dir", required=True, type=Path)
    p.add_argument("--output_csv", required=True, type=Path)
    p.add_argument("--weights", default=Path("resnet_18_23dataset.pth"), type=Path)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    a = p.parse_args()

    dev = torch.device(a.device)
    model = build_model(a.weights, dev)
    files = sorted(a.registered_dir.glob("*_registered.nii.gz"))

    rows = []
    with torch.no_grad():
        for i, f in enumerate(files, 1):
            sid = f.name.replace("_registered.nii.gz", "")
            print(f"[{i}/{len(files)}] {sid}")
            v = model(load_volume(f).to(dev)).cpu().numpy().ravel()
            rows.append({"ID": sid, **{f"f{j}": x for j, x in enumerate(v)}})

    a.output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(a.output_csv, index=False)


if __name__ == "__main__":
    main()