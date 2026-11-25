import argparse, glob, os, numpy as np, pandas as pd, nibabel as nib

ROIS = {"hipp_L": "Left Hippocampus", "hipp_R": "Right Hippocampus", "vent_L": "Left Lateral Ventricle", "vent_R": "Right Lateral Ventricle"}

def load_atlas(ref_img):
    from nilearn import datasets, image
    a = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-1mm")
    img = image.load_img(a.maps)
    if img.shape != ref_img.shape[:3]:
        img = image.resample_to_img(img, ref_img, interpolation="nearest")
    return np.asarray(img.dataobj).round().astype(int), list(a.labels)

def roi_features(vol, atlas, labels, csf_z):
    out = {}
    for k, name in ROIS.items():
        v = vol[atlas == labels.index(name)]
        out[f"{k}_mean"] = float(v.mean()) if v.size else np.nan
        out[f"{k}_csf"] = float((v < csf_z).mean()) if v.size else np.nan
        if k.startswith("hipp"): out[f"{k}_tissue"] = int((v >= csf_z).sum())
    return out

def qc_png(vol, atlas, labels, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m = (atlas == labels.index(ROIS["hipp_L"])) | (atlas == labels.index(ROIS["hipp_R"]))
    c = [int(round(i.mean())) for i in np.nonzero(m)]
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    for a, (img, mk, t) in zip(ax, [(vol[c[0]], m[c[0]], "axis 0 slice"), (vol[:, c[1]], m[:, c[1]], "axis 1 slice"), (vol[:, :, c[2]], m[:, :, c[2]], "axis 2 slice")]):
        a.imshow(np.rot90(img), cmap="gray"); a.contour(np.rot90(mk), levels=[0.5], colors="r", linewidths=1); a.set_title(t); a.axis("off")
    fig.savefig(path, dpi=100, bbox_inches="tight")

def main():
    p = argparse.ArgumentParser(description="Hippocampus and ventricle features from registered scans.")
    p.add_argument("-i", "--input_dir", required=True)
    p.add_argument("-o", "--output_csv", default="hippo_features.csv")
    p.add_argument("--csf_z", type=float, default=-0.8, help="z-score below which a voxel counts as CSF-like")
    p.add_argument("--axes", default="0,1,2", help="permute scan axes to match the atlas, e.g. 1,0,2")
    p.add_argument("--qc", default="hippo_qc.png", help="overlay image for the first scan")
    p.add_argument("--append", action="store_true", help="add to an existing output csv (use for the second dataset)")
    a = p.parse_args()
    perm = [int(i) for i in a.axes.split(",")]
    files = sorted(glob.glob(os.path.join(a.input_dir, "**", "*_registered.nii.gz"), recursive=True))
    if not files: raise SystemExit("No *_registered.nii.gz files found.")
    ref = nib.load(files[0])
    atlas, labels = load_atlas(ref)
    rows = []
    for k, f in enumerate(files, 1):
        vol = np.transpose(np.squeeze(nib.load(f).get_fdata()).astype(np.float32), perm)
        if vol.shape != atlas.shape:
            print("skipped (shape differs):", f); continue
        if k == 1: qc_png(vol, atlas, labels, a.qc); print("overlay saved to", a.qc, "- check that the red outline sits on the hippocampi")
        rows.append(dict(ID=os.path.basename(f).replace("_registered.nii.gz", ""), **roi_features(vol, atlas, labels, a.csf_z)))
        if k % 25 == 0: print(f"{k}/{len(files)}")
    df = pd.DataFrame(rows)
    if a.append and os.path.exists(a.output_csv): df = pd.concat([pd.read_csv(a.output_csv), df]).drop_duplicates("ID", keep="last")
    df.to_csv(a.output_csv, index=False)
    print("saved", a.output_csv, len(rows), "scans")

if __name__ == "__main__":
    main()