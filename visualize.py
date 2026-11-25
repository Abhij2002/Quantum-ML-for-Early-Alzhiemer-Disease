import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

img = nib.load(
    "D:/Projects/Quantum ML for Early Alzhiemer Disease/"
    "oasis1_registered/OAS1_0010_MR1_registered.nii.gz"
)

data = img.get_fdata()

print("Shape:", data.shape)
print("Min:", np.min(data))
print("Max:", np.max(data))
print("Non-zero voxels:", np.count_nonzero(data))

# Find the slice with the most non-zero voxels
slice_counts = np.count_nonzero(data, axis=(0, 1))
slice_idx = np.argmax(slice_counts)

print("Selected slice:", slice_idx)

plt.figure(figsize=(7, 7))
plt.imshow(data[:, :, slice_idx].T, cmap="gray", origin="lower")
plt.axis("off")
plt.show()