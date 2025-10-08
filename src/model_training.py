import numpy as np
import rasterio
import json


# Get hypercube
with np.load('outputs/reflectance_016003.npz') as data:
    hypercube = data["hypercube"]

# Get clipped+aligned tiff
with rasterio.open("outputs/cdl_2024_aligned_016003.tif") as cdl:
    cdl = cdl.read(1)

bands, H, W = hypercube.shape

with open('outputs/cdl_code_to_name.json') as f:
    # Convert dict keys from string to ints
    cdl_legend = {int(k): v for k, v in json.load(f).items()}

# Move axis 0 to axis -1 (end of np array), moving spectral bands to end
# Reshape to flatten for ML models
# i.e. x[1] -> band values for pixel (0,1) (x,y)
X = np.moveaxis(hypercube, 0, -1).reshape(-1, bands)
# Flattens CDL (groundtruths) into 1D array
y = cdl.reshape(-1)

# Build a mask to remove cdl values that dont have respective spectral information
valid_reflectance = np.isfinite(X).all(axis=1) # Check np.isfinite across all bands
valid_labels = y > 0 # valid ground truths CDL nodata (0) are excluded
valid_mask = valid_reflectance & valid_labels

# Apply filter
X_valid = X[valid_mask]
y_valid = y[valid_mask]

# 42.68% valid. 
print(f"Valid CDL pixels: {len(y_valid)} / {len(y)} ({len(y_valid)/len(y):.2%})")
print(f"Valid SR pixels: {len(X_valid)} / {len(X)} ({len(X_valid)/len(X):.2%})")

keep_idx = np.flatnonzero(valid_mask)

# print out values and categories of clipped cdl
vals, cnts = np.unique(y_valid, return_counts=True)
for v, c in zip(vals, cnts):
    print(f"class {v} ({cdl_legend[v]}): {c}")