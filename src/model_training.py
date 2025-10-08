import numpy as np
import rasterio
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

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

# Filter out rare classes; instances > 500 count
min_count = 500
valid_classes = [v for v, c in zip(vals,cnts) if c >= min_count]
mask = np.isin(y_valid, valid_classes)

#
X_sub = X_valid[mask].astype(np.float32, copy=False)
y_sub = y_valid[mask].astype(np.int32, copy=False)

# Downsample per class to cap ram & speed
cap_per_class = 50_000 
rng = np.random.default_rng(42)

classes, counts = np.unique(y_sub, return_counts=True)
take_idx_parts = []
for cls, cnt in zip(classes, counts):
    idx = np.flatnonzero(y_sub == cls)
    if cnt > cap_per_class:
        idx = rng.choice(idx, size=cap_per_class, replace=False)
    take_idx_parts.append(idx)

take_idx = np.concatenate(take_idx_parts)
rng.shuffle(take_idx)

X_small = X_sub[take_idx]
y_small = y_sub[take_idx]

# Small holdout without expensive copies
n = X_small.shape[0]
test_n = int(0.2 * n)
X_train, X_test = X_small[:-test_n], X_small[-test_n:]
y_train, y_test = y_small[:-test_n], y_small[-test_n:]

# Train RF (trees don’t need scaling)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    n_jobs=-1,
    class_weight="balanced_subsample",  # helps with imbalance
    random_state=42
)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

# Pretty report with class names
unique_test = np.unique(y_test)
target_names = [cdl_legend.get(int(c), str(c)) for c in unique_test]
print(classification_report(y_test, y_pred, labels=unique_test, target_names=target_names, digits=3))

report = classification_report(y_test, y_pred, labels=unique_test, target_names=target_names, digits=3)

with open("outputs/performance/classification_report.txt", "w") as f:
    f.write(report)

print("Classification report saved to outputs/performance/classification_report.txt")
