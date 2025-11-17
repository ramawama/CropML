import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from collections import Counter

def print_top_15_common_pixels(geotiff_path):
    """
    Reads a GeoTIFF file and prints the top 10 most common pixel values.

    Args:
        geotiff_path (str): The file path to the GeoTIFF image.
    """
    try:
        with rasterio.open(geotiff_path) as src:
            # Read the first band as a numpy array
            image_array = src.read(1) 
            
            # Get the NoData value if specified in the GeoTIFF metadata
            nodata_value = src.nodata

        # Flatten the array into a 1D list of pixel values
        pixel_values = image_array.flatten()

        # Filter out the NoData values
        if nodata_value is not None:
            pixel_values = pixel_values[pixel_values != nodata_value]
            print(f"NoData value ({nodata_value}) excluded from count.")

        # Count the frequency of each pixel value
        pixel_counts = Counter(pixel_values)

        # Get the top 10 most common values and their counts
        top_10 = pixel_counts.most_common(15)

        # Print the results
        print(f"\nTop 15 most common pixel values in '{geotiff_path}':")
        print("{:<15} {:<15}".format('Pixel Value', 'Count'))
        print("-" * 30)
        for value, count in top_10:
            print(f"{value:<15} {count:<15}")

    except rasterio.errors.RasterioIOError:
        print(f"Error: Could not open or read the file '{geotiff_path}'. Check the file path and permissions.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


#print_top_15_common_pixels('outputs/cdl_2024_clipped_to_aviris_f240604t01p00r12_rfl.tif')
# Choose a small subset of crops first
# 75, 69, 204, 72, 1, 67  (almonds, grapes, pistachios, citrus, corn, peaches) all in top 15 of crops

AVIRIS_IMG = "data/raw/aviris/f240604t01p00r12_rfl/f240604t01p00r12_rfl"
CDL_MATCH  = "data/processed/cdl_2024_matched_to_aviris_grid_f240604t01p00r12.tif"
OUT_NPZ    = "outputs/train_samples_aviris_crops.npz"

crop_classes = np.array([1, 67, 69, 72, 75, 204])  # Corn, Peaches, Grapes, Citrus, Almonds, Pistachios

xs, ys = [], []

with rasterio.open(CDL_MATCH) as cdl_src, rasterio.open(AVIRIS_IMG) as av_src:
    assert cdl_src.crs == av_src.crs
    assert cdl_src.transform == av_src.transform
    assert (cdl_src.width, cdl_src.height) == (av_src.width, av_src.height)

    B = av_src.count

    # iterate over native blocks/tiles (efficient)
    for _, window in cdl_src.block_windows(1):
        labels = cdl_src.read(1, window=window)          # (h, w) int labels
        m = np.isin(labels, crop_classes)
        if not m.any():
            continue

        cube = av_src.read(window=window)                # (B, h, w)
        # flatten + mask + finite check
        X = cube.reshape(B, -1).T                        # (Npix, B)
        y = labels.reshape(-1)
        keep = m.reshape(-1) & np.isfinite(X).all(axis=1)
        if keep.any():
            xs.append(X[keep])
            ys.append(y[keep])

# concat
if not xs:
    raise RuntimeError("No pixels for the requested crop classes were found.")
X_all = np.concatenate(xs, axis=0)
y_all = np.concatenate(ys, axis=0)

# map USDA codes -> 0..K-1 for ML
classes_sorted = np.sort(crop_classes)
code2idx = {c:i for i,c in enumerate(classes_sorted)}
y_enc = np.vectorize(code2idx.get)(y_all)

print("Raw counts (USDA codes):", dict(Counter(y_all)))
print("Encoded counts (0..K-1):", dict(Counter(y_enc)))
print("Feature matrix:", X_all.shape, "(pixels x bands)")

# [X, Y, Classes]
# X.shape = (N, B), N = num pixels, B = bands (224)
# Y.shape = (N,) Integer Codes that represent crop classification

np.savez_compressed(OUT_NPZ, X=X_all, y=y_enc, classes=classes_sorted)
print("✓ Saved training arrays →", OUT_NPZ)

# Visualize
code_to_name = {
    1: "Corn",
    67: "Peaches",
    69: "Grapes",
    72: "Citrus",
    75: "Almonds",
    204: "Pistachios",
}

X = X_all
y = y_enc
classes = classes_sorted

# mapping: encoded -> CDL code / crop name
enc2code = {i: int(code) for i, code in enumerate(classes.tolist())}
enc2name = {i: code_to_name.get(int(code), f"CDL {int(code)}") for i, code in enc2code.items()}

cnt_enc = Counter(y.tolist())
total = int(len(y))

rows = []
for enc in sorted(enc2code.keys()):
    code = enc2code[enc]
    name = enc2name[enc]
    count = int(cnt_enc.get(enc, 0))
    rows.append({
        "Encoded ID": enc,
        "CDL Code": code,
        "Crop": name,
        "Count": count,
        "Percent": 100.0 * count / total if total else 0.0
    })

df = pd.DataFrame(rows).sort_values("Count", ascending=False).reset_index(drop=True)
print("\nClass breakdown:")
print(df.to_string(index=False, formatters={"Percent": lambda v: f"{v:.2f}%"}))

os.makedirs("outputs/figs", exist_ok=True)

plt.figure(figsize=(9,5))
plt.bar(df["Crop"], df["Count"])
plt.title("Pixel Counts by Crop (current run)")
plt.xlabel("Crop"); plt.ylabel("Pixel Count")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("outputs/figs/pixel_counts_by_crop.png", dpi=150)
plt.close()

plt.figure(figsize=(7,7))
# Use percentages from df (already calculated)
plt.pie(
    df["Count"],
    labels=df["Crop"],
    autopct="%1.1f%%",
    startangle=90,
    counterclock=False
)
plt.title("Crop Composition (% of Total Extracted Pixels)")
plt.tight_layout()
plt.savefig("outputs/figs/pixel_share_pie.png", dpi=150)
plt.close()


print("Saved:")
print(" - outputs/figs/pixel_counts_by_crop.png")
print(" - outputs/figs/pixel_share_pie.png")

