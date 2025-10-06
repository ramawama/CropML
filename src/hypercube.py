import rasterio
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


# Path to landsard folder + out file
landsat_dir = Path("data/raw/landsat/LC09_CU_016003_20241013_20241017_02")
out_file = "data/processed/landsat_compsite.tiff"

# Get bands B1-B7 (Surface Reflectance) and QA
# Use glob to pattern match
band_files = sorted((landsat_dir.glob("*SR_B*.TIF")))
band_names = ['AEROSOL', 'BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2']
qa_file = next(landsat_dir.glob("*QA_PIXEL.TIF"))

print("Bands found:", [file.name for file in band_files])
print("QA file: ", qa_file)

def decode_bit(qa_arr: np.ndarray, bit: int) -> np.ndarray:
    """
       Decodes the QA Bit
    """
    return (qa_arr & 1 << bit) > 0  

def qa_mask(qa_arr: np.ndarray, mask_type: str) -> np.ndarray:
    """
    Creates a boolean mask based on the specified mask type.
     -   Args:
            qa_arr (np.ndarray): The quality assessment array.
            mask_type (str): The type of mask to create. Valid options are:
                "fill", "dilated", "cirrus", "cloud", "shadow", "snow", "clear", "water", 
                the high, mid and low masks refer to confidence levels.

     -   Returns:
            np.ndarray: The boolean mask with True and False values.
    """

    mask_type = mask_type.lower()  # Convert mask type to lowercase
    
    if mask_type == "fill":
        return decode_bit(qa_arr, 0)
    elif mask_type == "dilated":
        return decode_bit(qa_arr, 1)
    elif mask_type == "cirrus":
        return decode_bit(qa_arr, 2)
    elif mask_type == "cloud":
        return decode_bit(qa_arr, 3)
    elif mask_type == "shadow":
        return decode_bit(qa_arr, 4)
    elif mask_type == "snow":
        return decode_bit(qa_arr, 5)
    elif mask_type == "clear":
        return decode_bit(qa_arr, 6)
    elif mask_type == "water":
        return decode_bit(qa_arr, 7)
    elif mask_type == "high cloud":
        return decode_bit(qa_arr, 8) & decode_bit(qa_arr, 9)
    elif mask_type == "mid cloud":
        return ~decode_bit(qa_arr, 8) & decode_bit(qa_arr, 9)
    elif mask_type == "low cloud":
        return decode_bit(qa_arr, 8) & ~(decode_bit(qa_arr, 9))
    elif mask_type == "high shadow":
        return decode_bit(qa_arr, 10) & decode_bit(qa_arr, 11)
    elif mask_type == "mid shadow":
        return ~decode_bit(qa_arr, 10) & decode_bit(qa_arr, 11)
    elif mask_type == "low shadow":
        return decode_bit(qa_arr, 10) & ~decode_bit(qa_arr, 11)
    elif mask_type == "high snow/ice":
        return decode_bit(qa_arr, 12) & decode_bit(qa_arr, 13)
    elif mask_type == "mid snow/ice":
        return ~decode_bit(qa_arr, 12) & decode_bit(qa_arr, 13)
    elif mask_type == "low snow/ice":
        return decode_bit(qa_arr, 12) & ~decode_bit(qa_arr, 13)
    elif mask_type == "high cirrus":
        return decode_bit(qa_arr, 14) & decode_bit(qa_arr, 15)
    elif mask_type == "mid cirrus":
        return ~decode_bit(qa_arr, 14) & decode_bit(qa_arr, 15)
    elif mask_type == "low cirrus":
        return decode_bit(qa_arr, 14) & ~decode_bit(qa_arr, 15)
    else:
        raise ValueError(f"Invalid mask type: {mask_type}")
    

# Get QA mask
with rasterio.open(qa_file) as src:
    qa = src.read(1)
    profile = src.profile

qa_mask_arr = (qa_mask(qa,'cloud') | qa_mask(qa,'fill') | qa_mask(qa, 'snow') | qa_mask(qa, 'shadow'))
# print(qa_mask_arr)


sepctral_bands = []
for b in band_files:
    with rasterio.open(b) as src:
        arr = src.read(1).astype("float32")
        sepctral_bands.append(arr)

sepctral_bands = np.array(sepctral_bands)
# print(sepctral_bands)

qa_mask_broadcasted = np.broadcast_to(qa_mask_arr, sepctral_bands.shape)
# print(qa_mask_broadcasted)

spectral_array_masked = np.where(~qa_mask_broadcasted, sepctral_bands, 0)
# print(spectral_array_masked)

# --- helper for stretching ---
def stretch(arr):
    valid = arr[arr > 0]  # ignore zeros
    p2, p98 = np.percentile(valid, (2, 98))
    return np.clip((arr - p2) / (p98 - p2), 0, 1)


# --- extract RGB (B4=Red, B3=Green, B2=Blue) ---
def make_rgb(stack):
    red   = stack[3] / 10000.0   # SR_B4
    green = stack[2] / 10000.0   # SR_B3
    blue  = stack[1] / 10000.0   # SR_B2
    return np.dstack([stretch(red), stretch(green), stretch(blue)])

rgb_original = make_rgb(sepctral_bands)
rgb_masked   = make_rgb(spectral_array_masked)

# --- plot side by side ---
fig, axes = plt.subplots(1, 2, figsize=(14,7))

axes[0].imshow(rgb_original)
axes[0].set_title("Original Landsat RGB")
axes[0].axis("off")

axes[1].imshow(rgb_masked)
axes[1].set_title("Masked Landsat RGB (QA applied)")
axes[1].axis("off")

plt.show()

# ML

# feature stack (what you train on): float32 reflectance with NaNs for masked pixels
features = sepctral_bands.astype("float32") 
features = np.where(qa_mask_broadcasted, np.nan, features)   # NO in-place edits
# --- Quick checks to prove stretching didn't alter features ---
print("Features shape:", features.shape, "dtype:", features.dtype)
print("Feature band mins/maxes (ignore NaN):",
      [ (np.nanmin(features[i]), np.nanmax(features[i])) for i in [1,2,3] ])

print(spectral_array_masked)

reflectance = (spectral_array_masked * 0.0000275) - 0.2
reflectance[spectral_array_masked == 0] = np.nan  

print(reflectance)

print("Reflectance range:", np.nanmin(reflectance), np.nanmax(reflectance))
print("Proportion NaN:", np.mean(np.isnan(reflectance)))
print("vals less than 0: ", (reflectance < 0).sum())
print("vals more than 1: ", (reflectance > 1).sum())

qa_any = qa_mask(qa, 'cloud') | qa_mask(qa, 'shadow')
print("Pixels flagged by QA:", qa_any.sum(), "/", qa_any.size, f"({100*qa_any.mean():.2f}%)")

# Compare against your reflectance NaNs:
nan_mask = np.isnan(reflectance[0])  # pick one band
print("NaN pixels (any band):", nan_mask.sum(), "/", nan_mask.size, f"({100*nan_mask.mean():.2f}%)")

# Mask non-physical reflectance values
reflectance[(reflectance < 0) | (reflectance > 1)] = np.nan

print("After range screen:")
print("Valid proportion:", np.mean(~np.isnan(reflectance))) # how many pixels are valid

print(reflectance.shape)
# (7, 5000, 5000) (band, y, x). each pixel has a 7 band vector describing vector

np.savez_compressed('outputs/reflectance_016003.npy', reflectance)
# Align CDL and Reflectance 
cdl_dir = Path('data/raw/cdl/2024_30m_cdls/2024_30m_cdls.tif')
