#!/usr/bin/env python3
"""
PROPERLY FIXED AVIRIS Data Extraction
Handles NoData values (-9999) and extracts valid reflectance data
"""

import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from collections import Counter

# File paths
AVIRIS_IMG = "data/raw/aviris/f240604t01p00r12_rfl/f240604t01p00r12_rfl"
CDL_MATCH  = "data/processed/cdl_2024_matched_to_aviris_grid_f240604t01p00r12.tif"
OUT_NPZ    = "data/processed/train_samples_aviris_crops.npz"

print("="*70)
print("FIXED AVIRIS DATA EXTRACTION - HANDLING NODATA")
print("="*70)

# Define crop classes and names
crop_classes = np.array([1, 67, 69, 72, 75, 204])  # Corn, Peaches, Grapes, Citrus, Almonds, Pistachios

code_to_name = {
    1: "Corn",
    67: "Peaches",
    69: "Grapes",
    72: "Citrus",
    75: "Almonds",
    204: "Pistachios",
}

# CRITICAL: Define NoData value for AVIRIS
NODATA_VALUE = -9999

print("\n1. OPENING FILES AND CHECKING COMPATIBILITY")
print("-"*50)

xs, ys = [], []
sample_spectra = []
skipped_windows = 0
processed_windows = 0
total_pixels_extracted = 0

with rasterio.open(CDL_MATCH) as cdl_src, rasterio.open(AVIRIS_IMG) as av_src:
    # Verify alignment
    assert cdl_src.crs == av_src.crs, "CRS mismatch!"
    assert cdl_src.transform == av_src.transform, "Transform mismatch!"
    assert (cdl_src.width, cdl_src.height) == (av_src.width, av_src.height), "Dimension mismatch!"
    
    B = av_src.count
    print(f"✓ Files aligned properly")
    print(f"  Shape: {av_src.height} x {av_src.width}")
    print(f"  Number of bands: {B}")
    print(f"  Data type: {av_src.dtypes[0]}")
    print(f"  NoData value: {NODATA_VALUE}")
    
    print("\n2. EXTRACTING VALID PIXELS")
    print("-"*50)
    
    # Iterate over native blocks/tiles
    for window_idx, (_, window) in enumerate(cdl_src.block_windows(1)):
        # Read CDL labels for this window
        labels = cdl_src.read(1, window=window)  # (h, w)
        
        # Check if any of our crop classes are in this window
        crop_mask = np.isin(labels, crop_classes)
        if not crop_mask.any():
            skipped_windows += 1
            continue
        
        # Read AVIRIS hyperspectral cube for this window
        cube = av_src.read(window=window)  # (B, h, w)
        
        # ============================================
        # CRITICAL FIX: Handle NoData values properly
        # ============================================
        
        # Method 1: Create a mask for valid pixels (no NoData in any band)
        # A pixel is valid only if ALL its bands are valid
        valid_pixel_mask = np.all(cube != NODATA_VALUE, axis=0)  # (h, w)
        
        # Combine with crop mask - we want pixels that are both:
        # 1. One of our target crops
        # 2. Have valid spectral data across all bands
        combined_mask = crop_mask & valid_pixel_mask
        
        if not combined_mask.any():
            skipped_windows += 1
            continue
        
        # The data is already in reflectance format (0-1 range)
        # No scaling needed - AVIRIS float32 data is pre-calibrated
        
        # Clip to ensure valid reflectance range (safety check)
        cube = np.clip(cube, 0, 1.5)  # Allow slightly >1 for calibration artifacts
        
        # Reshape for ML format
        X = cube.reshape(B, -1).T  # (n_pixels, n_bands)
        y_flat = labels.reshape(-1)
        combined_mask_flat = combined_mask.reshape(-1)
        
        # Extract only the valid crop pixels
        X_valid = X[combined_mask_flat]
        y_valid = y_flat[combined_mask_flat]
        
        # Double-check for any remaining NoData or invalid values
        # Remove any pixels that still have NoData or non-finite values
        finite_mask = np.all(np.isfinite(X_valid), axis=1)
        X_valid = X_valid[finite_mask]
        y_valid = y_valid[finite_mask]
        
        if len(X_valid) > 0:
            xs.append(X_valid)
            ys.append(y_valid)
            total_pixels_extracted += len(X_valid)
            
            # Save some sample spectra for visualization
            if len(sample_spectra) < 20 and len(X_valid) > 0:
                n_samples = min(5, len(X_valid))
                for i in range(n_samples):
                    sample_spectra.append(X_valid[i])
        
        processed_windows += 1
        
        # Progress update
        if processed_windows % 100 == 0:
            print(f"  Processed {processed_windows} windows, "
                  f"extracted {total_pixels_extracted} pixels...")

print(f"\nProcessing complete:")
print(f"  Windows processed: {processed_windows}")
print(f"  Windows skipped: {skipped_windows}")
print(f"  Total pixels extracted: {total_pixels_extracted}")

# Concatenate all samples
if not xs:
    raise RuntimeError("No valid pixels found! Check if data has valid regions.")

X_all = np.concatenate(xs, axis=0)
y_all = np.concatenate(ys, axis=0)

print(f"\nFinal dataset shape: {X_all.shape}")

# ============================================
# 3. VALIDATE THE EXTRACTED DATA
# ============================================
print("\n3. VALIDATING SPECTRAL SIGNATURES")
print("-"*50)

# Calculate mean spectrum
mean_spectrum = np.mean(X_all, axis=0)
std_spectrum = np.std(X_all, axis=0)

print(f"Overall statistics:")
print(f"  Data range: [{X_all.min():.4f}, {X_all.max():.4f}]")
print(f"  Mean reflectance: {X_all.mean():.4f}")
print(f"  Std reflectance: {X_all.std():.4f}")

print(f"\nMean spectrum statistics:")
print(f"  Range: [{mean_spectrum.min():.4f}, {mean_spectrum.max():.4f}]")
print(f"  Mean: {mean_spectrum.mean():.4f}")
print(f"  Std across bands: {np.std(mean_spectrum):.4f}")

# Check for proper spectral variation
if np.std(mean_spectrum) < 0.01:
    print("\n⚠ WARNING: Low spectral variation detected!")
    print("  Mean spectrum appears flat - this is still a problem")
else:
    print("\n✓ Good spectral variation detected")

# Check vegetation pattern
if B >= 150:
    # Approximate band ranges for AVIRIS
    vis_bands = slice(30, 70)   # Visible
    nir_bands = slice(100, 140)  # NIR
    
    vis_mean = np.mean(mean_spectrum[vis_bands])
    nir_mean = np.mean(mean_spectrum[nir_bands])
    
    print(f"\nVegetation pattern check:")
    print(f"  VIS mean (bands 30-70): {vis_mean:.4f}")
    print(f"  NIR mean (bands 100-140): {nir_mean:.4f}")
    
    if nir_mean > 0 and vis_mean > 0:
        ratio = nir_mean / vis_mean
        print(f"  NIR/VIS ratio: {ratio:.2f}")
        
        if ratio > 2:
            print("  ✓ Strong vegetation signal!")
        elif ratio > 1.5:
            print("  ✓ Moderate vegetation signal")
        else:
            print("  ⚠ Weak or unusual vegetation signal")

# Check individual pixel variation
sample_pixels = X_all[:min(100, len(X_all))]
pixel_stds = [np.std(pixel) for pixel in sample_pixels]

print(f"\nPixel-level variation (first 100 pixels):")
print(f"  Mean std: {np.mean(pixel_stds):.4f}")
print(f"  Min std: {np.min(pixel_stds):.4f}")
print(f"  Max std: {np.max(pixel_stds):.4f}")

if np.mean(pixel_stds) < 0.01:
    print("  ⚠ WARNING: Individual pixels have flat spectra!")
    print("  This indicates a fundamental data problem")
else:
    print("  ✓ Good variation within individual pixels")

# ============================================
# 4. ENCODE LABELS AND SAVE
# ============================================
print("\n4. ENCODING LABELS AND SAVING")
print("-"*50)

# Map USDA codes to 0..K-1 for ML
classes_sorted = np.sort(crop_classes)
code2idx = {c: i for i, c in enumerate(classes_sorted)}
y_enc = np.vectorize(code2idx.get)(y_all)

print("Class distribution:")
counter = Counter(y_all)
for code in crop_classes:
    count = counter[code]
    name = code_to_name[code]
    percent = (count / len(y_all)) * 100
    print(f"  {name:12s} (code {code:3d}): {count:7d} pixels ({percent:5.1f}%)")

# Save the properly processed data
np.savez_compressed(OUT_NPZ, 
                   X=X_all, 
                   y=y_enc, 
                   classes=classes_sorted,
                   extraction_info={
                       'nodata_value': NODATA_VALUE,
                       'total_pixels': total_pixels_extracted,
                       'data_range': (X_all.min(), X_all.max()),
                       'mean_spectrum_std': np.std(mean_spectrum)
                   })

print(f"\n✓ Saved properly processed data → {OUT_NPZ}")

# ============================================
# 5. VISUALIZATION
# ============================================
print("\n5. CREATING DIAGNOSTIC VISUALIZATIONS")
print("-"*50)

# Create wavelength array
wavelengths = np.linspace(380, 2510, B)

# Mapping for visualization
enc2code = {i: int(code) for i, code in enumerate(classes_sorted.tolist())}
enc2name = {i: code_to_name.get(int(code), f"CDL {int(code)}") for i, code in enc2code.items()}

# Create comprehensive visualization
fig, axes = plt.subplots(3, 3, figsize=(16, 14))

# Plot 1: Mean spectra by crop
ax = axes[0, 0]
colors = plt.cm.Set2(np.linspace(0, 1, len(np.unique(y_enc))))
for idx, class_idx in enumerate(np.unique(y_enc)):
    class_pixels = X_all[y_enc == class_idx]
    mean_spec = np.mean(class_pixels, axis=0)
    ax.plot(wavelengths, mean_spec, label=enc2name[class_idx], 
            linewidth=2, color=colors[idx])
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Reflectance')
ax.set_title('Mean Spectral Signatures by Crop', fontweight='bold')
ax.legend(loc='best', fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, max(0.8, np.max(mean_spectrum) * 1.2))

# Plot 2: Sample individual spectra
ax = axes[0, 1]
for i, spec in enumerate(sample_spectra[:10]):
    ax.plot(wavelengths, spec, alpha=0.6, linewidth=1)
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Reflectance')
ax.set_title('Sample Individual Spectra', fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_ylim(0, max(0.8, np.max(sample_spectra) * 1.2) if sample_spectra else 1)

# Plot 3: Overall spectral statistics
ax = axes[0, 2]
ax.plot(wavelengths, mean_spectrum, 'b-', linewidth=2, label='Mean')
ax.fill_between(wavelengths, 
                mean_spectrum - std_spectrum,
                mean_spectrum + std_spectrum,
                alpha=0.3, label='±1 STD')
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Reflectance')
ax.set_title('Overall Spectral Statistics', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(0, max(0.8, (mean_spectrum + std_spectrum).max() * 1.2))

# Plot 4: Spectral variation by crop
ax = axes[1, 0]
for idx, class_idx in enumerate(np.unique(y_enc)):
    class_pixels = X_all[y_enc == class_idx]
    std_spec = np.std(class_pixels, axis=0)
    ax.plot(wavelengths, std_spec, label=enc2name[class_idx], 
            linewidth=2, color=colors[idx], alpha=0.7)
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Std Dev')
ax.set_title('Spectral Variation by Crop', fontweight='bold')
ax.legend(loc='best', fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 5: Class distribution
ax = axes[1, 1]
crop_counts = Counter(y_enc)
crops = [enc2name[i] for i in sorted(crop_counts.keys())]
counts = [crop_counts[i] for i in sorted(crop_counts.keys())]
bars = ax.bar(crops, counts, color=colors[:len(crops)])
ax.set_xlabel('Crop Type')
ax.set_ylabel('Number of Pixels')
ax.set_title('Pixel Distribution by Crop', fontweight='bold')
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
# Add count labels on bars
for bar, count in zip(bars, counts):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{count:,}', ha='center', va='bottom', fontsize=8)

# Plot 6: Histogram of all reflectance values
ax = axes[1, 2]
all_values = X_all.flatten()
ax.hist(all_values[all_values > 0], bins=100, edgecolor='none', alpha=0.7, color='green')
ax.set_xlabel('Reflectance')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of All Reflectance Values', fontweight='bold')
ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Min valid')
ax.axvline(x=1, color='red', linestyle='--', alpha=0.5, label='Max valid')
ax.set_xlim(-0.05, 1.5)
ax.legend()

# Plot 7: Band-wise statistics
ax = axes[2, 0]
band_means = np.mean(X_all, axis=0)
band_stds = np.std(X_all, axis=0)
ax.plot(wavelengths, band_means, 'b-', linewidth=2, label='Mean')
ax.fill_between(wavelengths,
                band_means - band_stds,
                band_means + band_stds,
                alpha=0.3, color='blue', label='±1 STD')
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Reflectance')
ax.set_title('Band-wise Statistics', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(0, max(0.8, (band_means + band_stds).max() * 1.2))

# Plot 8: Correlation matrix between bands (subset)
ax = axes[2, 1]
# Select subset of bands for correlation matrix
band_subset = np.arange(0, B, B//20)  # ~20 bands
corr_matrix = np.corrcoef(X_all[:, band_subset].T)
im = ax.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
ax.set_title('Band Correlation Matrix (Subset)', fontweight='bold')
ax.set_xlabel('Band Index')
ax.set_ylabel('Band Index')
plt.colorbar(im, ax=ax)

# Plot 9: Vegetation indices
ax = axes[2, 2]
if B >= 150:
    # Calculate NDVI for each pixel
    red_band = X_all[:, 60]  # ~680nm
    nir_band = X_all[:, 120]  # ~850nm
    ndvi = (nir_band - red_band) / (nir_band + red_band + 1e-10)
    
    # Plot NDVI distribution by crop
    for idx, class_idx in enumerate(np.unique(y_enc)):
        class_ndvi = ndvi[y_enc == class_idx]
        ax.hist(class_ndvi, bins=30, alpha=0.5, label=enc2name[class_idx],
               color=colors[idx])
    
    ax.set_xlabel('NDVI')
    ax.set_ylabel('Count')
    ax.set_title('NDVI Distribution by Crop', fontweight='bold')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle('AVIRIS Data Extraction Validation - NoData Properly Handled', 
            fontsize=16, fontweight='bold')
plt.tight_layout()

os.makedirs("outputs/figs", exist_ok=True)
plt.savefig("outputs/figs/aviris_extraction_validation_PROPER.png", dpi=150)
plt.show()

print("✓ Saved validation plots")

# ============================================
# 6. FINAL DIAGNOSIS
# ============================================
print("\n" + "="*70)
print("EXTRACTION COMPLETE - FINAL DIAGNOSIS")
print("="*70)

if np.std(mean_spectrum) < 0.01:
    print("\n❌ WARNING: DATA STILL HAS FLAT SPECTRA")
    print("Even after proper NoData handling, the spectra are flat.")
    print("This suggests:")
    print("  1. The AVIRIS data itself may be corrupted")
    print("  2. Wrong file or wrong bands may have been used")
    print("  3. Pre-processing may have averaged out spectral variation")
    print("\nRecommendation: Check the original AVIRIS data files")
elif X_all.max() > 1.5:
    print("\n⚠ Some reflectance values exceed 1.0")
    print("This can happen with atmospheric correction artifacts")
    print("Consider clipping to [0, 1] for classification")
else:
    print("\n✓ SUCCESS: Data extracted properly!")
    print(f"  • {total_pixels_extracted} valid pixels extracted")
    print(f"  • Reflectance range: [{X_all.min():.3f}, {X_all.max():.3f}]")
    print(f"  • Good spectral variation: std = {np.std(mean_spectrum):.4f}")
    if B >= 150 and nir_mean > vis_mean * 1.5:
        print(f"  • Vegetation pattern confirmed (NIR/VIS = {nir_mean/vis_mean:.2f})")

print("\nYou can now use this data for classification!")
print(f"File saved: {OUT_NPZ}")
