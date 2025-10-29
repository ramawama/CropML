import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from scipy.ndimage import binary_erosion

AVIRIS_IMG = "data/raw/aviris/f230912t01p00r10rdn_g/f230912t01p00r10rdn_g_sc01_ort_img"   # AVIRIS orthocorrected file (.hdr in same folder)
CDL_TIF    = "data/raw/cdl/2023_30m_cdls/2023_30m_cdls.tif" # ground truth crop layer
OUT_TIF    = "outputs/cdl_2023_to_aviris_footprint.tif" # dst file
ERODE_PIXELS = 1  # set to 0 to disable edge shrink

# --- 1. Open AVIRIS reference ---
with rasterio.open(AVIRIS_IMG) as av:
    av_crs = av.crs
    av_transform = av.transform
    av_h, av_w = av.height, av.width
    av_profile = av.profile
    # pick a representative band (AVIRIS = 224 bands)
    av_ref = av.read(1)

# --- 2. Define valid AVIRIS pixels (footprint) ---
# Adjust threshold depending on your preprocessing:
# e.g. av_ref > 0 or use a QA mask if you built one earlier.
av_valid = av_ref > 0

if ERODE_PIXELS and ERODE_PIXELS > 0:
    av_valid = binary_erosion(av_valid, iterations=ERODE_PIXELS)

# --- 3. Open CDL and metadata ---
with rasterio.open(CDL_TIF) as src:
    cdl = src.read(1)
    cdl_crs = src.crs
    cdl_transform = src.transform
    cdl_dtype = src.dtypes[0]
    cdl_nodata = src.nodata if src.nodata is not None else 0

# --- 4. Reproject CDL to AVIRIS grid ---
cdl_reproj = np.full((av_h, av_w), cdl_nodata, dtype=cdl_dtype)

reproject(
    source=cdl,
    destination=cdl_reproj,
    src_transform=cdl_transform, src_crs=cdl_crs,
    dst_transform=av_transform, dst_crs=av_crs,
    dst_width=av_w, dst_height=av_h,
    resampling=Resampling.nearest
)

# --- 5. Apply AVIRIS footprint mask ---
cdl_reproj[~av_valid] = cdl_nodata

# --- 6. Save result ---
out_meta = av_profile.copy()
out_meta.update(
    count=1,
    dtype=cdl_reproj.dtype,
    nodata=cdl_nodata,
    compress="lzw"
)
with rasterio.open(OUT_TIF, "w", **out_meta) as dst:
    dst.write(cdl_reproj, 1)

print(f"✅ Saved AVIRIS-footprint clipped CDL → {OUT_TIF}")
