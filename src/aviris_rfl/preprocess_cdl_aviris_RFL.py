import rasterio
from rasterio.mask import mask
from shapely.geometry import box, mapping
from shapely.ops import transform as shp_transform
import pyproj
import numpy as np
import rioxarray as rxr
from rasterio.enums import Resampling

# ---------- INPUTS ----------
AVIRIS_IMG = "data/raw/aviris/f240604t01p00r12_rfl/f240604t01p00r12_rfl"  # ENVI .hdr/.img OK
CDL_TIF    = "data/raw/cdl/2024_10m_cdls/2024_10m_cdls.tif"               # native 10 m CDL (EPSG:5070)
OUT_EXTENT = "data/processed/cdl_2024_clip_to_aviris_extent_f240604t01p00r12.tif"
OUT_MATCH  = "data/processed/cdl_2024_matched_to_aviris_grid_f240604t01p00r12.tif"
# ----------------------------

# 1) Read AVIRIS bounds + CRS
with rasterio.open(AVIRIS_IMG) as av:
    av_bounds = av.bounds
    av_crs = av.crs
    av_transform = av.transform
    av_size = (av.width, av.height)

# 2) Build AVIRIS extent polygon
av_box = box(av_bounds.left, av_bounds.bottom, av_bounds.right, av_bounds.top)

# 3) Clip CDL to AVIRIS extent (still on CDL's native grid)
with rasterio.open(CDL_TIF) as src:
    if src.crs != av_crs:
        # reproject AVIRIS extent into CDL CRS
        proj = pyproj.Transformer.from_crs(av_crs, src.crs, always_xy=True).transform
        geom_in_cdl = [mapping(shp_transform(proj, av_box))]
    else:
        geom_in_cdl = [mapping(av_box)]

    # indexes=1 makes a 2D (H,W) array so use shape[0], shape[1]
    cdl_clip, cdl_transform = mask(src, geom_in_cdl, crop=True, indexes=1)
    cdl_meta = src.meta.copy()
    cdl_meta.update({
        "height": cdl_clip.shape[0],
        "width":  cdl_clip.shape[1],
        "transform": cdl_transform,
        "count": 1,
        "compress": "LZW"
    })

with rasterio.open(OUT_EXTENT, "w", **cdl_meta) as dst:
    dst.write(cdl_clip, 1)

print(f"✓ Saved extent-clipped CDL → {OUT_EXTENT}")

# 4) Snap CDL to the AVIRIS grid exactly (CRS, transform, width, height)
av_xr  = rxr.open_rasterio(AVIRIS_IMG, masked=True)  # (B,H,W) — does NOT load full 30 GB
cdl_xr = rxr.open_rasterio(OUT_EXTENT).squeeze()     # (H?,W?)

# Reproject CRS first (nearest, categorical)
if cdl_xr.rio.crs != av_xr.rio.crs:
    cdl_xr = cdl_xr.rio.reproject(av_xr.rio.crs, resampling=Resampling.nearest)

# Reproject to match the AVIRIS pixel grid (same transform/size)
cdl_matched = cdl_xr.rio.reproject_match(av_xr, resampling=Resampling.nearest)
cdl_matched.rio.to_raster(OUT_MATCH, compress="LZW")
print(f"✓ Snapped CDL to AVIRIS grid → {OUT_MATCH}")

# 5) Normalize tiny float/sign diffs in transform (optional but kills strict-assert noise)
#    We replace the CDL transform with the AVIRIS transform exactly after confirming near-equality.
with rasterio.open(AVIRIS_IMG) as av_src, rasterio.open(OUT_MATCH, "r+") as cdl_src:
    # check with tolerance
    a = tuple(av_src.transform)[:6]
    b = tuple(cdl_src.transform)[:6]
    if not np.allclose(a, b, rtol=0, atol=1e-9):
        raise RuntimeError("Matched CDL is not sufficiently close to AVIRIS grid; something went off.")
    # force exact equality (handles -0.0 vs 0.0 pretty-prints)
    cdl_src.transform = av_src.transform

# 6) Final strict verification
with rasterio.open(AVIRIS_IMG) as av_src, rasterio.open(OUT_MATCH) as cdl_src:
    assert av_src.crs == cdl_src.crs, "CRS mismatch after match()"
    assert av_src.transform == cdl_src.transform, "Transform mismatch after normalization"
    assert (av_src.width, av_src.height) == (cdl_src.width, cdl_src.height), "Size mismatch"
    print("✓ Final checks passed: CRS/transform/size match exactly.")
