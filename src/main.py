import fiona
import rasterio
from matplotlib import pyplot

with fiona.open("classifier/IowaCounties.shp", "r") as shapefile:
    shapes = [feature["geometry"] for feature in shapefile]

with rasterio.open('/Users/ramajanco/Documents/GIS/CDL_2024_10m/2024_10m_cdls/2024_10m_cdls.tif') as src:
    out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True)
    out_meta = src.meta

