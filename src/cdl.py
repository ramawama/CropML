import numpy as np
import rasterio

with np.load('outputs/reflectance_016003.npz') as data:
    hypercube = data["hypercube"]

with rasterio.open("outputs/cdl_2024_aligned_016003.tif") as cdl:
    cdl = cdl.read(1)