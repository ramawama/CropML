import numpy as np
import rasterio

with np.load('outputs/reflectance_016003.npz', allow_pickle=True) as data:
    hypercube = data["hypercube"]
    hypercube_profile = data["hyper_profile"]

# print(hypercube)
print(type(hypercube_profile))
print(hypercube_profile)


# Align CDL and Reflectance 
with rasterio.open('data/raw/cdl/2024_30m_cdls/2024_30m_cdls.tif') as src:
    cdl = src.read(1)
    cdl_profile = src.profile

# print(cdl)
# print(cdl_profile)

H, W = hypercube.shape[1], hypercube[2]
dst_crs = hypercube_profile['crs']
dst_transform = hypercube_profile['transform']

# FIX HOW PROFILE IS OBTAINED NOT WORKING