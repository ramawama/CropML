import rasterio
import glob
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


def decode_qa_pixel(qa_array):
    '''
    Decode QA_pixel bitmask

    0 Fill
    1 Dilated Cloud
    2 Cirrus
    3 Cloud
    4 Cloud Shadow
    5 Snow
    6 Clear
    7 Water
    8-9 Cloud Confidence
    10-11 Cloud Shadow Confidence
    12-13 Snow/Ice Confidence
    14-15 Cirrus Confidence
    '''
    # Naive implementation, we will mask fill, cloud, shadow, snow
    # Use bit shifting bc qa is stored as bit index
    # 1 << 0 (Left shift 0001)
    # qa_array & (1 << n) checks if that bit is set
    fill = (qa_array & (1 << 0)) != 0 # (0: image, 1: fill)
    cloud = (qa_array & (1 << 3)) != 0 # (0: low conf, 1: high)
    shadow = (qa_array & (1 << 4)) != 0 # (0: low conf, 1: high conf)
    snow = (qa_array & (1 << 5)) != 0 # (0: low conf, 1: high conf)

    return fill | cloud
# Get QA mask
with rasterio.open(qa_file) as src:
    qa = src.read(1)
    profile = src.profile

mask = decode_qa_pixel(qa)

band_stack = []
for b in band_files:
    with rasterio.open(b) as src:
        arr = src.read(1).astype("float32")
        arr[mask] = np.nan # set mask
        band_stack.append(arr)


# Extract bands (remember: B4=Red, B3=Green, B2=Blue)
red   = band_stack[3]  
green = band_stack[2]
blue  = band_stack[1] 

# Stretch for contrast
def stretch(arr):
    p2, p98 = np.nanpercentile(arr, (2, 98))
    return np.clip((arr - p2) / (p98 - p2), 0, 1)

rgb = np.dstack([stretch(red), stretch(green), stretch(blue)])

plt.figure(figsize=(8,8))
plt.imshow(rgb)
plt.title("Landsat True Color (B4=Red, B3=Green, B2=Blue)")
plt.axis("off")
plt.show()
