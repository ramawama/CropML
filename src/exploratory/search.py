import hypercoast

hypercoast.nasa_earth_login()


# Display results for types of products from aviris in nasa earthdata
'''
results = hypercoast.search_datasets(instrument="aviris")
datasets = set()
for item in results:
    summary = item.summary()
    short_name = summary["short-name"]
    if short_name not in datasets:
        print(short_name)
    datasets.add(short_name)
print(f"\nFound {len(datasets)} unique datasets")
'''

# search_nasa_data
# Args:
#         bbox (List[float], optional): The bounding box coordinates [xmin, ymin, xmax, ymax].
#         temporal (str, optional): The temporal extent of the data.
#         count (int, optional): The number of granules to retrieve. Defaults to -1 (retrieve all).
#         short_name (str, optional): The short name of the dataset. Defaults to "ECO_L2T_LSTE".
#         output (str, optional): The output file path to save the GeoDataFrame as a file.
#         crs (str, optional): The coordinate reference system (CRS) of the GeoDataFrame. Defaults to "EPSG:4326".
#         return_gdf (bool, optional): Whether to return the GeoDataFrame in addition to the granules. Defaults to False.
#         **kwargs: Additional keyword arguments for the earthaccess.search_data() function.

tspan = ("2024-01-01", "2024-12-31")
results = hypercoast.search_nasa_data(
    # AVIRIS-NG_L2_Reflectance_2110 (modern hyperspectral)
    # AVIRIS-Classic_L2_Reflectance_2154 
    short_name=["AVIRIS-NG_L2_Reflectance_2110", "AVIRIS-Classic_L2_Reflectance_2154"],
    temporal=tspan,
)

print(f"\nFound {len(results)} reflectance datasets in 2024")
item = results[0]
print(item)

