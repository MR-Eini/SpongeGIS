from pathlib import Path
import tempfile
import zipfile

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin


# =========================
# USER SETTINGS
# =========================
zip_path = r"C:\Users\p101727\Downloads\SPU250_3035.zip"
output_raster = r"X:\download\Mamad\!SpongeGIStest\Kamienna\SPU\SPU_unique_ids_3035.tif"

# Name of the field in the shapefile that contains the SPU IDs
id_field = "Id"

# Output CRS and cell size
target_crs = "EPSG:3035"
cell_size = 250  # meters

# NoData value for cells outside polygons
nodata_value = 0
# =========================


def find_shp_in_folder(folder: Path) -> Path:
    shp_files = list(folder.rglob("*.shp"))
    if not shp_files:
        raise FileNotFoundError("No .shp file found inside the ZIP.")
    if len(shp_files) > 1:
        print("Multiple shapefiles found. Using:", shp_files[0])
    return shp_files[0]


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Extract ZIP
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmpdir)

        shp_path = find_shp_in_folder(tmpdir)
        print("Shapefile:", shp_path)

        # Read polygons
        gdf = gpd.read_file(shp_path)

        if gdf.empty:
            raise ValueError("The shapefile is empty.")

        if id_field not in gdf.columns:
            raise ValueError(f"Field '{id_field}' not found. Available fields: {list(gdf.columns)}")

        if gdf.crs is None:
            raise ValueError("Input shapefile has no CRS defined.")

        # Reproject
        gdf = gdf.to_crs(target_crs)

        # Keep only valid geometries
        gdf = gdf[gdf.geometry.notnull()].copy()
        gdf = gdf[gdf.is_valid].copy()

        if gdf.empty:
            raise ValueError("No valid geometries found after filtering.")

        # Check IDs
        if gdf[id_field].isnull().any():
            raise ValueError(f"Field '{id_field}' contains null values.")

        # Convert IDs to int32
        gdf[id_field] = gdf[id_field].astype(np.int32)

        # Check uniqueness
        n_total = len(gdf)
        n_unique = gdf[id_field].nunique()
        print(f"Polygons: {n_total}")
        print(f"Unique IDs in '{id_field}': {n_unique}")

        if n_total != n_unique:
            raise ValueError(
                f"IDs are not unique: {n_total} polygons but only {n_unique} unique values in '{id_field}'."
            )

        # Bounds
        minx, miny, maxx, maxy = gdf.total_bounds

        width = int(np.ceil((maxx - minx) / cell_size))
        height = int(np.ceil((maxy - miny) / cell_size))

        # Align top-left corner
        transform = from_origin(minx, maxy, cell_size, cell_size)

        # Prepare shapes: (geometry, value)
        shapes = [(geom, val) for geom, val in zip(gdf.geometry, gdf[id_field])]

        # Rasterize
        raster = rasterize(
            shapes=shapes,
            out_shape=(height, width),
            transform=transform,
            fill=nodata_value,
            dtype="int32",
            all_touched=False
        )

        # Save raster
        output_raster_path = Path(output_raster)
        output_raster_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(
            output_raster_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype="int32",
            crs=target_crs,
            transform=transform,
            nodata=nodata_value,
            compress="lzw"
        ) as dst:
            dst.write(raster, 1)

        print("Done.")
        print("Output raster:", output_raster)
        print("CRS:", target_crs)
        print("Cell size:", cell_size)
        print("NoData:", nodata_value)


if __name__ == "__main__":
    main()