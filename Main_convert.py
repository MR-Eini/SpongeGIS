import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from pyproj import CRS
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import calculate_default_transform, reproject, Resampling


# ============================================================
# USER SETTINGS
# ============================================================
input_path = r"X:\download\Mamad\!SpongeGIStest\Kamienna\SPU\SPU2km.shp"
output_raster = r"X:\download\Mamad\!SpongeGIStest\Kamienna\SPU\SPU_unique_ids_3035.tif"

# Allowed range: 100 m to 1000 m
# 100 m  = 1 ha
# 1000 m = 1 km²
cell_size_m = 1000

# For vector input only:
# False = include cells whose center is inside the polygon
# True  = include any cell touched by the polygon
all_touched = False

# Output NoData value
# Real cell IDs will start from 1, so 0 is reserved only for NoData
nodata_value = 0


# ============================================================
# CONSTANTS
# ============================================================
TARGET_CRS = CRS.from_epsg(3035)  # ETRS89 / LAEA Europe


# ============================================================
# HELPERS
# ============================================================
def validate_cell_size(cell_size: float) -> None:
    """Check that cell size is between 100 and 1000 meters."""
    if not (100 <= cell_size <= 1000):
        raise ValueError(
            f"Invalid cell size: {cell_size} m. "
            "Allowed range is 100 m to 1000 m "
            "(1 ha to 1 km² for square cells)."
        )


def detect_input_type(path: Path) -> str:
    """
    Detect whether the input is a shapefile or a raster.
    Returns: 'vector' or 'raster'
    """
    ext = path.suffix.lower()

    if ext == ".shp":
        return "vector"

    raster_exts = {".tif", ".tiff", ".img", ".asc", ".bil", ".vrt"}
    if ext in raster_exts:
        return "raster"

    raise ValueError(
        f"Unsupported input format: {ext}. "
        "Supported vector: .shp | Supported raster: .tif, .tiff, .img, .asc, .bil, .vrt"
    )


def build_unique_id_array(mask: np.ndarray, nodata: int = 0) -> np.ndarray:
    """
    Create an output raster where every valid cell gets a unique ID starting from 1.
    Invalid/outside cells remain NoData (stored as 0 in the raster).
    """
    valid = mask > 0
    out = np.full(mask.shape, nodata, dtype=np.uint32)

    n_valid = int(valid.sum())
    if n_valid == 0:
        raise ValueError("No valid cells found to assign unique IDs.")

    out[valid] = np.arange(1, n_valid + 1, dtype=np.uint32)
    return out


def prepare_vector_mask(vector_path: Path, cell_size: float, all_touched: bool = False):
    """
    Read shapefile, check CRS, reproject to EPSG:3035, and rasterize the geometry footprint.
    """
    gdf = gpd.read_file(vector_path)

    if gdf.empty:
        raise ValueError("Input shapefile is empty.")

    if gdf.crs is None:
        raise ValueError(
            "Input shapefile has no CRS defined. "
            "It cannot be safely converted to EPSG:3035."
        )

    # Remove null/empty geometries
    gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty].copy()
    if gdf.empty:
        raise ValueError("No valid geometries found in the shapefile.")

    # Repair invalid geometries if needed
    invalid = ~gdf.geometry.is_valid
    if invalid.any():
        gdf.loc[invalid, "geometry"] = gdf.loc[invalid, "geometry"].buffer(0)

    # Reproject if necessary
    src_crs = CRS.from_user_input(gdf.crs)
    if src_crs != TARGET_CRS:
        gdf = gdf.to_crs(TARGET_CRS)

    # Build raster grid
    minx, miny, maxx, maxy = gdf.total_bounds

    width = math.ceil((maxx - minx) / cell_size)
    height = math.ceil((maxy - miny) / cell_size)

    if width <= 0 or height <= 0:
        raise ValueError("Calculated raster dimensions are invalid.")

    transform = from_origin(minx, maxy, cell_size, cell_size)

    # Rasterize: 1 inside geometry, 0 outside
    shapes = ((geom, 1) for geom in gdf.geometry)

    mask = rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        all_touched=all_touched,
        dtype="uint8"
    )

    return mask, transform, TARGET_CRS


def prepare_raster_mask(raster_path: Path, cell_size: float):
    """
    Read raster, check CRS, reproject to EPSG:3035, and create a valid-data mask.
    Valid cells become 1, invalid/NoData become 0.
    """
    with rasterio.open(raster_path) as src:
        if src.crs is None:
            raise ValueError(
                "Input raster has no CRS defined. "
                "It cannot be safely converted to EPSG:3035."
            )

        src_crs = CRS.from_user_input(src.crs)

        # Valid data mask from input raster
        # 255 = valid, 0 = invalid
        src_valid_mask = (src.dataset_mask() > 0).astype(np.uint8)

        # Reproject and resample to target CRS and requested resolution
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src_crs,
            TARGET_CRS,
            src.width,
            src.height,
            *src.bounds,
            resolution=cell_size
        )

        if dst_width <= 0 or dst_height <= 0:
            raise ValueError("Calculated raster dimensions are invalid.")

        dst_mask = np.zeros((dst_height, dst_width), dtype=np.uint8)

        reproject(
            source=src_valid_mask,
            destination=dst_mask,
            src_transform=src.transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=TARGET_CRS,
            src_nodata=0,
            dst_nodata=0,
            resampling=Resampling.nearest
        )

    return dst_mask, dst_transform, TARGET_CRS


def save_raster(array: np.ndarray, output_path: Path, transform, crs, nodata: int = 0) -> None:
    """
    Save the unique-ID raster as GeoTIFF.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=array.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
        compress="lzw"
    ) as dst:
        dst.write(array, 1)


# ============================================================
# MAIN
# ============================================================
def main():
    in_path = Path(input_path)
    out_path = Path(output_raster)

    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    validate_cell_size(cell_size_m)
    input_type = detect_input_type(in_path)

    if input_type == "vector":
        mask, transform, crs = prepare_vector_mask(
            vector_path=in_path,
            cell_size=cell_size_m,
            all_touched=all_touched
        )
    else:
        mask, transform, crs = prepare_raster_mask(
            raster_path=in_path,
            cell_size=cell_size_m
        )

    unique_ids = build_unique_id_array(mask, nodata=nodata_value)
    n_cells = int((unique_ids != nodata_value).sum())

    save_raster(
        array=unique_ids,
        output_path=out_path,
        transform=transform,
        crs=crs,
        nodata=nodata_value
    )

    print("Done.")
    print(f"Input file: {in_path}")
    print(f"Input type: {input_type}")
    print(f"Output raster: {out_path}")
    print("Output CRS: EPSG:3035")
    print(f"Cell size: {cell_size_m} m")
    print(f"NoData value: {nodata_value}")
    print(f"Valid cells with unique IDs: {n_cells}")
    print("Cell IDs start from 1.")


if __name__ == "__main__":
    main()