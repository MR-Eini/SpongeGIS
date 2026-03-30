import math
import tempfile
import zipfile
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
input_path = r"X:\download\Mamad\!SpongeGIsStest\Kamienna\SPU\SPU2km - Copy.zip"
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
# Valid cell IDs start from 1, so 0 is reserved only for NoData
nodata_value = 0


# ============================================================
# CONSTANTS
# ============================================================
TARGET_CRS = CRS.from_epsg(3035)   # ETRS89 / LAEA Europe
MAX_RASTER_PIXELS = 2_000_000
MAX_VECTOR_AREA_KM2 = 200_000      # Change this if needed


# ============================================================
# VALIDATION FUNCTIONS
# ============================================================
def validate_cell_size(cell_size: float) -> None:
    """Check that the cell size is between 100 and 1000 meters."""
    if not (100 <= cell_size <= 1000):
        raise ValueError(
            f"Invalid cell size: {cell_size} m. "
            "Allowed range is 100 m to 1000 m "
            "(1 ha to 1 km² for square cells)."
        )


def validate_raster_file(raster_path: Path) -> None:
    """
    Validate raster input.
    Rules:
    - CRS must exist
    - number of pixels must not exceed MAX_RASTER_PIXELS
    """
    with rasterio.open(raster_path) as src:
        if src.crs is None:
            raise ValueError("Missing coordinate reference system (CRS).")

        pixel_count = src.width * src.height
        if pixel_count > MAX_RASTER_PIXELS:
            raise ValueError(
                f"The number of pixels cannot exceed {MAX_RASTER_PIXELS:,}."
            )


def validate_and_prepare_vector_file(vector_path: Path) -> gpd.GeoDataFrame:
    """
    Validate vector input and return a cleaned GeoDataFrame in EPSG:3035.
    Rules:
    - CRS must exist
    - total area must not exceed MAX_VECTOR_AREA_KM2
    """
    gdf = gpd.read_file(vector_path)

    if gdf.empty:
        raise ValueError("The input shapefile is empty.")

    if gdf.crs is None:
        raise ValueError("Missing coordinate reference system (CRS).")

    # Remove null/empty geometries
    gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty].copy()
    if gdf.empty:
        raise ValueError("No valid geometries were found in the shapefile.")

    # Repair invalid geometries if possible
    invalid = ~gdf.geometry.is_valid
    if invalid.any():
        gdf.loc[invalid, "geometry"] = gdf.loc[invalid, "geometry"].buffer(0)

    # Reproject to target CRS for area check and output processing
    source_crs = CRS.from_user_input(gdf.crs)
    if source_crs != TARGET_CRS:
        gdf = gdf.to_crs(TARGET_CRS)

    total_area_km2 = float(gdf.geometry.area.sum()) / 1_000_000.0
    if total_area_km2 > MAX_VECTOR_AREA_KM2:
        raise ValueError(
            f"The total area cannot exceed {MAX_VECTOR_AREA_KM2:,} km²."
        )

    return gdf


def input_data_control(path: Path) -> dict:
    """
    Validate the input and return a standardized description of the dataset.

    Allowed direct input:
    - *.shp
    - *.tif
    - *.tiff
    - *.zip

    ZIP rules:
    - it must contain exactly one main spatial dataset:
        - one shapefile (*.shp), or
        - one TIFF raster (*.tif or *.tiff)
    - if both shapefile and TIFF are present, an error is raised
    - if more than one shapefile or more than one TIFF is present, an error is raised
    """
    extension = path.suffix.lower()

    if extension not in {".shp", ".tif", ".tiff", ".zip"}:
        raise ValueError("Only *.shp, *.tif, *.tiff, or *.zip files are allowed.")

    # --------------------------------------------------------
    # Case 1: Direct shapefile input
    # --------------------------------------------------------
    if extension == ".shp":
        gdf = validate_and_prepare_vector_file(path)
        return {
            "input_type": "vector",
            "vector_gdf": gdf,
            "raster_path": None,
            "temp_dir": None
        }

    # --------------------------------------------------------
    # Case 2: Direct TIFF input
    # --------------------------------------------------------
    if extension in {".tif", ".tiff"}:
        validate_raster_file(path)
        return {
            "input_type": "raster",
            "vector_gdf": None,
            "raster_path": path,
            "temp_dir": None
        }

    # --------------------------------------------------------
    # Case 3: ZIP input
    # --------------------------------------------------------
    temp_dir = tempfile.TemporaryDirectory()
    extract_dir = Path(temp_dir.name)

    try:
        with zipfile.ZipFile(path, "r") as zf:
            zf.extractall(extract_dir)
    except zipfile.BadZipFile:
        temp_dir.cleanup()
        raise ValueError("The ZIP file is invalid or corrupted.")

    shp_files = sorted(extract_dir.rglob("*.shp"))
    tif_files = sorted(list(extract_dir.rglob("*.tif")) + list(extract_dir.rglob("*.tiff")))

    if len(shp_files) == 0 and len(tif_files) == 0:
        temp_dir.cleanup()
        raise ValueError("The ZIP file must contain one shapefile (*.shp) or one TIFF raster (*.tif or *.tiff).")

    if len(shp_files) > 0 and len(tif_files) > 0:
        temp_dir.cleanup()
        raise ValueError("The ZIP file must contain either one shapefile or one TIFF raster, not both.")

    if len(shp_files) > 1:
        temp_dir.cleanup()
        raise ValueError("The ZIP file must contain only one shapefile (*.shp).")

    if len(tif_files) > 1:
        temp_dir.cleanup()
        raise ValueError("The ZIP file must contain only one TIFF raster (*.tif or *.tiff).")

    # ZIP contains one shapefile
    if len(shp_files) == 1:
        try:
            gdf = validate_and_prepare_vector_file(shp_files[0])
        except Exception:
            temp_dir.cleanup()
            raise

        return {
            "input_type": "vector",
            "vector_gdf": gdf,
            "raster_path": None,
            "temp_dir": temp_dir
        }

    # ZIP contains one TIFF
    try:
        validate_raster_file(tif_files[0])
    except Exception:
        temp_dir.cleanup()
        raise

    return {
        "input_type": "raster",
        "vector_gdf": None,
        "raster_path": tif_files[0],
        "temp_dir": temp_dir
    }


# ============================================================
# PROCESSING FUNCTIONS
# ============================================================
def build_unique_id_array(mask: np.ndarray, nodata: int = 0) -> np.ndarray:
    """
    Create an output raster where each valid cell gets a unique ID starting from 1.
    Invalid/outside cells remain NoData (stored as 0 in the raster).
    """
    valid = mask > 0
    output = np.full(mask.shape, nodata, dtype=np.uint32)

    valid_count = int(valid.sum())
    if valid_count == 0:
        raise ValueError("No valid cells were found to assign unique IDs.")

    output[valid] = np.arange(1, valid_count + 1, dtype=np.uint32)
    return output


def prepare_vector_mask(gdf: gpd.GeoDataFrame, cell_size: float, all_touched: bool = False):
    """
    Rasterize a validated GeoDataFrame that is already in EPSG:3035.
    """
    minx, miny, maxx, maxy = gdf.total_bounds

    width = math.ceil((maxx - minx) / cell_size)
    height = math.ceil((maxy - miny) / cell_size)

    if width <= 0 or height <= 0:
        raise ValueError("Calculated raster dimensions are invalid.")

    transform = from_origin(minx, maxy, cell_size, cell_size)

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
    Read raster, reproject to EPSG:3035, and create a valid-data mask.
    Valid cells become 1, invalid/NoData become 0.
    """
    with rasterio.open(raster_path) as src:
        if src.crs is None:
            raise ValueError("Missing coordinate reference system (CRS).")

        source_crs = CRS.from_user_input(src.crs)

        # Valid data mask from the input raster
        src_valid_mask = (src.dataset_mask() > 0).astype(np.uint8)

        # Reproject and resample to the target CRS and requested resolution
        dst_transform, dst_width, dst_height = calculate_default_transform(
            source_crs,
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
            src_crs=source_crs,
            dst_transform=dst_transform,
            dst_crs=TARGET_CRS,
            src_nodata=0,
            dst_nodata=0,
            resampling=Resampling.nearest
        )

    return dst_mask, dst_transform, TARGET_CRS


def save_raster(array: np.ndarray, output_path: Path, transform, crs, nodata: int = 0) -> None:
    """Save the unique-ID raster as a GeoTIFF."""
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

    control_result = None
    temp_dir = None

    try:
        control_result = input_data_control(in_path)
        temp_dir = control_result["temp_dir"]

        if control_result["input_type"] == "vector":
            mask, transform, crs = prepare_vector_mask(
                gdf=control_result["vector_gdf"],
                cell_size=cell_size_m,
                all_touched=all_touched
            )
        else:
            mask, transform, crs = prepare_raster_mask(
                raster_path=control_result["raster_path"],
                cell_size=cell_size_m
            )

        unique_ids = build_unique_id_array(mask, nodata=nodata_value)
        valid_cell_count = int((unique_ids != nodata_value).sum())

        save_raster(
            array=unique_ids,
            output_path=out_path,
            transform=transform,
            crs=crs,
            nodata=nodata_value
        )

        print("Done.")
        print(f"Input file: {in_path}")
        print(f"Input type: {control_result['input_type']}")
        if control_result["input_type"] == "raster":
            print(f"Processed raster: {control_result['raster_path']}")
        print(f"Output raster: {out_path}")
        print("Output CRS: EPSG:3035")
        print(f"Cell size: {cell_size_m} m")
        print(f"NoData value: {nodata_value}")
        print(f"Valid cells with unique IDs: {valid_cell_count}")
        print("Cell IDs start from 1.")

    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


if __name__ == "__main__":
    main()