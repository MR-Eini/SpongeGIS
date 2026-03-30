import tempfile
import zipfile
from pathlib import Path

import numpy as np
import rasterio
from pyproj import CRS
from rasterio.warp import (
    calculate_default_transform,
    reproject,
    Resampling,
    transform_bounds,
)


# ============================================================
# USER SETTINGS
# ============================================================
input_path = r"X:\download\Mamad\!SpongeGIStest\Kamienna\SPU\SPU2km.zip"
output_raster = r"X:\download\Mamad\!SpongeGIStest\Kamienna\SPU\SPU_unique_ids_3035.tif"

# Allowed range: 100 m to 1000 m
# 100 m  = 1 ha
# 1000 m = 1 km²
cell_size_m = 1000

# Output NoData value
# Valid cell IDs start from 1, so 0 is reserved only for NoData
nodata_value = 0


# ============================================================
# CONSTANTS
# ============================================================
TARGET_CRS = CRS.from_epsg(3035)   # ETRS89 / LAEA Europe
MAX_TIF_PIXELS = 2_000_000
MAX_ZIPPED_TIF_AREA_KM2 = 200_000


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def validate_cell_size(cell_size: float) -> None:
    """Check that the cell size is between 100 and 1000 meters."""
    if not (100 <= cell_size <= 1000):
        raise ValueError(
            f"Invalid cell size: {cell_size} m. "
            "Allowed range is 100 m to 1000 m "
            "(1 ha to 1 km² for square cells)."
        )


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


def calculate_raster_area_km2(src: rasterio.io.DatasetReader) -> float:
    """
    Calculate raster bounding-box area in km² after transforming the bounds
    to EPSG:3035. This is used for the ZIP/TIFF area limit.
    """
    if src.crs is None:
        raise ValueError("Missing coordinate reference system (CRS).")

    bounds_3035 = transform_bounds(src.crs, TARGET_CRS, *src.bounds, densify_pts=21)
    minx, miny, maxx, maxy = bounds_3035
    area_m2 = max(0.0, (maxx - minx) * (maxy - miny))
    return area_m2 / 1_000_000.0


def input_data_control(path: Path) -> dict:
    """
    Validate the input file before processing.

    Rules:
    - Only *.tif, *.tiff, or *.zip are allowed.
    - If *.tif or *.tiff:
        - the number of pixels must not exceed 2,000,000.
        - the raster must have a CRS.
    - If *.zip:
        - the ZIP must contain exactly one TIFF raster (*.tif or *.tiff),
        - the raster must have a CRS,
        - the total area must not exceed MAX_ZIPPED_TIF_AREA_KM2.
    """
    extension = path.suffix.lower()

    if extension not in {".tif", ".tiff", ".zip"}:
        raise ValueError("Only *.tif, *.tiff, or *.zip files are allowed.")

    # --------------------------------------------------------
    # Case 1: Direct TIFF input
    # --------------------------------------------------------
    if extension in {".tif", ".tiff"}:
        with rasterio.open(path) as src:
            if src.crs is None:
                raise ValueError("Missing coordinate reference system (CRS).")

            pixel_count = src.width * src.height
            if pixel_count > MAX_TIF_PIXELS:
                raise ValueError("The number of pixels cannot exceed 2,000,000.")

        return {
            "input_type": "raster",
            "raster_path": path,
            "temp_dir": None
        }

    # --------------------------------------------------------
    # Case 2: ZIP input -> must contain exactly one TIFF raster
    # --------------------------------------------------------
    # --------------------------------------------------------
    # Case 2: ZIP input -> must contain at least one TIFF raster
    # --------------------------------------------------------
    temp_dir = tempfile.TemporaryDirectory()
    extract_dir = Path(temp_dir.name)

    with zipfile.ZipFile(path, "r") as zf:
        zf.extractall(extract_dir)

    tif_files = sorted(list(extract_dir.rglob("*.tif")) + list(extract_dir.rglob("*.tiff")))

    if len(tif_files) == 0:
        temp_dir.cleanup()
        raise ValueError("The ZIP file must contain at least one TIFF raster (*.tif or *.tiff).")

    print("TIFF files found in ZIP:")
    for f in tif_files:
        print(f" - {f}")

    # Choose the TIFF with the largest number of pixels
    best_tif = None
    best_pixels = -1

    for tif_path in tif_files:
        with rasterio.open(tif_path) as src:
            if src.crs is None:
                temp_dir.cleanup()
                raise ValueError(f"Missing coordinate reference system (CRS): {tif_path}")

            pixel_count = src.width * src.height
            if pixel_count > best_pixels:
                best_pixels = pixel_count
                best_tif = tif_path

    print(f"Selected TIFF: {best_tif}")

    with rasterio.open(best_tif) as src:
        total_area_km2 = calculate_raster_area_km2(src)
        if total_area_km2 > MAX_ZIPPED_TIF_AREA_KM2:
            temp_dir.cleanup()
            raise ValueError(
                f"The total area cannot exceed {MAX_ZIPPED_TIF_AREA_KM2:,} km²."
            )

    return {
        "input_type": "raster",
        "raster_path": best_tif,
        "temp_dir": temp_dir
    }


def prepare_raster_mask(raster_path: Path, cell_size: float):
    """
    Read raster, check CRS, reproject to EPSG:3035,
    and create a valid-data mask.
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