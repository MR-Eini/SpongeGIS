# -*- coding: utf-8 -*-
"""
Rasterize a polygon layer from a ZIP shapefile to a 10 m raster
using the SPU raster extent and EPSG:3035.

What this script does
---------------------
1) Reads a polygon shapefile from a ZIP file.
2) Checks whether the vector has a CRS.
3) If CRS is missing, it uses SOURCE_CRS_IF_MISSING.
4) Cleans geometries BEFORE reprojection.
5) Reprojects to EPSG:3035 if needed.
6) Performs a second light geometry cleaning AFTER reprojection.
7) Creates a new raster with:
   - SPU top-left origin
   - 10 m resolution
   - SPU-based extent
   - CRS EPSG:3035
8) Rasterizes polygons:
   - constant value BURN_VALUE, or
   - numeric attribute field ATTRIBUTE_FIELD
9) Writes GeoTIFF with NoData = 0.

Required packages
-----------------
- geopandas
- rasterio
- shapely
- numpy

Example conda install
---------------------
conda install -c conda-forge geopandas rasterio shapely numpy
"""

import math
import os
import tempfile
import zipfile
from typing import Optional, Union

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely.geometry import box


# ============================================================
# USER-DEFINED SETTINGS (EDIT ONLY THIS BLOCK)
# ============================================================

# Template raster: output extent/origin will be based on this raster
SPU_PATH = r"X:\download\Mamad\!SpongeGIStest\Kamienna\SPU\SPU_unique_ids_3035.tif"

# Input ZIP polygon and final output raster
INPUT_ZIP = r"X:\download\Mamad\!SpongeGIStest\Kamienna\input\shp\Arable.zip"
OUTPUT_RASTER = r"X:\download\Mamad\!SpongeGIStest\Kamienna\output\Arable_10m_on_SPU_3035.tif"

# Optional: shapefile name inside the ZIP without ".shp"
# Example: "Arable"
# Keep None to use the first .shp found in the ZIP
ZIP_LAYER = None

# Optional CRS if the input vector has no CRS at all.
# Keep None if the input vector already has CRS.
# Example: "EPSG:2180"
SOURCE_CRS_IF_MISSING = None

# Required target CRS
TARGET_CRS = CRS.from_epsg(3035)

# Output raster resolution in map units (meters)
OUTPUT_RESOLUTION = 10.0

# Rasterization settings
# If ATTRIBUTE_FIELD is None, polygons get BURN_VALUE
# If ATTRIBUTE_FIELD is set, that numeric field is burned instead
ATTRIBUTE_FIELD = None
BURN_VALUE = 1

# Rasterization behavior
ALL_TOUCHED = False

# Output settings
OUTPUT_DTYPE = "uint8"   # Examples: "uint8", "uint16", "int32", "float32"
OUTPUT_NODATA = 0


# ============================================================
# INTERNAL HELPERS
# ============================================================

DTypeLike = Union[str, np.dtype]


def ensure_parent_dir(path: str) -> None:
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)


def normalize_dtype(dtype_value: Optional[DTypeLike]) -> str:
    if dtype_value is None:
        raise ValueError("OUTPUT_DTYPE cannot be None in this script.")
    return np.dtype(dtype_value).name


def crs_equal(a: CRS, b: CRS) -> bool:
    try:
        return CRS.from_user_input(a) == CRS.from_user_input(b)
    except Exception:
        return False


def load_polygon_zip(zip_path: str, layer: Optional[str] = None) -> gpd.GeoDataFrame:
    """
    Read a zipped shapefile by extracting it to a temporary folder first.
    This avoids /vsizip path issues on Windows.
    """
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")

    with tempfile.TemporaryDirectory(prefix="zip_shp_") as temp_dir:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(temp_dir)

        shp_files = []
        for root, _, files in os.walk(temp_dir):
            for f in files:
                if f.lower().endswith(".shp"):
                    shp_files.append(os.path.join(root, f))

        if not shp_files:
            raise FileNotFoundError(f"No .shp file found inside ZIP: {zip_path}")

        shp_files = sorted(shp_files)

        if layer is not None:
            layer_lc = layer.lower()
            matches = [
                p for p in shp_files
                if os.path.splitext(os.path.basename(p))[0].lower() == layer_lc
            ]
            if not matches:
                available = [os.path.basename(p) for p in shp_files]
                raise FileNotFoundError(
                    f"Layer '{layer}' not found inside ZIP.\n"
                    f"Available shapefiles: {available}"
                )
            shp_path = matches[0]
        else:
            shp_path = shp_files[0]

        print(f"[INFO] Reading extracted shapefile: {shp_path}")
        gdf = gpd.read_file(shp_path)

    return gdf


def prepare_gdf(
    gdf: gpd.GeoDataFrame,
    target_crs: CRS,
    source_crs_if_missing: Optional[str],
    clip_bounds,
) -> gpd.GeoDataFrame:
    """
    Validate CRS, clean geometries BEFORE reprojection,
    reproject, clean again lightly, and filter to SPU extent.
    """
    if gdf.empty:
        raise ValueError("Input vector is empty.")

    # --------------------------------------------------------
    # 1) Ensure CRS exists first
    # --------------------------------------------------------
    if gdf.crs is None:
        if source_crs_if_missing is None:
            raise ValueError(
                "Input vector has no CRS.\n"
                "Set SOURCE_CRS_IF_MISSING, for example: 'EPSG:2180'"
            )
        gdf = gdf.set_crs(source_crs_if_missing)
        print(f"[INFO] Input CRS missing. Using user-defined CRS: {gdf.crs}")

    # --------------------------------------------------------
    # 2) Basic cleaning BEFORE reprojection
    # --------------------------------------------------------
    gdf = gdf[gdf.geometry.notnull()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()

    if gdf.empty:
        raise ValueError("No valid geometries remain after removing empty/null geometry.")

    # Keep only Polygon / MultiPolygon
    poly_mask = gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    gdf = gdf[poly_mask].copy()

    if gdf.empty:
        raise ValueError("No Polygon or MultiPolygon features found in input data.")

    # Fix invalid geometries before reprojection
    invalid_mask = ~gdf.is_valid
    if invalid_mask.any():
        print(
            f"[INFO] Fixing {int(invalid_mask.sum())} invalid geometries "
            f"before reprojection with buffer(0)"
        )
        gdf.loc[invalid_mask, "geometry"] = gdf.loc[invalid_mask, "geometry"].buffer(0)

    # Remove anything still invalid/empty after fixing
    gdf = gdf[gdf.geometry.notnull()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()
    gdf = gdf[gdf.is_valid].copy()

    if gdf.empty:
        raise ValueError("No valid polygon geometries remain after pre-reprojection cleaning.")

    # --------------------------------------------------------
    # 3) Reproject AFTER cleaning
    # --------------------------------------------------------
    if not crs_equal(gdf.crs, target_crs):
        print(f"[INFO] Reprojecting vector from {gdf.crs} to {target_crs}")
        gdf = gdf.to_crs(target_crs)

    # --------------------------------------------------------
    # 4) Light cleaning AFTER reprojection
    # --------------------------------------------------------
    gdf = gdf[gdf.geometry.notnull()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()

    if gdf.empty:
        raise ValueError("No geometries remain after reprojection.")

    invalid_mask = ~gdf.is_valid
    if invalid_mask.any():
        print(
            f"[INFO] Fixing {int(invalid_mask.sum())} invalid geometries "
            f"after reprojection with buffer(0)"
        )
        gdf.loc[invalid_mask, "geometry"] = gdf.loc[invalid_mask, "geometry"].buffer(0)

    gdf = gdf[gdf.geometry.notnull()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()
    gdf = gdf[gdf.is_valid].copy()

    if gdf.empty:
        raise ValueError("No valid polygon geometries remain after reprojection.")

    # --------------------------------------------------------
    # 5) Filter to SPU extent
    # --------------------------------------------------------
    left, bottom, right, top = clip_bounds
    bbox_geom = box(left, bottom, right, top)
    gdf = gdf[gdf.intersects(bbox_geom)].copy()

    if gdf.empty:
        raise ValueError("No polygon features intersect the SPU extent.")

    return gdf


def build_shapes(
    gdf: gpd.GeoDataFrame,
    attribute_field: Optional[str],
    burn_value: Union[int, float],
):
    """
    Build (geometry, value) pairs for rasterization.
    """
    if attribute_field is None:
        return [(geom, burn_value) for geom in gdf.geometry]

    if attribute_field not in gdf.columns:
        raise ValueError(
            f"ATTRIBUTE_FIELD '{attribute_field}' not found.\n"
            f"Available fields: {list(gdf.columns)}"
        )

    shapes = []
    for geom, val in zip(gdf.geometry, gdf[attribute_field]):
        if val is None:
            continue
        if isinstance(val, float) and np.isnan(val):
            continue
        try:
            numeric_val = float(val)
        except Exception as exc:
            raise ValueError(
                f"ATTRIBUTE_FIELD '{attribute_field}' must be numeric. "
                f"Bad value: {val!r}"
            ) from exc
        shapes.append((geom, numeric_val))

    if not shapes:
        raise ValueError(
            f"No valid numeric values found in ATTRIBUTE_FIELD '{attribute_field}'."
        )

    return shapes


# ============================================================
# MAIN FUNCTION
# ============================================================

def polygon_zip_to_raster_on_spu(
    zip_path: str,
    spu_path: str,
    out_path: str,
    zip_layer: Optional[str] = None,
    source_crs_if_missing: Optional[str] = None,
    target_crs: CRS = CRS.from_epsg(3035),
    resolution: float = 10.0,
    attribute_field: Optional[str] = None,
    burn_value: Union[int, float] = 1,
    all_touched: bool = False,
    output_dtype: DTypeLike = "uint8",
    output_nodata: Union[int, float] = 0,
) -> str:
    """
    Rasterize polygon ZIP shapefile to a raster over SPU extent.
    """
    ensure_parent_dir(out_path)
    out_dtype = normalize_dtype(output_dtype)

    with rasterio.open(spu_path) as spu_ds:
        if spu_ds.crs is None:
            raise ValueError(f"SPU raster has no CRS: {spu_path}")

        if not crs_equal(spu_ds.crs, target_crs):
            raise ValueError(
                f"SPU raster CRS is {spu_ds.crs}, but expected {target_crs}."
            )

        left, bottom, right, top = spu_ds.bounds

    if resolution <= 0:
        raise ValueError("Resolution must be greater than 0.")

    # Use SPU top-left origin and requested cell size.
    # Width/height use ceil so the raster fully covers the SPU extent.
    width = int(math.ceil((right - left) / resolution))
    height = int(math.ceil((top - bottom) / resolution))
    transform = from_origin(left, top, resolution, resolution)

    gdf = load_polygon_zip(zip_path, layer=zip_layer)

    gdf = prepare_gdf(
        gdf=gdf,
        target_crs=target_crs,
        source_crs_if_missing=source_crs_if_missing,
        clip_bounds=(left, bottom, right, top),
    )

    shapes = build_shapes(
        gdf=gdf,
        attribute_field=attribute_field,
        burn_value=burn_value,
    )

    arr = rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=output_nodata,
        all_touched=all_touched,
        dtype=out_dtype,
    )

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": out_dtype,
        "crs": target_crs,
        "transform": transform,
        "nodata": output_nodata,
        "compress": "lzw",
    }

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr, 1)

    return out_path


# ============================================================
# RUN
# ============================================================

def main():
    print("SPU template:", SPU_PATH)
    print("Input ZIP polygon:", INPUT_ZIP)
    print("Output raster:", OUTPUT_RASTER)
    print("Target CRS:", TARGET_CRS)
    print("Resolution:", OUTPUT_RESOLUTION)
    print("NoData:", OUTPUT_NODATA)

    out = polygon_zip_to_raster_on_spu(
        zip_path=INPUT_ZIP,
        spu_path=SPU_PATH,
        out_path=OUTPUT_RASTER,
        zip_layer=ZIP_LAYER,
        source_crs_if_missing=SOURCE_CRS_IF_MISSING,
        target_crs=TARGET_CRS,
        resolution=OUTPUT_RESOLUTION,
        attribute_field=ATTRIBUTE_FIELD,
        burn_value=BURN_VALUE,
        all_touched=ALL_TOUCHED,
        output_dtype=OUTPUT_DTYPE,
        output_nodata=OUTPUT_NODATA,
    )

    print(f"Saved: {out}")
    print("Done.")


if __name__ == "__main__":
    main()