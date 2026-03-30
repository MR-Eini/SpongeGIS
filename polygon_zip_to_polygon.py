# -*- coding: utf-8 -*-
"""
Convert polygon ZIP shapefile -> polygon shapefile,
matched to the SPU area.

What this script does
---------------------
1) Reads a polygon shapefile from a ZIP file.
2) Checks whether the vector has a CRS.
3) If CRS is missing, it uses SOURCE_CRS_IF_MISSING.
4) Cleans geometries BEFORE reprojection.
5) Reprojects to EPSG:3035 if needed.
6) Builds an SPU footprint polygon from the SPU raster:
   - can use valid cells only
   - can optionally require cell value > 0
7) Clips the input polygons to the SPU footprint.
8) Saves the result as a shapefile in EPSG:3035.

Notes
-----
- A shapefile does not have a raster-like "extent" in the same sense.
- So the practical equivalent is:
  - same CRS as SPU
  - clipped to the SPU footprint / inside SPU area
- This script does that.

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

import os
import tempfile
import zipfile
from typing import Optional

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.features import shapes
from shapely.geometry import box, shape
from shapely.ops import unary_union


# ============================================================
# USER-DEFINED SETTINGS (EDIT ONLY THIS BLOCK)
# ============================================================

# SPU raster template
SPU_PATH = r"X:\download\Mamad\!SpongeGIStest\Kamienna\SPU\SPU_unique_ids_3035.tif"

# Input polygon ZIP and output shapefile
INPUT_ZIP = r"X:\download\Mamad\!SpongeGIStest\Kamienna\input\shp\Arable.zip"
OUTPUT_SHP = r"X:\download\Mamad\!SpongeGIStest\Kamienna\output\Arable_on_SPU_3035.shp"

# Optional: shapefile name inside ZIP without ".shp"
# Keep None to use the first .shp found
ZIP_LAYER = None

# Optional CRS if input vector has no CRS
# Example: "EPSG:2180"
SOURCE_CRS_IF_MISSING = None

# Required output CRS
TARGET_CRS = CRS.from_epsg(3035)

# How to define SPU area from raster
# True  -> only cells that are not nodata and > 0 are treated as SPU area
# False -> all cells that are not nodata are treated as SPU area
SPU_REQUIRE_VALUE_GT_ZERO = True

# Clip behavior
# True  -> clip to actual SPU footprint polygon
# False -> clip only to SPU bounding box
CLIP_TO_SPU_FOOTPRINT = True

# If True, explode multipart output into single parts
EXPLODE_MULTIPART = False

# Output driver
OUTPUT_DRIVER = "ESRI Shapefile"


# ============================================================
# INTERNAL HELPERS
# ============================================================

def ensure_parent_dir(path: str) -> None:
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)


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


def clean_polygons_before_reprojection(
    gdf: gpd.GeoDataFrame,
    source_crs_if_missing: Optional[str],
) -> gpd.GeoDataFrame:
    """
    Ensure CRS exists, keep only polygonal geometries,
    and repair invalid geometries before reprojection.
    """
    if gdf.empty:
        raise ValueError("Input vector is empty.")

    if gdf.crs is None:
        if source_crs_if_missing is None:
            raise ValueError(
                "Input vector has no CRS.\n"
                "Set SOURCE_CRS_IF_MISSING, for example: 'EPSG:2180'"
            )
        gdf = gdf.set_crs(source_crs_if_missing)
        print(f"[INFO] Input CRS missing. Using user-defined CRS: {gdf.crs}")

    gdf = gdf[gdf.geometry.notnull()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()

    if gdf.empty:
        raise ValueError("No valid geometries remain after removing empty/null geometry.")

    poly_mask = gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    gdf = gdf[poly_mask].copy()

    if gdf.empty:
        raise ValueError("No Polygon or MultiPolygon features found in input data.")

    invalid_mask = ~gdf.is_valid
    if invalid_mask.any():
        print(
            f"[INFO] Fixing {int(invalid_mask.sum())} invalid geometries "
            f"before reprojection with buffer(0)"
        )
        gdf.loc[invalid_mask, "geometry"] = gdf.loc[invalid_mask, "geometry"].buffer(0)

    gdf = gdf[gdf.geometry.notnull()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()
    gdf = gdf[gdf.is_valid].copy()

    if gdf.empty:
        raise ValueError("No valid polygon geometries remain after pre-reprojection cleaning.")

    return gdf


def clean_polygons_after_reprojection(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Light second cleaning pass after reprojection.
    """
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

    return gdf


def build_spu_footprint(
    spu_path: str,
    require_value_gt_zero: bool = True,
) -> tuple:
    """
    Build SPU footprint polygon from raster cells.

    Returns
    -------
    footprint_geom : shapely geometry
        Polygon/MultiPolygon of the SPU area.
    left, bottom, right, top : float
        SPU bounds.
    spu_crs : CRS
        CRS of the SPU raster.
    """
    with rasterio.open(spu_path) as ds:
        if ds.crs is None:
            raise ValueError(f"SPU raster has no CRS: {spu_path}")

        arr = ds.read(1)
        nodata = ds.nodata
        transform = ds.transform
        left, bottom, right, top = ds.bounds
        spu_crs = ds.crs

        if nodata is None:
            valid_mask = np.ones(arr.shape, dtype=bool)
        else:
            valid_mask = arr != nodata

        if require_value_gt_zero:
            valid_mask = valid_mask & (arr > 0)

        if not np.any(valid_mask):
            raise ValueError("No valid SPU cells found to build footprint.")

        geom_list = []
        for geom, val in shapes(
            valid_mask.astype(np.uint8),
            mask=valid_mask,
            transform=transform
        ):
            if val == 1:
                geom_list.append(shape(geom))

    if not geom_list:
        raise ValueError("Could not build SPU footprint polygon.")

    footprint = unary_union(geom_list)

    if footprint.is_empty:
        raise ValueError("SPU footprint geometry is empty.")

    return footprint, left, bottom, right, top, spu_crs


def clip_to_spu(
    gdf: gpd.GeoDataFrame,
    spu_geom,
    left: float,
    bottom: float,
    right: float,
    top: float,
    clip_to_footprint: bool = True,
) -> gpd.GeoDataFrame:
    """
    Filter and clip polygons to SPU bbox or SPU footprint.
    """
    bbox_geom = box(left, bottom, right, top)

    gdf = gdf[gdf.intersects(bbox_geom)].copy()
    if gdf.empty:
        raise ValueError("No polygon features intersect the SPU extent.")

    clip_geom = spu_geom if clip_to_footprint else bbox_geom

    gdf = gpd.clip(gdf, clip_geom, keep_geom_type=False)

    gdf = gdf[gdf.geometry.notnull()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()

    if gdf.empty:
        raise ValueError("No polygon features remain after clipping to SPU.")

    gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()

    if gdf.empty:
        raise ValueError("No Polygon or MultiPolygon features remain after clipping.")

    return gdf


def save_vector(gdf: gpd.GeoDataFrame, out_path: str, driver: str) -> None:
    """
    Save output vector.
    """
    ensure_parent_dir(out_path)

    if driver == "ESRI Shapefile":
        base, ext = os.path.splitext(out_path)
        if ext.lower() != ".shp":
            out_path = base + ".shp"

    gdf.to_file(out_path, driver=driver)


# ============================================================
# MAIN FUNCTION
# ============================================================

def polygon_zip_to_polygon_on_spu(
    zip_path: str,
    spu_path: str,
    out_path: str,
    zip_layer: Optional[str] = None,
    source_crs_if_missing: Optional[str] = None,
    target_crs: CRS = CRS.from_epsg(3035),
    require_value_gt_zero: bool = True,
    clip_to_footprint: bool = True,
    explode_multipart: bool = False,
    output_driver: str = "ESRI Shapefile",
) -> str:
    """
    Convert ZIP polygon -> polygon shapefile clipped to SPU area.
    """
    gdf = load_polygon_zip(zip_path, layer=zip_layer)

    gdf = clean_polygons_before_reprojection(
        gdf=gdf,
        source_crs_if_missing=source_crs_if_missing,
    )

    if not crs_equal(gdf.crs, target_crs):
        print(f"[INFO] Reprojecting vector from {gdf.crs} to {target_crs}")
        gdf = gdf.to_crs(target_crs)

    gdf = clean_polygons_after_reprojection(gdf)

    spu_geom, left, bottom, right, top, spu_crs = build_spu_footprint(
        spu_path=spu_path,
        require_value_gt_zero=require_value_gt_zero,
    )

    if not crs_equal(spu_crs, target_crs):
        raise ValueError(
            f"SPU raster CRS is {spu_crs}, but expected {target_crs}."
        )

    gdf = clip_to_spu(
        gdf=gdf,
        spu_geom=spu_geom,
        left=left,
        bottom=bottom,
        right=right,
        top=top,
        clip_to_footprint=clip_to_footprint,
    )

    if explode_multipart:
        gdf = gdf.explode(index_parts=False).reset_index(drop=True)

    gdf = gdf[gdf.geometry.notnull()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()

    if gdf.empty:
        raise ValueError("Nothing to save after final cleaning.")

    save_vector(gdf, out_path, output_driver)

    return out_path


# ============================================================
# RUN
# ============================================================

def main():
    print("SPU template:", SPU_PATH)
    print("Input ZIP polygon:", INPUT_ZIP)
    print("Output shapefile:", OUTPUT_SHP)
    print("Target CRS:", TARGET_CRS)
    print("Clip to SPU footprint:", CLIP_TO_SPU_FOOTPRINT)
    print("SPU valid area rule (>0):", SPU_REQUIRE_VALUE_GT_ZERO)

    out = polygon_zip_to_polygon_on_spu(
        zip_path=INPUT_ZIP,
        spu_path=SPU_PATH,
        out_path=OUTPUT_SHP,
        zip_layer=ZIP_LAYER,
        source_crs_if_missing=SOURCE_CRS_IF_MISSING,
        target_crs=TARGET_CRS,
        require_value_gt_zero=SPU_REQUIRE_VALUE_GT_ZERO,
        clip_to_footprint=CLIP_TO_SPU_FOOTPRINT,
        explode_multipart=EXPLODE_MULTIPART,
        output_driver=OUTPUT_DRIVER,
    )

    print(f"Saved: {out}")
    print("Done.")


if __name__ == "__main__":
    main()