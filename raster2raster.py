# -*- coding: utf-8 -*-
"""
Align one input raster to the SPU raster grid.

What this script does
---------------------
1) Reads one input raster.
2) Checks whether the raster has a CRS.
3) If the input CRS is missing, it uses SOURCE_CRS_IF_MISSING.
4) If the input CRS is not EPSG:3035, it reprojects it to EPSG:3035.
5) Matches the output exactly to the SPU raster:
   - CRS
   - extent
   - resolution
   - transform
   - width / height
6) Preserves NoData as NoData as far as the source metadata allows.

Required packages
-----------------
- rasterio
- numpy

Example conda install
---------------------
conda install -c conda-forge rasterio numpy
"""

import os
from typing import Optional, Union

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.warp import reproject


# ============================================================
# USER-DEFINED SETTINGS (EDIT ONLY THIS BLOCK)
# ============================================================

# Template raster: output will match this raster exactly
SPU_PATH = r"X:\download\Mamad\!SpongeGIStest\Kamienna\SPU\SPU_unique_ids_3035.tif"

# Input raster and final output raster
INPUT_RASTER = r"X:\download\Mamad\!SpongeGIStest\Kamienna\input\Arable.tif"
OUTPUT_RASTER = r"X:\download\Mamad\!SpongeGIStest\Kamienna\output\Arable_on_SPU_3035.tif"

# Optional CRS if the input raster has no CRS at all.
# Keep None if the input raster already has CRS.
# Example: "EPSG:2180"
SOURCE_CRS_IF_MISSING = None

# Resampling method:
# categorical rasters -> "nearest"
# continuous rasters  -> "bilinear" or "cubic"
RESAMPLING_METHOD = "nearest"

# Band number to read from the input raster
INPUT_BAND = 1

# Optional output dtype. Keep None to use input dtype.
# Examples: "float32", "uint16", "int32"
OUTPUT_DTYPE = None

# Optional output NoData. Keep None to let the script choose:
# - source nodata if available
# - otherwise a safe default
OUTPUT_NODATA = None

# Target CRS for all outputs
TARGET_CRS = CRS.from_epsg(3035)


# ============================================================
# INTERNAL HELPERS
# ============================================================

ResamplingLike = Union[str, Resampling]


def get_resampling(method: ResamplingLike) -> Resampling:
    """Convert a string or Resampling enum to a rasterio Resampling enum."""
    if isinstance(method, Resampling):
        return method

    method = str(method).strip().lower()
    mapping = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
        "cubic_spline": Resampling.cubic_spline,
        "lanczos": Resampling.lanczos,
        "average": Resampling.average,
        "mode": Resampling.mode,
        "max": Resampling.max,
        "min": Resampling.min,
        "med": Resampling.med,
        "q1": Resampling.q1,
        "q3": Resampling.q3,
    }

    if method not in mapping:
        raise ValueError(
            f"Unsupported resampling method: {method!r}. "
            f"Supported: {', '.join(mapping.keys())}"
        )

    return mapping[method]


def normalize_dtype(dtype_value: Optional[Union[str, np.dtype]]) -> Optional[str]:
    """Normalize dtype to a rasterio-friendly string, or return None."""
    if dtype_value is None:
        return None
    return np.dtype(dtype_value).name


def choose_default_nodata(dtype_name: str):
    """
    Choose a default nodata value if source/output nodata is not provided.

    Notes:
    - If the source raster has a real nodata value, that is preferred.
    - If the source raster has no nodata metadata, this is only a fallback.
    """
    dt = np.dtype(dtype_name)

    if np.issubdtype(dt, np.floating):
        return -9999.0

    if np.issubdtype(dt, np.unsignedinteger):
        return 0

    if np.issubdtype(dt, np.signedinteger):
        info = np.iinfo(dt)
        return info.min

    raise ValueError(f"Unsupported dtype for nodata selection: {dtype_name}")


def ensure_parent_dir(path: str) -> None:
    """Create parent folder if it does not exist."""
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)


def crs_equal(a: CRS, b: CRS) -> bool:
    """Safe CRS comparison."""
    try:
        return CRS.from_user_input(a) == CRS.from_user_input(b)
    except Exception:
        return False


# ============================================================
# MAIN FUNCTION
# ============================================================

def align_raster_to_spu(
    src_path: str,
    spu_path: str,
    out_path: str,
    band: int = 1,
    source_crs_if_missing: Optional[str] = None,
    resampling: ResamplingLike = "nearest",
    dst_dtype: Optional[Union[str, np.dtype]] = None,
    dst_nodata=None,
    target_crs: CRS = TARGET_CRS,
) -> str:
    """
    Align one raster to the SPU raster grid.

    Parameters
    ----------
    src_path : str
        Input raster path.
    spu_path : str
        SPU raster path. Output will match this raster exactly.
    out_path : str
        Output raster path.
    band : int
        Input raster band to read.
    source_crs_if_missing : Optional[str]
        CRS to assign if input raster has no CRS metadata at all.
        Example: "EPSG:2180"
    resampling : str or rasterio.enums.Resampling
        Resampling method.
    dst_dtype : Optional[str]
        Output dtype. If None, input dtype is used.
    dst_nodata :
        Output nodata value. If None, source nodata is used if available;
        otherwise a default nodata is chosen based on dtype.
    target_crs : CRS
        Required target CRS. Default is EPSG:3035.

    Returns
    -------
    str
        Output raster path.
    """
    resampling_enum = get_resampling(resampling)
    ensure_parent_dir(out_path)

    with rasterio.open(spu_path) as spu_ds:
        if spu_ds.crs is None:
            raise ValueError(
                f"SPU raster has no CRS: {spu_path}"
            )

        if not crs_equal(spu_ds.crs, target_crs):
            raise ValueError(
                f"SPU raster CRS is {spu_ds.crs}, but expected {target_crs}. "
                f"The SPU template should already be EPSG:3035."
            )

        dst_transform = spu_ds.transform
        dst_width = spu_ds.width
        dst_height = spu_ds.height
        dst_crs = target_crs

    with rasterio.open(src_path) as src_ds:
        if band < 1 or band > src_ds.count:
            raise ValueError(
                f"Band {band} is out of range. "
                f"Input raster has {src_ds.count} band(s)."
            )

        src_crs = src_ds.crs
        if src_crs is None:
            if source_crs_if_missing is None:
                raise ValueError(
                    f"Input raster has no CRS: {src_path}\n"
                    f"Set SOURCE_CRS_IF_MISSING, for example: 'EPSG:2180'"
                )
            src_crs = CRS.from_user_input(source_crs_if_missing)
            print(f"[INFO] Input CRS missing. Using user-defined CRS: {src_crs}")

        src_array = src_ds.read(band)
        src_transform = src_ds.transform
        src_dtype = src_ds.dtypes[band - 1]
        src_nodata = src_ds.nodata

        out_dtype = normalize_dtype(dst_dtype) or normalize_dtype(src_dtype)

        if dst_nodata is None:
            if src_nodata is not None:
                out_nodata = src_nodata
            else:
                out_nodata = choose_default_nodata(out_dtype)
        else:
            out_nodata = dst_nodata

        destination = np.full(
            (dst_height, dst_width),
            fill_value=out_nodata,
            dtype=np.dtype(out_dtype),
        )

        reproject(
            source=src_array,
            destination=destination,
            src_transform=src_transform,
            src_crs=src_crs,
            src_nodata=src_nodata,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_nodata=out_nodata,
            resampling=resampling_enum,
        )

        profile = {
            "driver": "GTiff",
            "height": dst_height,
            "width": dst_width,
            "count": 1,
            "dtype": out_dtype,
            "crs": dst_crs,
            "transform": dst_transform,
            "nodata": out_nodata,
            "compress": "lzw",
        }

        with rasterio.open(out_path, "w", **profile) as dst_ds:
            dst_ds.write(destination, 1)

    return out_path


# ============================================================
# RUN
# ============================================================

def main():
    print("SPU template:", SPU_PATH)
    print("Input raster:", INPUT_RASTER)
    print("Output raster:", OUTPUT_RASTER)
    print("Target CRS:", TARGET_CRS)
    print("Resampling:", RESAMPLING_METHOD)

    out = align_raster_to_spu(
        src_path=INPUT_RASTER,
        spu_path=SPU_PATH,
        out_path=OUTPUT_RASTER,
        band=INPUT_BAND,
        source_crs_if_missing=SOURCE_CRS_IF_MISSING,
        resampling=RESAMPLING_METHOD,
        dst_dtype=OUTPUT_DTYPE,
        dst_nodata=OUTPUT_NODATA,
        target_crs=TARGET_CRS,
    )

    print(f"Saved: {out}")
    print("Done.")


if __name__ == "__main__":
    main()