#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SpongeGIS (raster-based indicators by SPU)

This version updates SpongeGIS3.py with fixes that directly address the common
vector↔raster mismatches observed in the Kamienna test run:

Key fixes / changes
- Stream-derived metrics (DrainageD, RiverSlope, MeanderRatio) are now set to NaN
  for SPUs with no stream cells (previously they were 0 due to bincount fill). This
  removes artificial 0 minima and inflated variance.
- FlowMinMaxRatio is computed as mean(MHQ/MLQ) on the raster (ratio-then-zonal-mean),
  matching the original raster-ratio definition. Configurable via --flow_ratio_mode.
- DEM hydrology arrays (fdir/acc/streams) are computed in-memory and do NOT require
  --write_intermediates (previously the code tried to read files that were only
  written when --write_intermediates was enabled).
- Input file resolution supports aliases and case-insensitive matching, so you can
  keep exact filenames (e.g., floodExtent.tif vs FloodExtent.tif) without edits.
- Optional NoData handling when rasters do not advertise nodata metadata:
  --assume_nodata_value and --treat_zero_as_nodata_for.
- RainFallErodibility and SoilErodibility use zonal MEDIAN aggregation to match original R indicators.
- If a stream mask raster is provided (e.g., streamlink.tif), it is preferred over DEM-derived streams for stream metrics.

Notes
- Landcover ratios treat presence as (value > 0). NoData is treated as "absence"
  for ratio computations (i.e., it contributes 0 to the numerator and counts in the
  denominator via the SPU zone raster).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Sequence, Union

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
from scipy.ndimage import distance_transform_edt
from pyproj import Transformer
from pysheds.grid import Grid


# ---------------------------
# Indicators (fixed output schema)
# ---------------------------

ALL_FACTORS = [
    # Drought
    "cwb", "swr", "grr", "sri", "FlowMinMaxRatio",
    # Flood
    "RiverSlope", "LandSlope", "MeanderRatio", "FloodRiskAreaRatio", "NonForestedRatio",
    # Flood/Drought
    "DrainageD", "twi", "ForestRatio", "LakeRatio", "WetlandRatio",
    "OrchVegRatio", "UrbanRatio", "ArableRatio", "ReclaimedRatio",
    # Sediment
    "GraniteRatio", "RainFallErodibility", "SoilErodibility",
]


# ---------------------------
# Input files (aliases supported)
# ---------------------------

# Each key maps to a list of candidate filenames. First match wins.
INPUT_CANDIDATES: Dict[str, List[str]] = {
    # binary landcover masks (>0 = present)
    "ForestRatio": ["Forest.tif", "forest.tif", "forest_dsm.tif", "Forest_dsm.tif"],
    "LakeRatio": ["Lake.tif", "lake.tif", "lake_dsm.tif", "Lake_dsm.tif"],
    "WetlandRatio": ["Wetland.tif", "wetland.tif"],
    "OrchVegRatio": ["Orchard.tif", "orchard.tif"],
    "UrbanRatio": ["Urban.tif", "urban.tif"],
    "ArableRatio": ["Arable.tif", "arable.tif"],
    "FloodRiskAreaRatio": ["FloodExtent.tif", "floodExtent.tif", "floodextent.tif"],

    # optional stream mask to match legacy river network (preferred over DEM-derived streams if present)
    "streams": ["streams.tif", "Streams.tif", "stream.tif", "Stream.tif", "streamlink.tif", "subcatch_streamlink.tif"],

    # keep GraniteRatio in schema
    "GraniteRatio": ["Granite.tif", "granite.tif"],

    # reclaimed
    "ditches": ["Ditches.tif", "ditches.tif"],
    "meadPastur": ["MeadPastur.tif", "meadPastur.tif", "meadpastur.tif"],

    # non-forest
    "nonForest": ["NonForest.tif", "nonForest.tif", "nonforest.tif"],

    # optional hydrology/climate (continuous)
    "cwb": ["cwb.tif", "CWB.tif"],
    "swr": ["swr.tif", "SWR.tif"],
    "grr": ["grr.tif", "GRR.tif"],
    "pAvgAnn": ["pAvgAnn.tif", "pavgann.tif", "pavgAnn.tif"],
    "pAvgVeg": ["pAvgVeg.tif", "pavgveg.tif", "pavgVeg.tif"],
    "swMMQ": ["swMMQ.tif", "swmmq.tif"],
    "swMLQ": ["swMLQ.tif", "swmlq.tif"],
    "swMHQ": ["swMHQ.tif", "swmhq.tif"],

    # soil fractions
    "sand": ["SoilSand_Fraction.tif", "soilsand_fraction.tif", "SoilSandFraction.tif"],
    "silt": ["SoilSilt_Fraction.tif", "soilsilt_fraction.tif", "SoilSiltFraction.tif"],
    "clay": ["SoilClay_Fraction.tif", "soilclay_fraction.tif", "SoilClayFraction.tif"],
    "orgC": ["SoilOrgC_Fraction.tif", "soilorgc_fraction.tif", "SoilOrgCFraction.tif"],
}


# ---------------------------
# Logging
# ---------------------------

def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)

def tic() -> float:
    return time.time()

def toc(t0: float, label: str) -> None:
    log(f"{label} | elapsed={time.time() - t0:.1f}s")


# ---------------------------
# File resolution (aliases + case-insensitive)
# ---------------------------

def _case_insensitive_lookup(folder: Path, name: str) -> Optional[Path]:
    """Find a file in `folder` whose name matches `name` case-insensitively."""
    target = name.lower()
    for p in folder.iterdir():
        if p.is_file() and p.name.lower() == target:
            return p
    return None

def resolve_input(input_dir: Path, key: str) -> Optional[str]:
    """Resolve an input raster path for a given logical key, using aliases and case-insensitive match."""
    if key not in INPUT_CANDIDATES:
        return None
    for fname in INPUT_CANDIDATES[key]:
        p = input_dir / fname
        if p.exists():
            return str(p)
        q = _case_insensitive_lookup(input_dir, fname)
        if q is not None:
            return str(q)
    return None


# ---------------------------
# Raster I/O + NoData handling
# ---------------------------

def read_raster(path: str) -> Tuple[np.ndarray, dict]:
    with rasterio.open(path) as ds:
        arr = ds.read(1)
        prof = ds.profile.copy()
        prof["crs"] = ds.crs
        prof["transform"] = ds.transform
        prof["width"] = ds.width
        prof["height"] = ds.height
        prof["nodata"] = ds.nodata
    return arr, prof

def write_raster(path: str, arr: np.ndarray, prof: dict, nodata=None, dtype=None) -> None:
    prof2 = prof.copy()
    if dtype is not None:
        prof2["dtype"] = dtype
    if nodata is not None:
        prof2["nodata"] = nodata
    prof2["count"] = 1
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **prof2) as ds:
        ds.write(arr, 1)

def cellsize_from_profile(prof: dict) -> Tuple[float, float]:
    a = prof["transform"].a
    e = prof["transform"].e
    return float(abs(a)), float(abs(e))

def _apply_assumed_nodata(arr: np.ndarray, nodata_assumed: Optional[float], treat_zero_as_nodata: bool) -> np.ndarray:
    """Convert assumed nodata values to NaN (float output)."""
    out = arr.astype(np.float32, copy=False)
    if nodata_assumed is not None:
        out = np.where(out == float(nodata_assumed), np.nan, out)
    if treat_zero_as_nodata:
        out = np.where(out == 0.0, np.nan, out)
    return out

def reproject_to_match(
    src_path: str,
    match_prof: dict,
    resampling: Resampling,
    dst_dtype=np.float32,
    dst_nodata=np.nan,
    nodata_assumed: Optional[float] = None,
    treat_zero_as_nodata: bool = False,
) -> np.ndarray:
    dst = np.empty((match_prof["height"], match_prof["width"]), dtype=dst_dtype)
    dst[:] = dst_nodata
    with rasterio.open(src_path) as src:
        # If nodata metadata is missing, optionally apply assumed nodata later (post-reproject)
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=match_prof["transform"],
            dst_crs=match_prof["crs"],
            dst_nodata=dst_nodata,
            resampling=resampling,
        )
    if dst_dtype in (np.float32, np.float64):
        dst = _apply_assumed_nodata(dst, nodata_assumed, treat_zero_as_nodata)
    return dst

def reproject_zones_to_match(src_path: str, match_prof: dict, zone_nodata_out: int = 0) -> np.ndarray:
    dst = np.full((match_prof["height"], match_prof["width"]), zone_nodata_out, dtype=np.int32)
    with rasterio.open(src_path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=match_prof["transform"],
            dst_crs=match_prof["crs"],
            dst_nodata=zone_nodata_out,
            resampling=Resampling.nearest,
        )
    z = dst.astype(np.int64)
    z[z <= 0] = 0
    return z


# ---------------------------
# Zonal stats (compressed IDs)
# ---------------------------

def sanitize_zones(zones: np.ndarray) -> np.ndarray:
    z = zones.astype(np.int64, copy=False)
    z = np.where(z > 0, z, 0)
    return z

def zonal_mean(zones: np.ndarray, values: np.ndarray) -> pd.Series:
    z = sanitize_zones(zones)
    v = values.astype(np.float64, copy=False)
    valid = np.isfinite(v) & (z > 0)
    if not np.any(valid):
        return pd.Series(dtype=float)
    zid = z[valid]
    vv = v[valid]
    uniq, inv = np.unique(zid, return_inverse=True)
    sums = np.bincount(inv, weights=vv).astype(np.float64)
    cnts = np.bincount(inv).astype(np.float64)
    out = sums / np.maximum(cnts, 1.0)
    return pd.Series(out, index=uniq)


def zonal_median(zones: np.ndarray, values: np.ndarray) -> pd.Series:
    """Zonal median (NaN-aware).

    Returns a Series indexed by ALL zone IDs present in `zones` (z > 0). Zones with
    no finite values get NaN.

    Implementation detail: sorts by (zone_id, value) and picks the middle element(s).
    """
    z = sanitize_zones(zones)
    v = values.astype(np.float64, copy=False)

    all_ids = np.unique(z[z > 0])
    if all_ids.size == 0:
        return pd.Series(dtype=float)

    valid = np.isfinite(v) & (z > 0)
    if not np.any(valid):
        return pd.Series(np.full(all_ids.size, np.nan, dtype=np.float64), index=all_ids)

    zid = z[valid].astype(np.int64, copy=False)
    vv = v[valid].astype(np.float64, copy=False)

    # Sort by zone then by value
    order = np.lexsort((vv, zid))
    zid_s = zid[order]
    vv_s = vv[order]

    # Group boundaries
    starts = np.flatnonzero(np.r_[True, zid_s[1:] != zid_s[:-1]])
    ends = np.r_[starts[1:], zid_s.size]
    n = ends - starts

    mid1 = starts + (n - 1) // 2
    mid2 = starts + n // 2
    med = (vv_s[mid1] + vv_s[mid2]) / 2.0

    uniq = zid_s[starts]
    out = np.full(all_ids.size, np.nan, dtype=np.float64)
    pos = np.searchsorted(all_ids, uniq)
    out[pos] = med
    return pd.Series(out, index=all_ids)


def zonal_percent_area(zones: np.ndarray, hit01: np.ndarray) -> pd.Series:
    """Percent of zone cells where hit01 > 0. Denominator is all zone cells (z > 0)."""
    z = sanitize_zones(zones)
    hit = (hit01 > 0) & (z > 0)
    z_all = z[z > 0]
    if z_all.size == 0:
        return pd.Series(dtype=float)

    uniq, inv_all = np.unique(z_all, return_inverse=True)
    total = np.bincount(inv_all).astype(np.float64)

    hit_ids = z[hit]
    if hit_ids.size > 0:
        pos = np.searchsorted(uniq, hit_ids)
        hit_counts = np.bincount(pos, minlength=uniq.size).astype(np.float64)
    else:
        hit_counts = np.zeros_like(total)

    pct = 100.0 * hit_counts / np.maximum(total, 1.0)
    return pd.Series(pct, index=uniq)

def merge_series(series_dict: Dict[str, pd.Series]) -> pd.DataFrame:
    df = None
    for k, s in series_dict.items():
        s = s.rename(k)
        df = s.to_frame() if df is None else df.join(s, how="outer")
    df.index.name = "ID"
    return df.reset_index()


# ---------------------------
# DEM hydrology (streams derived from DEM) - in memory
# ---------------------------

def derive_hydro_from_dem(
    dem_path: str,
    stream_threshold_km2: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid = Grid.from_raster(dem_path, data_name="dem")
    dem = grid.read_raster(dem_path)

    dem_filled = grid.fill_depressions(dem)
    dem_filled = grid.resolve_flats(dem_filled)

    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
    fdir = grid.flowdir(dem_filled, dirmap=dirmap)
    acc = grid.accumulation(fdir, dirmap=dirmap)

    with rasterio.open(dem_path) as ds:
        xres = abs(ds.transform.a)
        yres = abs(ds.transform.e)

    cell_area_m2 = xres * yres
    thr_cells = (stream_threshold_km2 * 1_000_000.0) / cell_area_m2
    streams = (acc >= thr_cells).astype(np.uint8)

    return fdir.astype(np.int32), acc.astype(np.float32), streams.astype(np.uint8)

def slope_from_dem(dem: np.ndarray, prof: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Returns slope in radians and degrees. DEM must already contain NaN for nodata."""
    xres, yres = cellsize_from_profile(prof)
    demf = dem.astype(np.float64, copy=False)
    dzdy, dzdx = np.gradient(demf, yres, xres)
    slope_rad = np.arctan(np.sqrt(dzdx**2 + dzdy**2))
    slope_deg = np.degrees(slope_rad)
    return slope_rad.astype(np.float32), slope_deg.astype(np.float32)

def safe_twi(acc_cells: np.ndarray, slope_rad: np.ndarray, cell_area: float) -> np.ndarray:
    eps = 1e-6
    sca = (acc_cells.astype(np.float64) + 1.0) * float(cell_area)
    den = np.tan(slope_rad.astype(np.float64)) + eps
    num = sca + eps
    twi = np.full(slope_rad.shape, np.nan, dtype=np.float32)
    ok = np.isfinite(num) & np.isfinite(den) & (num > 0) & (den > 0)
    twi[ok] = np.log(num[ok] / den[ok]).astype(np.float32)
    return twi

def d8_map() -> Dict[int, Tuple[int, int, float]]:
    return {
        64:  (-1, 0, 1.0),
        128: (-1, 1, np.sqrt(2)),
        1:   (0,  1, 1.0),
        2:   (1,  1, np.sqrt(2)),
        4:   (1,  0, 1.0),
        8:   (1, -1, np.sqrt(2)),
        16:  (0, -1, 1.0),
        32:  (-1,-1, np.sqrt(2)),
    }

def stream_metrics_by_spu(
    zones: np.ndarray,
    dem: np.ndarray,
    fdir: np.ndarray,
    acc: np.ndarray,
    streams01: np.ndarray,
    prof: dict,
    min_stream_cells_per_spu: int = 1,
    min_meander_mainlen_m: float = 0.0,
) -> Dict[str, pd.Series]:
    """
    DrainageD [km/km^2], RiverSlope [%], MeanderRatio [% straight / along-stream].
    SPUs without at least `min_stream_cells_per_spu` stream cells are returned as NaN.
    """
    z = sanitize_zones(zones)
    H, W = dem.shape
    xres, yres = cellsize_from_profile(prof)
    cell = float((xres + yres) / 2.0)
    cell_area_m2 = xres * yres
    dm = d8_map()

    stream = (streams01 > 0) & (z > 0) & np.isfinite(dem)
    if not np.any(stream):
        return {k: pd.Series(dtype=float) for k in ["DrainageD", "RiverSlope", "MeanderRatio"]}

    # zone list + areas for all zones
    z_all = z[z > 0]
    uniq_z, inv_all = np.unique(z_all, return_inverse=True)
    tot_cells = np.bincount(inv_all).astype(np.float64)
    area_km2 = (tot_cells * cell_area_m2) / 1_000_000.0

    # stream pixels
    r, c = np.where(stream)
    codes = fdir[r, c].astype(np.int32)

    dr = np.zeros_like(codes, dtype=np.int32)
    dc = np.zeros_like(codes, dtype=np.int32)
    step = np.zeros_like(codes, dtype=np.float64)

    for code, (drr, dcc, mul) in dm.items():
        m = codes == code
        dr[m] = drr
        dc[m] = dcc
        step[m] = mul * cell

    r2 = r + dr
    c2 = c + dc
    inside = (r2 >= 0) & (r2 < H) & (c2 >= 0) & (c2 < W) & (step > 0)
    r = r[inside]; c = c[inside]; r2 = r2[inside]; c2 = c2[inside]; step = step[inside]

    zid = z[r, c]
    pos = np.searchsorted(uniq_z, zid)

    # stream cell counts per SPU (used to set NaN for SPUs with no streams)
    stream_counts = np.bincount(pos, minlength=uniq_z.size).astype(np.int64)
    has_stream = stream_counts >= int(min_stream_cells_per_spu)

    # Drainage density
    length_m = step
    sum_len = np.bincount(pos, weights=length_m, minlength=uniq_z.size).astype(np.float64)
    DrainageD = (sum_len / 1000.0) / np.maximum(area_km2, 1e-12)
    DrainageD[~has_stream] = np.nan

    # River slope (%)
    z_up = dem[r, c].astype(np.float64)
    z_dn = dem[r2, c2].astype(np.float64)
    ok_dn = np.isfinite(z_dn)
    slope = np.full_like(z_up, np.nan, dtype=np.float64)
    slope[ok_dn] = (z_up[ok_dn] - z_dn[ok_dn]) / np.maximum(length_m[ok_dn], 1e-9)
    slope = np.where(np.isfinite(slope), np.maximum(slope, 0.001), np.nan)

    sum_w = np.bincount(pos, weights=length_m, minlength=uniq_z.size).astype(np.float64)
    sum_sw = np.bincount(pos, weights=np.where(np.isfinite(slope), slope * length_m, 0.0), minlength=uniq_z.size).astype(np.float64)
    RiverSlope = 100.0 * (sum_sw / np.maximum(sum_w, 1e-12))
    RiverSlope[~has_stream] = np.nan

    # Meander ratio (main path inside zone)
    is_stream = streams01 > 0
    dist = np.zeros((H, W), dtype=np.float64)

    sr, sc = np.where(is_stream & (z > 0) & np.isfinite(dem))
    if sr.size == 0:
        MeanderRatio = np.full(uniq_z.size, np.nan)
    else:
        sacc = acc[sr, sc].astype(np.float64)
        order = np.argsort(-sacc)
        sr = sr[order]; sc = sc[order]

        def downstream(rr: int, cc: int):
            code = int(fdir[rr, cc])
            if code not in dm:
                return None
            drr, dcc, mul = dm[code]
            rr2 = rr + drr
            cc2 = cc + dcc
            if rr2 < 0 or rr2 >= H or cc2 < 0 or cc2 >= W:
                return None
            return rr2, cc2, mul * cell

        for rr, cc in zip(sr, sc):
            dn = downstream(rr, cc)
            if dn is None:
                continue
            rr2, cc2, lm = dn
            zid0 = int(z[rr, cc])
            if is_stream[rr2, cc2] and int(z[rr2, cc2]) == zid0:
                dist[rr, cc] = lm + dist[rr2, cc2]
            else:
                dist[rr, cc] = lm

        main_len = np.zeros(uniq_z.size, dtype=np.float64)
        main_r = np.full(uniq_z.size, -1, dtype=np.int32)
        main_c = np.full(uniq_z.size, -1, dtype=np.int32)

        rr_all, cc_all = np.where(dist > 0)
        for rr0, cc0 in zip(rr_all, cc_all):
            zid0 = int(z[rr0, cc0])
            i = int(np.searchsorted(uniq_z, zid0))
            d0 = dist[rr0, cc0]
            if d0 > main_len[i]:
                main_len[i] = d0
                main_r[i] = rr0
                main_c[i] = cc0

        tr = prof["transform"]
        def center_xy(rr: int, cc: int) -> Tuple[float, float]:
            x = tr.c + (cc + 0.5) * tr.a + (rr + 0.5) * tr.b
            y = tr.f + (cc + 0.5) * tr.d + (rr + 0.5) * tr.e
            return float(x), float(y)

        MeanderRatio = np.full(uniq_z.size, np.nan)
        for i in range(uniq_z.size):
            if not has_stream[i]:
                continue
            if main_len[i] <= 0 or main_r[i] < 0:
                continue
            if float(main_len[i]) < float(min_meander_mainlen_m):
                continue

            zid0 = int(uniq_z[i])
            rr0, cc0 = int(main_r[i]), int(main_c[i])
            rr1, cc1 = rr0, cc0
            steps_guard = 0
            while steps_guard < 2_000_000:
                dn = downstream(rr1, cc1)
                if dn is None:
                    break
                rr2, cc2, _ = dn
                if not (0 <= rr2 < H and 0 <= cc2 < W):
                    break
                if int(z[rr2, cc2]) != zid0 or not is_stream[rr2, cc2]:
                    break
                rr1, cc1 = rr2, cc2
                steps_guard += 1

            x0, y0 = center_xy(rr0, cc0)
            x1, y1 = center_xy(rr1, cc1)
            straight = np.hypot(x1 - x0, y1 - y0)

            # Avoid degenerate cases where straight distance is ~0 for very short segments
            if straight <= 0.0:
                continue

            MeanderRatio[i] = 100.0 * straight / np.maximum(main_len[i], 1e-12)

    # also NaN out meander where no streams
    MeanderRatio[~has_stream] = np.nan

    return {
        "DrainageD": pd.Series(DrainageD, index=uniq_z),
        "RiverSlope": pd.Series(RiverSlope, index=uniq_z),
        "MeanderRatio": pd.Series(MeanderRatio, index=uniq_z),
    }


# ---------------------------
# Rainfall / soil erodibility
# ---------------------------

def to_fraction_if_percent(a: np.ndarray) -> np.ndarray:
    v = a[np.isfinite(a)]
    if v.size == 0:
        return a
    return (a / 100.0) if float(np.nanmedian(v)) > 1.0 else a


def compute_lat_weight(match_prof: dict) -> np.ndarray:
    """Latitude-based fuzzy weight used by RainFallErodibility (0..1).

    Weighting:
      - <=42°N  -> 0
      - >=47°N  -> 1
      - linear between 42..47
    """
    crs_src = match_prof["crs"]
    tr = match_prof["transform"]
    h, w = int(match_prof["height"]), int(match_prof["width"])

    cols = np.arange(w, dtype=np.float64)
    rows = np.arange(h, dtype=np.float64)
    xs = tr.c + (cols + 0.5) * tr.a
    ys = tr.f + (rows + 0.5) * tr.e
    X, Y = np.meshgrid(xs, ys)

    tfm = Transformer.from_crs(crs_src, "EPSG:4326", always_xy=True)
    _, lat = tfm.transform(X, Y)

    wgt = np.zeros_like(lat, dtype=np.float32)
    wgt[lat <= 42.0] = 0.0
    wgt[lat >= 47.0] = 1.0
    mid = (lat > 42.0) & (lat < 47.0)
    wgt[mid] = (lat[mid] - 42.0) / 5.0
    return wgt


def compute_rainfall_erodibility(Pann: np.ndarray, Pveg: np.ndarray, match_prof: dict) -> np.ndarray:
    erod_S = 1.3 * Pann
    erod_N = 10.0 * (-1.48 + 1.48 * Pveg)

    crs_src = match_prof["crs"]
    tr = match_prof["transform"]
    h, w = Pann.shape

    cols = np.arange(w)
    rows = np.arange(h)
    xs = tr.c + (cols + 0.5) * tr.a
    ys = tr.f + (rows + 0.5) * tr.e
    X, Y = np.meshgrid(xs, ys)

    tfm = Transformer.from_crs(crs_src, "EPSG:4326", always_xy=True)
    _, lat = tfm.transform(X, Y)

    wgt = np.zeros_like(lat, dtype=np.float32)
    wgt[lat <= 42.0] = 0.0
    wgt[lat >= 47.0] = 1.0
    mid = (lat > 42.0) & (lat < 47.0)
    wgt[mid] = (lat[mid] - 42.0) / 5.0

    erod = (wgt * erod_N + (1.0 - wgt) * erod_S) / 10.0
    return erod.astype(np.float32)

def compute_soil_erodibility_uslek(sand: np.ndarray, silt: np.ndarray, clay: np.ndarray, orgc: np.ndarray) -> np.ndarray:
    sand_f = np.clip(to_fraction_if_percent(sand), 0.0, 1.0)
    silt_f = np.clip(to_fraction_if_percent(silt), 0.0, 1.0)
    clay_f = np.clip(to_fraction_if_percent(clay), 0.0, 1.0)
    orgc_f = np.clip(to_fraction_if_percent(orgc), 0.0, 1.0)

    fcsand = 0.2 + 0.3 * np.exp(-25.6 * sand_f * (1.0 - silt_f))
    fcl_hi = np.power((silt_f / (clay_f + silt_f + 1e-12)), 0.3)
    forgc = 1.0 - (0.25 * orgc_f) / (orgc_f + np.exp(3.72 - 2.95 * orgc_f) + 1e-12)
    one_minus_sand = 1.0 - sand_f
    fhisand = 1.0 - (0.7 * one_minus_sand) / (one_minus_sand + np.exp(-5.51 + 22.9 * one_minus_sand) + 1e-12)

    return (fcsand * fcl_hi * forgc * fhisand).astype(np.float32)


# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="Folder containing all required rasters (SPU, DEM, inputs).")
    ap.add_argument("--output_dir", required=True, help="Output folder.")

    ap.add_argument("--spu_name", default="SPU100.tif", help="SPU raster used for zonal stats (default: SPU100.tif).")
    ap.add_argument("--dem_name", default="DEM10.tif", help="DEM raster filename (default: DEM10.tif).")

    ap.add_argument("--stream_threshold_km2", type=float, default=1.0)
    ap.add_argument("--write_intermediates", action="store_true")

    ap.add_argument(
        "--flow_ratio_mode",
        choices=["mhq_over_mlq", "mlq_over_mhq"],
        default="mhq_over_mlq",
        help="FlowMinMaxRatio definition (default: mhq_over_mlq).",
    )
    ap.add_argument(
        "--cwb_sign",
        choices=["as_is", "flip"],
        default="flip",
        help="CWB sign convention. Default flips sign to match the original indicators (PET-P style).",
    )

    ap.add_argument(
        "--assume_nodata_value",
        type=float,
        default=None,
        help="If an input raster has no nodata metadata, treat this value as nodata (converted to NaN for means).",
    )
    ap.add_argument(
        "--treat_zero_as_nodata_for",
        default="",
        help="Comma-separated logical keys for which 0 will be treated as nodata for zonal MEAN (e.g., swr,cwb,grr).",
    )

    ap.add_argument(
        "--min_stream_cells_per_spu",
        type=int,
        default=1,
        help="SPUs with fewer stream cells than this will get NaN for DrainageD/RiverSlope/MeanderRatio.",
    )
    ap.add_argument(
        "--min_meander_mainlen_m",
        type=float,
        default=0.0,
        help="Skip meander computation if the main-stream length inside SPU is shorter than this (meters).",
    )

    args = ap.parse_args()

    t_all = tic()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_dir = Path(args.input_dir)

    spu_path = input_dir / args.spu_name
    dem_path = input_dir / args.dem_name

    if not spu_path.exists():
        raise FileNotFoundError(str(spu_path))
    if not dem_path.exists():
        raise FileNotFoundError(str(dem_path))

    log(f"INPUT_DIR: {input_dir}")
    log(f"SPU: {spu_path}")
    log(f"DEM: {dem_path}")

    treat_zero_keys = {k.strip() for k in args.treat_zero_as_nodata_for.split(",") if k.strip()}

    # SPU grid used as landuse/zonal grid for landcover/climate/soil indicators
    zonesL_raw, profL = read_raster(str(spu_path))
    zonesL = sanitize_zones(zonesL_raw)

    # DEM grid for terrain + streams indicators
    dem_raw, dem_prof = read_raster(str(dem_path))
    dem = dem_raw.astype(np.float32, copy=False)
    if dem_prof.get("nodata") is not None:
        dem = np.where(dem == float(dem_prof["nodata"]), np.nan, dem)

    # Reproject SPU to DEM grid (nearest, nodata=0) if needed
    with rasterio.open(str(spu_path)) as ds:
        need_reproj = (ds.crs != dem_prof["crs"]) or (ds.transform != dem_prof["transform"]) or (ds.width != dem_prof["width"]) or (ds.height != dem_prof["height"])
    if need_reproj:
        log("Reprojecting SPU to DEM grid (nearest, nodata=0)...")
        zones_dem = reproject_zones_to_match(str(spu_path), dem_prof, zone_nodata_out=0)
    else:
        zones_dem = zonesL.copy()

    # DEM hydrology (flowdir/acc) derived from DEM - in memory
    t0 = tic()
    fdir, acc, streams01_dem = derive_hydro_from_dem(
        dem_path=str(dem_path),
        stream_threshold_km2=args.stream_threshold_km2,
    )
    toc(t0, "DEM hydrology derived (in-memory)")

    # Prefer an explicit stream mask if present (to better match the legacy river network),
    # otherwise fall back to DEM-derived thresholded streams.
    streams_tif = resolve_input(input_dir, "streams")
    if streams_tif:
        log(f"Streams source: {streams_tif} (using provided stream mask)")
        streams01 = reproject_to_match(
            streams_tif, dem_prof, Resampling.nearest,
            dst_dtype=np.float32, dst_nodata=np.nan,
            nodata_assumed=args.assume_nodata_value,
            treat_zero_as_nodata=False,
        )
        streams01 = ((streams01 > 0) & np.isfinite(streams01)).astype(np.uint8)
    else:
        log("Streams source: DEM-derived (thresholded flow accumulation)")
        streams01 = streams01_dem

    # Terrain metrics on DEM grid
    t0 = tic()
    slope_rad, slope_deg = slope_from_dem(dem, dem_prof)
    xres, yres = cellsize_from_profile(dem_prof)
    twi_arr = safe_twi(acc, slope_rad, cell_area=xres * yres)

    if args.write_intermediates:
        hydro_dir = out_dir / "hydro"
        hydro_dir.mkdir(parents=True, exist_ok=True)
        write_raster(str(hydro_dir / "streams.tif"), streams01.astype(np.uint8), dem_prof, nodata=0, dtype="uint8")
        write_raster(str(hydro_dir / "flowacc_cells.tif"), acc.astype(np.float32), dem_prof, nodata=np.nan, dtype="float32")
        write_raster(str(hydro_dir / "fdir.tif"), fdir.astype(np.int16), dem_prof, nodata=-1, dtype="int16")
        write_raster(str(hydro_dir / "slope_deg.tif"), slope_deg.astype(np.float32), dem_prof, nodata=np.nan, dtype="float32")
        write_raster(str(hydro_dir / "twi.tif"), twi_arr.astype(np.float32), dem_prof, nodata=np.nan, dtype="float32")
    toc(t0, "Slope + TWI computed")

    results: Dict[str, pd.Series] = {f: pd.Series(dtype=float) for f in ALL_FACTORS}
    missing: List[str] = []

    # ---- Area-% indicators (binary masks), computed on SPU grid ----
    area_fac = ["ForestRatio", "LakeRatio", "WetlandRatio", "OrchVegRatio", "UrbanRatio", "ArableRatio", "FloodRiskAreaRatio", "GraniteRatio"]
    for fac in area_fac:
        tif = resolve_input(input_dir, fac)
        log(f"{fac} source: {tif if tif else 'MISSING'}")
        if tif is None:
            missing.append(f"{fac}: missing one of {INPUT_CANDIDATES.get(fac, [])}")
            continue
        val = reproject_to_match(
            tif, profL, Resampling.nearest,
            dst_dtype=np.float32, dst_nodata=np.nan,
            nodata_assumed=None,
            treat_zero_as_nodata=False,
        )
        hit = (val > 0) & np.isfinite(val)
        results[fac] = zonal_percent_area(zonesL, hit.astype(np.uint8))

    # ---- ReclaimedRatio (ditches buffer 100m intersect meadPastur) on SPU grid ----
    ditches = resolve_input(input_dir, "ditches")
    mead = resolve_input(input_dir, "meadPastur")
    log(f"ReclaimedRatio sources: ditches={ditches if ditches else 'MISSING'}, meadPastur={mead if mead else 'MISSING'}")
    if ditches and mead:
        d_arr = reproject_to_match(ditches, profL, Resampling.nearest, dst_dtype=np.float32, dst_nodata=np.nan)
        m_arr = reproject_to_match(mead, profL, Resampling.nearest, dst_dtype=np.float32, dst_nodata=np.nan)

        xL, yL = cellsize_from_profile(profL)
        pix = float((xL + yL) / 2.0)

        ditch_mask = (d_arr > 0) & np.isfinite(d_arr)
        dist_m = distance_transform_edt(~ditch_mask) * pix
        buf100 = dist_m <= 100.0

        reclaimed = buf100 & (m_arr > 0) & np.isfinite(m_arr)
        results["ReclaimedRatio"] = zonal_percent_area(zonesL, reclaimed.astype(np.uint8))
    else:
        missing.append("ReclaimedRatio: missing ditches and/or meadPastur raster")

    # ---- NonForestedRatio (nonForest AND slope_rad>0.05) on DEM grid ----
    nonf = resolve_input(input_dir, "nonForest")
    log(f"NonForestedRatio source: {nonf if nonf else 'MISSING'}")
    if nonf:
        nonf_dem = reproject_to_match(nonf, dem_prof, Resampling.nearest, dst_dtype=np.float32, dst_nodata=np.nan)
        hit = (nonf_dem > 0) & np.isfinite(nonf_dem) & np.isfinite(slope_rad) & (slope_rad > 0.05)
        results["NonForestedRatio"] = zonal_percent_area(zones_dem, hit.astype(np.uint8))
    else:
        missing.append("NonForestedRatio: missing nonForest raster")

    # ---- LandSlope + TWI (mean) on DEM grid ----
    results["LandSlope"] = zonal_mean(zones_dem, slope_deg)
    results["twi"] = zonal_mean(zones_dem, twi_arr)

    # ---- Stream metrics from DEM on DEM grid ----
    stream_stats = stream_metrics_by_spu(
        zones=zones_dem,
        dem=dem,
        fdir=fdir,
        acc=acc,
        streams01=streams01,
        prof=dem_prof,
        min_stream_cells_per_spu=args.min_stream_cells_per_spu,
        min_meander_mainlen_m=args.min_meander_mainlen_m,
    )
    results["DrainageD"] = stream_stats["DrainageD"]
    results["RiverSlope"] = stream_stats["RiverSlope"]
    results["MeanderRatio"] = stream_stats["MeanderRatio"]

    # ---- Optional rasters (zonal mean on SPU grid) ----
    def mean_from_optional(key: str, resampling: Resampling = Resampling.bilinear) -> Optional[pd.Series]:
        tif = resolve_input(input_dir, key)
        log(f"{key} source: {tif if tif else 'MISSING'}")
        if not tif:
            missing.append(f"{key}: missing one of {INPUT_CANDIDATES.get(key, [])}")
            return None
        arr = reproject_to_match(
            tif, profL, resampling,
            dst_dtype=np.float32, dst_nodata=np.nan,
            nodata_assumed=args.assume_nodata_value,
            treat_zero_as_nodata=(key in treat_zero_keys),
        )
        return zonal_mean(zonesL, arr)

    # cwb / swr / grr (means)
    for fac in ["cwb", "swr", "grr"]:
        s = mean_from_optional(fac, resampling=Resampling.bilinear)
        if s is not None:
            if fac == "cwb" and args.cwb_sign == "flip":
                s = -1.0 * s
            results[fac] = s

    # sri = mean(swMMQ / pAvgAnn)  (matches original raster-ratio definition)
    pavgann = resolve_input(input_dir, "pAvgAnn")
    swmmq = resolve_input(input_dir, "swMMQ")
    log(f"sri sources: pAvgAnn={pavgann if pavgann else 'MISSING'}, swMMQ={swmmq if swmmq else 'MISSING'}")
    if pavgann and swmmq:
        P = reproject_to_match(
            pavgann, profL, Resampling.bilinear,
            dst_dtype=np.float32, dst_nodata=np.nan,
            nodata_assumed=args.assume_nodata_value,
            treat_zero_as_nodata=("pAvgAnn" in treat_zero_keys),
        )
        Q = reproject_to_match(
            swmmq, profL, Resampling.bilinear,
            dst_dtype=np.float32, dst_nodata=np.nan,
            nodata_assumed=args.assume_nodata_value,
            treat_zero_as_nodata=("swMMQ" in treat_zero_keys),
        )
        ratio_arr = np.full(P.shape, np.nan, dtype=np.float32)
        ok = np.isfinite(P) & np.isfinite(Q) & (P != 0.0)
        ratio_arr[ok] = (Q[ok] / P[ok]).astype(np.float32)
        results["sri"] = zonal_mean(zonesL, ratio_arr)
    else:
        missing.append("sri: missing pAvgAnn and/or swMMQ raster")

    # FlowMinMaxRatio
    swmlq = resolve_input(input_dir, "swMLQ")
    swmhq = resolve_input(input_dir, "swMHQ")
    log(f"FlowMinMaxRatio sources: swMLQ={swmlq if swmlq else 'MISSING'}, swMHQ={swmhq if swmhq else 'MISSING'}")
    if swmlq and swmhq:
        MLQ = reproject_to_match(
            swmlq, profL, Resampling.bilinear,
            dst_dtype=np.float32, dst_nodata=np.nan,
            nodata_assumed=args.assume_nodata_value,
            treat_zero_as_nodata=("swMLQ" in treat_zero_keys),
        )
        MHQ = reproject_to_match(
            swmhq, profL, Resampling.bilinear,
            dst_dtype=np.float32, dst_nodata=np.nan,
            nodata_assumed=args.assume_nodata_value,
            treat_zero_as_nodata=("swMHQ" in treat_zero_keys),
        )

        ratio_arr = np.full(MLQ.shape, np.nan, dtype=np.float32)
        ok = np.isfinite(MLQ) & np.isfinite(MHQ)

        if args.flow_ratio_mode == "mhq_over_mlq":
            ok = ok & (MLQ != 0.0)
            ratio_arr[ok] = (MHQ[ok] / MLQ[ok]).astype(np.float32)
        else:
            ok = ok & (MHQ != 0.0)
            ratio_arr[ok] = (MLQ[ok] / MHQ[ok]).astype(np.float32)

        results["FlowMinMaxRatio"] = zonal_mean(zonesL, ratio_arr).replace([np.inf, -np.inf], np.nan)
    else:
        missing.append("FlowMinMaxRatio: missing swMLQ and/or swMHQ raster")

    # RainFallErodibility needs pAvgAnn and pAvgVeg
    # Original definition uses a latitude-based fuzzy weighting between two erodibility formulations,
    # aggregated by SPU using MEDIAN (not mean).
    pavgveg = resolve_input(input_dir, "pAvgVeg")
    log(f"RainFallErodibility sources: pAvgAnn={pavgann if pavgann else 'MISSING'}, pAvgVeg={pavgveg if pavgveg else 'MISSING'}")
    if pavgann and pavgveg:
        Pann = reproject_to_match(
            pavgann, profL, Resampling.bilinear,
            dst_dtype=np.float32, dst_nodata=np.nan,
            nodata_assumed=args.assume_nodata_value,
            treat_zero_as_nodata=("pAvgAnn" in treat_zero_keys),
        )
        Pveg = reproject_to_match(
            pavgveg, profL, Resampling.bilinear,
            dst_dtype=np.float32, dst_nodata=np.nan,
            nodata_assumed=args.assume_nodata_value,
            treat_zero_as_nodata=("pAvgVeg" in treat_zero_keys),
        )

        # Component rasters
        erod_S = 1.3 * Pann
        erod_N = 10.0 * (-1.48 + 1.48 * Pveg)

        # Latitude weight raster (0..1)
        wgt = compute_lat_weight(profL)

        # Zonal medians of components (matches original R implementation)
        med_S = zonal_median(zonesL, erod_S)
        med_N = zonal_median(zonesL, erod_N)
        med_w = zonal_median(zonesL, wgt)

        erod_spu = (med_w * med_N + (1.0 - med_w) * med_S) / 10.0
        results["RainFallErodibility"] = erod_spu
    else:
        missing.append("RainFallErodibility: missing pAvgAnn and/or pAvgVeg raster")

    # SoilErodibility needs sand/silt/clay/orgC (fractions)
    sand = resolve_input(input_dir, "sand")
    silt = resolve_input(input_dir, "silt")
    clay = resolve_input(input_dir, "clay")
    orgc = resolve_input(input_dir, "orgC")
    log(f"SoilErodibility sources: sand={sand if sand else 'MISSING'}, silt={silt if silt else 'MISSING'}, clay={clay if clay else 'MISSING'}, orgC={orgc if orgc else 'MISSING'}")
    if sand and silt and clay and orgc:
        Sand = reproject_to_match(sand, profL, Resampling.bilinear, dst_dtype=np.float32, dst_nodata=np.nan, nodata_assumed=args.assume_nodata_value)
        Silt = reproject_to_match(silt, profL, Resampling.bilinear, dst_dtype=np.float32, dst_nodata=np.nan, nodata_assumed=args.assume_nodata_value)
        Clay = reproject_to_match(clay, profL, Resampling.bilinear, dst_dtype=np.float32, dst_nodata=np.nan, nodata_assumed=args.assume_nodata_value)
        OrgC = reproject_to_match(orgc, profL, Resampling.bilinear, dst_dtype=np.float32, dst_nodata=np.nan, nodata_assumed=args.assume_nodata_value)
        uslek = compute_soil_erodibility_uslek(Sand, Silt, Clay, OrgC)
        results["SoilErodibility"] = zonal_median(zonesL, uslek)
    else:
        missing.append("SoilErodibility: missing soil fraction rasters")

    # ---- Write outputs ----
    ordered = {f: results.get(f, pd.Series(dtype=float)) for f in ALL_FACTORS}
    out_df = merge_series(ordered)

    out_csv = out_dir / "indicators_by_spu.csv"
    out_df.to_csv(out_csv, index=False)

    report = out_dir / "report_missing_inputs.txt"
    report.write_text("\n".join(missing) if missing else "All required inputs were found.\n", encoding="utf-8")

    log(f"Saved: {out_csv}")
    log(f"Saved: {report}")
    toc(t_all, "TOTAL")


if __name__ == "__main__":
    main()
