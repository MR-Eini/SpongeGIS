#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SpongeGIS3 output validator

This script validates:
1) CSV structure + basic sanity checks, using ALL_FACTORS extracted from SpongeGIS3.py
2) Optional regression compare against a JSON baseline you create
3) Optional compare against hard-coded expected stats (baked into this script)

Hard-coded expectations:
- These values were extracted from the R test suite (test-indicators.R) and baked into this file.
- Only the overlap with SpongeGIS3 ALL_FACTORS is compared.
- GraniteRatio has no R expectation and is not compared in hard-coded mode.

Usage (hard-coded compare):
  python validate_spongegis3_outputs.py ^
    --spongegis_path SpongeGIS3.py ^
    --csv ...\indicators_by_spu.csv ^
    --missing-report ...\report_missing_inputs.txt ^
    --use_hardcoded_r --hc_abs_tol 1e-6 --hc_rel_tol 1e-6
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd


# -------------------------
# Hard-coded expected stats (baked from test-indicators.R)
# -------------------------
# Fields: min, max, mean, stdev (R-style). We compare stdev to sample std (ddof=1) by default.
HARDCODED_R_EXPECTED: Dict[str, Dict[str, float]] = {
  "cwb": {
    "min": -263.01,
    "max": -83.57,
    "mean": -165.22,
    "stdev": 47.31
  },
  "swr": {
    "min": 107.0,
    "max": 525.4,
    "mean": 431.13,
    "stdev": 61.47
  },
  "grr": {
    "min": 66.95,
    "max": 105.04,
    "mean": 84.89,
    "stdev": 14.13
  },
  "sri": {
    "min": 0.02,
    "max": 0.35,
    "mean": 0.23,
    "stdev": 0.07
  },
  "FlowMinMaxRatio": {
    "min": 0.01,
    "max": 149.36,
    "mean": 28.65,
    "stdev": 40.41
  },
  "RiverSlope": {
    "min": 0.1,
    "max": 3.6,
    "mean": 0.72,
    "stdev": 0.63
  },
  "LandSlope": {
    "min": 0.18,
    "max": 4.46,
    "mean": 1.69,
    "stdev": 0.81
  },
  "MeanderRatio": {
    "min": 59.64,
    "max": 99.75,
    "mean": 85.42,
    "stdev": 8.45
  },
  "FloodRiskAreaRatio": {
    "min": 0.0,
    "max": 90.52,
    "mean": 2.93,
    "stdev": 10.12
  },
  "NonForestedRatio": {
    "min": 0.0,
    "max": 59.39,
    "mean": 8.88,
    "stdev": 13.45
  },
  "DrainageD": {
    "min": 0.04,
    "max": 4.24,
    "mean": 0.69,
    "stdev": 0.54
  },
  "twi": {
    "min": 9.71,
    "max": 20.2,
    "mean": 11.26,
    "stdev": 1.38
  },
  "ForestRatio": {
    "min": 0.0,
    "max": 100.0,
    "mean": 34.52,
    "stdev": 33.08
  },
  "LakeRatio": {
    "min": 0.0,
    "max": 11.2,
    "mean": 0.18,
    "stdev": 1.18
  },
  "WetlandRatio": {
    "min": 0.0,
    "max": 39.57,
    "mean": 4.15,
    "stdev": 6.42
  },
  "OrchVegRatio": {
    "min": 0.0,
    "max": 8.3,
    "mean": 0.1,
    "stdev": 0.8
  },
  "UrbanRatio": {
    "min": 0.0,
    "max": 89.63,
    "mean": 9.07,
    "stdev": 14.2
  },
  "ArableRatio": {
    "min": 0.0,
    "max": 86.95,
    "mean": 33.49,
    "stdev": 30.1
  },
  "ReclaimedRatio": {
    "min": 0.0,
    "max": 29.69,
    "mean": 2.39,
    "stdev": 4.38
  },
  "RainFallErodibility": {
    "min": 591.26,
    "max": 1488.14,
    "mean": 749.36,
    "stdev": 214.84
  },
  "SoilErodibility": {
    "min": 0.08,
    "max": 0.18,
    "mean": 0.13,
    "stdev": 0.04
  }
}

# -------------------------
# Sanity rules
# -------------------------
PERCENT_FACTORS: Set[str] = {
    "ForestRatio", "LakeRatio", "WetlandRatio",
    "OrchVegRatio", "UrbanRatio", "ArableRatio",
    "ReclaimedRatio", "GraniteRatio",
    "FloodRiskAreaRatio", "NonForestedRatio",
    "MeanderRatio",
}

NONNEGATIVE_FACTORS: Set[str] = {
    "DrainageD", "RiverSlope", "LandSlope",
    "RainFallErodibility", "SoilErodibility",
    "sri", "FlowMinMaxRatio",
    "swr", "grr",
}

# cwb can be negative; do not force non-negative.


@dataclass
class Stat:
    count: int
    min: float
    max: float
    mean: float
    std: float  # population std (ddof=0) unless specified


def extract_all_factors(spongegis_path: Path) -> List[str]:
    """Extract ALL_FACTORS list from SpongeGIS3.py without importing it."""
    txt = spongegis_path.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"ALL_FACTORS\s*=\s*\[(.*?)\]\s*", txt, re.S)
    if not m:
        raise ValueError(f"Could not find ALL_FACTORS in {spongegis_path}")
    block = m.group(1)
    factors = re.findall(r'"([^"]+)"', block)
    if not factors:
        raise ValueError(f"Found ALL_FACTORS block but no factor names in {spongegis_path}")
    return factors


def parse_missing_report(path: Optional[Path]) -> Set[str]:
    if not path or (not path.exists()):
        return set()

    missing = set()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith("all required inputs were found"):
            continue
        if ":" in line:
            key = line.split(":", 1)[0].strip()
            if key:
                missing.add(key)
    return missing


def compute_stats(series: pd.Series, ddof: int = 0) -> Stat:
    v = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return Stat(count=0, min=float("nan"), max=float("nan"), mean=float("nan"), std=float("nan"))
    return Stat(
        count=int(v.size),
        min=float(np.min(v)),
        max=float(np.max(v)),
        mean=float(np.mean(v)),
        std=float(np.std(v, ddof=ddof)),
    )


def approx_equal(a: float, b: float, abs_tol: float, rel_tol: float) -> bool:
    if not (math.isfinite(a) and math.isfinite(b)):
        return (math.isnan(a) and math.isnan(b))
    return abs(a - b) <= max(abs_tol, rel_tol * max(abs(a), abs(b)))


def load_baseline(path: Path) -> Dict[str, Stat]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    stats = {}
    for k, v in obj.get("stats", {}).items():
        stats[k] = Stat(
            count=int(v["count"]),
            min=float(v["min"]),
            max=float(v["max"]),
            mean=float(v["mean"]),
            std=float(v["std"]),
        )
    return stats


def save_baseline(path: Path, factors: List[str], stats: Dict[str, Stat]) -> None:
    out = {
        "version": 1,
        "factors": factors,
        "stats": {k: {"count": v.count, "min": v.min, "max": v.max, "mean": v.mean, "std": v.std} for k, v in stats.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")


def validate_ranges(factor: str, st: Stat) -> List[str]:
    errs: List[str] = []
    if st.count == 0:
        errs.append("no finite values (count=0)")
        return errs

    if factor in PERCENT_FACTORS:
        if st.min < -1e-6 or st.max > 100.000001:
            errs.append(f"percent-range violation: min={st.min:.6g}, max={st.max:.6g} (expected 0..100)")

    if factor == "LandSlope":
        if st.min < -1e-6 or st.max > 90.000001:
            errs.append(f"slope-range violation: min={st.min:.6g}, max={st.max:.6g} (expected 0..90 deg)")

    if factor in NONNEGATIVE_FACTORS:
        if st.min < -1e-9:
            errs.append(f"non-negative violation: min={st.min:.6g}")

    if factor == "SoilErodibility":
        if st.min < -1e-9 or st.max > 2.0:
            errs.append(f"SoilErodibility out of expected range: min={st.min:.6g}, max={st.max:.6g} (expected ~0..2)")

    return errs


def compare_hardcoded_r(
    factors: List[str],
    stats_pop: Dict[str, Stat],
    stats_sample: Dict[str, Stat],
    missing_inputs: Set[str],
    abs_tol: float,
    rel_tol: float,
) -> List[str]:
    """Compare current stats against HARDCODED_R_EXPECTED (min/max/mean/stdev)."""
    failures: List[str] = []
    overlap = [f for f in factors if f in HARDCODED_R_EXPECTED]
    if not overlap:
        failures.append("Hard-coded R expected: no overlap with SpongeGIS3 ALL_FACTORS.")
        return failures

    for f in overlap:
        if f in missing_inputs:
            continue

        exp = HARDCODED_R_EXPECTED[f]
        cur = stats_pop[f]          # min/max/mean from population stats
        cur_sd = stats_sample[f].std  # compare to R stdev using sample std (ddof=1)

        checks = [
            ("min", cur.min, exp["min"]),
            ("max", cur.max, exp["max"]),
            ("mean", cur.mean, exp["mean"]),
            ("stdev", cur_sd, exp["stdev"]),
        ]

        for nm, a, b in checks:
            if not approx_equal(a, b, abs_tol=abs_tol, rel_tol=rel_tol):
                failures.append(
                    f"[HARDCODED_R] {f}: {nm} != expected (current={a:.6g}, expected={b:.6g}, abs_tol={abs_tol}, rel_tol={rel_tol})"
                )
    return failures


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spongegis_path", default="SpongeGIS3.py", help="Path to SpongeGIS3.py (used to read ALL_FACTORS).")
    ap.add_argument("--csv", required=True, help="Path to indicators_by_spu.csv produced by SpongeGIS3.")
    ap.add_argument("--missing-report", default=None, help="Path to report_missing_inputs.txt (optional).")

    ap.add_argument("--baseline", default=None, help="Path to baseline JSON (optional).")
    ap.add_argument("--write-baseline", action="store_true", help="Write baseline JSON from the current CSV and exit 0.")
    ap.add_argument("--abs-tol", type=float, default=1e-6, help="Absolute tolerance for baseline compare.")
    ap.add_argument("--rel-tol", type=float, default=1e-6, help="Relative tolerance for baseline compare.")

    ap.add_argument("--use_hardcoded_r", action="store_true", help="Compare against baked hard-coded expected stats from R tests.")
    ap.add_argument("--hc_abs_tol", type=float, default=1e-6, help="Absolute tolerance for hard-coded expected compare.")
    ap.add_argument("--hc_rel_tol", type=float, default=1e-6, help="Relative tolerance for hard-coded expected compare.")

    ap.add_argument("--strict", action="store_true",
                    help="Fail if any expected indicator is missing from the CSV, even if missing-report marks it missing.")
    args = ap.parse_args()

    spongegis_path = Path(args.spongegis_path)
    csv_path = Path(args.csv)

    if not spongegis_path.exists():
        raise FileNotFoundError(str(spongegis_path))
    if not csv_path.exists():
        raise FileNotFoundError(str(csv_path))

    factors = extract_all_factors(spongegis_path)

    missing_report_path: Optional[Path] = Path(args.missing_report) if args.missing_report else None
    missing_inputs = parse_missing_report(missing_report_path)

    df = pd.read_csv(csv_path)

    failures: List[str] = []

    # ID checks
    if "ID" not in df.columns:
        failures.append("CSV missing required column: ID")
    else:
        ids = pd.to_numeric(df["ID"], errors="coerce")
        if ids.isna().any():
            failures.append("ID column contains non-numeric or NaN values")
        if (ids <= 0).any():
            failures.append("ID column contains non-positive values")
        if ids.duplicated().any():
            failures.append("ID column contains duplicates")

    # Stats per factor (population + sample)
    stats_pop: Dict[str, Stat] = {}
    stats_sample: Dict[str, Stat] = {}

    for f in factors:
        if f not in df.columns:
            if args.strict or (f not in missing_inputs):
                failures.append(f"Missing expected indicator column: {f}")
            stats_pop[f] = Stat(0, float("nan"), float("nan"), float("nan"), float("nan"))
            stats_sample[f] = Stat(0, float("nan"), float("nan"), float("nan"), float("nan"))
            continue

        stats_pop[f] = compute_stats(df[f], ddof=0)
        # sample stdev used only for hard-coded R expected compare
        stats_sample[f] = compute_stats(df[f], ddof=1)

        if f in missing_inputs:
            continue

        for e in validate_ranges(f, stats_pop[f]):
            failures.append(f"{f}: {e}")

    # Baseline mode
    if args.baseline:
        baseline_path = Path(args.baseline)
        if args.write_baseline:
            save_baseline(baseline_path, factors, stats_pop)
            print(f"[OK] Baseline written: {baseline_path}")
            return 0

        if baseline_path.exists():
            base = load_baseline(baseline_path)
            for f in factors:
                if f not in base:
                    failures.append(f"Baseline missing factor: {f}")
                    continue
                cur = stats_pop[f]
                ref = base[f]

                if f in missing_inputs:
                    continue

                if cur.count != ref.count:
                    failures.append(f"{f}: count changed (current={cur.count}, baseline={ref.count})")

                for field in ["min", "max", "mean", "std"]:
                    a = getattr(cur, field)
                    b = getattr(ref, field)
                    if not approx_equal(a, b, abs_tol=args.abs_tol, rel_tol=args.rel_tol):
                        failures.append(
                            f"{f}: {field} changed (current={a:.6g}, baseline={b:.6g}, abs_tol={args.abs_tol}, rel_tol={args.rel_tol})"
                        )
        else:
            failures.append(f"Baseline file not found: {baseline_path} (use --write-baseline to create it)")

    # Hard-coded expected compare
    if args.use_hardcoded_r:
        failures.extend(compare_hardcoded_r(
            factors=factors,
            stats_pop=stats_pop,
            stats_sample=stats_sample,
            missing_inputs=missing_inputs,
            abs_tol=args.hc_abs_tol,
            rel_tol=args.hc_rel_tol
        ))

    # Print summary
    print(f"\nCSV: {csv_path}")
    if missing_report_path:
        print(f"Missing report: {missing_report_path} | missing indicators: {sorted(missing_inputs)}")
    print(f"Indicators expected (from {spongegis_path.name}): {len(factors)}")
    if args.use_hardcoded_r:
        print(f"Hard-coded expected compare: ON | abs={args.hc_abs_tol} rel={args.hc_rel_tol}")
    print("")

    # Print a compact table of stats (population std)
    view = []
    for f in factors:
        st = stats_pop[f]
        view.append([f, st.count, st.min, st.max, st.mean, st.std, ("MISSING_INPUT" if f in missing_inputs else "")])
    out = pd.DataFrame(view, columns=["factor", "count", "min", "max", "mean", "std(pop)", "note"])
    pd.set_option("display.max_rows", 200)
    print(out.to_string(index=False))

    if args.use_hardcoded_r:
        overlap = [f for f in factors if f in HARDCODED_R_EXPECTED]
        if overlap:
            rows = []
            for f in overlap:
                ex = HARDCODED_R_EXPECTED[f]
                cur = stats_pop[f]
                cur_sd = stats_sample[f].std
                rows.append([f, cur.min, ex["min"], cur.max, ex["max"], cur.mean, ex["mean"], cur_sd, ex["stdev"]])
            tab = pd.DataFrame(rows, columns=["factor","cur_min","exp_min","cur_max","exp_max","cur_mean","exp_mean","cur_stdev(ddof=1)","exp_stdev"])
            print("\nHARDCODED EXPECTED COMPARISON (overlap):")
            print(tab.to_string(index=False))

    if failures:
        print("\nFAILURES:")
        for x in failures:
            print(f" - {x}")
        return 1

    print("\n[OK] All validations passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
