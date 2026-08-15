"""
06_create_pfz_labels.py  (FIXED — leakage-aware + memory-efficient version)
Final version – NEVER loads 175M rows for merge / sort

KEY CHANGE vs original script:
--------------------------------------------------------------
The original label was:
    pfz(t) = f(sst(t), chl(t))
and Mode A then used sst(t), chl(t) as FEATURES to predict pfz(t).
That is circular (the label is a deterministic function of the
feature) and explains the near-perfect PR-AUC / Brier scores.

This version keeps the same-day label (pfz) for reference /
sanity-checking, but ALSO creates forward-shifted (leakage-free)
labels:
    pfz_lead1  = pfz(t+1)   -> "will conditions be favourable tomorrow?"
    pfz_lead3  = pfz(t+3)
    pfz_lead7  = pfz(t+7)

Models are then trained on features at time t to predict pfz at
t+lead, using ONLY information available at t. Because pfz(t+lead)
depends on sst(t+lead)/chl(t+lead) — values the model never sees —
raw sst(t)/chl(t) can be safely kept as predictive features (Mode A)
without being tautological. This is now a genuine forecasting task.

Shifting is done per (lat, lon) grid cell, sorted by time, so the
lag is a true temporal shift and does not leak across locations.

Memory note (2026-08-10 fix):
--------------------------------------------------------------
The previous version concatenated all ~175 M rows and called
sort_values, which required >24 GiB temporary arrays and crashed.
This version processes one spatial block at a time (24 blocks),
so peak RAM stays in the 5–12 GB range on typical machines.
"""

import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
from tqdm import tqdm
from sklearn.cluster import KMeans
import gc

# ====================== CONFIG ======================
FEATURE_ROOT = Path(r"D:\PFZ_BoB_ML\data\features")
MODELLING_ROOT = Path(r"D:\PFZ_BoB_ML\data\modelling")
MODELLING_ROOT.mkdir(parents=True, exist_ok=True)

YEARS = range(2012, 2026)

LABEL_METHOD = "threshold"
SST_MIN = 26.5
SST_MAX = 29.0
CHL_MIN = 0.2
MIN_DEPTH = 30
N_SPATIAL_BLOCKS = 24          # increased from 8: finer, more honest spatial CV blocks

# Forecast horizons (days ahead). 0 kept only for sanity-check comparisons.
LEAD_DAYS = [0, 1, 3, 7]

KEEP_VARS = [
    "sst", "ssh", "u_current", "v_current", "current_speed",
    "chl", "log_chl", "no3", "po4", "o2",
    "eastward_wind", "northward_wind", "wind_speed", "wind_stress",
    "sst_gradient", "chl_gradient", "bathymetry",
    "sst_front", "depth_category", "high_chl_flag", "optimal_sst_flag",
    "sst_anomaly", "ssh_anomaly", "chl_anomaly", "log_chl_anomaly",
    "current_speed_anomaly", "wind_speed_anomaly",
    "nino34", "dmi",
    "year", "month", "dayofyear", "sin_doy", "cos_doy",
]
# ====================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)


def create_threshold_label(ds):
    sst_ok = (ds["sst"] >= SST_MIN) & (ds["sst"] <= SST_MAX)
    chl_ok = (ds["chl"] >= CHL_MIN)
    return (sst_ok & chl_ok).astype("float32").rename("pfz")


def process_one_year(year: int) -> Path | None:
    """
    Same as before, but we DO NOT shift labels here — shifting has to
    happen after all years are concatenated (per grid cell, in time
    order), because a lead-7 label near Dec 31 needs data from the
    following year's file. We only build the raw per-year table here.
    """
    path = FEATURE_ROOT / f"features_{year}.nc"
    if not path.exists():
        log.warning(f"Missing: {path.name}")
        return None

    log.info(f"Processing {year} ...")
    ds = xr.open_dataset(path)
    ds["pfz"] = create_threshold_label(ds)

    if "bathymetry" in ds:
        ds = ds.where(ds["bathymetry"] >= MIN_DEPTH)

    available = [v for v in KEEP_VARS + ["pfz"] if v in ds.data_vars]
    ds = ds[available]

    for var in list(ds.data_vars):
        if np.issubdtype(ds[var].dtype, np.floating):
            ds[var] = ds[var].astype("float32")

    df = ds.to_dataframe().reset_index()

    rename_map = {}
    if "latitude" in df.columns:
        rename_map["latitude"] = "lat"
    if "longitude" in df.columns:
        rename_map["longitude"] = "lon"
    df = df.rename(columns=rename_map)
    df = df.loc[:, ~df.columns.duplicated()]
    if "quantile" in df.columns:
        df = df.drop(columns=["quantile"])

    df = df.dropna(subset=["sst", "chl", "pfz"])

    log.info(f"{year}: {len(df):,} rows | PFZ+ {df['pfz'].mean():.3f}")

    tmp_path = MODELLING_ROOT / f"tmp_{year}.parquet"
    df.to_parquet(tmp_path, index=False)

    ds.close()
    del ds, df
    gc.collect()
    return tmp_path


def build_spatial_lookup(tmp_files):
    """Build spatial blocks from unique cells across all years (very small)."""
    log.info("Building spatial block lookup from unique grid cells ...")

    grids = []
    for f in tmp_files:
        df = pd.read_parquet(f, columns=["lat", "lon"])
        grids.append(df.drop_duplicates())
        del df

    grid = pd.concat(grids, ignore_index=True).drop_duplicates().reset_index(drop=True)
    log.info(f"Unique grid cells: {len(grid):,}")

    kmeans = KMeans(n_clusters=N_SPATIAL_BLOCKS, random_state=42, n_init=10)
    grid["spatial_block"] = kmeans.fit_predict(grid[["lat", "lon"]].values).astype("int16")

    grid["lat_r"] = grid["lat"].round(5)
    grid["lon_r"] = grid["lon"].round(5)

    lookup = grid[["lat_r", "lon_r", "spatial_block"]].drop_duplicates()
    lookup.to_parquet(MODELLING_ROOT / "spatial_lookup.parquet", index=False)
    log.info("Spatial lookup saved.")
    return lookup


def add_lag_labels_all_years(tmp_files, lookup):
    """
    Memory-efficient version.
    Processes one spatial_block at a time so we never hold the full
    175 M-row table in RAM.  Logic is identical: sort by (lat, lon, time)
    inside each block and apply a pure temporal shift of the pfz column.
    """
    log.info("Building lag labels block-by-block (memory-efficient) ...")

    # ------------------------------------------------------------------
    # 1. Pre-compute which rows belong to which spatial block
    #    (we only need lat/lon + year for this step)
    # ------------------------------------------------------------------
    log.info("Indexing rows by spatial block ...")
    block_row_counts = {b: 0 for b in range(N_SPATIAL_BLOCKS)}
    block_row_counts[-1] = 0   # for any unmatched cells

    # We will write one intermediate parquet per spatial block
    block_dirs = {}
    for b in list(range(N_SPATIAL_BLOCKS)) + [-1]:
        d = MODELLING_ROOT / f"_block_{b:02d}"
        d.mkdir(exist_ok=True)
        block_dirs[b] = d

    # Stream every yearly file once and route rows to the correct block file
    for f in tqdm(tmp_files, desc="Routing years → blocks"):
        df = pd.read_parquet(f)
        df["lat_r"] = df["lat"].round(5)
        df["lon_r"] = df["lon"].round(5)
        df = df.merge(lookup, on=["lat_r", "lon_r"], how="left")
        df["spatial_block"] = df["spatial_block"].fillna(-1).astype("int16")

        for b, sub in df.groupby("spatial_block", sort=False):
            out = block_dirs[int(b)] / f"{f.stem}.parquet"
            # drop the temporary rounding columns; we keep original lat/lon
            sub = sub.drop(columns=["lat_r", "lon_r"])
            sub.to_parquet(out, index=False)
            block_row_counts[int(b)] += len(sub)

        del df
        gc.collect()

    log.info("Rows per spatial block:")
    for b, n in sorted(block_row_counts.items()):
        if n > 0:
            log.info(f"  block {b:2d}: {n:,}")

    # ------------------------------------------------------------------
    # 2. Process each spatial block independently
    # ------------------------------------------------------------------
    final_parts = []

    for b in tqdm(sorted(block_dirs.keys()), desc="Shifting labels per block"):
        files = sorted(block_dirs[b].glob("*.parquet"))
        if not files:
            continue

        # Load only this block (typically 5–10 M rows)
        dfs = [pd.read_parquet(f) for f in files]
        df = pd.concat(dfs, ignore_index=True)
        del dfs
        gc.collect()

        # Sort by grid cell + time (now small enough)
        df = df.sort_values(["lat", "lon", "time"]).reset_index(drop=True)

        # Forward-shifted labels
        grp = df.groupby(["lat", "lon"], sort=False)["pfz"]
        for lead in LEAD_DAYS:
            if lead == 0:
                continue
            col = f"pfz_lead{lead}"
            df[col] = grp.shift(-lead)

        # Quick diagnostics
        for lead in LEAD_DAYS:
            if lead == 0:
                continue
            col = f"pfz_lead{lead}"
            n_valid = df[col].notna().sum()
            rate = df[col].mean() if n_valid else float("nan")
            log.info(f"  block {b:2d} | {col}: {n_valid:,} valid | PFZ+ = {rate:.3f}")

        # Write the finished block
        out_path = MODELLING_ROOT / f"_labelled_block_{b:02d}.parquet"
        df.to_parquet(out_path, index=False)
        final_parts.append(out_path)

        del df
        gc.collect()

        # Clean the temporary per-year block files
        for f in files:
            f.unlink(missing_ok=True)
        try:
            block_dirs[b].rmdir()
        except OSError:
            pass

    # ------------------------------------------------------------------
    # 3. Concatenate the finished blocks (still one block at a time)
    #    and re-split by year
    # ------------------------------------------------------------------
    log.info("Re-assembling labelled blocks and writing yearly files ...")
    preferred = [
        "lat", "lon", "time", "year", "month", "dayofyear", "sin_doy", "cos_doy",
        "pfz", "pfz_lead1", "pfz_lead3", "pfz_lead7", "spatial_block",
        "sst", "ssh", "u_current", "v_current", "current_speed",
        "chl", "log_chl", "no3", "po4", "o2",
        "eastward_wind", "northward_wind", "wind_speed", "wind_stress",
        "sst_gradient", "chl_gradient", "bathymetry",
        "sst_front", "depth_category", "high_chl_flag", "optimal_sst_flag",
        "sst_anomaly", "ssh_anomaly", "chl_anomaly", "log_chl_anomaly",
        "current_speed_anomaly", "wind_speed_anomaly",
        "nino34", "dmi"
    ]

    # We will collect yearly dataframes on the fly
    yearly_dfs = {y: [] for y in YEARS}

    for part in tqdm(final_parts, desc="Collecting by year"):
        df = pd.read_parquet(part)
        cols = [c for c in preferred if c in df.columns]
        df = df[cols]

        for year, sub in df.groupby("year"):
            yearly_dfs[int(year)].append(sub)

        del df
        gc.collect()
        part.unlink(missing_ok=True)          # clean up

    final_files = []
    for year in sorted(yearly_dfs.keys()):
        parts = yearly_dfs[year]
        if not parts:
            continue
        df_year = pd.concat(parts, ignore_index=True)
        out_path = MODELLING_ROOT / f"modelling_{year}.parquet"
        df_year.to_parquet(out_path, index=False)
        log.info(f"Saved → {out_path.name} | shape {df_year.shape}")
        final_files.append(out_path)
        del df_year, parts
        gc.collect()

    return final_files


def main():
    log.info("========== CREATING TARGET + MODELLING TABLE (leakage-fixed) ==========")
    log.info(f"Label method : {LABEL_METHOD} | SST {SST_MIN}-{SST_MAX} | CHL ≥ {CHL_MIN} | depth ≥ {MIN_DEPTH}m")
    log.info(f"Lead times (days ahead): {LEAD_DAYS} | Spatial blocks: {N_SPATIAL_BLOCKS}")

    # Step 1: Create yearly tmp files (same-day label only, unchanged logic)
    tmp_files = []
    for year in tqdm(YEARS, desc="Creating yearly files"):
        tmp = process_one_year(year)
        if tmp:
            tmp_files.append(tmp)

    if not tmp_files:
        log.error("No data!")
        return

    # Step 2: Build spatial lookup (only ~34k rows)
    lookup = build_spatial_lookup(tmp_files)

    # Step 3 + 4: Build lag labels block-by-block and write yearly files
    final_files = add_lag_labels_all_years(tmp_files, lookup)

    # Save label definition (now documents the leakage fix + lead times)
    label_info = {
        "label_method": LABEL_METHOD,
        "sst_min": SST_MIN,
        "sst_max": SST_MAX,
        "chl_min": CHL_MIN,
        "min_depth_m": MIN_DEPTH,
        "n_spatial_blocks": N_SPATIAL_BLOCKS,
        "lead_days": LEAD_DAYS,
        "description": (
            "Binary PFZ from classic SST + CHL thresholds (INCOIS-style). "
            "'pfz' is the SAME-DAY label and is a deterministic function of "
            "sst(t)/chl(t) -- keep this ONLY as a sanity check, never use "
            "Mode A (raw sst/chl features) to predict same-day 'pfz' as a "
            "headline result, since that is tautological and explains the "
            "near-1.0 PR-AUC seen previously. 'pfz_lead1/3/7' are forward- "
            "shifted, per-grid-cell, leakage-free forecasting targets: "
            "features at day t predict conditions at t+lead, using only "
            "information available at t. These are the recommended targets "
            "for headline results."
        ),
        "known_leakage_note": (
            "Original 06_create_pfz_labels.py produced same-day-only "
            "labels; training Mode A (with raw sst/chl) on that same-day "
            "label is circular and inflates PR-AUC/Brier close to perfect. "
            "Fixed by adding forward-shifted lead labels above."
        ),
        "files": [str(f.name) for f in final_files]
    }
    with open(MODELLING_ROOT / "label_definition.json", "w") as f:
        json.dump(label_info, f, indent=2)

    # Clean tmp files
    for f in tmp_files:
        f.unlink(missing_ok=True)

    log.info("========== DONE ==========")
    log.info(f"Yearly modelling files saved in: {MODELLING_ROOT}")
    log.info("Next step: retrain 07A/07B/07C with TARGET = 'pfz_lead1' (or lead3 / lead7)")
    log.info("instead of TARGET = 'pfz'. Keep 'pfz' only for an explicit,")
    log.info("clearly-labelled 'same-day sanity check' table in your results.")


if __name__ == "__main__":
    main()