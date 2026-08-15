"""
External chlorophyll validation (LOG-SPACE): OC-CCI (monthly, CMEMS) vs.
the fused Bay of Bengal feature dataset (features_2019.nc).

Ocean-color chlorophyll is approximately log-normally distributed, so
validation statistics (bias, RMSE, correlation) are computed here on
log10(chl) rather than raw mg/m3 -- this is the standard approach in the
ocean-color validation literature (e.g. Seegers et al. 2018) and avoids a
few high-chl coastal cells dominating the error metrics.

Usage:
    python validate_chl_occci_log.py
"""

import numpy as np
import pandas as pd
import xarray as xr

# ------------------------- CONFIG -------------------------------------
FEATURES_NC = r"D:\PFZ_BoB_ML\data\features\features_2019.nc"
OCCCI_NC    = r"D:\PFZ_BoB_ML\data\external\oc_cci\occci_chl_bob_2019_monthly.nc"

MODEL_CHL_VAR = "chl"        # raw chlorophyll (mg/m3); adjust if named differently
                              # Set to "log_chl" + MODEL_IS_LOG=True to use your
                              # pre-computed log field instead.
MODEL_IS_LOG = False

OCCCI_CHL_VAR = "CHL"

LAT_NAME  = "latitude"
LON_NAME  = "longitude"
TIME_NAME = "time"

# Ocean-color convention: exclude non-physical / near-zero values before logging
MIN_VALID_CHL = 0.01  # mg/m3

OUT_CSV = r"D:\PFZ_BoB_ML\outputs\external_validation_rama\chl_validation_log_matched_2019.csv"
# ------------------------------------------------------------------------


def load_and_aggregate_model(path: str) -> xr.DataArray:
    ds = xr.open_dataset(path)
    chl = ds[MODEL_CHL_VAR]
    if MODEL_IS_LOG:
        chl = 10 ** chl  # back-transform to linear mg/m3; we'll re-log after averaging
    monthly = chl.resample({TIME_NAME: "1MS"}).mean()
    ds.close()
    return monthly


def load_occci(path: str) -> xr.DataArray:
    ds = xr.open_dataset(path)
    chl = ds[OCCCI_CHL_VAR]
    ds.close()
    return chl


def match_and_compare(model_monthly: xr.DataArray, occci: xr.DataArray) -> pd.DataFrame:
    occci_regridded = occci.interp(
        {LAT_NAME: model_monthly[LAT_NAME], LON_NAME: model_monthly[LON_NAME]},
        method="nearest",
    )

    records = []
    for t in model_monthly[TIME_NAME].values:
        month_label = pd.Timestamp(t).strftime("%Y-%m")
        try:
            model_slice = model_monthly.sel({TIME_NAME: t})
            occci_slice = occci_regridded.sel({TIME_NAME: t}, method="nearest")
        except Exception as e:
            print(f"Skipping {month_label}: {e}")
            continue

        m = model_slice.values.ravel()
        o = occci_slice.values.ravel()
        mask = ~np.isnan(m) & ~np.isnan(o) & (o > MIN_VALID_CHL) & (m > MIN_VALID_CHL)

        if mask.sum() == 0:
            continue

        log_m = np.log10(m[mask])
        log_o = np.log10(o[mask])
        diff = log_m - log_o

        records.append({
            "month": month_label,
            "n_cells": int(mask.sum()),
            "log_bias": float(diff.mean()),                       # decades
            "log_rmse": float(np.sqrt((diff ** 2).mean())),
            "log_mae": float(np.abs(diff).mean()),
            "corr_log": float(np.corrcoef(log_m, log_o)[0, 1]),
            "median_ratio_model_to_occci": float(np.median(10 ** diff)),  # e.g. 0.5 = model is half
            "model_geomean": float(10 ** log_m.mean()),
            "occci_geomean": float(10 ** log_o.mean()),
        })

    return pd.DataFrame(records)


def summarise(monthly_stats: pd.DataFrame) -> None:
    print("\n=== Per-month chlorophyll validation, LOG10 space (model vs. OC-CCI) ===")
    print(monthly_stats.to_string(index=False))

    total_n = monthly_stats["n_cells"].sum()
    weighted_log_bias = (monthly_stats["log_bias"] * monthly_stats["n_cells"]).sum() / total_n
    weighted_log_rmse = np.sqrt(
        (monthly_stats["log_rmse"] ** 2 * monthly_stats["n_cells"]).sum() / total_n
    )

    print("\n=== Annual summary (cell-count weighted, log10 space) ===")
    print(f"Total matched grid-cell-months : {total_n}")
    print(f"Weighted log10 bias            : {weighted_log_bias:+.4f} decades")
    print(f"  -> equivalent multiplicative factor: model = {10**weighted_log_bias:.3f} x OC-CCI")
    print(f"Weighted log10 RMSE            : {weighted_log_rmse:.4f} decades")
    print(f"Mean monthly log-space corr    : {monthly_stats['corr_log'].mean():.3f}")


if __name__ == "__main__":
    model_monthly = load_and_aggregate_model(FEATURES_NC)
    occci = load_occci(OCCCI_NC)
    stats = match_and_compare(model_monthly, occci)
    stats.to_csv(OUT_CSV, index=False)
    print(f"\nMatched monthly log-space stats written to {OUT_CSV}")
    summarise(stats)