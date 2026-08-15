import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from tqdm import tqdm

# ====================== CONFIG ======================
COMBINED_ROOT = Path(r"D:\PFZ_BoB_ML\data\combined")
FEATURE_ROOT = Path(r"D:\PFZ_BoB_ML\data\features")
CLIMATE_FILE = Path(r"D:\PFZ_BoB_ML\data\raw\climate_indices\climate_indices_monthly.csv")

FEATURE_ROOT.mkdir(parents=True, exist_ok=True)

YEARS = range(2012, 2026)

ANOMALY_VARS = ["sst", "ssh", "chl", "log_chl", "current_speed", "wind_speed"]
# ====================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger(__name__)


def load_climate_indices():
    """Load monthly Niño 3.4 and DMI"""
    if not CLIMATE_FILE.exists():
        raise FileNotFoundError(f"Climate indices file not found: {CLIMATE_FILE}")

    clim = pd.read_csv(CLIMATE_FILE)
    clim = clim.set_index(["year", "month"])
    log.info(f"Climate indices loaded → {len(clim)} months")
    return clim


def compute_climatology():
    """Compute monthly climatology for Z-score anomalies"""
    log.info("Computing climatology for Z-score anomalies ...")

    clim_list = []
    for year in tqdm(YEARS, desc="Loading for climatology"):
        path = COMBINED_ROOT / f"combined_{year}.nc"
        if not path.exists():
            continue

        ds = xr.open_dataset(path)

        if "log_chl" not in ds and "chl" in ds:
            ds["log_chl"] = np.log10(ds["chl"].where(ds["chl"] > 0))
        if "current_speed" not in ds:
            ds["current_speed"] = np.sqrt(ds["u_current"] ** 2 + ds["v_current"] ** 2)
        if "wind_speed" not in ds:
            ds["wind_speed"] = np.sqrt(ds["eastward_wind"] ** 2 + ds["northward_wind"] ** 2)

        keep = [v for v in ANOMALY_VARS if v in ds]
        ds = ds[keep]
        clim_list.append(ds)
        ds.close()

    full = xr.concat(clim_list, dim="time")
    monthly_mean = full.groupby("time.month").mean("time")
    monthly_std = full.groupby("time.month").std("time")
    monthly_std = monthly_std.where(monthly_std > 1e-6, 1e-6)

    log.info("Climatology computed successfully.")
    return monthly_mean, monthly_std


def add_features(ds: xr.Dataset, monthly_mean, monthly_std, clim_df) -> xr.Dataset:
    """Add all features including anomalies + climate indices"""

    # -------------------------------------------------
    # 1. Basic physical derived features
    # -------------------------------------------------
    ds["current_speed"] = np.sqrt(ds["u_current"] ** 2 + ds["v_current"] ** 2)

    Cd = 0.0013
    rho_air = 1.225
    ds["wind_stress"] = Cd * rho_air * ds["wind_speed"] ** 2

    # -------------------------------------------------
    # 2. Gradients (fronts)
    # -------------------------------------------------
    ds["sst_grad_x"] = ds["sst"].differentiate("longitude")
    ds["sst_grad_y"] = ds["sst"].differentiate("latitude")
    ds["sst_gradient"] = np.sqrt(ds["sst_grad_x"] ** 2 + ds["sst_grad_y"] ** 2)

    ds["chl_grad_x"] = ds["chl"].differentiate("longitude")
    ds["chl_grad_y"] = ds["chl"].differentiate("latitude")
    ds["chl_gradient"] = np.sqrt(ds["chl_grad_x"] ** 2 + ds["chl_grad_y"] ** 2)

    ds["sst_front"] = (ds["sst_gradient"] > ds["sst_gradient"].quantile(0.85, skipna=True)).astype("float32")

    # -------------------------------------------------
    # 3. Log chlorophyll
    # -------------------------------------------------
    ds["log_chl"] = np.log10(ds["chl"].where(ds["chl"] > 0))

    # -------------------------------------------------
    # 4. Bathymetry category
    # -------------------------------------------------
    ds["depth_category"] = xr.where(
        ds["bathymetry"] < 50, 1,
        xr.where(ds["bathymetry"] < 200, 2, 3)
    )

    # -------------------------------------------------
    # 5. Coordinates
    # -------------------------------------------------
    ds["lat"] = ds["latitude"]
    ds["lon"] = ds["longitude"]

    # -------------------------------------------------
    # 6. Temporal features
    # -------------------------------------------------
    time = ds["time"]
    ds["year"] = time.dt.year
    ds["month"] = time.dt.month
    ds["dayofyear"] = time.dt.dayofyear
    ds["sin_doy"] = np.sin(2 * np.pi * ds["dayofyear"] / 365.25)
    ds["cos_doy"] = np.cos(2 * np.pi * ds["dayofyear"] / 365.25)

    # -------------------------------------------------
    # 7. Threshold flags
    # -------------------------------------------------
    ds["high_chl_flag"] = (ds["chl"] > ds["chl"].quantile(0.75, skipna=True)).astype("float32")
    ds["optimal_sst_flag"] = ((ds["sst"] > 26.5) & (ds["sst"] < 29.0)).astype("float32")

    # -------------------------------------------------
    # 8. Z-score Anomalies
    # -------------------------------------------------
    for var in ANOMALY_VARS:
        if var not in ds or var not in monthly_mean:
            continue

        months = ds["time"].dt.month
        mean = monthly_mean[var].sel(month=months).drop_vars("month", errors="ignore")
        std = monthly_std[var].sel(month=months).drop_vars("month", errors="ignore")

        mean = mean.transpose(*ds[var].dims)
        std = std.transpose(*ds[var].dims)

        ds[f"{var}_anomaly"] = (ds[var] - mean) / std

    # -------------------------------------------------
    # 9. Climate Indices (Niño 3.4 + DMI)  ← NEW
    # -------------------------------------------------
    years = ds["time"].dt.year.values
    months = ds["time"].dt.month.values

    nino_vals = []
    dmi_vals = []

    for y, m in zip(years, months):
        try:
            nino_vals.append(clim_df.loc[(y, m), "nino34"])
            dmi_vals.append(clim_df.loc[(y, m), "dmi"])
        except KeyError:
            nino_vals.append(np.nan)
            dmi_vals.append(np.nan)

    ds["nino34"] = xr.DataArray(nino_vals, dims=["time"], coords={"time": ds["time"]})
    ds["dmi"] = xr.DataArray(dmi_vals, dims=["time"], coords={"time": ds["time"]})

    return ds


def process_year(year, monthly_mean, monthly_std, clim_df):
    input_path = COMBINED_ROOT / f"combined_{year}.nc"
    output_path = FEATURE_ROOT / f"features_{year}.nc"

    if not input_path.exists():
        log.warning(f"Missing: {input_path.name}")
        return

    log.info(f"Processing {year} ...")
    ds = xr.open_dataset(input_path)
    ds = add_features(ds, monthly_mean, monthly_std, clim_df)

    keep_vars = [
        # Coordinates
        "lat", "lon",
        # Temporal
        "year", "month", "dayofyear", "sin_doy", "cos_doy",
        # Physical
        "sst", "ssh", "u_current", "v_current", "current_speed",
        "chl", "log_chl", "no3", "po4", "o2",
        "eastward_wind", "northward_wind", "wind_speed", "wind_stress",
        "sst_gradient", "chl_gradient", "bathymetry",
        # Flags
        "sst_front", "depth_category", "high_chl_flag", "optimal_sst_flag",
        # Anomalies
        "sst_anomaly", "ssh_anomaly", "chl_anomaly", "log_chl_anomaly",
        "current_speed_anomaly", "wind_speed_anomaly",
        # Climate Indices
        "nino34", "dmi"
    ]

    keep_vars = [v for v in keep_vars if v in ds]
    ds = ds[keep_vars]

    ds.to_netcdf(output_path)
    ds.close()
    log.info(f"Saved → {output_path.name}  |  Features: {len(keep_vars)}")


def main():
    log.info("========== FEATURE ENGINEERING STARTED ==========")
    log.info("Full version: Physical + Anomalies + Temporal + Climate Indices (Niño3.4 + DMI)")

    clim_df = load_climate_indices()
    monthly_mean, monthly_std = compute_climatology()

    for year in tqdm(YEARS, desc="Feature Engineering"):
        process_year(year, monthly_mean, monthly_std, clim_df)

    log.info("========== FEATURE ENGINEERING FINISHED ==========")
    log.info(f"Feature files saved in → {FEATURE_ROOT}")
    log.info("Objective 1 (27+ feature set including climate indices) is now complete.")


if __name__ == "__main__":
    main()