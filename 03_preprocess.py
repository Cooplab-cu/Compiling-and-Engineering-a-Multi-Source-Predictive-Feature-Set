import xarray as xr
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm

# ====================== CONFIG ======================
RAW_ROOT = Path(r"D:\PFZ_BoB_ML\data\raw\copernicus")
PROCESSED_ROOT = Path(r"D:\PFZ_BoB_ML\data\processed")
PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)

YEARS = range(2012, 2026)

# Target grid = Physics grid (highest resolution among the 3D/2D products)
TARGET_LAT = None
TARGET_LON = None
# ====================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger(__name__)


def get_target_grid():
    """Use the Physics grid as the common target grid"""
    global TARGET_LAT, TARGET_LON
    sample = xr.open_dataset(RAW_ROOT / "physics" / "physics_2020.nc")
    TARGET_LAT = sample.latitude
    TARGET_LON = sample.longitude
    sample.close()
    log.info(f"Target grid set → lat: {len(TARGET_LAT)}, lon: {len(TARGET_LON)}")


def process_physics(year: int):
    path = RAW_ROOT / "physics" / f"physics_{year}.nc"
    ds = xr.open_dataset(path)

    # Keep only needed variables and squeeze depth
    ds = ds[["thetao", "zos", "uo", "vo"]].squeeze("depth", drop=True)

    # Rename for clarity
    ds = ds.rename({
        "thetao": "sst",
        "zos": "ssh",
        "uo": "u_current",
        "vo": "v_current"
    })

    out = PROCESSED_ROOT / f"physics_{year}.nc"
    ds.to_netcdf(out)
    ds.close()
    log.info(f"Physics {year} → {out.name}")


def process_bgc(year: int):
    path = RAW_ROOT / "bgc" / f"bgc_{year}.nc"
    ds = xr.open_dataset(path)

    ds = ds[["chl", "no3", "po4", "o2"]].squeeze("depth", drop=True)

    # Regrid to Physics grid
    ds = ds.interp(latitude=TARGET_LAT, longitude=TARGET_LON, method="linear")

    out = PROCESSED_ROOT / f"bgc_{year}.nc"
    ds.to_netcdf(out)
    ds.close()
    log.info(f"BGC {year} → {out.name}")


def process_wind(year: int):
    path = RAW_ROOT / "wind" / f"wind_{year}.nc"
    ds = xr.open_dataset(path)

    # Convert hourly → daily mean
    ds_daily = ds.resample(time="1D").mean()

    # Regrid to Physics grid
    ds_daily = ds_daily.interp(latitude=TARGET_LAT, longitude=TARGET_LON, method="linear")

    # Calculate wind speed
    ds_daily["wind_speed"] = np.sqrt(ds_daily.eastward_wind**2 + ds_daily.northward_wind**2)

    out = PROCESSED_ROOT / f"wind_{year}.nc"
    ds_daily.to_netcdf(out)
    ds.close()
    log.info(f"Wind {year} → {out.name}")


def process_gebco():
    path = RAW_ROOT / "gebco" / "gebco_bob_15arcsec.nc"
    ds = xr.open_dataset(path)

    # Rename coordinates if needed
    if "lat" in ds.coords:
        ds = ds.rename({"lat": "latitude", "lon": "longitude"})

    # Regrid to Physics grid
    ds = ds.interp(latitude=TARGET_LAT, longitude=TARGET_LON, method="linear")

    # Keep only elevation and make depth positive downward
    ds["bathymetry"] = -ds["elevation"]   # convert to positive depth
    ds = ds[["bathymetry"]]

    out = PROCESSED_ROOT / "gebco_bob.nc"
    ds.to_netcdf(out)
    ds.close()
    log.info(f"GEBCO → {out.name}")


def main():
    log.info("========== PREPROCESSING STARTED ==========")

    get_target_grid()

    # Process GEBCO once
    process_gebco()

    for year in tqdm(YEARS, desc="Processing years"):
        process_physics(year)
        process_bgc(year)
        process_wind(year)

    log.info("========== PREPROCESSING FINISHED ==========")
    log.info(f"Processed files saved in → {PROCESSED_ROOT}")


if __name__ == "__main__":
    main()