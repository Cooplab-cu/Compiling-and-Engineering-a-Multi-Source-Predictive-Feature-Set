import xarray as xr
from pathlib import Path
import logging
from tqdm import tqdm

# ====================== CONFIG ======================
PROCESSED_ROOT = Path(r"D:\PFZ_BoB_ML\data\processed")
COMBINED_ROOT = Path(r"D:\PFZ_BoB_ML\data\combined")
COMBINED_ROOT.mkdir(parents=True, exist_ok=True)

YEARS = range(2012, 2026)
# ====================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger(__name__)


def combine_year(year: int):
    """Merge physics + bgc + wind for one year"""

    physics_path = PROCESSED_ROOT / f"physics_{year}.nc"
    bgc_path = PROCESSED_ROOT / f"bgc_{year}.nc"
    wind_path = PROCESSED_ROOT / f"wind_{year}.nc"

    # Open datasets
    ds_phy = xr.open_dataset(physics_path)
    ds_bgc = xr.open_dataset(bgc_path)
    ds_wind = xr.open_dataset(wind_path)

    # Merge all variables
    ds = xr.merge([ds_phy, ds_bgc, ds_wind], compat="override")

    # Optional: add bathymetry (static)
    gebco_path = PROCESSED_ROOT / "gebco_bob.nc"
    if gebco_path.exists():
        ds_gebco = xr.open_dataset(gebco_path)
        ds["bathymetry"] = ds_gebco["bathymetry"]
        ds_gebco.close()

    # Save
    out_path = COMBINED_ROOT / f"combined_{year}.nc"
    ds.to_netcdf(out_path)

    # Close
    ds_phy.close()
    ds_bgc.close()
    ds_wind.close()
    ds.close()

    log.info(f"Combined {year} → {out_path.name}")
    return out_path


def main():
    log.info("========== COMBINING DATA STARTED ==========")

    for year in tqdm(YEARS, desc="Combining years"):
        combine_year(year)

    log.info("========== COMBINING DATA FINISHED ==========")
    log.info(f"Combined files saved in → {COMBINED_ROOT}")


if __name__ == "__main__":
    main()