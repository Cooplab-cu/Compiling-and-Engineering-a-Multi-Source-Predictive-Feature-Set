from pathlib import Path
import copernicusmarine
import logging

OUTPUT_ROOT = Path(r"D:\PFZ_BoB_ML\data\raw\copernicus")
BBOX = dict(
    minimum_longitude=78.0,
    maximum_longitude=100.0,
    minimum_latitude=5.0,
    maximum_latitude=25.0,
)
START_YEAR, END_YEAR = 2012, 2025
DEPTH = (0.0, 1.0)

DATASETS = {
    "physics": {
        "dataset_id": "cmems_mod_glo_phy_my_0.083deg_P1D-m",
        "variables": ["thetao", "zos", "uo", "vo"],
        "subdir": "physics",
    },
    "bgc": {
        "dataset_id": "cmems_mod_glo_bgc_my_0.25deg_P1D-m",
        "variables": ["chl", "no3", "po4", "o2"],
        "subdir": "bgc",
    },
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger(__name__)

def year_range(y):
    return f"{y}-01-01T00:00:00", f"{y}-12-31T23:59:59"

def download_one(cfg, start, end):
    out_dir = OUTPUT_ROOT / cfg["subdir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    year = start[:4]
    fname = f"{cfg['subdir']}_{year}.nc"
    out_path = out_dir / fname

    if out_path.exists():
        log.info(f"SKIP {out_path}")
        return

    log.info(f"Downloading {cfg['subdir']} {year} ...")
    copernicusmarine.subset(
        dataset_id=cfg["dataset_id"],
        variables=cfg["variables"],
        start_datetime=start,
        end_datetime=end,
        **BBOX,
        minimum_depth=DEPTH[0],
        maximum_depth=DEPTH[1],
        output_directory=str(out_dir),
        output_filename=fname,
        force_download=True,
    )
    log.info(f"OK -> {out_path}")

def main():
    for name, cfg in DATASETS.items():
        log.info(f"=== {name} ===")
        for y in range(START_YEAR, END_YEAR + 1):
            start, end = year_range(y)
            download_one(cfg, start, end)

if __name__ == "__main__":
    main()
    from pathlib import Path
    import logging
    import requests
    import xarray as xr
    import copernicusmarine

    # ====================== CONFIG ======================
    OUTPUT_ROOT = Path(r"D:\PFZ_BoB_ML\data\raw\copernicus")
    BBOX = dict(
        minimum_longitude=78.0,
        maximum_longitude=100.0,
        minimum_latitude=5.0,
        maximum_latitude=25.0,
    )
    START_YEAR, END_YEAR = 2012, 2025

    # ---------- WIND ----------
    # Best multi-year product currently available (hourly L4, 0.125°)
    # Variables: eastward_wind, northward_wind (stress-equivalent)
    WIND_CFG = {
        "dataset_id": "cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H",  # or "WIND_GLO_PHY_L4_MY_012_006"
        "variables": ["eastward_wind", "northward_wind"],
        "subdir": "wind",
    }

    # ---------- GEBCO ----------
    GEBCO_DIR = OUTPUT_ROOT / "gebco"
    GEBCO_DIR.mkdir(parents=True, exist_ok=True)
    # ====================================================

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    log = logging.getLogger(__name__)


    def year_range(y):
        return f"{y}-01-01T00:00:00", f"{y}-12-31T23:59:59"


    def download_wind():
        """Download yearly wind files (same pattern as physics/bgc)."""
        cfg = WIND_CFG
        out_dir = OUTPUT_ROOT / cfg["subdir"]
        out_dir.mkdir(parents=True, exist_ok=True)

        log.info("=== wind ===")
        for y in range(START_YEAR, END_YEAR + 1):
            start, end = year_range(y)
            fname = f"wind_{y}.nc"
            out_path = out_dir / fname

            if out_path.exists():
                log.info(f"SKIP {out_path}")
                continue

            log.info(f"Downloading wind {y} ...")
            try:
                copernicusmarine.subset(
                    dataset_id=cfg["dataset_id"],
                    variables=cfg["variables"],
                    start_datetime=start,
                    end_datetime=end,
                    **BBOX,
                    output_directory=str(out_dir),
                    output_filename=fname,
                    force_download=True,
                )
                log.info(f"OK -> {out_path}")
            except Exception as e:
                log.error(f"Failed wind {y}: {e}")
                # Fallback to the official product ID if the short name fails
                log.info("Trying alternative dataset_id ...")
                copernicusmarine.subset(
                    dataset_id="WIND_GLO_PHY_L4_MY_012_006",
                    variables=cfg["variables"],
                    start_datetime=start,
                    end_datetime=end,
                    **BBOX,
                    output_directory=str(out_dir),
                    output_filename=fname,
                    force_download=True,
                )
                log.info(f"OK -> {out_path}")


    from pathlib import Path
    import logging
    import xarray as xr
    import requests
    from tqdm import tqdm

    OUTPUT_ROOT = Path(r"D:\PFZ_BoB_ML\data\raw\copernicus")
    GEBCO_DIR = OUTPUT_ROOT / "gebco"
    GEBCO_DIR.mkdir(parents=True, exist_ok=True)

    BBOX = {
        "min_lon": 78.0,
        "max_lon": 100.0,
        "min_lat": 5.0,
        "max_lat": 25.0,
    }

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    log = logging.getLogger(__name__)


    def download_gebco():
        target = GEBCO_DIR / "gebco_bob_15arcsec.nc"
        if target.exists():
            log.info(f"SKIP {target}")
            return

        log.info("=== GEBCO (automatic) ===")

        # Current CEDA / OPeNDAP candidates (2026 → 2025 → 2024)
        candidates = [
            # GEBCO 2026 ice surface
            "https://dap.ceda.ac.uk/bodc/gebco/global/gebco_2026/ice_surface_elevation/netcdf/GEBCO_2026.nc",
            # GEBCO 2025
            "https://dap.ceda.ac.uk/bodc/gebco/global/gebco_2025/ice_surface_elevation/netcdf/GEBCO_2025.nc",
            # GEBCO 2024
            "https://dap.ceda.ac.uk/bodc/gebco/global/gebco_2024/ice_surface_elevation/netcdf/GEBCO_2024.nc",
        ]

        for url in candidates:
            try:
                log.info(f"Trying: {url}")
                # Open remotely and subset on the fly (no full download)
                ds = xr.open_dataset(url)

                # GEBCO variable is usually called 'elevation'
                var = "elevation" if "elevation" in ds else list(ds.data_vars)[0]

                subset = ds[var].sel(
                    lon=slice(BBOX["min_lon"], BBOX["max_lon"]),
                    lat=slice(BBOX["min_lat"], BBOX["max_lat"]),
                )

                # Save
                subset.to_netcdf(target)
                log.info(f"SUCCESS → {target}")
                log.info(f"Shape: {subset.shape}")
                return

            except Exception as e:
                log.warning(f"Failed: {e}")
                continue

        # ---------- Fallback: download global zip then crop ----------
        log.info("OPeNDAP failed. Trying global zip download + local crop ...")
        zip_url = "https://dap.ceda.ac.uk/bodc/gebco/global/gebco_2026/ice_surface_elevation/netcdf/GEBCO_2026.zip?download=1"
        zip_path = GEBCO_DIR / "GEBCO_2026.zip"

        try:
            log.info("Downloading global GEBCO_2026 (~4 GB) ... this takes time")
            with requests.get(zip_url, stream=True) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                with open(zip_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as pbar:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))

            # Extract and subset
            import zipfile
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(GEBCO_DIR)

            nc_file = next(GEBCO_DIR.glob("**/GEBCO_2026*.nc"))
            ds = xr.open_dataset(nc_file)
            var = "elevation" if "elevation" in ds else list(ds.data_vars)[0]
            subset = ds[var].sel(
                lon=slice(BBOX["min_lon"], BBOX["max_lon"]),
                lat=slice(BBOX["min_lat"], BBOX["max_lat"]),
            )
            subset.to_netcdf(target)
            log.info(f"SUCCESS → {target}")

            # Clean up big files
            zip_path.unlink(missing_ok=True)
            nc_file.unlink(missing_ok=True)

        except Exception as e:
            log.error(f"All automatic methods failed: {e}")
            log.info("Please download manually from https://download.gebco.net/")
            log.info(f"Save as → {target}")


    if __name__ == "__main__":
        download_gebco()

        import requests
        import pandas as pd
        from pathlib import Path
        import logging

        # ====================== CONFIG ======================
        OUTPUT_DIR = Path(r"D:\PFZ_BoB_ML\data\raw\climate_indices")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        NINO34_URL = "https://psl.noaa.gov/data/correlation/nina34.data"
        DMI_URL = "https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/dmi.had.long.data"
        # ====================================================

        logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
        log = logging.getLogger(__name__)


        def download_nino34():
            log.info("Downloading Niño 3.4 ...")
            r = requests.get(NINO34_URL)
            r.raise_for_status()

            lines = r.text.strip().split("\n")

            # Skip header lines until we find the first year
            data = []
            for line in lines:
                parts = line.split()
                if len(parts) >= 13 and parts[0].isdigit():
                    year = int(parts[0])
                    for month, val in enumerate(parts[1:13], start=1):
                        try:
                            v = float(val)
                            if v > -90:  # missing values are usually -99.99
                                data.append({"year": year, "month": month, "nino34": v})
                        except:
                            continue

            df = pd.DataFrame(data)
            out = OUTPUT_DIR / "nino34_monthly.csv"
            df.to_csv(out, index=False)
            log.info(f"Saved → {out}  ({len(df)} rows)")
            return df


        def download_dmi():
            log.info("Downloading DMI (IOD) ...")
            r = requests.get(DMI_URL)
            r.raise_for_status()

            lines = r.text.strip().split("\n")

            data = []
            for line in lines:
                parts = line.split()
                if len(parts) >= 13 and parts[0].isdigit():
                    year = int(parts[0])
                    for month, val in enumerate(parts[1:13], start=1):
                        try:
                            v = float(val)
                            if v > -90:
                                data.append({"year": year, "month": month, "dmi": v})
                        except:
                            continue

            df = pd.DataFrame(data)
            out = OUTPUT_DIR / "dmi_monthly.csv"
            df.to_csv(out, index=False)
            log.info(f"Saved → {out}  ({len(df)} rows)")
            return df


        def main():
            log.info("========== Downloading Climate Indices ==========")
            nino = download_nino34()
            dmi = download_dmi()

            # Merge both into one clean file
            merged = pd.merge(nino, dmi, on=["year", "month"], how="outer")
            merged = merged.sort_values(["year", "month"]).reset_index(drop=True)

            out = OUTPUT_DIR / "climate_indices_monthly.csv"
            merged.to_csv(out, index=False)
            log.info(f"Merged file → {out}")
            log.info(merged.tail(12))  # show latest year
            log.info("========== DONE ==========")


        if __name__ == "__main__":
            main()