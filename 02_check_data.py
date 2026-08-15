import xarray as xr
from pathlib import Path
import logging

# ================= CONFIG =================
ROOT = Path(r"D:\PFZ_BoB_ML\data\raw\copernicus")

DATASETS = {
    "physics": {
        "folder": "physics",
        "prefix": "physics_",
        "expected_vars": ["thetao", "zos", "uo", "vo"]
    },
    "bgc": {
        "folder": "bgc",
        "prefix": "bgc_",
        "expected_vars": ["chl", "no3", "po4", "o2"]
    },
    "wind": {
        "folder": "wind",
        "prefix": "wind_",
        "expected_vars": ["eastward_wind", "northward_wind"]
    },
    "gebco": {
        "folder": "gebco",
        "prefix": "gebco_bob",
        "expected_vars": ["elevation"]
    }
}

YEARS = range(2012, 2026)
# ==========================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger(__name__)


def check_file(path: Path, expected_vars: list):
    try:
        ds = xr.open_dataset(path)
        log.info(f"OK → {path.name}")
        log.info(f"   Dimensions : {dict(ds.dims)}")
        log.info(f"   Variables  : {list(ds.data_vars)}")

        missing = [v for v in expected_vars if v not in ds.data_vars]
        if missing:
            log.warning(f"   Missing variables: {missing}")
        else:
            log.info("   All expected variables present")

        ds.close()
        return True
    except Exception as e:
        log.error(f"FAILED → {path.name} | {e}")
        return False


def main():
    log.info("========== DATA CHECK STARTED ==========")

    for name, cfg in DATASETS.items():
        log.info(f"\n=== Checking {name.upper()} ===")
        folder = ROOT / cfg["folder"]

        if not folder.exists():
            log.error(f"Folder not found: {folder}")
            continue

        if name == "gebco":
            # Special case for GEBCO
            files = list(folder.glob("gebco_bob*.nc"))
            if not files:
                log.error("GEBCO file not found!")
            else:
                for f in files:
                    check_file(f, cfg["expected_vars"])
        else:
            for year in YEARS:
                fname = f"{cfg['prefix']}{year}.nc"
                fpath = folder / fname
                if fpath.exists():
                    check_file(fpath, cfg["expected_vars"])
                else:
                    log.warning(f"Missing file: {fname}")

    log.info("\n========== DATA CHECK FINISHED ==========")


if __name__ == "__main__":
    main()