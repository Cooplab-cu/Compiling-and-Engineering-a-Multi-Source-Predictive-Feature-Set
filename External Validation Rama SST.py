"""
External SST validation: RAMA buoy vs. fused Bay of Bengal feature dataset
---------------------------------------------------------------------------
Matches daily RAMA buoy SST observations against the nearest grid cell /
date in features_YYYY.nc, restricted to buoys that fall inside the study
domain (78-100E, 5-25N), and computes bias / RMSE / correlation.

Usage:
    python validate_sst_rama.py

Edit the CONFIG block below to point at your local files.
"""

import numpy as np
import pandas as pd
import xarray as xr

# ------------------------- CONFIG -------------------------------------
RAMA_CSV = r"D:\PFZ_BoB_ML\data\raw\sst_data_2019.csv"
NC_PATH  = r"D:\PFZ_BoB_ML\data\features\features_2019.nc"

# Variable / coordinate names in the fused dataset (per manuscript: thetao -> sst)
MODEL_SST_VAR = "sst"
LAT_NAME      = "latitude"
LON_NAME      = "longitude"
TIME_NAME     = "time"

# Study domain bounds (must match manuscript Section 2.1)
LON_MIN, LON_MAX = 78.0, 100.0
LAT_MIN, LAT_MAX = 5.0, 25.0

OUT_CSV = r"D:\PFZ_BoB_ML\outputs\external_validation_rama\sst_validation_matched_2019.csv"
# ------------------------------------------------------------------------


def load_rama(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])

    # Drop flagged-missing obs
    if "missing" in df.columns:
        df = df[df["missing"] == 0]

    # Keep only buoys physically inside the study domain
    in_domain = (
        (df["longitude"] >= LON_MIN) & (df["longitude"] <= LON_MAX) &
        (df["latitude"] >= LAT_MIN) & (df["latitude"] <= LAT_MAX)
    )
    dropped = df.loc[~in_domain, ["latitude", "longitude"]].drop_duplicates()
    if len(dropped):
        print("Dropping out-of-domain buoy(s):")
        print(dropped.to_string(index=False))

    df = df[in_domain].reset_index(drop=True)
    kept = df[["latitude", "longitude"]].drop_duplicates()
    print(f"\nBuoys retained inside domain ({len(kept)}):")
    print(kept.to_string(index=False))
    return df


def match_to_grid(rama: pd.DataFrame, nc_path: str) -> pd.DataFrame:
    ds = xr.open_dataset(nc_path)

    if LAT_NAME not in ds.coords or LON_NAME not in ds.coords:
        raise KeyError(
            f"Could not find '{LAT_NAME}'/'{LON_NAME}' in {list(ds.coords)}. "
            "Update LAT_NAME/LON_NAME in CONFIG."
        )

    records = []
    for (lat, lon), group in rama.groupby(["latitude", "longitude"]):
        try:
            point = ds[MODEL_SST_VAR].sel(
                {LAT_NAME: lat, LON_NAME: lon}, method="nearest"
            )
        except Exception as e:
            print(f"Skipping buoy ({lat},{lon}): {e}")
            continue

        # Snap to actual grid cell used, for the record
        grid_lat = float(point[LAT_NAME].values)
        grid_lon = float(point[LON_NAME].values)

        for _, row in group.iterrows():
            try:
                model_val = float(
                    point.sel({TIME_NAME: row["date"]}, method="nearest").values
                )
            except Exception:
                continue

            records.append({
                "buoy_lat": lat,
                "buoy_lon": lon,
                "grid_lat": grid_lat,
                "grid_lon": grid_lon,
                "date": row["date"],
                "obs_sst": row["sst_celsius"],
                "model_sst": model_val,
            })

    ds.close()
    return pd.DataFrame(records)


def summarise(matched: pd.DataFrame) -> None:
    matched = matched.dropna(subset=["obs_sst", "model_sst"])
    diff = matched["model_sst"] - matched["obs_sst"]

    bias = diff.mean()
    rmse = np.sqrt((diff ** 2).mean())
    mae = diff.abs().mean()
    r = matched["obs_sst"].corr(matched["model_sst"])

    print("\n=== Overall SST validation (RAMA vs. fused dataset) ===")
    print(f"N matched obs : {len(matched)}")
    print(f"Bias (model-obs): {bias:+.3f} degC")
    print(f"RMSE            : {rmse:.3f} degC")
    print(f"MAE             : {mae:.3f} degC")
    print(f"Correlation (r) : {r:.3f}")

    print("\n=== Per-buoy breakdown ===")
    for (lat, lon), g in matched.groupby(["buoy_lat", "buoy_lon"]):
        d = g["model_sst"] - g["obs_sst"]
        print(
            f"  ({lat:.1f}N, {lon:.1f}E)  n={len(g):4d}  "
            f"bias={d.mean():+.3f}  rmse={np.sqrt((d**2).mean()):.3f}  "
            f"r={g['obs_sst'].corr(g['model_sst']):.3f}"
        )


if __name__ == "__main__":
    rama = load_rama(RAMA_CSV)
    matched = match_to_grid(rama, NC_PATH)
    matched.to_csv(OUT_CSV, index=False)
    print(f"\nMatched records written to {OUT_CSV}")
    summarise(matched)