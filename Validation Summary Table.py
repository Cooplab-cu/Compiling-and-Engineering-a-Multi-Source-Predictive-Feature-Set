"""
Combine RAMA SST validation and OC-CCI chlorophyll validation (linear +
log-space) results into a single manuscript-ready summary table.

Reads the three matched-record CSVs already produced by:
  - validate_sst_rama.py       -> sst_validation_matched_2019.csv (per-obs)
  - validate_chl_occci.py      -> chl_validation_matched_2019.csv (per-month, linear)
  - validate_chl_occci_log.py  -> chl_validation_log_matched_2019.csv (per-month, log10)

Writes one summary CSV with one row per variable/space, ready to paste
into a Word table or LaTeX table for the manuscript's validation section.

Usage:
    python build_validation_summary_table.py
"""

import numpy as np
import pandas as pd

# ------------------------- CONFIG -------------------------------------
OUT_DIR = r"D:\PFZ_BoB_ML\outputs\external_validation_rama"

SST_CSV      = OUT_DIR + r"\sst_validation_matched_2019.csv"
CHL_LIN_CSV  = OUT_DIR + r"\chl_validation_matched_2019.csv"
CHL_LOG_CSV  = OUT_DIR + r"\chl_validation_log_matched_2019.csv"

OUT_TABLE = OUT_DIR + r"\Table_ExternalValidation_Summary_2019.csv"
# ------------------------------------------------------------------------


def summarise_sst(path: str) -> dict:
    df = pd.read_csv(path).dropna(subset=["obs_sst", "model_sst"])
    diff = df["model_sst"] - df["obs_sst"]
    n_buoys = df[["buoy_lat", "buoy_lon"]].drop_duplicates().shape[0]

    return {
        "Variable": "Sea surface temperature",
        "Reference dataset": "RAMA moorings",
        "Space": "Linear (degC)",
        "N": len(df),
        "Coverage": f"{n_buoys} buoys, daily, 2019",
        "Bias": round(diff.mean(), 3),
        "RMSE": round(np.sqrt((diff ** 2).mean()), 3),
        "Correlation (r)": round(df["obs_sst"].corr(df["model_sst"]), 3),
        "Notes": "Bias/RMSE in degC",
    }


def summarise_chl_linear(path: str) -> dict:
    df = pd.read_csv(path)
    total_n = df["n_cells"].sum()
    weighted_bias = (df["bias"] * df["n_cells"]).sum() / total_n
    weighted_rmse = np.sqrt((df["rmse"] ** 2 * df["n_cells"]).sum() / total_n)

    return {
        "Variable": "Chlorophyll-a",
        "Reference dataset": "OC-CCI (CMEMS 009_108)",
        "Space": "Linear (mg m-3)",
        "N": int(total_n),
        "Coverage": f"{len(df)} months, gridded, 2019",
        "Bias": round(weighted_bias, 3),
        "RMSE": round(weighted_rmse, 3),
        "Correlation (r)": round(df["corr"].mean(), 3),
        "Notes": "Large relative error typical of linear-space chl comparison",
    }


def summarise_chl_log(path: str) -> dict:
    df = pd.read_csv(path)
    total_n = df["n_cells"].sum()
    weighted_log_bias = (df["log_bias"] * df["n_cells"]).sum() / total_n
    weighted_log_rmse = np.sqrt((df["log_rmse"] ** 2 * df["n_cells"]).sum() / total_n)
    mult_factor = 10 ** weighted_log_bias

    return {
        "Variable": "Chlorophyll-a",
        "Reference dataset": "OC-CCI (CMEMS 009_108)",
        "Space": "log10 (decades)",
        "N": int(total_n),
        "Coverage": f"{len(df)} months, gridded, 2019",
        "Bias": round(weighted_log_bias, 3),
        "RMSE": round(weighted_log_rmse, 3),
        "Correlation (r)": round(df["corr_log"].mean(), 3),
        "Notes": f"Equivalent multiplicative bias: model = {mult_factor:.2f}x OC-CCI",
    }


if __name__ == "__main__":
    rows = [
        summarise_sst(SST_CSV),
        summarise_chl_linear(CHL_LIN_CSV),
        summarise_chl_log(CHL_LOG_CSV),
    ]
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_TABLE, index=False)

    print(f"Summary table written to {OUT_TABLE}\n")
    print(summary.to_string(index=False))