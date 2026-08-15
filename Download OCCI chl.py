"""
Download OC-CCI (ESA-CCI-based) chlorophyll-a from the Copernicus Marine
Service for the Bay of Bengal domain, 2019, using the same
`copernicusmarine` toolbox already used for your physics/bgc/wind pulls.

Product: OCEANCOLOUR_GLO_BGC_L4_MY_009_108 (Brockmann Consult, ESA-CCI inputs)
Dataset: c3s_obs-oc_glo_bgc-plankton_my_l4-multi-4km_P1M  (monthly, 4 km)
Variable: CHL

Note: this MY dataset is MONTHLY, not daily -- OC-CCI's daily gap-free
product is not distributed under this exact dataset id at time of writing.
For a like-for-like comparison against your daily fused `chl` field, either:
  (a) compare against monthly OC-CCI climatology (aggregate your daily chl
      to monthly means first), or
  (b) if you specifically need daily OC-CCI, download directly from the
      CEDA archive (https://data.ceda.ac.uk/neodc/esacci/ocean_colour/data)
      instead of CMEMS.
This script does (a), which is simpler and reuses your existing CMEMS login.

Requires: `copernicusmarine login` already run once (same credentials you
use for physics/bgc/wind), so no username/password needed here.
"""

import copernicusmarine

# ------------------------- CONFIG -------------------------------------
DATASET_ID = "c3s_obs-oc_glo_bgc-plankton_my_l4-multi-4km_P1M"
VARIABLES = ["CHL"]

# Study domain bounds (must match manuscript Section 2.1)
LON_MIN, LON_MAX = 78.0, 100.0
LAT_MIN, LAT_MAX = 5.0, 25.0

START_DATETIME = "2019-01-01T00:00:00"
END_DATETIME   = "2019-12-31T23:59:59"

OUTPUT_DIR = r"D:\PFZ_BoB_ML\data\external\oc_cci"
OUTPUT_FILENAME = "occci_chl_bob_2019_monthly.nc"
# ------------------------------------------------------------------------


if __name__ == "__main__":
    copernicusmarine.subset(
        dataset_id=DATASET_ID,
        variables=VARIABLES,
        minimum_longitude=LON_MIN,
        maximum_longitude=LON_MAX,
        minimum_latitude=LAT_MIN,
        maximum_latitude=LAT_MAX,
        start_datetime=START_DATETIME,
        end_datetime=END_DATETIME,
        output_directory=OUTPUT_DIR,
        output_filename=OUTPUT_FILENAME,
    )
    print(f"Done. Saved to {OUTPUT_DIR}\\{OUTPUT_FILENAME}")