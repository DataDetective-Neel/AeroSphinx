import pandas as pd, numpy as np
from load_insat_aod import load_insat_aod

def find_nearest(lat_pt, lon_pt, lat_arr, lon_arr, val_arr):
    # flatten and compute simple euclidean (approx)
    la, lo, va = lat_arr.flatten(), lon_arr.flatten(), val_arr.flatten()
    idx = np.argmin((la - lat_pt)**2 + (lo - lon_pt)**2)

    return float(va[idx])

def merge_ground_aod(pm_csv, aod_nc):
    df = pd.read_csv(pm_csv, parse_dates=['timestamp'])
    aod, latg, long = load_insat_aod(aod_nc)

    df['AOD'] = df.apply(
        lambda r: find_nearest(r.latitude, r.longitude, latg, long, aod),
        axis=1
    )
    return df

if __name__=="__main__":
    dfm = merge_ground_aod(
        "../data/ground_pm25_YYYYMMDD_HHMM.csv",
        "../data/sample_insat_aod.nc"
    )
    dfm.to_csv("../data/merged_pm_aod.csv", index=False)
    print("✅ merged → merged_pm_aod.csv")
