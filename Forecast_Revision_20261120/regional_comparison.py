from hashlib import sha256

import numpy as np
import pandas as pd

from forecast_analysis import ROOT, OUT, GFS_H, TARGET_H, degree, weights

IC = [f"icon_{h}" for h in range(14, 22)]
KEY = ["date", "city"]
SCORES = ["urban_max", "urban_18", "gfs_raw", "gfs_bias", "gfs_update", "icon_raw", "ceiling"]


def main():
    path = ROOT / "icon_d2"
    source = pd.read_csv(path/"sources.csv")
    for f in source.itertuples(index=False):
        assert sha256((path/f.file).read_bytes()).hexdigest() == f.sha256
    data = pd.read_csv(OUT/"nightly_forecast.csv.gz")
    obs = data.loc[data.year.eq(2026) & data.matched].merge(pd.read_csv(path/"points.csv"),
                                                         on=KEY, validate="one_to_one")
    obs = obs.dropna(subset=IC).copy()
    counts = obs.groupby("city").date.size().rename("matched_nights")
    counts.to_csv(OUT/"regional_cohort.csv")
    obs = obs.loc[obs.city.map(counts).ge(83)].copy()
    obs["ceiling"] = obs.target
    obs["icon_raw"] = degree(obs[IC].to_numpy())
    records = []
    for city, group in obs.groupby("city"):
        n = len(group)
        k = int(np.floor(.1*n+.5))
        heat = group.target.to_numpy()
        for name in SCORES:
            row = dict(city=city, score=name, n=n, slots=k, heat=heat.sum(),
                       captured=np.dot(heat, weights(group[name], k)))
            if name in ("gfs_raw", "icon_raw"):
                columns = GFS_H if name == "gfs_raw" else IC
                error = group[columns].to_numpy()-group[TARGET_H].to_numpy()
                row.update(sum_error=error.sum(), hourly_abs_error=np.abs(error).sum())
            records.append(row)
    by_city = pd.DataFrame(records)
    by_city.to_csv(OUT/"regional_city.csv", index=False)
    pooled = by_city.assign(region=np.where(by_city.city.eq("Vienna"), "Vienna", "Germany")).groupby(
        ["region","score"], as_index=False).sum(numeric_only=True, min_count=1)
    pooled["coverage_pct"] = 100*pooled.captured/pooled.heat
    pooled["mae_C"] = pooled.hourly_abs_error/(8*pooled.n)
    pooled["bias_C"] = pooled.sum_error/(8*pooled.n)
    pooled.to_csv(OUT/"regional_summary.csv", index=False)
    print(pooled[["region","score","n","slots","coverage_pct","mae_C","bias_C"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
