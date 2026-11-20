from hashlib import md5, sha256

import numpy as np
import pandas as pd

from analysis import NIGHT, RAW, OUT, RULES, rank

L2 = "Freiburg_AWS_20220901_20230831_gap_filled_data_ta_rh_Plein_et_al.csv"
META = "Freiburg-Street-Level-Weather-Station-Network-MetaData-V1-0.csv"


def main():
    dwd = pd.read_csv(OUT / "nightly.csv.gz", parse_dates=["date"])
    dwd = dwd.loc[dwd.city.eq("Freiburg") & dwd.year.eq(2023), ["date", *RULES[:-1]]]
    assert md5((RAW / L2).read_bytes()).hexdigest() == "840a2f677d43b1298f50f40f0a250d98"
    meta = pd.read_csv(RAW / META)
    meta["lcz"] = pd.to_numeric(meta.local_climate_zone.str.extract(r"^(\d+)", expand=False),
                               errors="coerce")
    built = meta.loc[meta.lcz.between(1, 10)]
    names = ["utc", "station_id", "variable", "value", "type"]
    raw = pd.read_csv(RAW / L2, names=names, header=0)
    observed = raw.loc[raw["variable"].eq("Ta_deg_C") & raw["type"].eq("observed") &
                       raw.station_id.isin(built.station_id) & raw.value.between(-30, 50)].copy()
    timestamp = pd.DatetimeIndex(pd.to_datetime(observed.utc, utc=True))
    observed["local"] = timestamp.tz_convert("Europe/Berlin").tz_localize(None)
    observed = observed.loc[observed.local.between("2023-06-01", "2023-09-01 06:00")]
    wide = observed.pivot(index="local", columns="station_id", values="value")
    times = pd.DatetimeIndex((dwd.date.to_numpy()[:, None] +
                              np.array(NIGHT, dtype="timedelta64[h]")).ravel())
    rows, details = [], []
    for station in sorted(built.station_id):
        group = dwd.copy()
        if station in wide:
            values = wide[station].reindex(times).to_numpy().reshape(len(dwd), len(NIGHT))
            group["target"] = np.maximum(values - 20, 0).sum(axis=1)
        else:
            group["target"] = np.nan
        group = group.dropna(subset=["target"]).copy()
        group["oracle"] = group.target
        row = dict(station_id=station, complete=len(group), included=len(group) >= 83)
        if row["included"]:
            _, result = rank(group, .10)
            row.update(result)
            group["station_id"] = station
            details.append(group)
        rows.append(row)
    study = pd.DataFrame(rows).merge(built[["station_id", "station_name", "lcz",
                                           "station_lat", "station_lon", "station_elevation"]],
                                     on="station_id", validate="one_to_one")
    for key in RULES:
        study[key + "_pct"] = 100 * study[key] / study.burden
    study["urban_timing"] = study.urban_18_pct - study.urban_max_pct
    study["evening_siting"] = study.urban_18_pct - study.ref_18_pct
    study.to_csv(OUT / "freiburg_sites.csv", index=False)
    pd.concat(details).to_csv(OUT / "freiburg_nightly.csv.gz", index=False,
                              compression={"method": "gzip", "mtime": 0})
    manifest = [
        (L2, f"https://zenodo.org/records/12732565/files/{L2}"),
        (META, f"https://zenodo.org/records/12732552/files/{META}"),
    ]
    pd.DataFrame([{"file": name, "url": url, "sha256": sha256((RAW / name).read_bytes()).hexdigest()}
                  for name, url in manifest]).to_csv(OUT / "freiburg_sources.csv", index=False)
    print(study.loc[study.included, ["station_id", "complete", "lcz", "burden", *[
        r + "_pct" for r in RULES], "urban_timing", "evening_siting"]].round(2).to_string(index=False))
    print(f"Included {study.included.sum()} of {len(study)} metadata sites in built LCZs")


if __name__ == "__main__":
    main()
