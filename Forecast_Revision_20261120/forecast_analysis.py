from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
V2 = ROOT / "base" if (ROOT / "base" / "review_analysis.py").is_file() else ROOT.parent / "Revision_20261120"
BASE = V2 / "original" if (V2 / "original" / "analysis.py").is_file() else ROOT.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(BASE))
from analysis import DAY, NIGHT, weights

OUT = ROOT / "results"
DATES = pd.DatetimeIndex([d for y in (2024, 2025, 2026)
                          for d in pd.date_range(f"{y}-06-01", f"{y}-08-31")])
HOURS = DAY + NIGHT
LEADS = (10, *range(14, 22))
TARGET_H = [f"t_{h}" for h in NIGHT]
GFS_H = [f"gfs_{h}" for h in range(14, 22)]
SCORES = ["urban_max", "urban_18", "gfs_raw", "gfs_bias", "gfs_update", "ceiling"]
KEY = ["city", "date"]


def sha_file(path, digest):
    assert sha256(path.read_bytes()).hexdigest() == digest, path


def temps(series, times):
    values = pd.Series(series.to_numpy(dtype=float), index=pd.DatetimeIndex(times)).sort_index()
    values = values[values.notna() & values.between(-30, 50)]
    assert not values.index.has_duplicates
    return values


def dwd(path, station):
    with ZipFile(path) as z:
        with z.open(next(n for n in z.namelist() if n.startswith("produkt"))) as stream:
            obs = pd.read_csv(stream, sep=";", encoding="latin1", skipinitialspace=True)
    obs.columns = obs.columns.str.strip()
    assert obs.STATIONS_ID.eq(int(station)).all()
    time = pd.to_datetime(obs.MESS_DATUM.astype(str).str.strip(), format="%Y%m%d%H", utc=True)
    col = "TT_TU" if "TT_TU" in obs else "LUFTTEMPERATUR"
    return temps(obs[col], time)


def stations(sites):
    raw = BASE / "raw"
    registry = pd.read_csv(raw / "sources.csv", dtype={"station_id": str})
    recent = ROOT / "obs_2026"
    extra = pd.read_csv(recent / "sources.csv")
    for r in extra.itertuples(index=False):
        sha_file(recent / r.file, r.sha256)
    austrian = pd.read_csv(V2 / "external" / "sources.csv").set_index("file")
    utc = pd.DatetimeIndex(
        (DATES.to_numpy()[:, None] + np.array([h-2 for h in HOURS], dtype="timedelta64[h]")).ravel(),
        tz="UTC")
    result = []
    for row in sites.itertuples(index=False):
        if row.city == "Vienna":
            chunks = []
            for year in (2024, 2025, 2026):
                name = f"geosphere_{year}.csv"
                path = V2 / "external" / name if year < 2026 else recent / "Vienna_5925_2026.csv"
                if year < 2026:
                    sha_file(path, austrian.loc[name, "sha256"])
                file = pd.read_csv(path)
                file = file.loc[file.station.eq(int(row.station_id)) & file.tl_flag.isin([10,11,12,20,21,22])]
                chunks.append(temps(file.tl, pd.to_datetime(file.time, utc=True)))
        else:
            selected = registry.loc[registry.role.eq("urban") & registry.station_id.eq(row.station_id)]
            chunks = []
            for entry in selected.itertuples(index=False):
                path = raw / entry.file
                sha_file(path, entry.sha256)
                series = dwd(path, row.station_id)
                chunks.append(series.loc[series.index.year >= 2024])
            if row.city in ("Frankfurt", "Munich"):
                src = dwd(recent / f"{row.city}_{row.station_id}_2026.zip", row.station_id)
                chunks.append(src.loc[src.index.year == 2026])
        cell = pd.concat(chunks)
        assert not cell.index.has_duplicates, row.city
        grid = cell.reindex(utc).to_numpy().reshape(len(DATES), len(HOURS))
        record = pd.DataFrame({"date": DATES, "city": row.city,
                               "urban_max": grid[:, :len(DAY)].max(axis=1),
                               "urban_18": grid[:, len(DAY)-1]})
        record[TARGET_H] = grid[:, len(DAY):]
        result.append(record)
    return pd.concat(result, ignore_index=True)


def forecasts(sites):
    rows, logs = [], []
    for day in DATES:
        d = day.date()
        path = ROOT / "forecast_points" / f"{d}.json"
        file = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"fields": []}
        fields = {f["lead"]: f for f in file["fields"]}
        assert len(fields) == len(file["fields"]) and set(fields).issubset(LEADS)
        ready_by_18 = False
        for lead, f in fields.items():
            assert f["init"] == f"{d}T06:00:00Z"
            assert f":d={d:%Y%m%d}06:TMP:2 m above ground:{lead} hour fcst:" in f["inventory"]
            assert len(f["sha256"]) == 64 and set(f["point_c"]) == set(sites.city)
        if len(fields) == len(LEADS):
            latest = max(parsedate_to_datetime(r["modified_http"]) for r in fields.values())
            decision = datetime(d.year, d.month, d.day, 16, tzinfo=timezone.utc)
            ready_by_18 = latest <= decision
            logs.append(dict(date=d, count=len(fields), latest_archive_utc=latest.isoformat(),
                             lag_hours=(latest-decision).total_seconds()/3600, ready_by_18=ready_by_18))
        else:
            logs.append(dict(date=d, count=len(fields), ready_by_18=False))
        for city in sites.city:
            rows.append(dict(date=day, city=city, archive_by_18=ready_by_18,
                             **{f"gfs_{k}": fields.get(k, {}).get("point_c", {}).get(city, np.nan)
                                for k in LEADS}))
    pd.DataFrame(logs).to_csv(OUT / "forecast_availability.csv", index=False)
    return pd.DataFrame(rows)


def degree(arr, base=20):
    return np.maximum(arr-base, 0).sum(axis=1)


def fit_and_predict(panel):
    train = panel.loc[panel.year.eq(2024) & panel.city.ne("Vienna") & panel.matched].copy()
    r = train[TARGET_H].to_numpy() - train[GFS_H].to_numpy()
    x = train.urban_18 - train.gfs_10
    inputs = np.column_stack((np.ones(len(train)), x))
    beta = np.linalg.lstsq(inputs, r, rcond=None)[0]
    raw = panel[GFS_H].to_numpy()
    local = panel.urban_18 - panel.gfs_10
    predicted = {"gfs_raw": raw, "gfs_bias": raw + r.mean(axis=0),
                 "gfs_update": raw + np.column_stack((np.ones(len(panel)), local)) @ beta}
    coefficients = pd.DataFrame({"lead": list(range(14, 22)), "hour_cest": [h % 24 for h in NIGHT],
                                 "bias_correction": r.mean(axis=0), "update_intercept": beta[0],
                                 "update_slope": beta[1]})
    coefficients.to_csv(OUT / "coefficients.csv", index=False)
    print(f"Calibrated using {len(train)} station nights across {train.city.nunique()} German cities.", flush=True)
    return predicted


def matched_cases(panel):
    sample = panel.loc[panel.matched & panel.year.ge(2025)].copy()
    return sample.loc[sample.groupby(["city", "year"]).date.transform("size").ge(83)].copy()


def ranks(sample, pred, bases=(18, 20, 22)):
    rows = []
    for base in bases:
        aligned = sample.copy()
        aligned["ceiling"] = degree(aligned[TARGET_H].to_numpy(), base)
        for name, x in pred.items():
            aligned[name] = degree(x[aligned.index], base)
        for (city, year), frame in aligned.groupby(["city", "year"]):
            k = int(np.floor(.1*len(frame) + .5))
            total = frame.ceiling.sum()
            for score in SCORES:
                flag = weights(frame[score], k)
                rows.append(dict(city=city, year=year, base=base, score=score, n=len(frame),
                                 slots=k, heat=total, captured=np.dot(flag, frame.ceiling)))
    return pd.DataFrame(rows)


def decision(frame, previous, name, k=9):
    flags = np.zeros(len(frame))
    remain = k
    assert len(previous) >= 20 and len(frame) == 92
    for i, (ready, value) in enumerate(zip(frame.ready, frame[name])):
        if not ready or not remain:
            continue
        days = len(frame) - i
        if remain >= days or value > previous[name].quantile(1-remain/days):
            flags[i] = 1
            remain -= 1
    return flags


def replay(panel, sample, pred):
    rows, log = [], []
    for name in pred:
        panel[name] = degree(pred[name])
    panel["target"] = degree(panel[TARGET_H].to_numpy())
    history = panel.loc[panel.year.eq(2024) & panel.ready]
    cols = ["city", "date", "year", "ready", "target", *SCORES[:-1]]
    history[cols].to_csv(OUT / "prior_scores.csv", index=False)
    for (city, year), _ in sample.groupby(["city", "year"]):
        frame = panel.loc[panel.city.eq(city) & panel.year.eq(year)].sort_values("date")
        past = history.loc[history.city.eq(city)]
        measured = frame.target.notna().to_numpy()
        heat = frame.target.fillna(0).to_numpy()
        detail = frame[cols].copy()
        for name in SCORES:
            if name == "ceiling":
                modes = {"whole_season": weights(frame.target.fillna(-np.inf), 9)}
            else:
                modes = {"whole_season": weights(frame[name].where(frame.ready, -np.inf), 9),
                         "chronological": decision(frame, past, name)}
            for mode, flag in modes.items():
                detail[f"{mode}_{name}"] = flag
                rows.append(dict(city=city, year=year, score=name, mode=mode, calendar_days=len(frame),
                                 known_nights=measured.sum(), slots=flag.sum(),
                                 unknown_targets=flag[~measured].sum(), heat=heat.sum(),
                                 captured=np.dot(flag, heat)))
        log.append(detail)
    pd.concat(log).to_csv(OUT / "decisions.csv", index=False)
    return pd.DataFrame(rows)


def grouped(data, dims):
    detail = data.assign(region=np.where(data.city.eq("Vienna"), "Vienna", "Germany"))
    tall = detail.groupby(["region", *dims], as_index=False).sum(numeric_only=True)
    tall["coverage_pct"] = 100*tall.captured/tall.heat
    for col in ("base", "year"):
        if col in tall.columns and col not in dims:
            tall = tall.drop(columns=col)
    return tall


def error_check(sample, pred):
    rows = []
    for (city, y), frame in sample.groupby(["city", "year"]):
        t = frame[TARGET_H].to_numpy()
        heat = degree(t)
        for name, array in pred.items():
            est = array[frame.index]
            delta = est-t
            rows.append(dict(city=city, year=y, forecast=name, n=len(t), nb_values=delta.size,
                             sum_error=delta.sum(), sum_ae=np.abs(delta).sum(),
                             sum_degree_ae=np.abs(degree(est)-heat).sum()))
    annual = pd.DataFrame(rows)
    annual.to_csv(OUT / "temperature_error_city_year.csv", index=False)
    grand = annual.assign(region=np.where(annual.city.eq("Vienna"), "Vienna", "Germany"))
    total = grand.groupby(["region","forecast"], as_index=False).sum(numeric_only=True).drop(columns="year")
    total["mae_C"] = total.sum_ae/total.nb_values
    total["bias_C"] = total.sum_error/total.nb_values
    total["heat_mae"] = total.sum_degree_ae/total.n
    total.to_csv(OUT / "temperature_error.csv", index=False)


def checks(rank):
    germany = rank.loc[rank.city.ne("Vienna") & rank.base.eq(20)]
    cities = sorted(germany.city.unique())
    checks = [(f"Without {city}", germany.loc[germany.city.ne(city)]) for city in cities]
    checks.append(("2026 excluding two downloaded national recent series",
                   germany.loc[germany.year.eq(2026) & ~germany.city.isin(["Frankfurt","Munich"])]))
    rows = []
    for label, group in checks:
        stats = grouped(group, ["score"]).set_index("score").coverage_pct
        rows.append({"sample": label, **stats.to_dict(), "update_minus_bias": stats.gfs_update-stats.gfs_bias})
    pd.DataFrame(rows).to_csv(OUT / "robustness.csv", index=False)


def main():
    OUT.mkdir(exist_ok=True)
    sites = pd.read_csv(ROOT / "forecast_sites.csv", dtype={"station_id": str})
    panel = stations(sites).merge(forecasts(sites), on=KEY, validate="one_to_one")
    panel["year"] = panel.date.dt.year
    panel["ready"] = panel.archive_by_18 & panel[["urban_max", "urban_18", "gfs_10", *GFS_H]].notna().all(axis=1)
    panel["night_observed"] = panel[TARGET_H].notna().all(axis=1)
    panel["matched"] = panel.ready & panel.night_observed
    panel.groupby(["city","year"])[["ready","night_observed","matched"]].sum().to_csv(OUT / "cohort_audit.csv")
    sample = matched_cases(panel)
    assert sample.city.nunique() == 8 and set(sample.year) == {2025, 2026}
    pred = fit_and_predict(panel)
    error_check(sample, pred)
    result = ranks(sample, pred)
    result.to_csv(OUT / "rank_city_year.csv", index=False)
    pooled = grouped(result, ["base", "score"])
    pooled.to_csv(OUT / "rank_summary.csv", index=False)
    grouped(result, ["year","base","score"]).to_csv(OUT / "rank_year.csv", index=False)
    checks(result)
    audit = replay(panel, sample, pred)
    audit.to_csv(OUT / "replay_city_year.csv", index=False)
    replay_total = grouped(audit, ["mode","score"])
    replay_total.to_csv(OUT / "replay_summary.csv", index=False)
    grouped(audit, ["year","mode","score"]).to_csv(OUT / "replay_year.csv", index=False)
    panel.to_csv(OUT / "nightly_forecast.csv.gz", index=False, compression={"method":"gzip","mtime":0})
    print(pooled.query("base==20")[["region","score","n","slots","coverage_pct"]].round(3).to_string(index=False))
    print(replay_total[["region","mode","score","known_nights","slots","coverage_pct"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
