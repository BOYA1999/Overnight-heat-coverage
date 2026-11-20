from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "original" if (ROOT / "original" / "analysis.py").is_file() else ROOT.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(BASE))
from analysis import DAY, NIGHT, RULES, group_rank, pooled, weights

OUT = ROOT / "results"
EXT = ROOT / "external"
DATES = pd.date_range("2017-06-01", "2025-08-31")
DATES = DATES[DATES.month.isin([6, 7, 8])]
UTC = pd.DatetimeIndex((DATES.to_numpy()[:, None] + np.array(DAY + NIGHT, dtype="timedelta64[h]")).ravel() -
                       np.timedelta64(2, "h"), tz="UTC")
CHECKED = (10, 11, 12, 20, 21, 22)
OVERNIGHT = [f"hour_{h}" for h in NIGHT]
MODEL_INPUT = {
    "linear_18": ["urban_18"],
    "linear_change": ["urban_18", "change"],
    "linear_moisture": ["urban_18", "change", "vapour"],
}
SCORES = ["urban_max", "ref_max", "urban_18", "ref_18", *MODEL_INPUT]


def pack(stamp, temperature, humidity):
    frame = pd.DataFrame({"t": np.array(temperature), "rh": np.array(humidity)},
                         index=pd.DatetimeIndex(stamp))
    frame = frame[frame.index.isin(UTC)]
    assert not frame.index.has_duplicates
    frame["t"] = frame.t.where(frame.t.between(-30, 50))
    frame["rh"] = frame.rh.where(frame.rh.between(0, 100))
    return frame.reindex(UTC).to_numpy().reshape(len(DATES), len(DAY + NIGHT), 2)


def dwd_array(row):
    path = BASE / "raw" / row.file
    assert sha256(path.read_bytes()).hexdigest() == row.sha256, row.file
    with ZipFile(path) as z:
        with z.open(next(n for n in z.namelist() if n.startswith("produkt"))) as stream:
            data = pd.read_csv(stream, sep=";", encoding="latin1", skipinitialspace=True)
    data.columns = data.columns.str.strip()
    assert data.STATIONS_ID.eq(int(row.station_id)).all()
    time = pd.to_datetime(data.MESS_DATUM.astype(str).str.strip(), format="%Y%m%d%H", utc=True)
    if row.product == "climate":
        return pack(time, data.TT_TU, data.RF_TU)
    return pack(time, data.LUFTTEMPERATUR, data.REL_FEUCHTE)


def panel(city, urban_sources, reference):
    arrays = np.stack(urban_sources)
    nday = len(DAY)
    day_ok = np.isfinite(arrays[:, :, :nday, 0]).all(axis=2)
    night_ok = np.isfinite(arrays[:, :, nday:, 0]).all(axis=2)
    assert (day_ok.sum(axis=0) <= 1).all() and (night_ok.sum(axis=0) <= 1).all(), city
    source = (2 * day_ok + night_ok).argmax(axis=0)
    urban = arrays[source, np.arange(len(DATES))]
    u, r = urban[:, :, 0], reference[:, :, 0]
    data = pd.DataFrame({"city": city, "date": DATES, "year": DATES.year,
                         "day": (DATES - pd.to_datetime(DATES.year.astype(str) + "-06-01")).days,
                         "urban_max": np.max(u[:, :nday], axis=1),
                         "ref_max": np.max(r[:, :nday], axis=1),
                         "urban_18": u[:, nday-1], "ref_18": r[:, nday-1],
                         "change": u[:, nday-1] - u[:, DAY.index(15)],
                         "rh18": urban[:, nday-1, 1],
                         "target": np.maximum(u[:, nday:] - 20, 0).sum(axis=1),
                         "ref_target": np.maximum(r[:, nday:] - 20, 0).sum(axis=1)})
    data["vapour"] = (data.rh18 / 100 * 6.112 *
                      np.exp(17.67 * data.urban_18 / (data.urban_18 + 243.5)))
    data["basic_complete"] = data[["urban_max", "ref_max", "target", "ref_target"]].notna().all(axis=1)
    data["day_ready"] = data[["urban_max", "ref_max", "rh18"]].notna().all(axis=1)
    data[OVERNIGHT] = u[:, nday:]
    data["oracle"] = data.target
    return data


def germany():
    catalog = pd.read_csv(BASE / "raw" / "sources.csv", dtype={"station_id": str})
    arrays = {row.file: dwd_array(row) for row in catalog.itertuples(index=False)}
    tables = []
    for city, group in catalog.groupby("city"):
        urban = [arrays[name] for name in group.loc[group.role.eq("urban"), "file"]]
        reference = arrays[group.loc[group.role.eq("reference"), "file"].item()]
        tables.append(panel(city, urban, reference))
    return pd.concat(tables, ignore_index=True)


def vienna():
    catalog = pd.read_csv(EXT / "sources.csv")
    values = {}
    for row in catalog.itertuples(index=False):
        path = EXT / row.file
        assert sha256(path.read_bytes()).hexdigest() == row.sha256, path
        if row.file.startswith("geosphere_"):
            values[row.file] = pd.read_csv(path)
    raw = pd.concat(values.values(), ignore_index=True)
    rows, arrays = [], {}
    for station, group in raw.groupby("station"):
        stamp = pd.to_datetime(group.time, utc=True)
        t = group.tl.where(group.tl_flag.isin(CHECKED))
        rh = group.rf.where(group.rf_flag.isin(CHECKED))
        rows.append(dict(station=station, rows=len(group), missing_T=group.tl.isna().sum(),
                         rejected_T=(group.tl.notna() & t.isna()).sum(), valid_T=t.notna().sum(),
                         rejected_RH=(group.rf.notna() & rh.isna()).sum(), valid_RH=rh.notna().sum(),
                         T_codes="/".join(map(str, sorted(group.tl_flag.dropna().unique()))),
                         RH_codes="/".join(map(str, sorted(group.rf_flag.dropna().unique())))))
        arrays[station] = pack(stamp, t, rh)
    pd.DataFrame(rows).to_csv(OUT / "vienna_quality.csv", index=False)
    return panel("Vienna", [arrays[5925]], arrays[5990]), panel("Vienna", [arrays[5925]], arrays[5972])


def included(data, mask):
    sample = data.loc[mask].copy()
    return sample.loc[sample.groupby(["city", "year"]).date.transform("size").ge(83)].copy()


def check_original(german):
    old = pd.read_csv(BASE / "results" / "nightly.csv.gz", parse_dates=["date"])
    new = included(german, german.basic_complete)
    joined = old.merge(new, on=["city", "date"], validate="one_to_one")
    assert len(joined) == len(old) == len(new) == 4878
    for col in RULES:
        assert np.allclose(joined[col+"_x"], joined[col+"_y"]), col
    return old


def allocation(old):
    records, annual = [], []
    for year, season in old.groupby("year"):
        season = season.copy()
        season["local"] = season.groupby("city")["target"].transform(
            lambda y: weights(y, int(np.floor(.1 * len(y) + .5))))
        k = round(season.local.sum())
        season["pooled"] = weights(season.target, k)
        for city, site in season.groupby("city"):
            records.append(dict(city=city, year=year, total=site.target.sum(),
                                local_slots=site.local.sum(), pooled_slots=site.pooled.sum(),
                                local_heat=np.dot(site.local, site.target),
                                pooled_heat=np.dot(site.pooled, site.target)))
        annual.append(dict(year=year, slots=k, n=len(season), heat=season.target.sum(),
                           local=np.dot(season.local, season.target),
                           pooled=np.dot(season.pooled, season.target)))
    pd.DataFrame(records).to_csv(OUT / "allocation_city_year.csv", index=False)
    pd.DataFrame(annual).to_csv(OUT / "allocation_year.csv", index=False)
    orig = group_rank(old, .10)
    rows = []
    for label, table in [(f"{y}", orig.loc[orig.year.eq(y)]) for y in sorted(orig.year.unique())] + [
        (f"Without {y}", orig.loc[orig.year.ne(y)]) for y in sorted(orig.year.unique())]:
        row = pooled(table)
        rows.append({"sample": label, **row, "urban_timing": row["urban_18"] - row["urban_max"]})
    city = pd.DataFrame([pooled(frame) for _, frame in orig.groupby("city")]).mean().to_dict()
    rows.append({"sample": "Equal city weights", **city, "urban_timing": city["urban_18"] - city["urban_max"]})
    pd.DataFrame(rows).to_csv(OUT / "aggregate_checks.csv", index=False)


def fit_profiles(train, data):
    coeff = []
    predict_hours = {}
    for name, terms in MODEL_INPUT.items():
        x = np.column_stack((np.ones(len(train)), train[terms].to_numpy()))
        y = train[OVERNIGHT].to_numpy()
        beta, _, rank_value, _ = np.linalg.lstsq(x, y, rcond=None)
        assert rank_value == x.shape[1]
        pred = np.column_stack((np.ones(len(data)), data[terms].to_numpy())) @ beta
        predict_hours[name] = pred
        data[name] = np.maximum(pred - 20, 0).sum(axis=1)
        for j, term in enumerate(["intercept", *terms]):
            coeff.append({"model": name, "term": term, **dict(zip(OVERNIGHT, beta[j]))})
    pd.DataFrame(coeff).to_csv(OUT / "profile_coefficients.csv", index=False)
    return predict_hours


def chronology(group, history, score, k, window=15):
    selected = np.zeros(len(group))
    remaining = k
    assert len(group) == 92 and group.day.tolist() == list(range(92))
    for i, pair in enumerate(zip(group[score], group.day_ready)):
        value, ready = pair
        if not ready or remaining == 0:
            continue
        days = len(group) - i
        available = history.loc[history.day.sub(i).abs().le(window), score]
        assert len(available) >= 20 and np.isfinite(value)
        if remaining >= days or value > available.quantile(1 - remaining / days):
            selected[i] = 1
            remaining -= 1
    return selected


def test_replay(data, test):
    items, details = [], []
    history = data.loc[data.year.le(2020) & data.day_ready]
    history[["city", "date", "day", *SCORES]].to_csv(OUT / "calibration_scores.csv", index=False)
    for (city, year), _ in test.groupby(["city", "year"]):
        group = data.loc[data.city.eq(city) & data.year.eq(year)].sort_values("date").copy()
        prior = history.loc[history.city.eq(city)]
        known = group.target.notna().to_numpy()
        y = group.target.fillna(0).to_numpy()
        k = 9
        ordered = {"oracle": {"batch": weights(group.target.fillna(-np.inf), k)}}
        for key in SCORES:
            unready = group[key].where(group.day_ready, -np.inf)
            ordered[key] = {"batch": weights(unready, k),
                            "chronological": chronology(group, prior, key, k)}
        detail = group[["city", "year", "date", "target", "day_ready", *SCORES]].copy()
        detail["target_observed"] = known
        for key, modes in ordered.items():
            for mode, flag in modes.items():
                assert 0 <= flag.sum() <= k + 1e-9
                detail[f"{mode}_{key}"] = flag
                items.append(dict(cohort="Vienna" if city == "Vienna" else "Germany",
                                  city=city, year=year, score=key, mode=mode,
                                  target_n=int(known.sum()), ready_n=int(group.day_ready.sum()),
                                  calendar_days=len(group), quota=k, selected=flag.sum(),
                                  unknown_target=flag[~known].sum(), heat=y.sum(), captured=np.dot(y, flag)))
        details.append(detail)
    pd.concat(details).to_csv(OUT / "test_calendar_decisions.csv", index=False)
    by_season = pd.DataFrame(items)
    by_season.to_csv(OUT / "replay_city_year.csv", index=False)
    totals = by_season.drop(columns=["year"]).groupby(["cohort", "score", "mode"]).sum(numeric_only=True)
    totals["capture_pct"] = 100 * totals.captured / totals.heat
    totals["selected_pct"] = 100 * totals.selected / totals.calendar_days
    totals.reset_index().to_csv(OUT / "replay_summary.csv", index=False)
    return totals


def threshold_check(data):
    complete = included(data, data.basic_complete)
    early = complete.loc[complete.year.le(2020)]
    late = complete.loc[complete.year.ge(2021)]
    site_base = early.groupby("city").apply(
        lambda g: np.quantile(g[OVERNIGHT].to_numpy(), .90), include_groups=False)
    site_base.rename("base_C").to_csv(OUT / "local_base.csv")
    summaries = []
    for name in ("18 C", "20 C", "22 C", "Local 90th percentile"):
        for population, test in [("Germany", late.loc[late.city.ne("Vienna")]),
                                 ("Vienna", late.loc[late.city.eq("Vienna")])]:
            test = test.copy()
            base = test.city.map(site_base).to_numpy() if name.startswith("Local") else float(name.split()[0])
            base = np.broadcast_to(base, len(test))
            test["target"] = np.maximum(test[OVERNIGHT].to_numpy() - base[:, None], 0).sum(axis=1)
            test["oracle"] = test.target
            summaries.append({"cohort": population, "base": name, **pooled(group_rank(test, .10))})
    pd.DataFrame(summaries).to_csv(OUT / "threshold_comparison.csv", index=False)


def replay_sensitivity(data, test):
    history = data.loc[data.year.le(2020) & data.day_ready]
    rows = []
    for (city, year), _ in test.groupby(["city", "year"]):
        group = data.loc[data.city.eq(city) & data.year.eq(year)].sort_values("date")
        prior = history.loc[history.city.eq(city)]
        y = group.target.fillna(0).to_numpy()
        for window in (10, 15, 30):
            values = {}
            for score in ("urban_max", "urban_18"):
                flags = chronology(group, prior, score, 9, window)
                values[f"{score}_slots"] = flags.sum()
                values[score] = np.dot(y, flags)
            rows.append(dict(cohort="Vienna" if city == "Vienna" else "Germany",
                             city=city, year=year, half_window=window, days_ready=group.day_ready.sum(),
                             heat=y.sum(), **values))
    pd.DataFrame(rows).to_csv(OUT / "replay_sensitivity.csv", index=False)


def main():
    OUT.mkdir(exist_ok=True)
    german = germany()
    old = check_original(german)
    allocation(old)
    v, alt = vienna()
    branches = []
    for label, universe in [("Schwechat", v), ("Gross-Enzersdorf", alt)]:
        sample = included(universe, universe.basic_complete)
        stats = group_rank(sample, .10)
        stats.insert(0, "reference", label)
        stats.to_csv(OUT / f"vienna_{label}_city_year.csv", index=False)
        branches.append({"reference": label, **pooled(stats)})
    pd.DataFrame(branches).to_csv(OUT / "vienna_summary.csv", index=False)

    # Model fitting and evaluation use the same coverage criterion with 18:00 humidity.
    all_nights = pd.concat([german, v], ignore_index=True)
    sample = included(all_nights, all_nights.basic_complete & all_nights.day_ready)
    train = sample.loc[sample.city.ne("Vienna") & sample.year.le(2020)]
    test = sample.loc[sample.year.ge(2021)]
    pred = fit_profiles(train, all_nights)
    err_rows = []
    for label, frame in [("Germany", test.loc[test.city.ne("Vienna")]), ("Vienna", test.loc[test.city.eq("Vienna")])]:
        for key, temp in pred.items():
            err = temp[frame.index] - frame[OVERNIGHT].to_numpy()
            err_rows.append(dict(cohort=label, model=key, nights=len(frame), summers=frame.year.nunique(),
                                 city_summers=len(frame[["city", "year"]].drop_duplicates()),
                                 mean_error=err.mean(), hourly_mae=np.abs(err).mean(),
                                 burden_mae=np.abs(all_nights.loc[frame.index, key] - frame.target).mean()))
    pd.DataFrame(err_rows).to_csv(OUT / "model_error.csv", index=False)
    coverage = all_nights.groupby(["city", "year"])[["basic_complete", "day_ready"]].sum()
    coverage["complete_with_rh"] = all_nights.assign(ok=all_nights.basic_complete & all_nights.day_ready).groupby(
        ["city", "year"]).ok.sum()
    coverage.to_csv(OUT / "calendar_coverage.csv")
    print(f"Original primary target reproduced for {len(old):,} nights. Train: {len(train):,}; test: {len(test):,}.")
    print(pd.DataFrame(branches).round(2).to_string(index=False))
    report = test_replay(all_nights, test)
    print(report[["target_n", "quota", "selected", "unknown_target", "capture_pct"]].round(2).to_string())
    threshold_check(all_nights)
    replay_sensitivity(all_nights, test)


if __name__ == "__main__":
    main()
