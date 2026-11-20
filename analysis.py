from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
OUT = ROOT / "results"
DAY = tuple(range(8, 19))
NIGHT = tuple(range(22, 30))
HOURS = tuple(range(8, 31))
RULES = ("ref_max", "urban_max", "ref_18", "urban_18", "oracle")
CONTRASTS = {
    "urban_timing": ("urban_18", "urban_max"),
    "reference_timing": ("ref_18", "ref_max"),
    "daytime_siting": ("urban_max", "ref_max"),
    "evening_siting": ("urban_18", "ref_18"),
    "urban_evening_vs_reference_max": ("urban_18", "ref_max"),
}
GROUP = ["city", "year"]


def read_temperature(entry):
    path = RAW / entry.file
    if sha256(path.read_bytes()).hexdigest() != entry.sha256:
        raise ValueError(f"Archive has changed: {entry.file}")
    with ZipFile(path) as archive:
        name = next(n for n in archive.namelist() if n.startswith("produkt"))
        with archive.open(name) as stream:
            data = pd.read_csv(stream, sep=";", encoding="latin1", skipinitialspace=True)
    data.columns = data.columns.str.strip()
    assert set(data.STATIONS_ID.unique()) == {int(entry.station_id)}, entry.file
    timestamp = pd.to_datetime(data.MESS_DATUM.astype(str).str.strip(), utc=True, format="%Y%m%d%H")
    temp_col = "TT_TU" if entry.product == "climate" else "LUFTTEMPERATUR"
    quality_col = "QN_9" if entry.product == "climate" else "QUALITAETS_NIVEAU"
    series = pd.Series(pd.to_numeric(data[temp_col]).to_numpy(), index=pd.DatetimeIndex(timestamp))
    selected_month = (series.index.year >= 2017) & (series.index.year <= 2025)
    selected_month &= (series.index.month >= 5) & (series.index.month <= 9)
    values = series.loc[selected_month]
    if values.index.has_duplicates:
        conflicting = values.groupby(level=0).nunique()
        if conflicting.gt(1).any():
            raise ValueError(f"Conflicting timestamps in {entry.file}")
        values = values[~values.index.duplicated()]
    bounds = values.between(-30, 50)
    stats = dict(file=entry.file, first_utc=timestamp.min(), last_utc=timestamp.max(),
                 rows=len(data), study_rows=int(selected_month.sum()),
                 study_valid=int(bounds.sum()),
                 study_nonmissing_out_of_bounds=int((~bounds & values.ne(-999) & values.notna()).sum()),
                 valid_min=values[bounds].min(), valid_max=values[bounds].max(),
                 quality_codes=",".join(map(str, sorted(data.loc[selected_month, quality_col].dropna().unique()))))
    values = values.where(bounds).dropna()
    values.index = values.index.tz_convert("Europe/Berlin").tz_localize(None)
    return values.sort_index(), stats


def hour_columns(prefix, hours):
    return [f"{prefix}_{h:02d}" for h in hours]


def paired_panel(sources):
    days = pd.date_range("2017-06-01", "2025-08-31")
    days = days[days.month.isin((6, 7, 8))]
    slots = pd.DatetimeIndex((days.to_numpy()[:, None] +
                              np.array(HOURS, dtype="timedelta64[h]")).ravel())
    series = {}
    quality = []
    for entry in sources.itertuples(index=False):
        values, stats = read_temperature(entry)
        series[entry.file] = values.reindex(slots).to_numpy().reshape(len(days), len(HOURS))
        quality.append(stats)
    out = []
    core_idx = [HOURS.index(h) for h in DAY + NIGHT]
    for city, catalog in sources.groupby("city"):
        urban = np.full((len(days), len(HOURS)), np.nan)
        used = np.full(len(days), "", dtype=object)
        for entry in catalog[catalog.role.eq("urban")].itertuples(index=False):
            values = series[entry.file]
            complete = np.isfinite(values[:, core_idx]).all(axis=1)
            if (complete & (used != "")).any():
                raise ValueError(f"Overlapping complete urban records at {city}")
            urban[complete] = values[complete]
            used[complete] = entry.product
        reference = series[catalog.loc[catalog.role.eq("reference"), "file"].item()]
        flags = pd.DataFrame({"date": days, "year": days.year, "city": city, "product": used,
                              "urban_ok": np.isfinite(urban[:, core_idx]).all(axis=1),
                              "ref_ok": np.isfinite(reference[:, core_idx]).all(axis=1)})
        flags["complete"] = flags.urban_ok & flags.ref_ok
        u = pd.DataFrame(urban, columns=hour_columns("u", HOURS))
        r = pd.DataFrame(reference, columns=hour_columns("r", HOURS))
        out.append(pd.concat([flags, u, r], axis=1))
    pd.DataFrame(quality).to_csv(OUT / "source_quality.csv", index=False)
    return pd.concat(out, ignore_index=True)


def eligible(panel, min_n=83, night=NIGHT):
    valid = panel[panel.complete].copy()
    target_slots = hour_columns("u", night) + hour_columns("r", night)
    valid = valid.dropna(subset=target_slots)
    return valid[valid.groupby(GROUP).date.transform("size").ge(min_n)].copy()


def scores(panel, night=NIGHT, threshold=20):
    data = panel.copy()
    data["target"] = data[hour_columns("u", night)].sub(threshold).clip(lower=0).sum(axis=1)
    data["ref_target"] = data[hour_columns("r", night)].sub(threshold).clip(lower=0).sum(axis=1)
    for prefix, label in (("u", "urban"), ("r", "ref")):
        data[label + "_max"] = data[hour_columns(prefix, DAY)].max(axis=1)
        data[label + "_18"] = data[f"{prefix}_18"]
    data["oracle"] = data.target
    return data


def weights(score, k):
    values = np.asarray(score)
    boundary = np.partition(values, len(values) - k)[len(values) - k]
    above = values > boundary
    ties = values == boundary
    flags = above + ties * ((k - above.sum()) / ties.sum())
    assert np.isclose(flags.sum(), k)
    return flags


def rank(group, fraction):
    k = max(1, int(np.floor(fraction * len(group) + 0.5)))
    heat = group.target.to_numpy()
    selected = {key: weights(group[key], k) for key in RULES}
    captured = {key: np.dot(flag, heat) for key, flag in selected.items()}
    assert max(captured.values()) <= captured["oracle"] + 1e-7
    return selected, dict(nights=len(group), slots=k, burden=heat.sum(),
                          positive_nights=(heat > 0).sum(), **captured)


def group_rank(data, fraction):
    rows = [dict(city=city, year=year, **rank(group, fraction)[1])
            for (city, year), group in data.groupby(GROUP, sort=True)]
    return pd.DataFrame(rows)


def pooled(table):
    totals = table.select_dtypes("number").sum()
    return dict(nights=int(totals.nights), seasons=len(table), slots=int(totals.slots),
                burden=totals.burden, positive_nights=int(totals.positive_nights),
                **{key: 100 * totals[key] / totals.burden for key in RULES})


def bootstrap(table):
    years = table.groupby("year").sum(numeric_only=True)
    values = years[["burden", *RULES]].to_numpy()
    draws = np.random.default_rng(481104).integers(0, len(years), (10000, len(years)))
    total = values[draws].sum(axis=1)
    samples = pd.DataFrame(100 * total[:, 1:] / total[:, [0]], columns=RULES)
    central = pooled(table)
    for key, (lhs, rhs) in CONTRASTS.items():
        samples[key] = samples[lhs] - samples[rhs]
        central[key] = central[lhs] - central[rhs]
    names = samples.columns.tolist()
    q = samples.quantile([.025, .975])
    return pd.DataFrame({"metric": names, "estimate": [central[k] for k in names],
                         "lower": q.iloc[0].to_numpy(), "upper": q.iloc[1].to_numpy()})


def fixed_cutoffs(data):
    rows = []
    for rule in RULES[:-1]:
        for city, group in data.groupby("city"):
            train = group[group.year.le(2020)]
            test = group[group.year.ge(2021)]
            cut = train[rule].quantile(.90)
            marked = test[rule].gt(cut)
            rows.append(dict(city=city, rule=rule, cutoff=cut, calibration_n=len(train),
                             test_n=len(test), flags=marked.sum(), burden=test.target.sum(),
                             captured=test.loc[marked, "target"].sum()))
    return pd.DataFrame(rows)


def discordance(data):
    chunks = []
    for (city, year), group in data.groupby(GROUP):
        flags, _ = rank(group, .10)
        chunk = group[["city", "year", "date", "target", "ref_target", *RULES[:-1]]].copy()
        chunk["u_night_mean"] = group[hour_columns("u", NIGHT)].mean(axis=1)
        chunk["r_night_mean"] = group[hour_columns("r", NIGHT)].mean(axis=1)
        chunk["delta_18"] = group.u_18 - group.r_18
        chunk["delta_night"] = chunk.u_night_mean - chunk.r_night_mean
        chunk["early_urban_change"] = group.u_22 - group.u_18
        chunk["early_ref_change"] = group.r_22 - group.r_18
        chunk["w_evening_only"] = np.maximum(flags["urban_18"] - flags["urban_max"], 0)
        chunk["w_max_only"] = np.maximum(flags["urban_max"] - flags["urban_18"], 0)
        chunk["w_day_urban"] = flags["urban_max"]
        chunk["w_evening_urban"] = flags["urban_18"]
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True)


def sensitivity(panel, data):
    specifications = [
        ("Primary", data, .10),
        ("5% budget", data, .05),
        ("20% budget", data, .20),
        ("18 C threshold", scores(eligible(panel), threshold=18), .10),
        ("22 C threshold", scores(eligible(panel), threshold=22), .10),
        ("21:00–06:59 window", scores(eligible(panel, night=range(21, 31)), night=range(21, 31)), .10),
        ("00:00–05:59 window", scores(eligible(panel), night=range(24, 30)), .10),
        ("95% seasonal coverage", scores(eligible(panel, min_n=88)), .10),
        ("Without provisional urban product", scores(eligible(panel[panel["product"].ne("climate_urban_prov")])), .10),
        ("Historical national product at both sites", data[data["product"].eq("climate")], .10),
        ("2017–2020", data[data.year.le(2020)], .10),
        ("2021–2025", data[data.year.ge(2021)], .10),
        *[("Without " + city, data[data.city.ne(city)], .10) for city in sorted(data.city.unique())],
    ]
    rows = []
    for name, frame, fraction in specifications:
        estimate = pooled(group_rank(frame, fraction))
        rows.append(dict(scenario=name, budget=fraction, cities=frame.city.nunique(),
                         **estimate, **{key: estimate[lhs] - estimate[rhs]
                                        for key, (lhs, rhs) in CONTRASTS.items()}))
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(exist_ok=True)
    sources = pd.read_csv(RAW / "sources.csv", dtype={"station_id": str})
    panel = paired_panel(sources)
    counts = panel.groupby(GROUP)[["urban_ok", "ref_ok", "complete"]].sum().reset_index()
    counts["included"] = counts.complete.ge(83)
    counts.to_csv(OUT / "season_coverage.csv", index=False)
    data = scores(eligible(panel))
    assert data.groupby(GROUP).size().ge(83).all() and data.city.nunique() == 7
    data.to_csv(OUT / "nightly.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    ranking = group_rank(data, .10)
    ranking.to_csv(OUT / "capture_city_year.csv", index=False)
    summary = pd.DataFrame([dict(city="Pooled", **pooled(ranking)),
                            *[dict(city=city, **pooled(group)) for city, group in ranking.groupby("city")]])
    summary.to_csv(OUT / "capture_summary.csv", index=False)
    bootstrap(ranking).to_csv(OUT / "uncertainty.csv", index=False)
    sensitivity(panel, data).to_csv(OUT / "sensitivity.csv", index=False)
    fixed_cutoffs(data).to_csv(OUT / "fixed_cutoffs.csv", index=False)
    discordance(data).to_csv(OUT / "discordance.csv", index=False)
    curve = pd.DataFrame([dict(budget=i / 100, **pooled(group_rank(data, i / 100)))
                          for i in range(1, 31)])
    curve.to_csv(OUT / "budget_curve.csv", index=False)
    print(summary.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
