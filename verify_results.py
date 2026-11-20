from collections import defaultdict
from csv import DictReader
from datetime import date, datetime, time, timedelta
from gzip import open as gzip_open
from math import isclose
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
OUT = ROOT / "results"


def read(path, opener=open):
    with opener(path, "rt", encoding="utf-8", newline="") as stream:
        return list(DictReader(stream))


def raw_hours(file):
    with ZipFile(RAW / file) as archive:
        name = next(n for n in archive.namelist() if n.startswith("produkt"))
        lines = archive.read(name).decode("latin1").splitlines()
    header = [s.strip() for s in lines[0].split(";")]
    stamp = header.index("MESS_DATUM")
    temp = header.index("TT_TU" if "TT_TU" in header else "LUFTTEMPERATUR")
    values = {}
    for line in lines[1:]:
        fields = line.split(";")
        key = fields[stamp].strip()
        if "20170530" <= key <= "20250902":
            value = float(fields[temp])
            if -30 <= value <= 50:
                values[key] = value
    return values


def expected_capture(pairs, k):
    by_score = defaultdict(list)
    for score, outcome in pairs:
        by_score[score].append(outcome)
    total = 0.0
    for score in sorted(by_score, reverse=True):
        outcomes = by_score[score]
        take = min(k, len(outcomes))
        total += sum(outcomes) * (take / len(outcomes))
        k -= take
        if not k:
            break
    return total


def main():
    sources = defaultdict(lambda: defaultdict(list))
    for row in read(RAW / "sources.csv"):
        sources[row["city"]][row["role"]].append(raw_hours(row["file"]))
    records = {(row["city"], row["date"]): row for row in
               read(OUT / "nightly.csv.gz", gzip_open)}
    observed_results = {(row["city"], int(row["year"])): row for row in
                        read(OUT / "capture_city_year.csv")}
    coverage = {(row["city"], int(row["year"])): int(row["complete"])
                for row in read(OUT / "season_coverage.csv")}
    keys = ["ref_max", "urban_max", "ref_18", "urban_18", "oracle"]
    hours = list(range(8, 19)) + list(range(22, 30))
    checked = 0
    for city, providers in sources.items():
        for year in range(2017, 2026):
            summer = []
            for offset in range(92):
                day = date(year, 6, 1) + timedelta(days=offset)
                midnight = datetime.combine(day, time.min)
                stamps = [(midnight + timedelta(hours=h-2)).strftime("%Y%m%d%H") for h in hours]
                valid_u = [series for series in providers["urban"]
                           if all(key in series for key in stamps)]
                ref = providers["reference"][0]
                if not valid_u or not all(key in ref for key in stamps):
                    assert (city, str(day)) not in records
                    continue
                assert len(valid_u) == 1
                u, r = ([series[key] for key in stamps] for series in [valid_u[0], ref])
                heat = sum(max(value - 20, 0) for value in u[-8:])
                summer.append((str(day), [max(r[:11]), max(u[:11]), r[10], u[10], heat], heat, u, r))
            assert len(summer) == coverage[(city, year)]
            if len(summer) < 83:
                assert all((city, item[0]) not in records for item in summer)
                continue
            reported = observed_results[(city, year)]
            assert float(reported["nights"]) == len(summer)
            assert isclose(float(reported["burden"]), sum(item[2] for item in summer))
            for day, indicators, heat, u, r in summer:
                res = records[(city, day)]
                assert isclose(float(res["target"]), heat, abs_tol=1e-9)
                for h, u_expected, r_expected in zip(hours, u, r):
                    assert float(res[f"u_{h:02d}"]) == u_expected
                    assert float(res[f"r_{h:02d}"]) == r_expected
                checked += 1
            k = int(len(summer) * .1 + .5)
            assert int(reported["slots"]) == k
            for i, key in enumerate(keys):
                val = expected_capture([(item[1][i], item[2]) for item in summer], k)
                assert isclose(val, float(reported[key]), abs_tol=1e-8), (city, year, key)
    assert checked == len(records)
    print(f"Verified coverage of 63 city-summers, {checked:,} paired nights, "
          f"{checked*len(hours)*2:,} source temperatures and all five selection totals per included season.")


if __name__ == "__main__":
    main()
