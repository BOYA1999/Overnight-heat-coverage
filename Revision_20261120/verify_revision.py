from collections import defaultdict
from csv import DictReader
from datetime import date, datetime, timedelta, timezone
from math import floor, isclose
from pathlib import Path

ROOT = Path(__file__).resolve().parent
R = ROOT / "results"

def rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(DictReader(stream))


def spread(total_by_score, k):
    heat = 0.
    for score in sorted(total_by_score, reverse=True):
        values = total_by_score[score]
        take = min(len(values), k)
        heat += sum(values) * take / len(values)
        k -= take
        if not k:
            return heat
    raise ValueError("Too few scores")


def verify_vienna():
    h = list(range(8, 19)) + list(range(22, 30))
    keys = ["ref_max", "urban_max", "ref_18", "urban_18", "oracle"]
    result = {int(row["year"]): row for row in rows(R / "vienna_Schwechat_city_year.csv")}
    nights = 0
    for year in range(2017, 2026):
        temperatures = defaultdict(dict)
        for r in rows(ROOT / "external" / f"geosphere_{year}.csv"):
            if r["tl_flag"] in {"10", "11", "12", "20", "21", "22"} and r["tl"]:
                value = float(r["tl"])
                assert -30 <= value <= 50
                temperatures[int(r["station"])][datetime.fromisoformat(r["time"])] = value
        sums = {k: defaultdict(list) for k in keys}
        n, burden = 0, 0
        for i in range(92):
            start = datetime(year, 6, 1, tzinfo=timezone.utc) + timedelta(days=i, hours=-2)
            times = [start + timedelta(hours=x) for x in h]
            u = [temperatures[5925].get(t) for t in times]
            ref = [temperatures[5990].get(t) for t in times]
            if None in u or None in ref:
                continue
            y = sum(max(t - 20, 0) for t in u[-8:])
            for k, score in zip(keys, [max(ref[:11]), max(u[:11]), ref[10], u[10], y]):
                sums[k][score].append(y)
            burden += y
            n += 1
        if n < 83:
            assert year not in result
            continue
        r = result[year]
        k = floor(n * .1 + .5)
        assert (int(r["nights"]), int(r["slots"])) == (n, k)
        assert isclose(float(r["burden"]), burden)
        for key in keys:
            assert isclose(spread(sums[key], k), float(r[key]), abs_tol=1e-8)
        nights += n
    assert len(result) == 8 and nights == 736
    return nights


def linear_quantile(values, q):
    x = sorted(values)
    index = (len(x) - 1) * q
    lo = floor(index)
    hi = min(lo+1, len(x)-1)
    return x[lo] + (index-lo)*(x[hi]-x[lo])


def verify_chronology():
    cal = defaultdict(list)
    for row in rows(R / "calibration_scores.csv"):
        cal[row["city"]].append(row)
    exams = defaultdict(list)
    for row in rows(R / "test_calendar_decisions.csv"):
        exams[(row["city"], int(row["year"]))].append(row)
    matrix = rows(R / "replay_city_year.csv")
    names = sorted({r["score"] for r in matrix if r["mode"] == "chronological"})
    stored = {(row["city"], int(row["year"]), row["score"]): row for row in matrix
              if row["mode"] == "chronological"}
    flags = 0
    for (city, year), cohort in exams.items():
        cohort.sort(key=lambda row: row["date"])
        assert len(cohort) == 92
        assert all(int(r["date"][:4]) < year for r in cal[city])
        for score in names:
            left = 9
            captured = 0.
            unknown = 0
            for i, r in enumerate(cohort):
                assert date.fromisoformat(r["date"]) == date(year, 6, 1) + timedelta(days=i)
                act = False
                if r["day_ready"] == "True" and left:
                    remain = 92 - i
                    if left >= remain:
                        act = True
                    else:
                        history = [float(r[score]) for r in cal[city] if abs(int(r["day"]) - i) <= 15]
                        act = float(r[score]) > linear_quantile(history, 1 - left / remain)
                assert act == (float(r["chronological_" + score]) == 1), (city, year, score, i)
                if act:
                    left -= 1
                    if r["target"]:
                        captured += float(r["target"])
                    else:
                        unknown += 1
            output = stored[(city, year, score)]
            assert isclose(captured, float(output["captured"]), abs_tol=1e-8)
            assert int(float(output["selected"])) == 9 - left
            assert int(float(output["unknown_target"])) == unknown
            flags += 9 - left
    return len(exams), flags, len(names)


if __name__ == "__main__":
    n = verify_vienna()
    g, flags, rules = verify_chronology()
    print(f"Rechecked {n} Vienna paired nights against source flags and hourly data.")
    print(f"Independently replayed {rules} rules over {g} full calendars and verified {flags} selections.")
