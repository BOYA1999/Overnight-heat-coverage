from collections import defaultdict
from csv import DictReader
from datetime import date, timedelta
from email.utils import parsedate_to_datetime
from gzip import open as open_gzip
from math import floor, isclose
from pathlib import Path
from statistics import mean
import json

ROOT = Path(__file__).resolve().parent
O = ROOT / "results"
NIGHT = tuple(range(22, 30))
LEADS = tuple(range(14, 22))
SCORE = ("urban_max", "urban_18", "gfs_raw", "gfs_bias", "gfs_update", "ceiling")


def csv(path, compressed=False):
    reader = open_gzip if compressed else open
    with reader(path, "rt", newline="", encoding="utf-8") as stream:
        return list(DictReader(stream))


def float_eq(x, y):
    return isclose(float(x), float(y), abs_tol=4e-8, rel_tol=1e-10)


def threshold(values, q):
    x = sorted(values)
    p = q*(len(x)-1)
    i = floor(p)
    return x[i] + (p-i)*(x[min(i+1,len(x)-1)]-x[i])


def check_coefficients(rows, coefficients):
    train = [r for r in rows if r["year"] == "2024" and r["city"] != "Vienna" and r["matched"] == "True"]
    assert len(train) == 582
    x = [float(r["urban_18"])-float(r["gfs_10"]) for r in train]
    center = mean(x)
    for r, hour, lead in zip(coefficients,NIGHT,LEADS):
        y = [float(row[f"t_{hour}"])-float(row[f"gfs_{lead}"]) for row in train]
        bias = mean(y)
        slope = sum((a-center)*(b-bias) for a,b in zip(x,y))/sum((a-center)**2 for a in x)
        assert float_eq(r["bias_correction"], bias)
        assert float_eq(r["update_slope"], slope)
        assert float_eq(r["update_intercept"], bias-slope*center)


def degree(vals, t):
    return sum(max(v-t,0) for v in vals)


def captured(scores, outcomes, k):
    group = defaultdict(list)
    for x, y in zip(scores, outcomes):
        group[x].append(y)
    total = 0
    for x in sorted(group, reverse=True):
        selected = min(k, len(group[x]))
        total += sum(group[x])*(selected/len(group[x]))
        k -= selected
        if k == 0:
            return total
    raise ValueError("Quota exceeds candidates")


def check_rank(rows, coefficients):
    groups = defaultdict(list)
    for row in rows:
        if row["year"] != "2024" and row["matched"] == "True":
            groups[(row["city"], row["year"])].append(row)
    results = {(r["city"],r["year"],r["base"],r["score"]):r for r in csv(O/"rank_city_year.csv")}
    verified = 0
    for (city, year), group in groups.items():
        if len(group) < 83:
            assert (city,year,"20","gfs_update") not in results
            continue
        obs = [[float(r[f"t_{h}"]) for h in NIGHT] for r in group]
        raw = [[float(r[f"gfs_{l}"]) for l in LEADS] for r in group]
        delta = [float(r["urban_18"])-float(r["gfs_10"]) for r in group]
        for t in (18,20,22):
            y = [degree(q,t) for q in obs]
            preds = {"gfs_raw": [degree(q,t) for q in raw]}
            preds["gfs_bias"] = [degree([v+float(c["bias_correction"]) for v,c in zip(q,coefficients)],t)
                                 for q in raw]
            preds["gfs_update"] = [
                degree([v+float(c["update_intercept"])+z*float(c["update_slope"])
                        for v,c in zip(q,coefficients)],t) for q,z in zip(raw,delta)]
            preds["urban_18"] = [float(r["urban_18"]) for r in group]
            preds["urban_max"] = [float(r["urban_max"]) for r in group]
            preds["ceiling"] = y
            k = floor(.1*len(group)+.5)
            for name in SCORE:
                row = results[(city,year,str(t),name)]
                assert (int(row["n"]),int(row["slots"])) == (len(group),k)
                assert float_eq(row["heat"],sum(y))
                assert float_eq(row["captured"],captured(preds[name],y,k)), (city,year,t,name)
                verified += 1
    assert verified == len(results)
    return verified


def check_decisions():
    other = defaultdict(list)
    past = defaultdict(list)
    for row in csv(O/"prior_scores.csv"):
        assert row["year"] == "2024" and row["ready"] == "True"
        past[row["city"]].append(row)
    for row in csv(O/"decisions.csv"):
        other[(row["city"],int(row["year"]))].append(row)
    steps = 0
    for (city, year), rows in other.items():
        rows.sort(key=lambda x:x["date"])
        for name in SCORE[:-1]:
            old = [float(r[name]) for r in past[city]]
            remain = 9
            assert len(rows) == 92
            for i,r in enumerate(rows):
                assert date.fromisoformat(r["date"]) == date(year,6,1)+timedelta(days=i)
                days = 92-i
                expected = r["ready"] == "True" and remain > 0 and (
                    remain >= days or float(r[name]) > threshold(old,1-remain/days))
                assert expected == bool(float(r["chronological_"+name])), (city,year,name,i)
                remain -= expected
                steps += 1
    return steps


def check_archive():
    checked = 0
    for p in sorted((ROOT/"forecast_points").glob("*.json")):
        fields = json.loads(p.read_text(encoding="utf-8"))["fields"]
        assert {f["lead"] for f in fields} == {10,*LEADS}
        for f in fields:
            last = parsedate_to_datetime(f["modified_http"])
            assert f["init"] == p.stem+"T06:00:00Z"
            assert last.date().isoformat() == p.stem and 6 <= last.hour < 16
            checked += 1
    assert checked == 2484
    return checked


if __name__ == "__main__":
    forecast_n = check_archive()
    rows = csv(O/"nightly_forecast.csv.gz",True)
    coefficients = csv(O/"coefficients.csv")
    check_coefficients(rows,coefficients)
    rank = check_rank(rows,coefficients)
    steps = check_decisions()
    print(f"{forecast_n} initialization-verified fields; {rank} independent ranking totals; "
          f"{steps} chronological decisions and the frozen 2024 coefficients independently verified.")
