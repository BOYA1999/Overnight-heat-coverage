from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from hashlib import sha256
from pathlib import Path
from threading import Lock, local
import csv
import json
import sys

import eccodes as ec
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parent
DEST = ROOT / "forecast_points"
AWS = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
HOURS = (10, *range(14, 22))
THREAD = local()
DECODER = Lock()

with (ROOT / "forecast_sites.csv").open(encoding="utf-8", newline="") as file:
    SITES = {r["city"]: (float(r["lat"]), float(r["lon"])) for r in csv.DictReader(file)}


def http():
    if not hasattr(THREAD, "session"):
        THREAD.session = requests.Session()
        adapter = HTTPAdapter(max_retries=Retry(total=3, backoff_factor=.6,
              status_forcelist=(429, 500, 502, 503, 504), allowed_methods={"GET"}))
        THREAD.session.mount("https://", adapter)
    return THREAD.session


def decode(message, day, lead, pilot=False):
    with DECODER:
        gid = ec.codes_new_from_message(message)
        try:
            getters = lambda *names: tuple(ec.codes_get(gid, name) for name in names)
            valid_day = day + timedelta(days=(6 + lead) // 24)
            assert getters("dataDate", "dataTime", "forecastTime") == (
                int(day.strftime("%Y%m%d")), 600, lead)
            assert getters("validityDate", "validityTime") == (
                int(valid_day.strftime("%Y%m%d")), ((6 + lead) % 24) * 100)
            assert getters("level", "typeOfLevel", "stepType", "units") == (
                2, "heightAboveGround", "instant", "K")
            assert getters("gridType", "Ni", "Nj", "jPointsAreConsecutive",
                           "alternativeRowScanning", "iScansNegatively", "jScansPositively") == (
                "regular_ll", 1440, 721, 0, 0, 0, 0)
            assert getters("latitudeOfFirstGridPointInDegrees", "longitudeOfFirstGridPointInDegrees",
                           "iDirectionIncrementInDegrees", "jDirectionIncrementInDegrees") == (
                90.0, 0.0, .25, .25)
            points = {}
            full = ec.codes_get_values(gid)
            for city, (lat, lon) in SITES.items():
                j, i = round((90 - lat)/.25), round((lon % 360)/.25) % 1440
                kelvin = float(full[1440*j+i])
                assert 230 <= kelvin <= 330, (city, day, lead, kelvin)
                if pilot:
                    nearest = ec.codes_grib_find_nearest(gid, lat, lon)[0]
                    assert abs(nearest["value"] - kelvin) < 1e-9, (city, nearest, kelvin)
                points[city] = round(kelvin - 273.15, 10)
        finally:
            ec.codes_release(gid)
    return points


def get_field(day, lead, pilot=False):
    url = f"{AWS}/gfs.{day:%Y%m%d}/06/atmos/gfs.t06z.pgrb2.0p25.f{lead:03d}"
    listing = http().get(url + ".idx", timeout=(8, 20))
    listing.raise_for_status()
    rows = listing.text.splitlines()
    key = f":d={day:%Y%m%d}06:TMP:2 m above ground:{lead} hour fcst:"
    matches = [i for i, line in enumerate(rows) if key in line]
    assert len(matches) == 1, (day, lead, matches)
    i = matches[0]
    start, stop = int(rows[i].split(":")[1]), int(rows[i+1].split(":")[1]) - 1
    res = http().get(url, headers={"Range": f"bytes={start}-{stop}"}, timeout=(8, 28))
    res.raise_for_status()
    assert res.status_code == 206 and res.headers["Content-Range"].startswith(f"bytes {start}-{stop}/")
    data = res.content
    assert len(data) == stop-start+1 and data.startswith(b"GRIB") and data.endswith(b"7777")
    points = decode(data, day, lead, pilot=pilot)
    if pilot:
        (ROOT / "gfs_example.grib2").write_bytes(data)
    return dict(lead=lead, init=f"{day}T06:00:00Z", url=url, byte_start=start, byte_stop=stop,
                inventory=rows[i], modified_http=res.headers["Last-Modified"],
                sha256=sha256(data).hexdigest(), point_c=points)


def download_day(day):
    path = DEST / f"{day}.json"
    cache = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"fields": [], "issues": {}}
    ready = {r["lead"] for r in cache["fields"]}
    for lead in HOURS:
        if lead in ready:
            continue
        try:
            cache["fields"].append(get_field(day, lead))
            cache["issues"].pop(str(lead), None)
        except requests.RequestException as exc:
            cache["issues"][str(lead)] = str(exc)
    cache["fields"].sort(key=lambda f: f["lead"])
    scratch = path.with_suffix(".tmp")
    scratch.write_text(json.dumps(cache, separators=(",", ":")), encoding="utf-8")
    scratch.replace(path)
    return len(cache["fields"]), len(cache["issues"])


def main():
    if sys.argv[1:] == ["--pilot"]:
        result = get_field(date(2025, 7, 1), 14, pilot=True)
        print(json.dumps(result))
        return
    DEST.mkdir(exist_ok=True)
    dates = [date(y, 6, 1) + timedelta(days=i) for y in (2024, 2025, 2026) for i in range(92)]
    fields = issues = 0
    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = [pool.submit(download_day, day) for day in dates]
        for j, future in enumerate(as_completed(futures), 1):
            n, m = future.result()
            fields, issues = fields+n, issues+m
            if j % 30 == 0 or j == len(dates):
                print(f"{j}/{len(dates)} run dates, {fields} fields, {issues} retrieval errors", flush=True)


if __name__ == "__main__":
    main()
