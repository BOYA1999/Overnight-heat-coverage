from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from hashlib import sha256
from pathlib import Path
import json
import time

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
DEST = ROOT / "icon_d2"
URL = "https://single-runs-api.open-meteo.com/v1/forecast"
SITES = pd.read_csv(ROOT / "forecast_sites.csv")
LEADS = (10, *range(14, 22))


def retrieve(day):
    request = dict(latitude=",".join(SITES.lat.astype(str)), longitude=",".join(SITES.lon.astype(str)),
                   models="icon_d2", run=f"{day}T06:00", hourly="temperature_2m",
                   forecast_days=2, cell_selection="nearest",
                   elevation=",".join("nan" for _ in SITES.city), timezone="GMT")
    path = DEST / f"{day}.json"
    if not path.is_file():
        for j in range(3):
            try:
                response = requests.get(URL, params=request, timeout=(8, 25))
                if response.status_code in (403, 429):
                    raise RuntimeError(f"Service limit {response.status_code}: {response.text[:220]}")
                response.raise_for_status()
                body = response.content
                break
            except requests.RequestException:
                if j == 2:
                    raise
                time.sleep(1+j)
        payload = json.loads(body)
        assert isinstance(payload, list) and len(payload) == len(SITES)
        path.write_bytes(body)
        time.sleep(.45)
    body = path.read_bytes()
    payload = json.loads(body)
    rows = []
    valid = pd.Timestamp(f"{day}T06:00") + pd.to_timedelta(LEADS, unit="h")
    for site, item in zip(SITES.itertuples(index=False), payload):
        assert abs(item["latitude"]-site.lat) < .06 and abs(item["longitude"]-site.lon) < .06, (day, site)
        time_index = pd.DatetimeIndex(item["hourly"]["time"])
        assert time_index[0] == pd.Timestamp(f"{day}T06:00") and not time_index.has_duplicates
        assert item["hourly_units"]["temperature_2m"] == "°C" and item["utc_offset_seconds"] == 0
        y = pd.Series(item["hourly"]["temperature_2m"], index=time_index).reindex(valid)
        rows.append({"date": str(day), "city": site.city, **dict(zip([f"icon_{l}" for l in LEADS], y))})
    prepared = requests.Request("GET", URL, params=request).prepare()
    return {"file": path.name, "run_utc": f"{day}T06:00:00Z",
            "url": prepared.url, "sha256": sha256(body).hexdigest()}, rows


def main():
    DEST.mkdir(exist_ok=True)
    dates = [date(2026, 6, 1)+timedelta(days=i) for i in range(92)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        result = list(pool.map(retrieve, dates))
    pd.DataFrame([r[0] for r in result]).to_csv(DEST/"sources.csv", index=False)
    raw = pd.DataFrame([row for _, rows in result for row in rows])
    raw.to_csv(DEST/"points.csv", index=False)
    print(f"ICON-D2 {len(result)} initializations, {len(raw)} station-dates, "
          f"{raw.filter(like='icon_').notna().all(axis=1).sum()} complete profiles.")
    return raw


if __name__ == "__main__":
    main()
