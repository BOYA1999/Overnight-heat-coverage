from concurrent.futures import ThreadPoolExecutor
from csv import DictWriter
from hashlib import sha256
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen
import time

ROOT = Path(__file__).resolve().parent
DEST = ROOT / "external"
URL = "https://dataset.api.hub.geosphere.at/v1/station/historical/klima-v2-1h"
PARAMS = "tl,tl_flag,rf,rf_flag"
STATIONS = "5925,5990,5972"


def download(job):
    name, url, header = job
    path = DEST / name
    if path.is_file():
        data = path.read_bytes()
    else:
        for attempt in range(3):
            try:
                with urlopen(url, timeout=30) as response:
                    data = response.read()
                break
            except OSError:
                if attempt == 2:
                    raise
                time.sleep(1)
        if not data.startswith(header):
            raise ValueError(f"Unexpected response from {url}")
        path.write_bytes(data)
    if not data.startswith(header):
        raise ValueError(f"Invalid data in {name}")
    return {"file": name, "url": url, "bytes": len(data), "sha256": sha256(data).hexdigest()}


def main():
    DEST.mkdir(exist_ok=True)
    jobs = [
        ("parameters.csv", URL + "/metadata/parameters", b"Parameter"),
        ("stations.csv", URL + "/metadata/stations", b"id,Synop"),
    ]
    for year in range(2017, 2026):
        query = urlencode(dict(parameters=PARAMS, station_ids=STATIONS, output_format="csv",
                               start=f"{year}-05-31T22:00:00Z", end=f"{year}-09-01T05:00:00Z"))
        jobs.append((f"geosphere_{year}.csv", f"{URL}?{query}", b"time,station,tl,tl_flag,rf,rf_flag"))
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(download, jobs))
    with (DEST / "sources.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = DictWriter(stream, list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"Preserved {len(results)} GeoSphere files for station IDs {STATIONS}.")


if __name__ == "__main__":
    main()
