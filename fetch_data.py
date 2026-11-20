from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
from urllib.request import urlopen
import csv
import time

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
BASE = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/"
H = "climate/hourly/air_temperature/historical/"
U = "climate_urban/hourly/air_temperature/recent/"
P = "climate_urban_prov/hourly/air_temperature/recent/"

SOURCES = [
    ("Berlin", "urban", "00399", U + "stundenwerte_air_temperature_399_akt.zip"),
    ("Berlin", "reference", "00427", H + "stundenwerte_TU_00427_19730101_20251231_hist.zip"),
    ("Dresden", "urban", "15818", P + "stundenwerte_air_temperature_15818_akt.zip"),
    ("Dresden", "urban", "15818", U + "stundenwerte_air_temperature_15818_akt.zip"),
    ("Dresden", "reference", "01048", H + "stundenwerte_TU_01048_19730101_20251231_hist.zip"),
    ("Frankfurt", "urban", "01424", H + "stundenwerte_TU_01424_20080801_20251231_hist.zip"),
    ("Frankfurt", "reference", "01420", H + "stundenwerte_TU_01420_19810101_20251231_hist.zip"),
    ("Freiburg", "urban", "13667", U + "stundenwerte_air_temperature_13667_akt.zip"),
    ("Freiburg", "reference", "01443", H + "stundenwerte_TU_01443_19510101_20251231_hist.zip"),
    ("Hamburg", "urban", "15973", P + "stundenwerte_air_temperature_15973_akt.zip"),
    ("Hamburg", "urban", "19844", U + "stundenwerte_air_temperature_19844_akt.zip"),
    ("Hamburg", "reference", "01975", H + "stundenwerte_TU_01975_19490101_20251231_hist.zip"),
    ("Hannover", "urban", "15811", P + "stundenwerte_air_temperature_15811_akt.zip"),
    ("Hannover", "urban", "15811", U + "stundenwerte_air_temperature_15811_akt.zip"),
    ("Hannover", "reference", "02014", H + "stundenwerte_TU_02014_19490101_20251231_hist.zip"),
    ("Munich", "urban", "03379", H + "stundenwerte_TU_03379_19970701_20251231_hist.zip"),
    ("Munich", "reference", "01262", H + "stundenwerte_TU_01262_19920517_20251231_hist.zip"),
]

DOCUMENTS = [
    "https://www.dwd.de/EN/ourservices/urban_heatisland/cdc_descrip_en.pdf?__blob=publicationFile&v=3",
    BASE + "climate/hourly/air_temperature/DESCRIPTION_obsgermany_climate_hourly_air_temperature_en.pdf",
    BASE + U + "DESCRIPTION_obsgermany_climate_urban_hourly_tu_recent_en.pdf",
    BASE + P + "DESCRIPTION_obsgermany-climate_urban_prov-hourly-tu-recent_en.pdf",
]

def fetch(url, path):
    if path.exists():
        return path.read_bytes()
    for attempt in range(3):
        try:
            with urlopen(url, timeout=25) as stream:
                content = stream.read()
            if not content.startswith(b"PK" if path.suffix == ".zip" else b"%PDF"):
                raise ValueError(f"Unexpected response for {url}")
            path.write_bytes(content)
            return content
        except OSError:
            if attempt == 2:
                raise
            time.sleep(1)

def download(entry):
    city, role, station, relative = entry
    product = relative.split("/")[0]
    name = product + "_" + relative.rsplit("/", 1)[-1]
    data = fetch(BASE + relative, RAW / name)
    return (city, role, station, product, name, BASE + relative, len(data), sha256(data).hexdigest())

if __name__ == "__main__":
    RAW.mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=6) as pool:
        rows = list(pool.map(download, SOURCES))
        for url in DOCUMENTS:
            fetch(url, RAW / url.split("?")[0].rsplit("/", 1)[-1])
    with (RAW / "sources.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("city", "role", "station_id", "product", "file", "url", "bytes", "sha256"))
        writer.writerows(rows)
    print(f"{len(rows)} DWD archives documented. Total size: {sum(r[-2] for r in rows):,} bytes.")
