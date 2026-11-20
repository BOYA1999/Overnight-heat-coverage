import argparse
from hashlib import sha256
from io import BytesIO
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
from zipfile import ZipFile

import pandas as pd
from pandas.testing import assert_frame_equal

ROOT = Path(__file__).resolve().parent
OBS = "Revision_20261120"
FORECAST = "Forecast_Revision_20261120"
FIGURES = "Reference_Revision_20261120"
V5 = "Reviewer_Revision_V5_20261120"
SCRIPTS = (
    "analysis.py", "verify_results.py", "spatial_check.py",
    f"{OBS}/review_analysis.py", f"{OBS}/verify_revision.py",
    f"{FORECAST}/forecast_analysis.py", f"{FORECAST}/fetch_icon.py",
    f"{FORECAST}/regional_comparison.py", f"{FORECAST}/verify_forecast.py",
    f"{V5}/reanalyse.py", f"{V5}/check_station_models.py", f"{V5}/plot_revision.py",
)
DATA_FOLDERS = (
    "raw/", f"{OBS}/external/", f"{FORECAST}/obs_2026/",
    f"{FORECAST}/forecast_points/", f"{FORECAST}/icon_d2/",
)
REFERENCE_INPUTS = (
    f"{FIGURES}/figures/freiburg_map_sites.csv",
    f"{V5}/results/gfs_grid_height.csv",
)


def digest(contents):
    return sha256(contents).hexdigest()


def stage(archive, hashes, output):
    for name in (*SCRIPTS, "stations.csv", f"{FORECAST}/forecast_sites.csv", f"{FIGURES}/plot_results.py"):
        dest = output / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)
    for name, checksum in hashes.items():
        if name not in REFERENCE_INPUTS and not name.startswith(DATA_FOLDERS):
            continue
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
            raise ValueError(f"Unsafe input path in archive: {name}")
        contents = archive.read(name)
        if digest(contents) != checksum:
            raise ValueError(f"Input checksum mismatch: {name}")
        dest = output / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(contents)
    if len(list((output / FORECAST / "icon_d2").glob("2026-*.json"))) != 92:
        raise ValueError("The frozen ICON-D2 snapshots must cover all 92 evaluation dates.")
    (output / V5 / "figures").mkdir()


def run(archive, output):
    with ZipFile(archive) as z:
        hashes = {name: checksum for checksum, name in
                  (line.split("  ", 1) for line in z.read("SHA256SUMS_V5.txt").decode("utf-8").splitlines())}
        if not all(name in hashes for name in REFERENCE_INPUTS):
            raise ValueError("Supply the V5 replication archive with the frozen review inputs.")
        output.mkdir(parents=True)
        stage(z, hashes, output)
        for script in SCRIPTS:
            print(f"Running {script}", flush=True)
            subprocess.run([sys.executable, str(output / script)], check=True, cwd=output)
        results = [name for name in hashes if name.startswith(f"{V5}/results/")
                   and name.endswith(".csv") and name not in REFERENCE_INPUTS]
        exact = 0
        for name in results:
            expected = z.read(name)
            if digest(expected) != hashes[name]:
                raise ValueError(f"Reference checksum mismatch: {name}")
            actual = (output / name).read_bytes()
            exact += actual == expected
            assert_frame_equal(pd.read_csv(BytesIO(actual)), pd.read_csv(BytesIO(expected)),
                               check_exact=False, rtol=1e-10, atol=1e-9, obj=name)
        print(f"Verified {len(results)} V5 result tables ({exact} byte-identical).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay the overnight heat study from its frozen V5 inputs.")
    parser.add_argument("archive", type=Path, help="Path to Urban_Climate_V5_Reproducibility_and_QGIS.zip")
    parser.add_argument("--output", type=Path, default=Path("replay-output"), help="New directory for data and results")
    args = parser.parse_args()
    run(args.archive.resolve(), args.output.resolve())
