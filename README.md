# Overnight heat coverage under fixed monitoring quotas

This repository contains the calculation code for the seven-city German study, the Vienna external comparison, the Freiburg station analysis, and the GFS and ICON-D2 forecast comparisons. Overnight heat is the sum of the hourly air-temperature exceedances above 20 °C across the eight observations from 22:00 through 05:00 local summer time. The analysis evaluates the heat captured by selecting a fixed number of nights.

## Reproduce the V5 results

Use Python 3.12. Obtain the frozen replication archive `Urban_Climate_V5_Reproducibility_and_QGIS.zip` from the study materials. The archive is distributed separately and supplies the observation snapshots, archived forecast fields and inputs for checking the Freiburg map classes and grid elevations.

```bash
python -m pip install -r requirements.txt
python replay.py /path/to/Urban_Climate_V5_Reproducibility_and_QGIS.zip
```

The command verifies source checksums, runs the calculation scripts from this repository in a new `replay-output/` directory and compares the V5 tables with the frozen reference. It keeps the original study files separate from the Git checkout. The offline replay uses the archive's extracted GFS grid-point temperatures and surface heights and its raw ICON-D2 run responses. It also regenerates the statistical Figures 2, 5 and 6. The two QGIS projects and their geographic layers are provided with the replication archive.

| Analysis | Entry point |
| --- | --- |
| German station matching, quotas, and main results | `analysis.py` |
| Freiburg station-level check | `spatial_check.py` |
| Vienna evaluation, shared allocation and calendar replay | `Revision_20261120/review_analysis.py` |
| GFS evaluation and matching of the regional forecast dates | `Forecast_Revision_20261120/forecast_analysis.py`, `Forecast_Revision_20261120/regional_comparison.py` |
| Paired uncertainty, forecast recalibration and grid-elevation sensitivity | `Reviewer_Revision_V5_20261120/reanalyse.py` |
| Cohort denominators and city-specific station models | `Reviewer_Revision_V5_20261120/check_station_models.py` |

## Data provenance

- German hourly observations and the urban–surroundings pairing are from the [DWD Climate Data Center](https://opendata.dwd.de/climate_environment/CDC/observations_germany/). The file URLs and snapshot hashes are in [the station manifest](raw/sources.csv).
- Vienna hourly observations are from [GeoSphere Austria hourly data v2](https://doi.org/10.60669/9bdm-yq93). Freiburg uses the [street-level observations](https://doi.org/10.5281/zenodo.12732565) and [station documentation](https://doi.org/10.5281/zenodo.12732552).
- GFS comes from the [NOAA Open Data archive](https://registry.opendata.aws/noaa-gfs-bdp-pds/). ICON-D2 comes from the [Open-Meteo Single Runs archive](https://open-meteo.com/en/docs/single-runs-api). Source fetchers are included for rebuilding inputs. GFS retrieval requires Python ecCodes with its native library. The offline replay uses cached point data and does not need ecCodes.

Re-downloading rolling station products can produce different files. Use the frozen inputs for an exact replay of the paper. Code uses the repository's [MIT license](LICENSE). Observation, forecast and map inputs retain their source terms.
