from pathlib import Path

import geopandas as gpd
import matplotlib.colors as colors
import numpy as np
import pandas as pd
from PIL import Image
import rasterio
from rasterio.transform import from_bounds, from_origin
from scipy.spatial.distance import cdist
from shapely import contains_xy
from shapely.geometry import MultiPoint, box

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures"
LAYERS = OUT / "freiburg_layers"
UTM = 25832
BOUNDS = (407900, 5309150, 419200, 5321900)
ZOOM = (412930, 5315880, 415360, 5317830)
STOPS = ["#f8e9c9", "#edba76", "#d47a3f", "#ac4d32", "#762738", "#47183a"]
RAMP = colors.LinearSegmentedColormap.from_list("night_heat", STOPS)
MATRIX = np.array([
    ["#e5e8e7", "#eebea0", "#c97c50"],
    ["#b6d8d6", "#b5a7ae", "#a26177"],
    ["#58a8ac", "#628497", "#56396c"],
])


def map_data():
    sites = pd.read_csv(ROOT.parent / "results/freiburg_sites.csv").query("included").set_index("station_id")
    nights = pd.read_csv(ROOT.parent / "results/freiburg_nightly.csv.gz")
    outcome = nights.pivot(index="date", columns="station_id", values="target")[sites.index].dropna()
    evening = nights.groupby("date").urban_18
    assert evening.nunique().max() == 1
    score = evening.first().loc[outcome.index]
    quota = int(np.floor(.1 * len(outcome) + .5))
    cutoff = score.nlargest(quota).iloc[-1]
    above, tied = score.gt(cutoff), score.eq(cutoff)
    weights = above.astype(float) + tied * ((quota - above.sum()) / tied.sum())
    assert len(sites) == 31 and len(outcome) == 82 and np.isclose(weights.sum(), 8)
    sites["mean_h20"] = outcome.mean()
    sites["uncovered_pct"] = 100 * (1 - outcome.mul(weights, axis=0).sum() / outcome.sum())
    sites = gpd.GeoDataFrame(sites, geometry=gpd.points_from_xy(sites.station_lon, sites.station_lat), crs=4326)
    sites = sites.to_crs(UTM)
    sites["easting_m"], sites["northing_m"] = sites.geometry.x, sites.geometry.y
    for column in ("mean_h20", "uncovered_pct"):
        cuts = np.round(sites[column].quantile([1/3, 2/3]).to_numpy(), 1)
        print(column, cuts)
        sites[column + "_class"] = np.searchsorted(cuts, sites[column], side="right")
    sites["heat_color"] = [colors.to_hex(RAMP((v - 1) / 13)) for v in sites.mean_h20]
    sites["joint_color"] = MATRIX[sites.uncovered_pct_class, sites.mean_h20_class]
    sites["map_label"] = sites.index.map({"FRWILD":"Wildtal","FRWITT":"Wittnau"}).fillna("")
    return sites, weights


def predict(distances, values):
    nearest = np.argsort(distances, axis=1)[:, :4]
    distance = np.take_along_axis(distances, nearest, axis=1)
    weights = 1 / np.maximum(distance, .01) ** 2
    return np.sum(weights * values[nearest], axis=1) / weights.sum(axis=1), distance


def check_interpolation(xy, sites):
    distances = cdist(xy, xy)
    rows = []
    for radius in (0, 1000):
        keep = distances > radius
        assert keep.sum(axis=1).min() >= 4
        for metric in ("mean_h20", "uncovered_pct"):
            values = sites[metric].to_numpy()
            estimate, _ = predict(np.where(keep, distances, np.inf), values)
            baseline = (keep @ values) / keep.sum(axis=1)
            rows.extend((sid, radius, metric, v, p, m) for sid,v,p,m
                        in zip(sites.index, values, estimate, baseline, strict=True))
    cols = ["station_id", "excluded_radius_m", "metric", "observed", "idw", "training_mean"]
    checks = pd.DataFrame(rows, columns=cols)
    checks["absolute_error_idw"] = (checks.idw - checks.observed).abs()
    checks["absolute_error_training_mean"] = (checks.training_mean - checks.observed).abs()
    checks.to_csv(OUT / "freiburg_interpolation_validation.csv", index=False, float_format="%.9f")
    print(checks.groupby(["metric","excluded_radius_m"])[
        ["absolute_error_idw","absolute_error_training_mean"]].agg(["mean","max"]).round(3).to_string())


def main():
    LAYERS.mkdir(exist_ok=True)
    sites, weights = map_data()
    xy = sites[["easting_m", "northing_m"]].to_numpy()
    check_interpolation(xy, sites)
    districts = gpd.read_file(OUT / "freiburg_districts.geojson").to_crs(UTM)
    assert len(districts) == 42 and districts.is_valid.all()
    city = districts.geometry.union_all()
    sites["within_municipality"] = sites.within(city)
    assert sites.within_municipality.sum() == 29
    columns = ["station_lat","station_lon","easting_m","northing_m","within_municipality",
               "mean_h20","uncovered_pct","mean_h20_class","uncovered_pct_class"]
    sites[columns].to_csv(OUT / "freiburg_map_sites.csv", float_format="%.9f")
    weights.rename("urban_18_selection_weight").to_csv(OUT / "freiburg_map_nights.csv")
    sites[columns + ["station_name","heat_color","joint_color","map_label","geometry"]].to_file(
        LAYERS / "stations.gpkg", layer="stations")
    districts[["nr","geometry"]].to_file(LAYERS / "districts.gpkg", layer="districts")
    gpd.GeoDataFrame({"name":["Freiburg im Breisgau"]}, geometry=[city], crs=UTM).to_file(
        LAYERS / "municipality.gpkg", layer="municipality")
    gpd.GeoDataFrame({"name":["Central station detail"]}, geometry=[box(*ZOOM)], crs=UTM).to_file(
        LAYERS / "detail_extent.gpkg", layer="detail_extent")
    xmin,ymin,xmax,ymax = BOUNDS
    gx, gy = np.arange(xmin+50,xmax,100), np.arange(ymax-100,ymin,-100)
    xx,yy = np.meshgrid(gx,gy)
    estimate, near = predict(cdist(np.c_[xx.ravel(),yy.ravel()],xy),sites.mean_h20.to_numpy())
    inside = contains_xy(city,xx,yy) & contains_xy(MultiPoint(xy).convex_hull,xx,yy)
    support = inside & ((near[:,0]<=1250)&(near[:,2]<=2500)).reshape(xx.shape)
    surface = np.where(support,estimate.reshape(xx.shape),-9999).astype("float32")
    with rasterio.open(LAYERS/"heat_surface.tif","w",driver="GTiff",width=len(gx),height=len(gy),
                       count=1,dtype=surface.dtype,crs=f"EPSG:{UTM}",nodata=-9999,
                       transform=from_origin(gx[0]-50,gy[0]+50,100,100),compress="deflate") as f:
        f.write(surface,1)
    with Image.open(OUT/"freiburg_orthophoto.jpg") as image:
        bands=np.array(image.convert("RGB"))
    height,width,_ = bands.shape
    with rasterio.open(LAYERS/"orthophoto.tif","w",driver="GTiff",width=width,height=height,
                       count=3,dtype=bands.dtype,crs=f"EPSG:{UTM}",photometric="RGB",
                       transform=from_bounds(*BOUNDS,width,height),compress="jpeg",jpeg_quality=93) as f:
        f.write(np.moveaxis(bands,-1,0))
    print("Grid cells displayed",int(support.sum()),"from",int(contains_xy(city,xx,yy).sum()))


if __name__ == "__main__":
    main()
