from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
V2 = BASE / "Revision_20261120"
V3 = BASE / "Forecast_Revision_20261120"
V4 = BASE / "Reference_Revision_20261120"
OUT = ROOT / "results"
DRAWS = 10_000
RNG = np.random.default_rng(8192026)

sys.dont_write_bytecode = True
sys.path.insert(0, str(BASE))
from analysis import group_rank, weights

SCORES = ["urban_max", "ref_max", "urban_18", "ref_18", "oracle"]
DIFFS = {"urban_timing": ("urban_18", "urban_max"), "reference_timing": ("ref_18", "ref_max"),
         "daytime_siting": ("urban_max", "ref_max"), "evening_siting": ("urban_18", "ref_18"),
         "time_by_site": ("urban_timing", "reference_timing")}
OBS = [f"t_{h}" for h in range(22, 30)]
GFS = [f"gfs_{h}" for h in range(14, 22)]
ICON = [f"icon_{h}" for h in range(14, 22)]


def csv(base, name):
    return pd.read_csv(base/"results"/(name+".csv"))


def save(records, name):
    pd.DataFrame(records).to_csv(OUT/(name+".csv"), index=False, float_format="%.10g")


def interval(estimate, sample):
    lo, hi = np.quantile(sample, [.025, .975])
    return {"estimate": float(estimate), "lower": float(lo), "upper": float(hi)}


def block_indices(n, block, n_draw=DRAWS):
    starts = RNG.integers(n, size=(n_draw, (n + block - 1)//block, 1))
    return ((starts + np.arange(block)) % n).reshape(n_draw, -1)[:, :n]


def quota_heat(target, score):
    n = np.isfinite(target).sum(axis=1)
    k = np.maximum(1, np.floor(.10*n + .5).astype(int))
    assert (k <= n).all()
    score = np.where(np.isfinite(target), score, -np.inf)
    cut = np.take_along_axis(np.sort(score, axis=1), (score.shape[1]-k)[:, None], axis=1)
    gt, eq = score > cut, score == cut
    frac = (k - gt.sum(axis=1))/eq.sum(axis=1)
    return np.sum(np.nan_to_num(target) * (gt + eq*frac[:, None]), axis=1)


def cluster_parts(parts, metrics, diffs, labels):
    columns = ["burden", *metrics]
    x=parts[columns].to_numpy()
    ix=np.random.default_rng(481104).integers(len(x),size=(DRAWS,len(x)))
    draws=x[ix].sum(axis=1)
    totals=x.sum(axis=0)
    boot={f: 100*draws[:,j]/draws[:,0] for j,f in enumerate(metrics,1)}
    point={f: 100*totals[j]/totals[0] for j,f in enumerate(metrics,1)}
    for f,(a,b) in diffs.items():
        boot[f],point[f]=boot[a]-boot[b],point[a]-point[b]
    return [{**labels,"clusters":len(parts),"metric":f,**interval(point[f],boot[f])} for f in point]


def historical():
    raw=pd.read_csv(BASE/"results/nightly.csv.gz")
    assert len(raw)==4878
    check=np.maximum(raw[[f"u_{h}" for h in range(22,30)]].to_numpy()-20,0).sum(axis=1)
    assert np.allclose(check,raw.target)
    rows=[]
    for fraction in (.05,.10,.20):
        tab=group_rank(raw,fraction)
        for unit,group in [("summer",tab.groupby("year").sum(numeric_only=True)),("city-summer",tab)]:
            rows.extend(cluster_parts(group,SCORES,DIFFS,{"cohort":"Germany","quota":fraction,"resample":unit}))
    vienna=csv(V2,"vienna_Schwechat_city_year")
    rows.extend(cluster_parts(vienna,SCORES,DIFFS,{"cohort":"Vienna","quota":.10,"resample":"summer"}))
    alloc=csv(V2,"allocation_year").rename(columns={"heat":"burden"})
    rows.extend(cluster_parts(alloc,["local","pooled"],{"shared_minus_local":("pooled","local")},
        {"cohort":"Germany","quota":.10,"resample":"summer, shared-slot comparison"}))
    save(rows,"historical_intervals")
    city=csv(V2,"allocation_city_year")
    save(city.groupby("city",as_index=False).sum(numeric_only=True).drop(columns="year"),"allocated_slots")
    legacy=csv(BASE,"uncertainty").set_index("metric")
    df=pd.DataFrame(rows)
    primary=df[(df.cohort=="Germany") & (df.quota==.10) & (df["resample"]=="summer")].set_index("metric")
    assert np.allclose(primary.loc[["urban_timing","oracle"],["estimate","lower","upper"]],
        legacy.loc[["urban_timing","oracle"],["estimate","lower","upper"]])


def evaluate(data, scores, profiles):
    rows=[]
    for (city,year),g in data.groupby(["city","year"]):
        y=g.target.to_numpy()
        k=int(np.floor(.10*len(g)+.5))
        for key in scores:
            row={"city":city,"year":year,"score":key,"n":len(g),"slots":k,"heat":y.sum(),
                 "captured": np.dot(y,weights(g[key],k))}
            if key in profiles:
                err=g[profiles[key]].to_numpy()-g[OBS].to_numpy()
                row.update(abs_error=np.abs(err).sum(),sum_error=err.sum())
            rows.append(row)
    return pd.DataFrame(rows)


def collapse(frame, label, gain):
    t=frame.assign(cohort=np.where(frame.city.eq("Vienna"),"Vienna","Germany")).groupby(
        ["cohort","score"],as_index=False).sum(numeric_only=True,min_count=1).drop(columns="year")
    t["case"]=label
    t["coverage_pct"]=100*t.captured/t.heat
    if "abs_error" in t:
        t["mae_C"]=t.abs_error/(8*t.n)
        t["bias_C"]=t.sum_error/(8*t.n)
    w=t.pivot(index="cohort",columns="score",values="coverage_pct")
    t["gain_pp"]=t.cohort.map(w[gain[0]]-w[gain[1]])
    return t


def degree(t):
    return np.maximum(t-20,0).sum(axis=1)


def forecast_data():
    all_=pd.read_csv(V3/"results/nightly_forecast.csv.gz")
    all_["date"]=pd.to_datetime(all_.date)
    data=all_.loc[all_.matched & all_.year.ge(2025)].copy()
    data=data.loc[data.groupby(["city","year"]).city.transform("size").ge(83)].copy()
    train=all_[all_.matched & all_.city.ne("Vienna") & all_.year.eq(2024)]
    assert len(train)==582 and len(train[train.city.eq("Hamburg")])==32
    forecast_rows,coefficient_rows=[],[]
    profiles={"gfs_raw":GFS}
    for label,cal in [("all 2024",train),("2024 without Hamburg",train[train.city.ne("Hamburg")])]:
        error=cal[OBS].to_numpy()-cal[GFS].to_numpy()
        beta=np.linalg.lstsq(np.c_[np.ones(len(cal)),cal.urban_18-cal.gfs_10],error,rcond=None)[0]
        coefficient_rows.extend({"calibration":label,"training_n":len(cal),"lead":lead,
                                 "bias_correction":mean,"update_intercept":intercept,"update_slope":slope}
            for lead,mean,intercept,slope in zip(range(14,22),error.mean(axis=0),beta[0],beta[1],strict=True))
        predictions={"bias":error.mean(axis=0),"update":np.c_[np.ones(len(data)),data.urban_18-data.gfs_10] @ beta}
        for score,correction in predictions.items():
            cols=[f"{score}_{h}" for h in range(8)]
            data[cols]=data[GFS].to_numpy()+correction
            if label=="all 2024":
                assert np.allclose(data["gfs_"+score],degree(data[cols].to_numpy()))
            data["gfs_"+score]=degree(data[cols].to_numpy())
            profiles["gfs_"+score]=cols
        f=evaluate(data,["gfs_raw","gfs_bias","gfs_update"],profiles)
        for year in ("Pooled",2025,2026):
            part=f if year=="Pooled" else f[f.year.eq(year)]
            rec=collapse(part,label,("gfs_update","gfs_bias"))
            rec["window"]=year
            rec["training_n"]=len(cal)
            forecast_rows.append(rec)
        if label=="all 2024":
            frozen=data.copy()
    save(pd.concat(forecast_rows,ignore_index=True),"hamburg_calibration")
    save(coefficient_rows,"gfs_review_coefficients")

    files=list(sorted((V3/"icon_d2").glob("2026-*.json")))
    sites=pd.read_csv(V3/"forecast_sites.csv",dtype={"station_id":str})
    elevation=[]
    for f in files:
        for s,r in zip(sites.itertuples(index=False),json.loads(f.read_bytes()),strict=True):
            elevation.append({"city":s.city,"icon_grid_lat":r["latitude"],"icon_grid_lon":r["longitude"],
                              "icon_grid_height":r["elevation"]})
    grid=pd.DataFrame(elevation).drop_duplicates()
    assert len(files)==92 and len(grid)==len(sites)==8
    z=pd.read_csv(BASE/"stations.csv",dtype={"station_id":str}).query("role == 'urban'").drop_duplicates("city",keep="last")
    station_z=dict(zip(z.city,z.elevation_m))
    station_z["Vienna"]=pd.read_csv(V2/"external/stations.csv").loc[lambda d:d.id.eq(5925),"Höhe [m]"].item()
    grid=grid.merge(csv(ROOT,"gfs_grid_height"),on="city",validate="one_to_one")
    grid["station_height"]=grid.city.map(station_z)
    for m in ("gfs","icon"):
        grid[m+"_minus_station_m"]=grid[m+"_grid_height"]-grid.station_height
    save(grid,"grid_heights")
    region=frozen[frozen.year.eq(2026)].merge(pd.read_csv(V3/"icon_d2/points.csv",parse_dates=["date"]),
        on=["date","city"],validate="one_to_one").dropna(subset=ICON)
    assert len(region)==644 and region.city.nunique()==7
    save(region[["city","date","target",*OBS,*GFS,*ICON]],"regional_matched")
    records=[]
    for c in (0.0,4.0,6.5,9.0):
        sample=region.copy()
        for key,cols in (("gfs",GFS),("icon",ICON)):
            zz=sample.city.map(grid.set_index("city")[key+"_minus_station_m"]).to_numpy()
            sample[cols]=sample[cols].to_numpy()+c/1000*zz[:,None]
            sample[key+"_raw"]=degree(sample[cols].to_numpy())
        r=collapse(evaluate(sample,["gfs_raw","icon_raw"],{"gfs_raw":GFS,"icon_raw":ICON}),
                   "raw" if c==0 else f"{c:g} K/km",("icon_raw","gfs_raw"))
        r["lapse_K_km"]=c
        records.append(r)
        if c==0:
            original=csv(V3,"regional_summary").rename(columns={"region":"cohort"})
            matched=r.merge(original,on=["cohort","score"],validate="one_to_one")
            for k in ("mae_C","coverage_pct"):
                assert len(matched)==4 and np.allclose(matched[k+"_x"],matched[k+"_y"])
    save(pd.concat(records,ignore_index=True),"lapse_sensitivity")
    f=region.copy()
    f["icon_raw"]=degree(f[ICON].to_numpy())
    follow=evaluate(f,["gfs_raw","gfs_bias","gfs_update","icon_raw"],profiles|{"icon_raw":ICON})
    restricted=follow.loc[~follow.city.isin(["Frankfurt","Munich","Vienna"])]
    summaries=[]
    for label,gain in [("update vs bias",("gfs_update","gfs_bias")),("ICON-D2 vs GFS",("icon_raw","gfs_raw"))]:
        a=collapse(restricted,"same four continuing-urban sites",gain)
        a["contrast"]=label
        summaries.append(a)
    save(pd.concat(summaries,ignore_index=True),"continuing_urban_2026")
    return frozen, f, profiles


def conditional_panel(data, scores, profiles, diffs, label, block):
    math={}
    years=sorted(data.year.unique())
    ix={y:block_indices(92,block) for y in years}
    for cohort,part in data.groupby(np.where(data.city.eq("Vienna"),"Vienna","Germany")):
        cols=["heat","n",*scores,*[s+"_mae" for s in profiles]]
        out={col: np.zeros(DRAWS) for col in cols}
        exact={col:0. for col in cols}
        for (city,year),g in part.groupby(["city","year"]):
            cal=pd.date_range(f"{year}-06-01",periods=92)
            a=g.set_index("date").reindex(cal)
            heat=a.target.to_numpy()
            ids=ix[year]
            y=heat[ids]
            out["heat"]+=np.nansum(y,axis=1)
            out["n"]+=np.isfinite(y).sum(axis=1)
            exact["heat"]+=g.target.sum()
            exact["n"]+=len(g)
            for score in scores:
                out[score]+=quota_heat(y,a[score].to_numpy()[ids])
                exact[score]+=np.dot(g.target,weights(g[score],int(np.floor(.1*len(g)+.5))))
            for score,profile in profiles.items():
                err=np.abs(a[profile].to_numpy()-a[OBS].to_numpy()).sum(axis=1)
                out[score+"_mae"]+=np.nansum(err[ids],axis=1)
                exact[score+"_mae"]+=np.nansum(err)
        reps={s:100*out[s]/out["heat"] for s in scores} | {s+"_mae":out[s+"_mae"]/(out["n"]*8) for s in profiles}
        centre={s:100*exact[s]/exact["heat"] for s in scores} | {s+"_mae":exact[s+"_mae"]/(exact["n"]*8) for s in profiles}
        for key,(a,b) in diffs.items():
            reps[key]=reps[a]-reps[b]
            centre[key]=centre[a]-centre[b]
        math[cohort]=[{"cohort":cohort,"sample":label,"block_days":block,"nights":int(exact["n"]),
                      "metric":s,**interval(centre[s],v)} for s,v in reps.items()]
    return [r for a in math.values() for r in a]


def freiburg():
    records=pd.read_csv(BASE/"results/freiburg_nightly.csv.gz",parse_dates=["date"])
    legacy=csv(BASE,"freiburg_sites").query("included")
    r=records.pivot(index="date",columns="station_id",values="target")
    common=r.dropna()
    score=records.groupby("date").urban_18.first()
    a=weights(score.loc[common.index],8)
    x,y=common.mean(),100*(1-common.mul(a,axis=0).sum()/common.sum())
    xcut,ycut=x.quantile([1/3,2/3]),y.quantile([1/3,2/3])
    rounded=(x>=round(xcut.iloc[-1],1)) & (y>=round(ycut.iloc[-1],1))
    exact=(x>=xcut.iloc[-1]) & (y>=ycut.iloc[-1])
    published=pd.read_csv(V4/"figures/freiburg_map_sites.csv").set_index("station_id").loc[x.index]
    assert len(x)==31 and len(common)==82 and np.allclose(published.mean_h20,x)
    assert np.allclose(published.uncovered_pct,y)
    assert np.array_equal(rounded,(published.mean_h20_class.eq(2) & published.uncovered_pct_class.eq(2)))
    save([{"n_sites":31,"n_common":len(common),"slots_common":8,"rho":spearmanr(x,y).statistic,
          "x_lo":xcut.iloc[0],"x_hi":xcut.iloc[1],"y_lo":ycut.iloc[0],"y_hi":ycut.iloc[1],
          "high_exact":int(exact.sum()),"high_rounded":int(rounded.sum()),
          "strictly_above_exact":int(((x>xcut.iloc[-1]) & (y>ycut.iloc[-1])).sum()),
          "on_upper_cut":",".join(x.index[(x==xcut.iloc[-1]) | (y==ycut.iloc[-1])]),
          "changed_high_high":int((rounded!=exact).sum()),"glassbach_h":x["FRGLBA"]}],"freiburg_map_audit")
    cal=pd.date_range("2023-06-01",periods=92)
    summaries=[]
    for days in (7,14):
        ix=block_indices(92,days)
        changes=[]
        for sid,ff in records.groupby("station_id"):
            ff=ff.set_index("date").reindex(cal)
            y=ff.target.to_numpy()[ix]
            ratio=100*(quota_heat(y,ff.urban_18.to_numpy()[ix])-quota_heat(y,ff.urban_max.to_numpy()[ix]))/np.nansum(y,axis=1)
            changes.append(ratio)
        summaries.append({"metric":"median station evening minus daytime","block_days":days,
                          **interval(legacy.urban_timing.median(),np.median(changes,axis=0))})
    save(summaries,"freiburg_intervals")


def main():
    historical()
    freiburg()
    frozen, region, profiles=forecast_data()
    rows=[]
    for days in (7,14):
        for label,part in [("GFS 2025–2026",frozen),("GFS 2025",frozen[frozen.year.eq(2025)]),
                           ("GFS 2026",frozen[frozen.year.eq(2026)])]:
            rows.extend(conditional_panel(part,["gfs_raw","gfs_bias","gfs_update","urban_18"],profiles,
                        {"update_gain":("gfs_update","gfs_bias"),
                         "raw_minus_station":("gfs_raw","urban_18"),
                         "bias_mae_minus_raw":("gfs_bias_mae","gfs_raw_mae")},label,days))
        rows.extend(conditional_panel(region,["gfs_raw","icon_raw"],{"gfs_raw":GFS,"icon_raw":ICON},
                    {"icon_gain":("icon_raw","gfs_raw"),
                     "icon_mae_minus_gfs":("icon_raw_mae","gfs_raw_mae")},"Raw products 2026",days))
    save(rows,"forecast_intervals")


if __name__=="__main__":
    main()
