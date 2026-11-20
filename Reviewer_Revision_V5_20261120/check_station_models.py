from io import BytesIO
from zipfile import ZipFile
import re

from reanalyse import BASE, V2, OUT, csv, save, cluster_parts, weights
import numpy as np
import pandas as pd
from openpyxl import load_workbook
import sys

sys.path.insert(0, str(V2))
import review_analysis as obs


def samples():
    obs.OUT = OUT
    g, (v, _) = obs.germany(), obs.vienna()
    all_ = pd.concat([g, v], ignore_index=True)
    complete = obs.included(all_, all_.basic_complete)
    with_rh = obs.included(all_, all_.basic_complete & all_.day_ready)
    train = with_rh.query("year <= 2020 and city != 'Vienna'")
    test = with_rh.query("year >= 2021").copy()
    calendar = pd.read_csv(V2/"results/test_calendar_decisions.csv",parse_dates=["date"])
    check = calendar.merge(all_,on=["city","date","year"],validate="one_to_one",suffixes=("_replay",""))
    assert np.allclose(check.target_replay,check.target,equal_nan=True)
    assert check.target_observed.equals(check.target.notna())
    assert check.day_ready_replay.equals(check.day_ready)
    status = []
    for key,selection in [("Historical matched temperatures",complete.query("year >= 2021")),
                          ("Matched temperature and humidity",test),("Calendar replay",check)]:
        for cohort, frame in selection.groupby(np.where(selection.city.eq("Vienna"),"Vienna","Germany")):
            seasons = frame.groupby(["city","year"]).size()
            k = np.floor(.1*seasons+.5)
            status.append({"cohort":cohort,"sample":key,"seasons":len(seasons),"days":len(frame),
                           "outcomes":frame.target.notna().sum(),"ready":frame.day_ready.sum(),
                           "slots":int(k.sum()),"heat":frame.target.sum()})
    audit=pd.DataFrame(status)
    for cohort,count in (("Germany",[2793,2792,2844]),("Vienna",[460,460,460])):
        s=audit.loc[audit.cohort.eq(cohort)]
        assert s.outcomes.tolist()==count
        assert s.seasons.nunique()==1
    save(audit,"cohort_denominators")
    early=complete.query("year <= 2020")
    bases=early.groupby("city")[obs.OVERNIGHT].apply(lambda d:np.quantile(d.to_numpy(),.9))
    assert np.allclose(bases,csv(V2,"local_base").set_index("city").loc[bases.index,"base_C"])
    save(bases.rename("base_C").reset_index(),"local_bases_verified")
    assert len(train)==2085
    return all_,train,test


def profiles(train,test):
    coeff=csv(V2,"profile_coefficients")
    profiles_={}
    for name,cols in obs.MODEL_INPUT.items():
        beta=coeff.loc[coeff.model.eq(name)].set_index("term").loc[["intercept",*cols],obs.OVERNIGHT]
        profiles_[name]=np.c_[np.ones(len(test)),test[cols]] @ beta.to_numpy()
    cols=obs.MODEL_INPUT["linear_moisture"]
    interacted=np.full((len(test),len(obs.OVERNIGHT)),np.nan)
    test=test.reset_index(drop=True)
    for city,s in test.query("city != 'Vienna'").groupby("city"):
        g=train.loc[train.city.eq(city)]
        b,_,rank,_=np.linalg.lstsq(np.c_[np.ones(len(g)),g[cols]],g[obs.OVERNIGHT],rcond=None)
        assert rank==4
        interacted[s.index]=np.c_[np.ones(len(s)),s[cols]] @ b
    profiles_["city_interactions"]=interacted
    y=test[obs.OVERNIGHT].to_numpy()
    target=test.target.to_numpy()
    errors={}
    for name,pred in profiles_.items():
        test[name]=np.maximum(pred-20,0).sum(axis=1)
        errors[name]=np.abs(pred-y).sum(axis=1)
    rows, holdout=[],[]
    original=csv(V2,"model_error")
    for cohort,frame in test.groupby(np.where(test.city.eq("Vienna"),"Vienna","Germany")):
        keys=[k for k in profiles_ if frame[k].notna().all()]
        a=frame[["year"]].assign(burden=8.)
        for k in keys:
            a[k]=errors[k][frame.index]/100
        diffs={"added_features":("linear_moisture","linear_18")}
        if cohort=="Germany":
            diffs["city_slopes_minus_pooled"]=("city_interactions","linear_moisture")
            score=frame.rename(columns={"target":"heat"})
            parts=[]
            for (city,year),gg in score.groupby(["city","year"]):
                n=len(gg)
                slots=int(np.floor(.1*n+.5))
                parts.append({"city":city,"year":year,"burden":gg.heat.sum(),
                    **{k:np.dot(gg.heat,weights(gg[k],slots)) for k in keys}})
            holdout.extend(cluster_parts(pd.DataFrame(parts).groupby("year").sum(numeric_only=True),
                keys,{"city_slopes_minus_pooled":diffs["city_slopes_minus_pooled"]},
                {"cohort":cohort,"units":"pp or percent", "nights":len(frame)}))
        result=cluster_parts(a.groupby("year").sum(),keys,diffs,{"cohort":cohort,"nights":len(frame)})
        for r in result:
            legacy=original[(original.cohort==cohort)&(original.model==r["metric"])]
            if len(legacy):
                assert np.isclose(legacy.hourly_mae.item(),r["estimate"])
        rows.extend(result)
    save(rows,"profile_error_intervals")
    save(holdout,"profile_matched_coverage")


def replay():
    old=csv(V2,"replay_city_year")
    rows=[]
    for cohort,frame in old.groupby("cohort"):
        a=frame.pivot_table(index=["year","city"],columns=["mode","score"],values="captured")
        a.columns=[mode+"_"+name for mode,name in a.columns]
        a["burden"]=frame.drop_duplicates(["year","city"]).set_index(["year","city"]).heat
        diffs={"evening_minus_day_replay":("chronological_urban_18","chronological_urban_max"),
               "evening_rank_minus_replay":("batch_urban_18","chronological_urban_18"),
               "added_features_replay":("chronological_linear_moisture","chronological_linear_18"),
               "added_features_batch":("batch_linear_moisture","batch_linear_18")}
        rows.extend(cluster_parts(a.groupby("year").sum(),a.columns.drop("burden"),diffs,{"cohort":cohort}))
    save(rows,"replay_intervals")


def station_table():
    meta=pd.read_csv(BASE/"stations.csv",dtype={"station_id":str})
    source=pd.read_csv(BASE/"raw/sources.csv",dtype={"station_id":str})
    qc=csv(BASE,"source_quality").set_index("file")
    seasons=csv(BASE,"season_coverage")
    epochs=[]
    for r in meta.itertuples(index=False):
        files=source[source.station_id.eq(r.station_id)]
        if len(files)>1:
            if r.city=="Hannover":
                product="climate_urban_prov" if "temporary" in r.epoch else "climate_urban"
                files=files[files["product"].eq(product)]
            else:
                assert r.city=="Dresden"
        for f in files.itertuples(index=False):
            with ZipFile(BASE/"raw"/f.file) as z:
                if f.product=="climate":
                    txt=next(n for n in z.namelist() if "Geraete_Lufttemperatur" in n and n.endswith(".txt"))
                    parts=[line.split(";") for line in z.read(txt).decode("latin1").splitlines()]
                    heights={float(fields[5]) for fields in parts if len(fields)>9 and
                             fields[6].isdigit() and (not fields[7] or fields[7]>="20170601")}
                    assert heights=={2.}
                    sensor="2.0 (PT100)"
                else:
                    x=[n for n in z.namelist() if n.endswith(".xlsx")]
                    sensor="Not specified"
                    if x:
                        sheet=load_workbook(BytesIO(z.read(x[0])),read_only=True,data_only=True).active
                        values=list(sheet.values)[1]
                        height=re.findall(r"([0-9]+(?:,[0-9]+)?)m",values[9])
                        sensor=f"{float(height[-1].replace(',','.')):.1f} (PT100)"
            avail=qc.loc[f.file]
            eligible=seasons.query("city == @r.city and included").year.astype(str).tolist()
            row=r._asdict()
            epochs.append({**row,"product":f.product,"sensor_height_m":sensor,
                           "archive_first_utc":avail.first_utc,"archive_last_utc":avail.last_utc,
                           "pair_eligible_summers":", ".join(eligible),"source":f.file})
    ext=pd.read_csv(V2/"external/stations.csv").set_index("id")
    for sid,rol in [(5925,"urban"),(5990,"reference"),(5972,"alternate reference")]:
        row=ext.loc[sid]
        epochs.append({"city":"Vienna","role":rol,"station_id":str(sid),"name":row.Stationsname,
                       "lon":row.iloc[2],"lat":row.iloc[3],"elevation_m":row.iloc[4],
                       "epoch":"2017–2025 extraction","product":"GeoSphere hourly v2",
                       "sensor_height_m":"Not specified in station table",
                       "pair_eligible_summers":"2017, 2018, 2020–2025","source":"10.60669/9bdm-yq93"})
    assert len(epochs)==21
    save(epochs,"station_metadata")
    bad=seasons.loc[~seasons.included,["city","year","complete"]].copy()
    bad["reason"]="Under 83 complete paired temperature nights"
    save(bad,"excluded_study_summers")


if __name__=="__main__":
    _,train,test=samples()
    profiles(train,test)
    replay()
    station_table()
