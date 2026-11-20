from pathlib import Path
from shutil import copy2
import sys

ROOT = Path(__file__).resolve().parent
V4 = ROOT.parent / "Reference_Revision_20261120"
sys.path.insert(0, str(V4))
import plot_results as prior
from plot_results import INK, GREY, NAVY, TEAL, COPPER, PLUM, CITY, panel, key, result, save
import matplotlib.pyplot as plt
import numpy as np

prior.OUT = ROOT / "figures"
REGIONS = [("Germany",NAVY,"o"),("Vienna",COPPER,"D")]


def dot(ax,row,pos,color,marker,vertical=False,solid=True):
    ax.plot([pos,pos] if vertical else [row.lower,row.upper],
            [row.lower,row.upper] if vertical else [pos,pos],color=color,lw=1,alpha=.7,zorder=2)
    ax.scatter(*((pos,row.estimate) if vertical else (row.estimate,pos)),
               facecolor=color if solid else "white",edgecolor=color,marker=marker,s=23,lw=.9,zorder=4)


def fig2():
    f=plt.figure(figsize=(7.45,6.20))
    g=f.add_gridspec(2,2,height_ratios=[1.15,1],left=.148,right=.974,bottom=.105,top=.931,hspace=.64,wspace=.42)
    a,b,c=f.add_subplot(g[0,:]),f.add_subplot(g[1,0]),f.add_subplot(g[1,1])
    d=result(prior.BASE,"capture_summary").set_index("city").loc[CITY]
    y=np.arange(len(d))
    panel(a,"a","Coverage by city at the 10% quota",grid="x")
    a.hlines(y,d[["urban_max","urban_18"]].min(axis=1),d.oracle,color="#b4bfc5",lw=1.3)
    a.scatter(d.urban_max,y,facecolor="white",edgecolor=COPPER,s=30,lw=1.15,zorder=4)
    a.scatter(d.urban_18,y,c=TEAL,s=25,zorder=5)
    a.scatter(d.oracle,y,c=NAVY,s=29,marker="D",zorder=5)
    a.annotate("69.3",(d.loc["Hamburg","oracle"],CITY.index("Hamburg")),
               xytext=(6,0),textcoords="offset points",fontsize=7.8,color=NAVY,va="center")
    a.set(xlim=(35,72),yticks=y,yticklabels=CITY,ylim=(6.6,-.6),xlabel="Share of measured overnight heat (%)")
    a.legend(handles=[key("Daytime maximum",COPPER,empty=True),key("Urban 18:00",TEAL),key("Local ceiling",NAVY,"D")],
             loc="upper center",bbox_to_anchor=(.5,-.17),frameon=False,ncol=3)
    panel(b,"b","Coverage by quota")
    bud=result(prior.BASE,"budget_curve")
    for fld,col,ls in [("oracle",NAVY,"--"),("urban_18",TEAL,"-"),("urban_max",COPPER,":")]:
        b.plot(100*bud.budget,bud[fld],color=col,ls=ls,lw=1.7)
    s=result(ROOT,"historical_intervals").query("metric == 'pooled'").estimate.item()
    b.scatter(10,s,c=PLUM,marker="^",s=38,zorder=5,edgecolor="white",lw=.45)
    b.axvline(10,c=GREY,lw=.65,ls=":")
    b.set(xlim=(0,31),ylim=(0,98),xticks=[0,10,20,30],yticks=[0,25,50,75],
          xlabel="Local quota (% of nights)",ylabel="Overnight heat covered (%)")
    b.legend(handles=[key("Shared-slot ceiling",PLUM,"^")],loc="lower right",frameon=False,
             handletextpad=.26,borderpad=.25,fontsize=7.5)
    panel(c,"c","Evening gain by summer")
    d=result(prior.V2,"aggregate_checks")
    d=d[d["sample"].str.fullmatch(r"20\d\d")]
    change=d.urban_timing.to_numpy()
    x=np.arange(len(change))
    pooled=result(prior.BASE,"uncertainty").set_index("metric").loc["urban_timing","estimate"]
    c.axhline(0,c=GREY,lw=.8)
    c.vlines(x,0,change,colors=np.where(change>=0,TEAL,COPPER),lw=1.5)
    c.scatter(x,change,c=np.where(change>=0,TEAL,COPPER),s=22,zorder=4)
    c.axhline(pooled,c=NAVY,lw=1,ls="--")
    c.annotate("−0.02",(4,change[4]),xytext=(6,-10),textcoords="offset points",ha="center",fontweight="medium",
               fontsize=7.5,color=COPPER)
    c.set(xlim=(-.55,len(x)-.45),ylim=(-2.4,4.85),xticks=x,xticklabels=d["sample"].str[-2:],
          xlabel="Summer (2017–2025)",ylabel="18:00 minus daytime (pp)")
    c.legend(handles=[key("Pooled +1.57 pp",NAVY,None,"--")],loc="upper left",frameon=False,
             fontsize=7.5,handletextpad=.35,borderpad=.25)
    save(f,"Figure_2_German_Coverage")


def fig5():
    f=plt.figure(figsize=(7.45,4.42))
    g=f.add_gridspec(2,2,left=.172,right=.969,bottom=.194,top=.876,wspace=.44,hspace=.56,
                     width_ratios=(1.22,1))
    a,b,c=f.add_subplot(g[:,0]),f.add_subplot(g[0,1]),f.add_subplot(g[1,1])
    rep=result(ROOT,"replay_intervals").set_index(["cohort","metric"])
    err=result(ROOT,"profile_error_intervals").set_index(["cohort","metric"])
    panel(a,"a","Coverage by decision rule",grid="x")
    rows=[(r,k,col,p) for r,col,_,p0 in [(*REGIONS[0],3.7),(*REGIONS[1],1.1)]
          for k,p in [("urban_max",p0),("urban_18",p0-.85)]]
    for r,k,col,p in rows:
        for method,offset,marker in [("batch",.13,"o"),("chronological",-.13,"s")]:
            dot(a,rep.loc[(r,method+"_"+k)],p+offset,col,marker,solid=method=="chronological")
    a.set(yticks=[p for _,_,_,p in rows],yticklabels=["Daytime max","18:00","Daytime max","18:00"],
          ylim=(-.24,4.53),xlim=(15,55),xlabel="Overnight heat covered (%)")
    for r,col,_,p in [(*REGIONS[0],4.25),(*REGIONS[1],1.65)]:
        a.text(.02,p,r,transform=a.get_yaxis_transform(),fontsize=8.5,color=col,weight="bold")
    a.legend(handles=[key("Full summer",INK,"o",empty=True),key("Calendar order",INK,"s")],
             loc="upper center",bbox_to_anchor=(.5,-.17),ncol=2,frameon=False,handletextpad=.27,columnspacing=.8)
    panel(b,"b","Added profile inputs",grid="x")
    b.axvline(0,c=GREY,lw=.8)
    for j,(r,col,m) in enumerate(REGIONS):
        dot(b,err.loc[(r,"added_features")],1-j,col,m)
    b.set(xlabel="Change in hourly MAE (°C)",xlim=(-.35,.045),xticks=[-.3,-.2,-.1,0],
          yticks=[0,1],yticklabels=["Vienna","Germany"],ylim=(-.48,1.48))
    c.grid(axis="x",color="#dce4e8",lw=.55)
    c.spines[["top","right"]].set_visible(False)
    c.spines[["bottom","left"]].set_color("#a2aeb5")
    c.axvline(0,c=GREY,lw=.8)
    for j,(r,col,m) in enumerate(REGIONS):
        dot(c,rep.loc[(r,"added_features_replay")],1-j,col,m)
    c.set(xlabel="Change in replay coverage (pp)",xlim=(-1.15,2.12),xticks=[-1,0,1,2],
          yticks=[0,1],yticklabels=["Vienna","Germany"],ylim=(-.48,1.48))
    save(f,"Figure_5_Daily_Decisions")


def fig6():
    f,a=plt.subplots(2,2,figsize=(7.45,6.16))
    a=a.ravel()
    f.subplots_adjust(left=.107,right=.968,bottom=.14,top=.876,wspace=.39,hspace=.64)
    ci=result(ROOT,"forecast_intervals").query("block_days == 7").set_index(["sample","cohort","metric"]).sort_index()
    keys=["gfs_raw","gfs_bias","gfs_update","urban_18"]
    labels=["Raw","Bias\ncorrected","18:00\nupdate","Observed\n18:00"]
    f.legend(handles=[key(r,col,m) for r,col,m in REGIONS],loc="upper center",bbox_to_anchor=(.52,.98),
             frameon=False,ncol=2,columnspacing=1.6)
    for n,(letter,title,lab) in enumerate([("a","GFS hourly accuracy","Hourly MAE (°C)"),
                                           ("b","GFS selection coverage","Overnight heat covered (%)")]):
        ax=a[n]
        panel(ax,letter,title)
        for j,(r,col,m) in enumerate(REGIONS):
            for i,k in enumerate(keys[:3] if n==0 else keys):
                row=ci.loc[("GFS 2025–2026",r,k+"_mae" if n==0 else k)]
                dot(ax,row,i+(.12 if j else -.12),col,m,True)
        lim=(.8,4.05) if n==0 else (19,55)
        xx=range(3 if n==0 else 4)
        ax.set(xticks=list(xx),xticklabels=labels[:len(xx)],ylabel=lab,xlim=(-.48,len(xx)-.54),ylim=lim)
    c,d=a[2:]
    panel(c,"c","Evening update gain")
    c.axhline(0,c=GREY,lw=.8)
    labels_c=["2025","2026","Pooled"]
    for j,(r,col,m) in enumerate(REGIONS):
        for i,label in enumerate(["GFS 2025","GFS 2026","GFS 2025–2026"]):
            dot(c,ci.loc[(label,r,"update_gain")],i+(.11 if j else -.11),col,m,True)
    c.set(xticks=range(3),xticklabels=labels_c,ylabel="Update minus bias correction (pp)",
          xlim=(-.43,2.43),ylim=(-5.3,3.8))
    panel(d,"d","Matched raw products, 2026")
    for r,col,_ in REGIONS:
        points=ci.loc[("Raw products 2026",r)]
        for k,m,solid in [("gfs_raw","o",False),("icon_raw","s",True)]:
            x,y=points.loc[k+"_mae"],points.loc[k]
            d.plot([x.lower,x.upper],[y.estimate,y.estimate],color=col,lw=.9,alpha=.62)
            dot(d,y,x.estimate,col,m,True,solid=solid)
    d.set(xlim=(.62,4.60),ylim=(17,55),xticks=[1,2,3,4],ylabel="Overnight heat covered (%)",
          xlabel="Hourly MAE (°C)")
    d.legend(handles=[key("GFS",INK,"o",empty=True),key("ICON-D2",INK,"s")],frameon=False,
             loc="upper center",bbox_to_anchor=(.5,-.30),ncol=2,handletextpad=.28)
    save(f,"Figure_6_Archived_Forecasts")


if __name__=="__main__":
    for n,directory in [(1,V4/"Figure1_QGIS_20261120"),(3,V4/"figures"),(4,V4/"figures")]:
        for f in directory.glob(f"Figure_{n}_*"):
            if f.suffix in (".png",".pdf"):
                copy2(f,prior.OUT/f.name)
    fig2()
    fig5()
    fig6()
