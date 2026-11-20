from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd
from pyproj import Transformer

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures"
BASE = ROOT.parent
V2 = BASE / "Revision_20261120"
V3 = BASE / "Forecast_Revision_20261120"
INK = "#223647"
MUTED = "#5a6a78"
GREY = "#89949b"
GRID = "#dce4e8"
NAVY = "#275c78"
TEAL = "#187873"
COPPER = "#bd5d39"
PLUM = "#775376"
CITY = ["Berlin", "Dresden", "Frankfurt", "Freiburg", "Hamburg", "Hannover", "Munich"]
COHORTS = ("Germany", "Vienna")
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 10,
    "axes.labelsize": 9, "xtick.labelsize": 8.3, "ytick.labelsize": 8.3,
    "pdf.fonttype": 42, "savefig.facecolor": "white", "legend.fontsize": 8,
})


def result(version, name):
    return pd.read_csv(version / "results" / (name + ".csv"))


def panel(ax, letter, title, grid="y"):
    ax.set_title(f"({letter})  {title}", loc="left", fontweight="bold", pad=9)
    ax.set_axisbelow(True)
    if grid != 'none':
        ax.grid(axis=grid, color=GRID, linewidth=.55)
    ax.tick_params(length=2.8, pad=3)
    for name in ("top", "right"):
        ax.spines[name].set_visible(False)
    for name in ("bottom", "left"):
        ax.spines[name].set_color("#a2aeb5")
        ax.spines[name].set_linewidth(.65)


def key(label, color, marker="o", line="none", empty=False):
    return Line2D([], [], color=color, marker=marker, linestyle=line, linewidth=1.4,
                  markerfacecolor="white" if empty else color, markersize=5.6, label=label)


def save(fig, name):
    fig.savefig(OUT / (name + ".pdf"), dpi=450)
    fig.savefig(OUT / (name + ".png"), dpi=500)
    plt.close(fig)


def fig1():
    fig = plt.figure(figsize=(7.45, 4.55))
    layout = fig.add_gridspec(1, 2, width_ratios=(1.04, 1), left=.043, right=.969,
                              bottom=.123, top=.867, wspace=.23)
    a, b = [fig.add_subplot(layout[0, i]) for i in range(2)]
    world = gpd.read_file(BASE / "raw/ne_110m_admin_0_countries.geojson")
    world = world[world.CONTINENT.eq("Europe")].to_crs(3035)
    world.plot(ax=a, color="#e9edef", edgecolor="white", linewidth=.5, zorder=1)
    world[world.ADMIN.eq("Germany")].plot(ax=a, color="#d5e1e3", edgecolor="#93a9ad", linewidth=.85, zorder=2)
    coords = Transformer.from_crs(4326, 3035, always_xy=True)
    st = pd.read_csv(BASE / "stations.csv").drop_duplicates(["city", "role"], keep="last")
    st["x"], st["y"] = coords.transform(st.lon, st.lat)
    for city, pair in st.groupby("city"):
        u = pair.set_index("role")
        a.plot(u.x, u.y, c=GREY, lw=.85, zorder=3)
        x,y = u.loc["urban", ["x", "y"]]
        dx,dy = (7,2) if city != "Dresden" else (7,-7)
        a.annotate(city, (x,y), xytext=(dx,dy), textcoords="offset points", fontsize=8.3,
                   color=INK, weight="medium", zorder=7)
    a.scatter(st.query("role == 'reference'").x, st.query("role == 'reference'").y,
              c="white", edgecolor=NAVY, lw=.85, s=20, zorder=4)
    a.scatter(st.query("role == 'urban'").x, st.query("role == 'urban'").y, c=NAVY,
              edgecolor="white", lw=.55, s=27, zorder=5)
    external = pd.read_csv(V2 / "external/stations.csv")
    # GeoSphere ID in the first column; the archival headings for coordinates are explicit.
    v = external[external.iloc[:, 0].eq(5925)].iloc[0]
    vx,vy = coords.transform(float(v.iloc[3]), float(v.iloc[4]))
    a.scatter(vx,vy,c=COPPER,s=30,marker="D",edgecolor="white",lw=.55,zorder=6)
    a.annotate("Vienna", (vx,vy), xytext=(-8,-2), ha="right", textcoords="offset points",
               fontsize=8.3,color=COPPER,weight="medium")
    a.set(xlim=(3980000, 4850000), ylim=(2650000, 3520000), aspect="equal")
    a.set_axis_off()
    a.set_title("(a)  Observation networks", loc="left", fontweight="bold", pad=9)
    a.legend(handles=[key("DWD urban",NAVY), key("DWD reference",NAVY,empty=True),
                      key("GeoSphere external",COPPER,"D")],
             frameon=False, loc="upper center", bbox_to_anchor=(.49,.017), ncol=2,
             borderpad=.1, columnspacing=1.0, handletextpad=.34)
    vals = result(BASE,"season_coverage")
    panel(b,"b","Complete paired nights",grid="none")
    palette = {False: "#f5e8e1", True: "#d4e6e6"}
    years = sorted(vals.year.unique())
    for row in vals.itertuples():
        x,y = years.index(row.year), CITY.index(row.city)
        b.add_patch(Rectangle((x-.45,y-.42),.9,.84,facecolor=palette[bool(row.included)],edgecolor="none"))
        b.text(x,y,str(row.complete),ha="center",va="center",fontsize=8.2,
               color=INK if row.included else "#925e4c")
    b.set(xlim=(-.5,8.5),ylim=(6.58,-.58),xticks=range(9),xticklabels=[str(y)[2:] for y in years],
          yticks=range(7),yticklabels=CITY,xlabel="Summer (2017–2025)")
    b.tick_params(length=0)
    for s in b.spines.values():
        s.set_visible(False)
    b.legend(handles=[Patch(facecolor=palette[True],label="At least 83 nights"),
                      Patch(facecolor=palette[False],label="Fewer than 83")],
             loc="upper center",bbox_to_anchor=(.5,-.17),frameon=False,ncol=2,handlelength=1.2,columnspacing=1.1)
    save(fig,"Figure_1_Observation_Network")


def fig2():
    fig = plt.figure(figsize=(7.45, 6.23))
    gs = fig.add_gridspec(2,2,height_ratios=[1.15,1],left=.153,right=.975,bottom=.118,top=.929,
                          hspace=.63,wspace=.42)
    a=fig.add_subplot(gs[0,:])
    b=fig.add_subplot(gs[1,0])
    c=fig.add_subplot(gs[1,1])
    summary = result(BASE,"capture_summary").set_index("city").loc[CITY]
    panel(a,"a","Coverage by city at the 10% quota",grid="x")
    y=np.arange(len(summary))
    a.hlines(y, summary[["urban_max","urban_18"]].min(axis=1), summary.oracle, color="#b4bfc5", lw=1.3)
    a.scatter(summary.urban_max,y,facecolor="white",edgecolor=COPPER,s=32,lw=1.05,zorder=4)
    a.scatter(summary.urban_18,y,c=TEAL,s=27,zorder=5)
    a.scatter(summary.oracle,y,c=NAVY,s=31,marker="D",zorder=5)
    a.set(xlim=(32,72),yticks=y,yticklabels=CITY,ylim=(6.6,-.6),xlabel="Share of measured overnight heat (%)")
    a.legend(handles=[key("Daytime maximum",COPPER,empty=True),key("Urban 18:00",TEAL),key("Ceiling",NAVY,"D")],
             loc="upper center",bbox_to_anchor=(.5,-.17),frameon=False,ncol=3)
    panel(b,"b","Coverage by quota")
    bud=result(BASE,"budget_curve")
    for fld,col,ls in [("oracle",NAVY,"--"),("urban_18",TEAL,"-"),("urban_max",COPPER,":")]:
        b.plot(100*bud.budget,bud[fld],color=col,ls=ls,lw=1.8)
    alloc=result(V2,"allocation_year")
    shared=100*alloc.pooled.sum()/alloc.heat.sum()
    b.scatter(10,shared,c=PLUM,marker="^",s=37,zorder=5,edgecolor="white",lw=.45)
    b.axvline(10,c=GREY,lw=.65,ls=":")
    b.set(xlim=(0,31),ylim=(0,98),xticks=[0,10,20,30],yticks=[0,25,50,75],
          xlabel="Local quota (% of nights)",ylabel="Overnight heat covered (%)")
    b.legend(handles=[key("Shared-slot ceiling",PLUM,"^")],loc="lower right",frameon=False,
             handletextpad=.26,borderpad=.25,fontsize=7.5)
    panel(c,"c","Evening gain by summer")
    annual=result(V2,"aggregate_checks")
    annual=annual[annual["sample"].astype(str).str.fullmatch(r"20\d\d")]
    delta=annual.urban_timing.to_numpy()
    x=np.arange(len(delta))
    pooled=result(BASE,"uncertainty").set_index("metric").loc["urban_timing","estimate"]
    c.axhline(0,c=GREY,lw=.8)
    c.vlines(x,0,delta,colors=np.where(delta>=0,TEAL,COPPER),lw=1.5)
    c.scatter(x,delta,c=np.where(delta>=0,TEAL,COPPER),s=24,zorder=4)
    c.axhline(pooled,c=NAVY,lw=1.05,ls="--")
    c.set(xlim=(-.55,len(x)-.45),ylim=(-2.4,4.8),xticks=x,
          xticklabels=[str(v)[2:] for v in annual["sample"].astype(str)],xlabel="Summer (2017–2025)",
          ylabel="18:00 minus daytime (pp)")
    c.legend(handles=[key("Pooled +1.57 pp",NAVY,None,"--")],loc="upper left",frameon=False,
             fontsize=7.5,handletextpad=.35,borderpad=.24)
    save(fig,"Figure_2_German_Coverage")


def fig4():
    fig, (a,b) = plt.subplots(1,2,figsize=(7.45,3.8))
    fig.subplots_adjust(left=.096,right=.973,bottom=.242,top=.85,wspace=.36)
    panel(a,"a","Evening gain in Vienna")
    vi=result(V2,"vienna_Schwechat_city_year")
    vi["delta"] = 100*(vi.urban_18-vi.urban_max)/vi.burden
    pooled=100*(vi.urban_18.sum()-vi.urban_max.sum())/vi.burden.sum()
    a.axhline(0,c=GREY,lw=.82)
    a.vlines(vi.year,0,vi.delta,colors=np.where(vi.delta>=0,TEAL,COPPER),lw=1.6)
    a.scatter(vi.year,vi.delta,s=29,c=np.where(vi.delta>=0,TEAL,COPPER),zorder=4)
    a.axhline(pooled,c=NAVY,ls="--",lw=1)
    a.axvspan(2018.67,2019.34,facecolor="#eef0f1",zorder=0)
    a.set(xlim=(2016.66,2025.35),ylim=(-3.1,3.65),
          xticks=range(2017,2026),xticklabels=[str(y)[2:] for y in range(2017,2026)],
          xlabel="Summer (2017–2025)",ylabel="18:00 minus daytime (pp)")
    a.legend(handles=[key("Pooled +0.37 pp",NAVY,None,"--"),Patch(facecolor="#eef0f1",label="Ineligible")],
             loc="upper center",bbox_to_anchor=(.5,-.18),frameon=False,handlelength=1.4,ncol=2)
    panel(b,"b","Ceiling by temperature base")
    t=result(V2,"threshold_comparison").pivot(index="base",columns="cohort",values="oracle")
    t=t.loc[["18 C","20 C","22 C","Local 90th percentile"]]
    x=np.arange(4)
    for y1,y2,pos in zip(t.Germany,t.Vienna,x,strict=True):
        b.plot([pos,pos],[y1,y2],color="#bdc7cb",lw=1)
    for cohort,color,marker in [("Germany",NAVY,"o"),("Vienna",COPPER,"D")]:
        b.scatter(x,t[cohort],c=color,marker=marker,s=29,zorder=4)
    b.set(xlim=(-.38,3.39),ylim=(15,89),xticks=x,xticklabels=["18 °C","20 °C","22 °C","Site\n90th percentile"],
          ylabel="Ceiling (% of heat)")
    b.legend(handles=[key("Germany",NAVY),key("Vienna",COPPER,"D")],frameon=False,
             loc="upper left",handletextpad=.32)
    save(fig,"Figure_4_Vienna_and_Heat_Definition")


def fig5():
    fig,(a,b) = plt.subplots(1,2,figsize=(7.45,4.18))
    fig.subplots_adjust(left=.150,right=.973,bottom=.233,top=.83,wspace=.43)
    panel(a,"a","Decisions in calendar order",grid="x")
    replay=result(V2,"replay_summary").set_index(["cohort","score","mode"])
    positions=[4,3,1,0]
    for (region,score,color),y in zip(
            [(r,s,c) for r in COHORTS for s,c in [("urban_max",COPPER),("urban_18",TEAL)]],
            positions,strict=True):
        full=replay.loc[(region,score,"batch"),"capture_pct"]
        daily=replay.loc[(region,score,"chronological"),"capture_pct"]
        a.plot([daily,full],[y,y],c=color,lw=1.3)
        a.scatter(daily,y,c=color,s=32,zorder=4)
        a.scatter(full,y,facecolor="white",edgecolor=color,s=31,lw=1.1,zorder=5)
    a.set(xlim=(19,46.5),ylim=(-.55,4.55),yticks=positions,
          yticklabels=["Germany\nDaytime max","18:00","Vienna\nDaytime max","18:00"],
          xlabel="Overnight heat covered (%)")
    a.legend(handles=[key("Calendar order",INK),key("Full summer",INK,empty=True)],
             frameon=False,loc="upper center",bbox_to_anchor=(.5,-.18),ncol=2,
             handletextpad=.22,columnspacing=.7)
    panel(b,"b","Profile accuracy and coverage")
    errors=result(V2,"model_error").set_index(["cohort","model"])
    for cohort,color in [("Germany",NAVY),("Vienna",COPPER)]:
        x=[errors.loc[(cohort,m),"hourly_mae"] for m in ("linear_18","linear_moisture")]
        y=[replay.loc[(cohort,m,"chronological"),"capture_pct"] for m in ("linear_18","linear_moisture")]
        b.plot(x,y,c=color,lw=1.35)
        b.scatter(x[0],y[0],facecolor="white",edgecolor=color,s=34,lw=1.1,zorder=5)
        b.scatter(x[1],y[1],c=color,s=34,zorder=4)
        b.annotate(cohort,(np.mean(x),np.mean(y)),xytext=(0,10),textcoords="offset points",
                   ha="center",color=color,fontsize=8.6)
    b.set(xlim=(1.14,1.78),ylim=(20,45),ylabel="Heat covered in replay (%)",xlabel="Hourly mean absolute error (°C)")
    b.legend(handles=[key("18:00 model",INK,empty=True),key("+ Change and vapour pressure",INK)],frameon=False,
             loc="upper center",bbox_to_anchor=(.5,-.18),ncol=1,handletextpad=.35,borderpad=.25)
    save(fig,"Figure_5_Daily_Decisions")


def fig6():
    fig,a=plt.subplots(2,2,figsize=(7.45,6.27))
    a=a.ravel()
    fig.subplots_adjust(left=.105,right=.965,bottom=.126,top=.902,wspace=.41,hspace=.59)
    rank=result(V3,"rank_summary").query("base == 20").set_index(["region","score"])
    t=result(V3,"temperature_error").set_index(["region","forecast"])
    scores=("gfs_raw","gfs_bias","gfs_update")
    markers=("o","s","D")
    colours=(GREY,NAVY,TEAL)
    for i,region in enumerate(COHORTS):
        ax=a[i]
        panel(ax,"ab"[i],f"{region} · GFS test summers")
        xx=[t.loc[(region,score),"mae_C"] for score in scores]
        yy=[rank.loc[(region,score),"coverage_pct"] for score in scores]
        ax.plot(xx,yy,c="#bcc7cc",lw=1,zorder=2)
        for x,y,m,col in zip(xx,yy,markers,colours,strict=True):
            ax.scatter(x,y,c=col,marker=m,s=44,edgecolor="white",lw=.65,zorder=4)
        obs=rank.loc[(region,"urban_18"),"coverage_pct"]
        ax.axhline(obs,c=COPPER,ls="--",lw=.8)
        ax.set(xlim=(.85,4.2),ylim=(25,46),xticks=[1,2,3,4],yticks=[25,30,35,40,45],
               xlabel="Hourly mean absolute error (°C)",ylabel="Overnight heat covered (%)")
    fig.legend(handles=[key("Raw",GREY,"o"),key("Mean-bias correction",NAVY,"s"),key("18:00 update",TEAL,"D"),
                        key("Observed 18:00",COPPER,None,"--")],frameon=False,loc="upper center",
               bbox_to_anchor=(.51,.988),ncol=4,columnspacing=1.1,handletextpad=.38)
    panel(a[2],"c","Gain from the evening update")
    annual=result(V3,"rank_year").query("base == 20").pivot(index=["region","year"],columns="score",values="coverage_pct")
    vlim=(-1.55,2.15)
    for region,sep,col,m in [("Germany",-.095,NAVY,"o"),("Vienna",.095,COPPER,"D")]:
        vals=annual.loc[region]
        delta=(vals.gfs_update-vals.gfs_bias).to_numpy()
        x=np.arange(len(vals))+sep
        a[2].vlines(x,0,delta,color=col,lw=1.2)
        a[2].scatter(x,delta,c=col,marker=m,s=33,zorder=4)
    a[2].axhline(0,c=GREY,lw=.8)
    a[2].set(xticks=[0,1],xticklabels=annual.loc["Germany"].index,
             xlim=(-.46,1.46),ylim=vlim,xlabel="Evaluation summer",ylabel="Update minus mean bias (pp)")
    a[2].legend(handles=[key("Germany",NAVY),key("Vienna",COPPER,"D")],frameon=False,
                loc="upper left",handletextpad=.28,borderpad=.25)
    panel(a[3],"d","Regional comparison in 2026")
    reg=result(V3,"regional_summary").set_index(["region","score"])
    for region,color in [("Germany",NAVY),("Vienna",COPPER)]:
        g=reg.loc[region].loc[["gfs_raw","icon_raw"]]
        a[3].plot(g.mae_C,g.coverage_pct,c=color,lw=1.2)
        a[3].scatter(g.iloc[0].mae_C,g.iloc[0].coverage_pct,facecolor="white",edgecolor=color,s=38,lw=1,zorder=5)
        a[3].scatter(g.iloc[1].mae_C,g.iloc[1].coverage_pct,c=color,s=36,marker="s",zorder=5)
    a[3].set(xlim=(.57,4.2),ylim=(23,42),xticks=[1,2,3,4],yticks=[25,30,35,40],
             xlabel="Hourly mean absolute error (°C)",ylabel="Overnight heat covered (%)")
    a[3].legend(handles=[key("GFS",INK,empty=True),key("ICON-D2",INK,"s")],frameon=False,
                loc="upper center",bbox_to_anchor=(.5,-.24),ncol=2)
    save(fig,"Figure_6_Archived_Forecasts")


if __name__ == "__main__":
    fig1()
    fig2()
    fig4()
    fig5()
    fig6()
