from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
from csv import DictReader
import csv
import json

import eccodes as ec
import requests

ROOT=Path(__file__).resolve().parent
V3=ROOT.parent/"Forecast_Revision_20261120"
SITES=list(DictReader((V3/"forecast_sites.csv").open(encoding="utf-8")))
OUT=ROOT/"results"


def retrieve(day):
    url=f"https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{day}/06/atmos/gfs.t06z.pgrb2.0p25.f000"
    with requests.Session() as session:
        response=session.get(url+".idx",timeout=(8,20))
        response.raise_for_status()
        listing=response.text.splitlines()
        i,= [i for i,line in enumerate(listing) if ":HGT:surface:" in line]
        begin=int(listing[i].split(":")[1])
        end=int(listing[i+1].split(":")[1])-1
        path=OUT/f"gfs_terrain_{day}.grib2"
        if not path.is_file():
            r=session.get(url,headers={"Range":f"bytes={begin}-{end}"},timeout=(8,25))
            r.raise_for_status()
            assert r.status_code==206 and r.headers["Content-Range"].startswith(f"bytes {begin}-{end}/")
            path.write_bytes(r.content)
        raw=path.read_bytes()
        assert len(raw)==end-begin+1 and raw.startswith(b"GRIB") and raw.endswith(b"7777")
        return raw, {"url":url,"inventory":listing[i],"byte_start":begin,"byte_stop":end,
                     "file":path.name,"sha256":sha256(raw).hexdigest()}


def main():
    with ThreadPoolExecutor(max_workers=3) as ex:
        files=list(ex.map(retrieve,["20260601","20260701","20260831"]))
    arrays=[]
    for content,meta in files:
        g=ec.codes_new_from_message(content)
        try:
            get=lambda *keys: tuple(ec.codes_get(g,k) for k in keys)
            assert get("gridType","Ni","Nj","typeOfLevel","level","dataTime")==("regular_ll",1440,721,"surface",0,600)
            assert get("latitudeOfFirstGridPointInDegrees","longitudeOfFirstGridPointInDegrees","iDirectionIncrementInDegrees","jDirectionIncrementInDegrees")== (90.0,0.0,.25,.25)
            z=[]
            vals=ec.codes_get_values(g)
            for s in SITES:
                lat,lon=float(s["lat"]),float(s["lon"])
                j,i=round((90-lat)/.25),round(lon/.25)%1440
                a=ec.codes_grib_find_nearest(g,lat,lon)[0]
                assert abs(a["value"]-vals[j*1440+i])<1e-7
                z.append((90-j*.25,i*.25,float(a["value"])))
            arrays.append(z)
        finally:
            ec.codes_release(g)
    assert max(abs(arrays[0][j][2]-arr[j][2]) for arr in arrays for j in range(len(SITES)))<=.1
    (OUT/"gfs_terrain_sources.json").write_text(json.dumps([meta for _,meta in files],indent=2),encoding="utf-8")
    with (OUT/"gfs_grid_height.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f)
        w.writerow(["city","gfs_grid_lat","gfs_grid_lon","gfs_grid_height"])
        w.writerows([[s["city"],*a] for s,a in zip(SITES,arrays[0],strict=True)])
    print(arrays[0])


if __name__=="__main__":
    main()
