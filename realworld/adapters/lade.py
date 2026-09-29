"""LaDe: last-mile delivery / pick-up logs from Cainiao (Wu et al., KDD 2024).

10.7M packages, 21k couriers, 5 Chinese cities, May-Oct 2022; per package the
stop location, AOI, the courier's accept event and finish (delivery / pick-up)
event with timestamps and GPS; LaDe-P also has the customer's pick-up time
window.  Apache-2.0, https://huggingface.co/datasets/Cainiao-AI/LaDe-D and
.../LaDe-P (parquet, one file per city).

Decision process (per courier-day SMDP)
    One episode = one courier on one day.  At each decision epoch (a task
    completion) the courier picks the next open task; open = accepted (released)
    and not yet finished.  The logged action is the next task finished; the
    realized sojourn is the time until the next decision epoch.  Tasks accepted
    during the day (mostly LaDe-P) make the arrival process exogenous and
    replayable.  Because only completions are logged, the decision time of step
    t+1 is taken as max(f_t, release of the logged next task) -- the latest time
    consistent with the log (see realworld/README.md).  Travel for rollouts is
    ``detour * distance / speed`` with speed and service time calibrated per
    city from the logged legs (stored in the episode-set metadata).

Cleaning mirrors LaDe's own preprocessing: GPS-drift stops (implied speed >
500 m/min on both adjacent legs) are dropped, and packages finished at the
same location within ``merge_minutes`` are merged into one stop.
Splits are by date (the same cut for every courier): train / val / test.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from realworld.episodes import (InstanceRecord, RoutingEpisodeSet, TrajectoryRecord, concat_sets,
                                haversine_km, latlng_to_local_km)
from realworld.registry import DatasetSpec, register

HF_REPOS = {"delivery": "Cainiao-AI/LaDe-D", "pickup": "Cainiao-AI/LaDe-P"}
CITIES = {"sh": "Shanghai", "hz": "Hangzhou", "cq": "Chongqing", "jl": "Jilin", "yt": "Yantai"}
YEAR = 2022
# Column names of LaDe's original internal release, mapped as in LaDe's preprocess.py.
LEGACY_NAMES = {"from_dipan_id": "region_id", "from_city_name": "city", "delivery_user_id": "courier_id",
                "poi_lng": "lng", "poi_lat": "lat", "typecode": "aoi_type", "got_time": "pickup_time",
                "got_gps_time": "pickup_gps_time", "got_gps_lng": "pickup_gps_lng",
                "got_gps_lat": "pickup_gps_lat", "book_start_time": "time_window_start",
                "expect_got_time": "time_window_end", "package_id": "order_id"}
DETOUR = 1.3


@dataclass
class LaDeOptions:
    task: str = "delivery"                 # delivery (LaDe-D) | pickup (LaDe-P)
    cities: Sequence[str] = ("jl", "yt")   # city codes; the two small cities by default
    min_stops: int = 5
    max_stops: int = 100
    merge_minutes: float = 1.0             # merge same-location completions within this gap
    max_speed_m_per_min: float = 500.0     # LaDe's GPS-drift threshold
    max_first_leg_min: float = 120.0       # cap on the dispatch time before the first stop
    static: bool = False                   # release every task at the episode start
    n_aoi_types: int = 8                   # one-hot the most frequent AOI types (+ "other")
    split_fractions: Sequence[float] = (0.7, 0.1, 0.2)
    max_episodes_per_city: Optional[int] = None
    seed: int = 0


def find_city_file(raw_dir: str, task: str, city: str) -> str:
    pats = [os.path.join(raw_dir, "**", f"{task}_{city}*.parquet"),
            os.path.join(raw_dir, "**", f"{task}_{city}*.csv")]
    for p in pats:
        hits = sorted(glob.glob(p, recursive=True))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"no {task}_{city}*.parquet/csv under {raw_dir}; run "
                            f"`python prepare_realworld.py download --dataset {'lade_d' if task == 'delivery' else 'lade_p'}`")


def read_city(path: str):
    import pandas as pd
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    return df.rename(columns={k: v for k, v in LEGACY_NAMES.items() if k in df.columns})


def to_minutes(series):
    """'MM-DD HH:MM:SS' (LaDe) -> minutes since YEAR-01-01 00:00 (NaN if missing)."""
    import pandas as pd
    s = series.astype(str).str.strip()
    dt = pd.to_datetime(f"{YEAR}-" + s, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    return ((dt - pd.Timestamp(f"{YEAR}-01-01")).dt.total_seconds() / 60.0).to_numpy()


def _nan_min(a: float, b: float) -> float:
    return b if a != a else (a if b != b else min(a, b))


def _nan_max(a: float, b: float) -> float:
    return b if a != a else (a if b != b else max(a, b))


def _local_ids(values: np.ndarray) -> np.ndarray:
    _, inv = np.unique(values, return_inverse=True)
    return inv.astype(np.int32)


def clean_courier_day(g, task: str, opt: LaDeOptions) -> Optional[Dict[str, np.ndarray]]:
    """One courier-day group -> merged, drift-filtered stop arrays in finish order."""
    fin = g["_finish"].to_numpy()
    order = np.argsort(fin, kind="stable")
    col = lambda c: g[c].to_numpy()[order] if c in g.columns else np.full(len(g), np.nan)
    lat, lng, fin = col("lat").astype(float), col("lng").astype(float), fin[order]
    acc = col("_accept")
    tws, twe = col("_tw_start"), col("_tw_end")
    aoi, aoi_type = col("aoi_id"), col("aoi_type")
    stops: List[Dict] = []
    for i in range(len(fin)):
        if (stops and abs(stops[-1]["lat"] - lat[i]) < 1e-6 and abs(stops[-1]["lng"] - lng[i]) < 1e-6
                and fin[i] - stops[-1]["finish"] <= opt.merge_minutes):
            s = stops[-1]
            s["n_pkg"] += 1
            s["finish"] = max(s["finish"], fin[i])
            s["accept"] = _nan_min(s["accept"], acc[i])
            s["tw_start"] = _nan_max(s["tw_start"], tws[i])
            s["tw_end"] = _nan_min(s["tw_end"], twe[i])
            continue
        stops.append(dict(lat=lat[i], lng=lng[i], finish=fin[i], accept=acc[i], tw_start=tws[i],
                          tw_end=twe[i], aoi=aoi[i], aoi_type=aoi_type[i], n_pkg=1))
    if len(stops) >= 3:
        la = np.array([s["lat"] for s in stops])
        ln = np.array([s["lng"] for s in stops])
        f = np.array([s["finish"] for s in stops])
        speed = haversine_km(la[:-1], ln[:-1], la[1:], ln[1:]) * 1000.0 / np.maximum(np.diff(f), 0.1)
        fast = speed > opt.max_speed_m_per_min
        drift = np.zeros(len(stops), bool)
        drift[1:-1] = fast[:-1] & fast[1:]
        stops = [s for s, d in zip(stops, drift) if not d]
    if not (opt.min_stops <= len(stops) <= opt.max_stops):
        return None
    out = {k: np.array([s[k] for s in stops]) for k in stops[0]}
    # depot: where the courier accepted the earliest task (the station for LaDe-D), if plausible
    dep_lat, dep_lng = out["lat"][0], out["lng"][0]
    if "accept_gps_lat" in g.columns and np.isfinite(acc).any():
        j = int(np.nanargmin(acc))
        a_lat, a_lng = float(col("accept_gps_lat")[j]), float(col("accept_gps_lng")[j])
        if (np.isfinite(a_lat) and np.isfinite(a_lng) and 15 < a_lat < 55 and 70 < a_lng < 140
                and haversine_km(a_lat, a_lng, out["lat"].mean(), out["lng"].mean()) < 30.0):
            dep_lat, dep_lng = a_lat, a_lng
    out["depot"] = np.array([dep_lat, dep_lng])
    return out


def calibrate(days: List[Dict[str, np.ndarray]]) -> Dict[str, float]:
    """Per-city service time (min) and travel speed (km/min) from logged legs.

    Only legs whose next task was already accepted when the previous one
    finished are used, so waiting for new requests does not bias the speed.
    """
    dist, dt = [], []
    for d in days:
        known = ~(d["accept"][1:] > d["finish"][:-1])     # NaN accept counts as known
        dist.append(haversine_km(d["lat"][:-1], d["lng"][:-1], d["lat"][1:], d["lng"][1:])[known])
        dt.append(np.diff(d["finish"])[known])
    dist, dt = np.concatenate(dist), np.concatenate(dt)
    near = dist < 0.05
    service = float(np.clip(np.median(dt[near]) if near.sum() > 20 else 2.0, 0.5, 15.0))
    move = (dist > 0.3) & (dt - service > 0.5)
    speed = float(np.clip(np.median(DETOUR * dist[move] / (dt[move] - service)) if move.sum() > 20 else 0.25,
                          0.05, 0.6))
    return dict(service_min=service, speed_km_per_min=speed, legs=int(len(dist)))


def build_instance(d: Dict[str, np.ndarray], meta: Dict, calib: Dict[str, float], aoi_vocab: List,
                   task: str, opt: LaDeOptions, rng: np.random.Generator):
    n = len(d["finish"]) + 1
    first_fin = d["finish"][0]
    acc = d["accept"]
    t0 = first_fin - opt.max_first_leg_min
    if np.isfinite(acc).any():
        t0 = max(float(np.nanmin(acc)), t0)
    perm = rng.permutation(n - 1)                   # node order must not reveal the visit order
    node_of_stop = np.empty(n - 1, np.int64)
    node_of_stop[perm] = np.arange(1, n)

    def place(values, depot_value, dtype=np.float32):
        out = np.empty(n, dtype)
        out[0] = depot_value
        out[node_of_stop] = values
        return out

    lat = place(d["lat"], d["depot"][0], np.float64)
    lng = place(d["lng"], d["depot"][1], np.float64)
    x, y = latlng_to_local_km(lat, lng, lat[0], lng[0])
    finish = d["finish"] - t0
    release = np.where(np.isfinite(acc), acc - t0, 0.0).clip(min=0.0)
    release = np.minimum(release, finish)           # a task cannot finish before it is accepted
    if opt.static:
        release[:] = 0.0
    tw_start = np.where(np.isfinite(d["tw_end"]), np.nan_to_num(d["tw_start"] - t0, nan=0.0), 0.0)
    tw_end = np.where(np.isfinite(d["tw_end"]), d["tw_end"] - t0, np.inf)
    zone = place(_local_ids(np.asarray(d["aoi"]).astype(str)), -1, np.int32)
    zone[0] = -1
    grid = np.floor(np.stack([x[1:], y[1:]], -1)).astype(np.int64)
    zone2 = np.full(n, -1, np.int32)
    zone2[1:] = _local_ids(grid[:, 0] * 100003 + grid[:, 1])
    onehot = np.zeros((n - 1, len(aoi_vocab) + 1), np.float32)
    types = np.asarray(d["aoi_type"]).astype(str)
    for i, t in enumerate(types):
        onehot[i, aoi_vocab.index(t) if t in aoi_vocab else len(aoi_vocab)] = 1.0
    node_feats = np.zeros((n, 1 + onehot.shape[1]), np.float32)
    node_feats[node_of_stop, 0] = np.log1p(d["n_pkg"])
    node_feats[node_of_stop, 1:] = onehot
    date = np.datetime64(f"{YEAR}-01-01") + np.timedelta64(int(first_fin // 1440), "D")
    dow = (date.astype("datetime64[D]").view("int64") - 4) % 7   # 1970-01-01 was a Thursday
    inst_global = np.array([np.log1p(d["n_pkg"].sum()), np.sin(2 * np.pi * dow / 7),
                            np.cos(2 * np.pi * dow / 7)], np.float32)
    rec = InstanceRecord(coords=np.stack([x, y], -1), node_feats=node_feats, zone=zone, zone2=zone2,
                         release=place(release, 0.0), tw_start=place(tw_start, 0.0),
                         tw_end=place(tw_end, np.inf), service=place(np.full(n - 1, calib["service_min"]), 0.0),
                         speed=calib["speed_km_per_min"], start_tod=float(t0 % 1440),
                         inst_global=inst_global, meta=dict(meta, date=str(date), n_packages=int(d["n_pkg"].sum())))
    return rec, node_of_stop, finish.astype(np.float32)


def convert_city(raw_dir: str, city: str, opt: LaDeOptions, aoi_vocab: List, name: str):
    path = find_city_file(raw_dir, opt.task, city)
    print(f"[{name}] reading {path}")
    df = read_city(path)
    fin_col = "delivery_time" if opt.task == "delivery" else "pickup_time"
    df["_finish"] = to_minutes(df[fin_col])
    df["_accept"] = to_minutes(df["accept_time"])
    if opt.task == "pickup" and "time_window_end" in df.columns:
        df["_tw_start"] = to_minutes(df["time_window_start"])
        df["_tw_end"] = to_minutes(df["time_window_end"])
    else:
        df["_tw_start"] = np.nan
        df["_tw_end"] = np.nan
    n0 = len(df)
    df = df.drop_duplicates(subset=["order_id"]) if "order_id" in df.columns else df.drop_duplicates()
    ok = (np.isfinite(df["_finish"]) & df["lat"].between(15, 55) & df["lng"].between(70, 140))
    df = df[ok]
    days, metas = [], []
    stats = dict(rows=int(n0), rows_valid=int(len(df)), episodes_raw=0, episodes_kept=0)
    for (courier, ds), g in df.groupby(["courier_id", "ds"], sort=True):
        stats["episodes_raw"] += 1
        d = clean_courier_day(g, opt.task, opt)
        if d is None:
            continue
        days.append(d)
        metas.append(dict(courier_id=int(courier), ds=int(ds), city=CITIES.get(city, city),
                          region_id=int(g["region_id"].iloc[0]) if "region_id" in g.columns else -1,
                          task=opt.task))
    if not days:
        raise ValueError(f"[{name}] no episodes left for city {city}")
    calib = calibrate(days)
    rng = np.random.default_rng(opt.seed)
    if opt.max_episodes_per_city is not None and len(days) > opt.max_episodes_per_city:
        keep = np.sort(rng.choice(len(days), opt.max_episodes_per_city, replace=False))
        days, metas = [days[i] for i in keep], [metas[i] for i in keep]
    stats["episodes_kept"] = len(days)
    print(f"[{name}] {city}: {stats}, calibration {calib}")
    instances, trajs = [], []
    for d, m in zip(days, metas):
        rec, actions, times = build_instance(d, m, calib, aoi_vocab, opt.task, opt, rng)
        trajs.append(TrajectoryRecord(inst=len(instances), actions=actions, times=times, source="courier"))
        instances.append(rec)
    node_names = ["log_n_packages"] + [f"aoi_type_{t}" for t in aoi_vocab] + ["aoi_type_other"]
    eps = RoutingEpisodeSet.from_records(
        name, "all", "haversine", instances, trajs, node_feat_names=node_names,
        glob_feat_names=["log_n_packages_day", "sin_dow", "cos_dow"], detour=DETOUR,
        meta=dict(dataset=name, part=city, calibration={city: calib}, cleaning={city: stats},
                  license="Apache-2.0", source=HF_REPOS[opt.task], time_unit="minutes"),
        max_nodes=opt.max_stops + 1)
    return eps


def aoi_vocabulary(raw_dir: str, opt: LaDeOptions) -> List[str]:
    import pandas as pd
    counts = None
    for city in opt.cities:
        path = find_city_file(raw_dir, opt.task, city)
        col = (pd.read_parquet(path, columns=["aoi_type"]) if path.endswith(".parquet")
               else pd.read_csv(path, usecols=["aoi_type"]))["aoi_type"].astype(str)
        c = col.value_counts()
        counts = c if counts is None else counts.add(c, fill_value=0)
    return list(counts.sort_values(ascending=False).index[:opt.n_aoi_types])


def split_by_date(eps: RoutingEpisodeSet, fractions: Sequence[float]) -> Dict[str, RoutingEpisodeSet]:
    days = sorted({m["date"] for m in eps.inst_meta})
    cuts = np.round(np.cumsum(fractions) / np.sum(fractions) * len(days)).astype(int)
    names = ["train", "val", "test"]
    bounds = dict(zip(names, zip([0] + list(cuts[:-1]), cuts)))
    out = {}
    for split, (a, b) in bounds.items():
        chosen = set(days[a:b])
        idx = [i for i, m in enumerate(eps.inst_meta) if m["date"] in chosen]
        if idx:
            out[split] = eps.select_instances(idx, split=split)
            out[split].meta["dates"] = [days[a], days[b - 1]]
    return out


def _prepare(name: str, task: str, raw_dir: str, out_dir: str, **options) -> Dict[str, str]:
    opt = LaDeOptions(task=task, **options)
    if isinstance(opt.cities, str):
        opt.cities = [c.strip() for c in opt.cities.split(",") if c.strip()]
    os.makedirs(out_dir, exist_ok=True)
    vocab = aoi_vocabulary(raw_dir, opt)
    per_city = [convert_city(raw_dir, c, opt, vocab, name) for c in opt.cities]
    full = concat_sets(per_city)
    calib, cleaning = {}, {}
    for s in per_city:
        calib.update(s.meta["calibration"])
        cleaning.update(s.meta["cleaning"])
    full.meta.update(calibration=calib, cleaning=cleaning, cities=list(opt.cities),
                     options={k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(opt).items()})
    paths = {}
    for split, eps in split_by_date(full, opt.split_fractions).items():
        paths[split] = eps.save(os.path.join(out_dir, f"{split}.npz"))
        print(f"[{name}] wrote {paths[split]}: {eps.summary()}")
    return paths


def prepare_delivery(raw_dir: str, out_dir: str, **options) -> Dict[str, str]:
    return _prepare("lade_d", "delivery", raw_dir, out_dir, **options)


def prepare_pickup(raw_dir: str, out_dir: str, **options) -> Dict[str, str]:
    return _prepare("lade_p", "pickup", raw_dir, out_dir, **options)


def _download(task: str, raw_dir: str, **_) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError("pip install huggingface_hub") from e
    repo = HF_REPOS[task]
    target = os.path.join(raw_dir, repo.split("/")[-1])
    print(f"[lade] downloading {repo} to {target}")
    snapshot_download(repo_id=repo, repo_type="dataset", local_dir=target)


_COMMON = dict(
    family="routing", license="Apache-2.0",
    citation="Wu et al., LaDe: The First Comprehensive Last-mile Express Dataset from Industry, KDD 2024, "
             "doi:10.1145/3637528.3671548",
    logged_source="courier", splits=["train", "val", "test"], return_to_depot=False,
    smdp_discount=True, time_source="logged", reference_metrics=["hr3", "krc", "lsd", "ed"])

register(DatasetSpec(
    name="lade_d",
    description="LaDe-D: Cainiao courier delivery logs (2022, 5 Chinese cities), one episode per courier-day",
    mdp="per-courier SMDP: next delivery among accepted packages, realized sojourns, exogenous arrivals",
    url="https://huggingface.co/datasets/Cainiao-AI/LaDe-D",
    prepare=prepare_delivery, download=lambda raw_dir, **kw: _download("delivery", raw_dir),
    reward=dict(time=1.0, travel=0.0, lateness=0.0, zone_leave=0.0, latency=0.0), **_COMMON))

register(DatasetSpec(
    name="lade_p",
    description="LaDe-P: Cainiao courier pick-up logs with customer time windows, one episode per courier-day",
    mdp="per-courier SMDP with dynamic request arrivals and time windows (exogenous, replayable)",
    url="https://huggingface.co/datasets/Cainiao-AI/LaDe-P",
    prepare=prepare_pickup, download=lambda raw_dir, **kw: _download("pickup", raw_dir),
    reward=dict(time=1.0, travel=0.0, lateness=1.0, zone_leave=0.0, latency=0.0), **_COMMON))
