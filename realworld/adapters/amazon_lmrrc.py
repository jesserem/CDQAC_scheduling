"""2021 Amazon Last Mile Routing Research Challenge (Merchan et al., Transp. Sci. 2024).

9,184 real routes driven by Amazon drivers in 2018 (Seattle, Los Angeles,
Austin, Chicago, Boston): route-, stop- and package-level features, a travel
time matrix per route (historical averages) and the driver's executed stop
sequence.  CC BY-NC 4.0, s3://amazon-last-mile-challenges/almrrc2021/ (public).

Decision process (Option A of our MDP analysis)
    Deterministic finite-horizon MDP per route: state = (remaining stops,
    current stop, clock), action = next stop, transition deterministic given
    the travel matrix.  Default reward = -(travel + planned service time)
    [minutes], incl. the return to the station, i.e. the return is minus the
    route duration.  The *driver's* objective is latent (tacit knowledge, zone
    conventions), so evaluation reports both route duration and the official
    challenge score against the driver sequence.  One logged trajectory per
    route (source ``driver``); synthetic sources can be added with
    ``prepare_realworld.py augment``.

Raw layout after ``aws s3 sync --no-sign-request s3://amazon-last-mile-challenges/almrrc2021/ RAW``::

    RAW/almrrc2021-data-training/model_build_inputs/{route_data,actual_sequences,travel_times,
                                                     package_data,invalid_sequence_scores}.json
    RAW/almrrc2021-data-evaluation/model_apply_inputs/{new_route_data,new_travel_times,new_package_data}.json
    RAW/almrrc2021-data-evaluation/model_score_inputs/{new_actual_sequences,new_invalid_sequence_scores}.json

Splits written by :func:`prepare`: ``train`` / ``val`` (random 10% of the
challenge's training routes) and ``test`` (the challenge's 3,072 evaluation
routes).
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import tqdm

from realworld.adapters._io import is_missing, iter_json_object, load_json, s3_public_sync
from realworld.episodes import (InstanceRecord, RoutingEpisodeSet, TrajectoryRecord,
                                latlng_to_local_km)
from realworld.registry import DatasetSpec, register

NAME = "amazon_lmrrc"
S3_BUCKET = "amazon-last-mile-challenges"
S3_PREFIX = "almrrc2021/"

PARTS = {
    "train": dict(dir="almrrc2021-data-training",
                  route="model_build_inputs/route_data.json",
                  seq="model_build_inputs/actual_sequences.json",
                  travel="model_build_inputs/travel_times.json",
                  package="model_build_inputs/package_data.json",
                  invalid="model_build_inputs/invalid_sequence_scores.json"),
    "test": dict(dir="almrrc2021-data-evaluation",
                 route="model_apply_inputs/new_route_data.json",
                 seq="model_score_inputs/new_actual_sequences.json",
                 travel="model_apply_inputs/new_travel_times.json",
                 package="model_apply_inputs/new_package_data.json",
                 invalid="model_score_inputs/new_invalid_sequence_scores.json"),
}

# Summer-2018 (DST) UTC offsets of the five metro areas, keyed by the two
# letters after the "D" of the station code (DLA3 -> LA).  Used only for the
# time-of-day feature; unknown codes fall back to UTC.
METRO_UTC_OFFSET_H = {"LA": -7, "SE": -7, "AU": -5, "CH": -5, "BO": -4}

NODE_FEATS = ["log_n_packages", "log_volume_l", "tw_width_h"]
GLOB_FEATS = ["capacity_m3", "load_factor", "log_n_packages_route"]


def _parse_utc(s: str) -> Optional[datetime]:
    if is_missing(s):
        return None
    try:
        return datetime.strptime(str(s).strip(), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def _stop_packages(pkgs: Dict, depart: datetime):
    """(n_packages, service_min, volume_l, tw_start_min, tw_end_min) of one stop."""
    n, svc, vol = 0, 0.0, 0.0
    tw_s, tw_e = -np.inf, np.inf
    for p in (pkgs or {}).values():
        n += 1
        s = p.get("planned_service_time_seconds")
        svc += 0.0 if is_missing(s) else float(s) / 60.0
        d = p.get("dimensions") or {}
        try:
            vol += float(d["depth_cm"]) * float(d["height_cm"]) * float(d["width_cm"]) / 1000.0
        except (KeyError, TypeError, ValueError):
            pass
        tw = p.get("time_window") or {}
        a, b = _parse_utc(tw.get("start_time_utc")), _parse_utc(tw.get("end_time_utc"))
        if a is not None and b is not None:
            tw_s = max(tw_s, (a - depart).total_seconds() / 60.0)
            tw_e = min(tw_e, (b - depart).total_seconds() / 60.0)
    if not np.isfinite(tw_e):
        tw_s = 0.0
    return n, svc, vol, tw_s, tw_e


def _local_ids(labels):
    ids, table = [], {}
    for v in labels:
        if v is None:
            ids.append(-1)
        else:
            ids.append(table.setdefault(v, len(table)))
    return np.array(ids, np.int32)


def build_route(route_id: str, route: Dict, actual: Dict, travel: Dict, packages: Dict,
                invalid_score: Optional[float]):
    """One route -> (InstanceRecord, driver action sequence) or (None, reason)."""
    stops = route["stops"]
    stations = [s for s, v in stops.items() if str(v.get("type", "")).lower() == "station"]
    if len(stations) != 1:
        return None, "station"
    station = stations[0]
    order = [station] + sorted(s for s in stops if s != station)
    n = len(order)
    pos = {s: i for i, s in enumerate(order)}
    if set(actual) != set(order):
        return None, "sequence_stops"
    seq_nodes = [pos[s] for s, _ in sorted(actual.items(), key=lambda kv: kv[1])]
    if seq_nodes[0] != 0:
        return None, "sequence_start"

    lat = np.array([float(stops[s]["lat"]) for s in order])
    lng = np.array([float(stops[s]["lng"]) for s in order])
    x, y = latlng_to_local_km(lat, lng, lat[0], lng[0])
    try:
        tt = np.array([[float(travel[a][b]) for b in order] for a in order], np.float32) / 60.0
    except (KeyError, TypeError, ValueError):
        return None, "travel"
    if not np.isfinite(tt).all():
        return None, "travel"

    date = str(route["date_YYYY_MM_DD"])
    depart = datetime.strptime(f"{date} {route['departure_time_utc']}", "%Y-%m-%d %H:%M:%S")
    code = str(route.get("station_code", ""))
    offset = METRO_UTC_OFFSET_H.get(code[1:3], 0)
    start_tod = ((depart.hour + offset) * 60 + depart.minute + depart.second / 60.0) % 1440

    agg = np.array([_stop_packages((packages or {}).get(s), depart) for s in order], np.float64)
    n_pkg, svc, vol, tw_s, tw_e = agg.T
    tw_s[0], tw_e[0], svc[0] = 0.0, np.inf, 0.0
    has_tw = np.isfinite(tw_e)
    zone_raw = [None if is_missing(stops[s].get("zone_id")) else str(stops[s]["zone_id"]) for s in order]
    zone_raw[0] = None
    zone = _local_ids(zone_raw)
    zone2 = _local_ids([None if z is None else z.split(".")[0] for z in zone_raw])
    cap_l = float(route.get("executor_capacity_cm3") or 0.0) / 1000.0
    node_feats = np.stack([np.log1p(n_pkg), np.log1p(vol),
                           np.where(has_tw, (tw_e - tw_s) / 60.0, 0.0).clip(0, 24)], -1).astype(np.float32)
    inst_global = np.array([cap_l / 1000.0, vol.sum() / max(cap_l, 1e-6), np.log1p(n_pkg.sum())], np.float32)
    score = route.get("route_score")
    meta = dict(route_id=route_id, station_code=code, metro=code[1:3], date=date,
                departure_time_utc=str(route["departure_time_utc"]),
                route_score=None if is_missing(score) else str(score),
                invalid_sequence_score=None if invalid_score is None else float(invalid_score),
                n_packages=int(n_pkg.sum()), n_zones=int(zone.max() + 1))
    rec = InstanceRecord(coords=np.stack([x, y], -1), node_feats=node_feats, zone=zone, zone2=zone2,
                         release=np.zeros(n, np.float32), tw_start=tw_s.astype(np.float32),
                         tw_end=tw_e.astype(np.float32), service=svc.astype(np.float32), travel=tt,
                         start_tod=float(start_tod), inst_global=inst_global, meta=meta)
    return (rec, np.array(seq_nodes[1:], np.int64)), None


def convert_part(raw_dir: str, part: str, max_routes: Optional[int] = None,
                 max_stops: Optional[int] = None) -> RoutingEpisodeSet:
    files = {k: os.path.join(raw_dir, PARTS[part]["dir"], v) for k, v in PARTS[part].items() if k != "dir"}
    for k, p in files.items():
        if not os.path.exists(p):
            raise FileNotFoundError(f"missing {p}; run `python prepare_realworld.py download --dataset {NAME}`")
    print(f"[{NAME}] reading {part} route data and sequences")
    routes = load_json(files["route"])
    seqs = load_json(files["seq"])
    invalid = load_json(files["invalid"]) if os.path.exists(files["invalid"]) else {}
    wanted = [r for r in routes if r in seqs]
    if max_routes is not None:
        wanted = wanted[:max_routes]
    wanted_set = set(wanted)

    print(f"[{NAME}] streaming package data")
    packages = {}
    for rid, pk in tqdm.tqdm(iter_json_object(files["package"]), desc="packages", leave=False):
        if rid in wanted_set:
            packages[rid] = pk

    instances, trajs, dropped = [], [], {}
    print(f"[{NAME}] streaming travel times and building routes")
    for rid, tt in tqdm.tqdm(iter_json_object(files["travel"]), desc="routes", total=len(routes), leave=False):
        if rid not in wanted_set:
            continue
        out, reason = build_route(rid, routes[rid], seqs[rid]["actual"], tt, packages.pop(rid, {}),
                                  invalid.get(rid))
        if out is None or (max_stops is not None and len(out[0].coords) > max_stops + 1):
            reason = reason or "max_stops"
            dropped[reason] = dropped.get(reason, 0) + 1
            continue
        rec, actions = out
        trajs.append(TrajectoryRecord(inst=len(instances), actions=actions, source="driver"))
        instances.append(rec)
    meta = dict(dataset=NAME, part=part, dropped=dropped, license="CC BY-NC 4.0",
                source=f"s3://{S3_BUCKET}/{S3_PREFIX}", time_unit="minutes")
    print(f"[{NAME}] {part}: {len(instances)} routes kept, dropped {dropped}")
    return RoutingEpisodeSet.from_records(NAME, part, "matrix", instances, trajs, node_feat_names=NODE_FEATS,
                                          glob_feat_names=GLOB_FEATS, meta=meta)


def prepare(raw_dir: str, out_dir: str, val_fraction: float = 0.1, seed: int = 0,
            max_routes: Optional[int] = None, max_stops: Optional[int] = None) -> Dict[str, str]:
    """Convert the raw challenge files into train / val / test episode sets."""
    os.makedirs(out_dir, exist_ok=True)
    train_full = convert_part(raw_dir, "train", max_routes, max_stops)
    parts = train_full.split_by({"train": 1.0 - val_fraction, "val": val_fraction}, seed=seed)
    parts["test"] = convert_part(raw_dir, "test", max_routes, max_stops)
    paths = {}
    for split, eps in parts.items():
        eps.split = split
        paths[split] = eps.save(os.path.join(out_dir, f"{split}.npz"))
        print(f"[{NAME}] wrote {paths[split]}: {eps.summary()}")
    return paths


def download(raw_dir: str, **_) -> None:
    print(f"[{NAME}] downloading s3://{S3_BUCKET}/{S3_PREFIX} (~3.1 GB) to {raw_dir}")
    print("         (equivalent: aws s3 sync --no-sign-request "
          f"s3://{S3_BUCKET}/{S3_PREFIX} {raw_dir})")
    n = s3_public_sync(S3_BUCKET, S3_PREFIX, raw_dir)
    print(f"[{NAME}] {n} files downloaded")


register(DatasetSpec(
    name=NAME, family="routing",
    description="Amazon Last Mile Routing Research Challenge 2021: 9,184 real driver routes (2018, 5 US metros)",
    mdp="deterministic finite-horizon MDP over next-stop choices (travel-time matrix, planned service times)",
    license="CC BY-NC 4.0",
    citation="Merchan et al., 2021 Amazon Last Mile Routing Research Challenge: Data Set, "
             "Transportation Science 58(1), 2024, doi:10.1287/trsc.2022.1173",
    url="https://registry.opendata.aws/amazon-last-mile-challenges/",
    prepare=prepare, download=download, logged_source="driver", splits=["train", "val", "test"],
    reward=dict(time=1.0, travel=0.0, lateness=0.0, zone_leave=0.0, latency=0.0),
    return_to_depot=True, smdp_discount=False, time_source="model", amazon_score=True,
    reference_metrics=["amazon_score", "hr3", "krc", "lsd", "ed"],
))
