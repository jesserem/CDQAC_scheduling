"""Tiny synthetic raw files in the exact published layouts of the real datasets.

Used by the tests (and handy to try the pipeline before downloading ~3 GB):

    python -m tests.realworld_fixtures ./raw_fixtures

writes ``./raw_fixtures/amazon`` (the almrrc2021 JSON layout, including the
non-standard ``NaN`` tokens of the challenge files) and ``./raw_fixtures/lade``
(LaDe-D / LaDe-P per-city parquet files with the Hugging Face column names and
``MM-DD HH:MM:SS`` time strings; one city as CSV to exercise that path).
The numbers are random; only the schemas are real.
"""
import json
import os
import string
import sys
from datetime import datetime, timedelta

import numpy as np

METROS = {"LA": (34.05, -118.25), "SE": (47.61, -122.33), "AU": (30.27, -97.74), "CH": (41.88, -87.63),
          "BO": (42.36, -71.06)}


def _stop_ids(rng, n):
    letters = string.ascii_uppercase
    pool = [a + b for a in letters for b in letters]
    return list(rng.choice(pool, n, replace=False))


def write_amazon(root: str, n_train: int = 24, n_test: int = 10, seed: int = 0) -> str:
    rng = np.random.default_rng(seed)
    parts = {"train": ("almrrc2021-data-training", "model_build_inputs", "model_build_inputs", ""),
             "test": ("almrrc2021-data-evaluation", "model_apply_inputs", "model_score_inputs", "new_")}
    for part, n_routes in (("train", n_train), ("test", n_test)):
        top, apply_dir, score_dir, pre = parts[part]
        route_data, seqs, travel, packages, invalid = {}, {}, {}, {}, {}
        for r in range(n_routes):
            rid = f"RouteID_{rng.integers(1 << 62):016x}-{part}{r}"
            metro = list(METROS)[r % len(METROS)]
            lat0, lng0 = METROS[metro]
            n = int(rng.integers(8, 21))
            ids = _stop_ids(rng, n + 1)
            station = ids[0]
            date = datetime(2018, 7, 20) + timedelta(days=int(rng.integers(0, 30)))
            depart = date + timedelta(hours=int(rng.integers(14, 17)), minutes=int(rng.integers(0, 60)))
            stops = {station: dict(lat=lat0 + rng.normal(0, 0.01), lng=lng0 + rng.normal(0, 0.01),
                                   type="Station", zone_id=float("nan"))}
            centers = rng.normal(0, 0.03, size=(3, 2))
            for s in ids[1:]:
                z = int(rng.integers(0, 3))
                zone = float("nan") if rng.random() < 0.05 else f"{'ABC'[z]}-{10 + z}.{rng.integers(1, 3)}{'ABCD'[rng.integers(0, 4)]}"
                stops[s] = dict(lat=lat0 + 0.1 + centers[z, 0] + rng.normal(0, 0.004),
                                lng=lng0 + 0.1 + centers[z, 1] + rng.normal(0, 0.004), type="Dropoff", zone_id=zone)
            rd = dict(station_code=f"D{metro}{rng.integers(1, 9)}", date_YYYY_MM_DD=date.strftime("%Y-%m-%d"),
                      departure_time_utc=depart.strftime("%H:%M:%S"), executor_capacity_cm3=int(3313071),
                      stops=stops)
            if part == "train":
                rd["route_score"] = str(rng.choice(["High", "Medium", "Low"], p=[0.45, 0.5, 0.05]))
            route_data[rid] = rd
            xy = np.array([[stops[s]["lat"], stops[s]["lng"]] for s in ids])
            d = np.abs(xy[:, None, :] - xy[None, :, :]).sum(-1) * 111.0  # km (manhattan-ish)
            tt = d / 30.0 * 3600 + rng.uniform(20, 90, size=(n + 1, n + 1))  # seconds, asymmetric
            np.fill_diagonal(tt, 0.0)
            travel[rid] = {a: {b: round(float(tt[i, j]), 1) for j, b in enumerate(ids)} for i, a in enumerate(ids)}
            order = [station] + list(rng.permutation(ids[1:]))
            seqs[rid] = {"actual": {s: i for i, s in enumerate(order)}}
            pk = {}
            for s in ids[1:]:
                pk[s] = {}
                for _ in range(int(rng.integers(1, 4))):
                    if rng.random() < 0.3:
                        a = depart + timedelta(hours=int(rng.integers(0, 3)))
                        tw = dict(start_time_utc=a.strftime("%Y-%m-%d %H:%M:%S"),
                                  end_time_utc=(a + timedelta(hours=int(rng.integers(1, 4)))).strftime("%Y-%m-%d %H:%M:%S"))
                    else:
                        tw = dict(start_time_utc=float("nan"), end_time_utc=float("nan"))
                    pkg = dict(time_window=tw, planned_service_time_seconds=float(rng.integers(10, 120)),
                               dimensions=dict(depth_cm=float(rng.uniform(5, 50)), height_cm=float(rng.uniform(2, 30)),
                                               width_cm=float(rng.uniform(5, 40))))
                    if part == "train":
                        pkg = dict(scan_status="DELIVERED", **pkg)
                    pk[s][f"PackageID_{rng.integers(1 << 62):016x}"] = pkg
            packages[rid] = pk
            invalid[rid] = float(rng.uniform(0.7, 1.3))
        base = os.path.join(root, top)
        os.makedirs(os.path.join(base, apply_dir), exist_ok=True)
        os.makedirs(os.path.join(base, score_dir), exist_ok=True)
        files = {os.path.join(base, apply_dir, f"{pre}route_data.json"): route_data,
                 os.path.join(base, apply_dir, f"{pre}travel_times.json"): travel,
                 os.path.join(base, apply_dir, f"{pre}package_data.json"): packages,
                 os.path.join(base, score_dir, f"{pre}actual_sequences.json"): seqs,
                 os.path.join(base, score_dir, f"{pre}invalid_sequence_scores.json"): invalid}
        for path, obj in files.items():
            with open(path, "w") as f:
                json.dump(obj, f)  # allow_nan=True -> NaN tokens, like the challenge files
    return root


def _fmt(minutes: float) -> str:
    t = datetime(2022, 1, 1) + timedelta(minutes=float(minutes))
    return t.strftime("%m-%d %H:%M:%S")


def write_lade(root: str, couriers: int = 4, days: int = 12, seed: int = 0, csv_city: str = "yt") -> str:
    import pandas as pd
    rng = np.random.default_rng(seed)
    cities = {"jl": ("Jilin", 43.84, 126.55), "yt": ("Yantai", 37.46, 121.45)}
    for task, repo in (("delivery", "LaDe-D"), ("pickup", "LaDe-P")):
        for code, (city, lat0, lng0) in cities.items():
            rows, oid = [], 0
            for c in range(couriers):
                courier = int(1000 + c + 100 * (code == "yt"))
                region = int(rng.integers(1, 60))
                base = np.array([lat0, lng0]) + rng.normal(0, 0.05, 2)
                aois = rng.normal(0, 0.01, size=(6, 2))
                for dday in range(days):
                    date = datetime(2022, 6, 1) + timedelta(days=dday)
                    day0 = (date - datetime(2022, 1, 1)).days * 1440
                    n = int(rng.integers(3, 25)) if rng.random() > 0.1 else 2
                    t = day0 + 9 * 60 + rng.uniform(0, 60)
                    pos = base.copy()
                    prev_loc = base.copy()
                    for i in range(n):
                        a = int(rng.integers(0, 6))
                        loc = base + aois[a] + rng.normal(0, 0.002, 2)
                        same_stop = bool(i and rng.random() < 0.15)
                        if same_stop:
                            loc = prev_loc  # same stop, another package, finished within a minute
                        dist = np.abs(loc - pos).sum() * 111.0
                        t = t + (rng.uniform(0.0, 0.8) if same_stop else dist / 0.25 + rng.uniform(1, 6))
                        if task == "delivery":
                            accept = day0 + 7 * 60 + rng.uniform(0, 60)
                        else:
                            accept = t - rng.uniform(5, 90)
                        rec = dict(order_id=oid, region_id=region, city=city, courier_id=courier,
                                   lng=float(loc[1]), lat=float(loc[0]), aoi_id=int(a + 10 * c), aoi_type=int(a % 3 + 1),
                                   accept_time=_fmt(accept), accept_gps_time=_fmt(accept),
                                   accept_gps_lng=float(base[1] if task == "delivery" else loc[1] + 0.001),
                                   accept_gps_lat=float(base[0] if task == "delivery" else loc[0] + 0.001))
                        if task == "delivery":
                            rec.update(delivery_time=_fmt(t), delivery_gps_time=_fmt(t),
                                       delivery_gps_lng=float(loc[1]), delivery_gps_lat=float(loc[0]))
                        else:
                            ws = accept + rng.uniform(0, 60)
                            rec.update(time_window_start=_fmt(ws), time_window_end=_fmt(ws + 120),
                                       pickup_time=_fmt(t), pickup_gps_time=_fmt(t),
                                       pickup_gps_lng=float(loc[1]), pickup_gps_lat=float(loc[0]))
                        rec["ds"] = int(date.strftime("%m%d"))
                        rows.append(rec)
                        oid += 1
                        pos, prev_loc = loc, loc
                    if n >= 5:  # a GPS-drift stop in the middle of the day
                        j = len(rows) - n // 2
                        rows[j] = dict(rows[j], lat=rows[j]["lat"] + 0.5, lng=rows[j]["lng"] + 0.5)
            df = pd.DataFrame(rows)
            out = os.path.join(root, repo, "data")
            os.makedirs(out, exist_ok=True)
            if code == csv_city:
                df.to_csv(os.path.join(out, f"{task}_{code}.csv"), index=False)
            else:
                df.to_parquet(os.path.join(out, f"{task}_{code}-00000-of-00001-fixture.parquet"), index=False)
    return root


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "./raw_fixtures"
    print(write_amazon(os.path.join(root, "amazon")))
    print(write_lade(os.path.join(root, "lade")))
