# Real-world offline-RL datasets for combinatorial optimization

This extension turns real-world routing logs into offline-RL datasets and
trains every method of this repository on them (CDQAC, IQL, TD3-AWR,
Offline-LD's d-mSAC / mQRDQN, BC) without changing the method code.

| Dataset | Registry name | Behaviour | Decision process | License |
|---|---|---|---|---|
| 2021 Amazon Last Mile Routing Research Challenge (9,184 routes, 2018, 5 US metros) | `amazon_lmrrc` | real drivers (`driver`) | deterministic finite-horizon MDP, next-stop choice | CC BY-NC 4.0 |
| LaDe-D, Cainiao delivery logs (2022, 5 Chinese cities) | `lade_d` | real couriers (`courier`) | per-courier-day SMDP, realized sojourns | Apache-2.0 |
| LaDe-P, Cainiao pick-up logs with time windows | `lade_p` | real couriers (`courier`) | per-courier-day SMDP with dynamic arrivals + time windows | Apache-2.0 |

Synthetic behaviour (random, nearest-neighbour, eps-greedy, zone-aware, soft-min)
can be added on top of the real instances, so real, synthetic and mixed
datasets can be compared the way the paper compares PDR / GA / random data.

## Quick start

```bash
pip install -r requirements.txt   # + pyarrow (LaDe parquet), ijson (streams Amazon's multi-GB JSON)

python prepare_realworld.py list

# Amazon (~3.1 GB, public S3 bucket, no credentials needed)
python prepare_realworld.py download --dataset amazon_lmrrc          # -> ./raw/amazon_lmrrc
python prepare_realworld.py convert  --dataset amazon_lmrrc          # -> ./realworld_data/amazon_lmrrc/{train,val,test}.npz

# LaDe (Hugging Face; the two small cities by default, --opt cities=sh,hz,cq,jl,yt for all)
python prepare_realworld.py download --dataset lade_d
python prepare_realworld.py convert  --dataset lade_d --opt cities=jl,yt max_stops=100

# optional: add synthetic behaviour to a split (coverage augmentation)
python prepare_realworld.py augment  --dataset amazon_lmrrc --split train --behaviors random:5,eps_nn_0.1:5,nn:1
python prepare_realworld.py describe --dataset amazon_lmrrc --split train   # datasheet numbers per source

# train (same methods / hyperparameters / CLI as train.py)
python train_realworld.py --dataset amazon_lmrrc                         # CDQAC on the real driver routes
python train_realworld.py --method iql --dataset lade_d --sources courier,random
python train_realworld.py --dataset amazon_lmrrc --sources driver,eps_nn* --instance_filter route_score=High
python train_realworld.py --dataset lade_p --gamma 0.999 --smdp_discount true   # per-minute discounting

# evaluate checkpoints, heuristics and the logged behaviour on the test split
python eval_realworld.py checkpoints/amazon_lmrrc/<run> logged nn zone_nn random --split test
```

No download? `python -m tests.realworld_fixtures ./raw_fixtures` writes tiny
files in the exact raw layouts of both datasets; point `--raw_dir` at
`./raw_fixtures/amazon` or `./raw_fixtures/lade` to try the whole pipeline.

## How it fits into the repository

```
raw files ──adapter──▶ RoutingEpisodeSet (.npz) ──RoutingBuffer──▶ batches ──▶ cdqac.methods.* (unchanged)
 (per dataset)          instances + trajectories    lazy states,            ▲
                                                    rewards, gamma^tau      │ networks from
                                         RoutingEnv (rollouts) ◀── policy ──┘ realworld.nets.RoutingNets
```

| Piece | File | Role |
|---|---|---|
| Episode format | `realworld/episodes.py` | instances (stops, travel model, time windows, releases, zones) + trajectories (action sequence, logged completion times, behaviour source) |
| Adapters | `realworld/adapters/{amazon_lmrrc,lade}.py` | raw files → episode sets, download helpers, registration |
| Registry | `realworld/registry.py` | `DatasetSpec` per dataset: license, MDP, default reward, SMDP flag, metrics |
| State / replay / env | `realworld/routing/core.py` | one batched state builder shared by buffer and env; trajectory replay; rollout env |
| Buffer | `realworld/routing/buffer.py` | `cdqac.buffer.Buffer`-compatible batches, states rebuilt lazily, per-transition discounts |
| Networks | `realworld/routing/network.py`, `realworld/nets.py` | LEHD-style encoder; actor / quantile critic / value / mQRDQN heads with the FJSP interfaces |
| Behaviours | `realworld/routing/behaviors.py` | synthetic behaviour sources |
| Metrics / evaluation | `realworld/routing/metrics.py`, `realworld/evaluation.py` | official Amazon score, LaDe route metrics, route time, lateness, zone backtracking |
| CLIs | `prepare_realworld.py`, `train_realworld.py`, `eval_realworld.py` | download / convert / augment / describe, train, evaluate |

The only change to existing code is in `train.py`: the `build_*` functions take
a network factory (`nets=FJSPNets` by default, bit-identical networks for the
JSP/FJSP path; checked for all six methods).

## Decision processes

Common to both datasets: node 0 is the depot / start, every action is the next
stop, times are minutes since the episode start, a state is fully determined by
(instance, served set, current stop, clock).

**Serving and decision epochs.** At decision epoch `d_t` the vehicle is at the
current stop; serving stop `a` finishes at `f_t = d_t + travel(cur, a) + service(a)`
(or at the logged completion time). The next epoch is `d_{t+1} = f_t`, or the
next request arrival if nothing is open. The SMDP sojourn is
`tau_t = d_{t+1} - d_t`, so `sum_t tau_t` is the episode duration.

**Amazon (`amazon_lmrrc`).** Option A of our MDP analysis: a deterministic
finite-horizon MDP with the challenge's travel-time matrix (historical
averages) and planned service times. Default reward: `-(travel + service)` in
hours, including the return to the station, i.e. the return is minus the route
duration. The driver's real objective is latent (tacit knowledge, zones), so
evaluation reports both the duration and the official challenge score against
the driver's route. Zone structure is in the state (same-zone / open-in-zone
features, zone pooling in the network) and optionally in the reward
(`--reward_zone_leave`, a backtracking penalty for leaving a zone that still has
open stops). Splits: the challenge's 6,112 training routes → `train` / `val`
(random 10%), its 3,072 evaluation routes → `test`. Route-quality labels
(`route_score` High / Medium / Low) are kept in the metadata
(`--instance_filter route_score=High`).

**LaDe (`lade_d`, `lade_p`).** Option B: one courier on one day is an episode
of a per-courier SMDP with exogenous, replayable request arrivals (a task is
released when the courier accepts it). The logged action is the next task
finished, the realized sojourn comes from the log (`--time_source logged`), and
rollouts use `detour × distance / speed + service`, with speed and service time
calibrated per city from logged legs (stored in the metadata). Default reward:
`-tau` (time), plus `-lateness` w.r.t. the customer time window for LaDe-P;
`--reward_latency` gives the sum-of-completion-times objective instead.
`--opt static=True` freezes the day (Option A: all tasks known at the start).
Cleaning follows LaDe's own preprocessing (GPS-drift stops removed, packages
finished at the same place within a minute merged into one stop), node order is
shuffled so it cannot leak the route, and splits are by date.

One modelling assumption to keep in mind: only completions are logged, not
when the courier committed to the next task. When the logged next task was
released *after* the previous completion, its decision epoch is taken as
`max(f_t, release)`, the latest time consistent with the log. The rollout env
never idles voluntarily (it only waits when nothing is open), so a policy cannot
reproduce "waiting for a request that has not arrived yet".

**Discounting.** With `--smdp_discount true` the buffer returns
`gamma ** (tau / sojourn_unit)` per transition and the training loop sets it as
the trainer's discount for that batch. Every method only uses `self.discount`
inside `r + (1 - done) * discount * next_value`, so a `[B, 1]` tensor gives the
exact SMDP target (tested). With the repository default `gamma = 1` all
discounts are 1, as in the paper's finite-horizon setting.

## Episode-set format (`<data_root>/<dataset>/<split>.npz`)

Plain arrays plus a JSON header (no pickle), padded to the largest instance `N`:

| Field | Shape | Meaning |
|---|---|---|
| `n_nodes` | `[E]` | nodes incl. depot |
| `coords` | `[E, N, 2]` | local km, depot at the origin |
| `node_feats` | `[E, N, F]` | dataset-specific static features (`node_feat_names`) |
| `zone`, `zone2` | `[E, N]` | fine / coarse zone ids local to the instance (-1 = none) |
| `release`, `tw_start`, `tw_end`, `service` | `[E, N]` | minutes (`tw_end = inf`: no window) |
| `speed`, `start_tod` | `[E]` | km/min (distance-based travel), local minute of day at clock 0 |
| `inst_global` | `[E, G]` | dataset-specific instance features (`glob_feat_names`) |
| `travel_flat`, `travel_offsets` | `[sum n²]`, `[E]` | travel matrices (minutes), ragged, `travel_mode == "matrix"` |
| `traj_inst`, `traj_len` | `[K]` | trajectory → instance, length |
| `traj_actions` | `[K, N-1]` | visit order (-1 padded) |
| `traj_times` | `[K, N-1]` | logged completion times, NaN if not logged |
| `traj_source` | `[K]` | index into `source_names` (`driver`, `courier`, `random`, `eps_nn_0.1`, ...) |

`RoutingEpisodeSet.load(path).summary()` gives the datasheet numbers;
`inst_meta` holds per-instance metadata (route id, station, date, route score,
courier, city, ...).

## State and network

The state is an 8-tuple whose slots mirror the FJSP state (mask in slot 6,
`get_embedding` takes slots 0-5 and 7), which is what lets all methods run
unchanged: `node_feat [B,N,Fn]`, `node_valid [B,N]`, `current [B]`,
`global_feat [B,Fg]`, `node_open [B,N]`, `zone [B,N]`, `action_mask [B,N]`,
`edge_feat [B,N,Fe]`. Features (names in `realworld/routing/core.py`) cover
geometry, time windows and slack, release / age, travel time and its rank from
the current stop, same-zone indicators and open-stops-per-zone, time of day and
progress. They are standardized with statistics estimated on the training
buffer and stored in every checkpoint.

The encoder follows LEHD (light encoder over all stops, heavier decoder over the
open stops + a context token), with optional zone pooling. Size it with
`--d_model --n_heads --n_enc_layers --n_dec_layers --ff_dim`; the MLP heads use
the method's usual `num_mlp_layers_*` / `hidden_dim_*` fields. Amazon routes
have up to ~250 stops, so attention is the dominant cost: start with
`--batch_size 128` on a 16 GB GPU.

## Evaluation

`eval_realworld.py` (and the validation during training) reports per episode
and on average: `return`, route `time`, `time_ratio` (vs. the logged route
replayed under the same travel model), `lateness`, `late_stops`, zone leaves and
revisits (backtracking), and route similarity to the logged route: the official
Amazon challenge score (`amazon_score`, exact re-implementation of
`scoring/score.py` of MIT-CAVE/rc-cli, 0 = identical, ~1 = random) and LaDe's
route-prediction metrics (`hr3`, `krc`, `lsd`, `ed`). For LaDe the logged route's
real duration is reported as `logged_time`. Baselines: `logged`, `nn`, `zone_nn`,
`random`, and any behaviour name. `--num_samples N` keeps the best of N sampled
rollouts per instance.

Checkpoint selection on the validation split uses `--select_metric` (default
`return`; `amazon_score`, `time`, `krc`, ... also work).

## Adding a dataset

**Another routing-type dataset** (e.g. Meituan, EURO-NeurIPS VRP, NYC TLC
driver trips): only an adapter is needed.

1. Create `realworld/adapters/<name>.py` with
   `prepare(raw_dir, out_dir, **options) -> {split: path}` that builds
   `InstanceRecord`s (node 0 = depot; coords in local km via
   `latlng_to_local_km`; times in minutes; a travel matrix *or* a speed for
   distance-based travel) and `TrajectoryRecord`s (the logged sequences, with
   completion times if the log has them), then
   `RoutingEpisodeSet.from_records(...).save(...)` per split.
   Optionally `download(raw_dir)`.
2. Register it at the bottom of the module with
   `register(DatasetSpec(name=..., logged_source=..., reward=..., smdp_discount=..., ...))`
   and add the module to `_load_builtin()` in `realworld/registry.py`.
3. Add a writer for a tiny raw fixture in `tests/realworld_fixtures.py` and a
   test that checks the conversion against the raw values (see
   `TestAdapters`).

Instance-style datasets without logged behaviour work the same way: write
the instances with no trajectories and generate behaviour with
`prepare_realworld.py augment` (random / heuristics), or add a stronger
heuristic to `realworld/routing/behaviors.py`.

**A new problem family** (Azure packing, Alibaba / Borg / PWA scheduling, ACN
charging): mirror `realworld/routing/`.

1. An episode container (or reuse `RoutingEpisodeSet`'s pattern: instances +
   trajectories + sources, saved as `.npz`).
2. A tensor/state builder producing an 8-tuple with the feasibility mask in slot
   6 and `get_embedding(slots 0-5, 7)`, a replay function (rewards, sojourns,
   decision clocks) and a rollout env for evaluation. Exogenous traces (Azure
   arrivals) replay exactly; endogenous ones (Alibaba placement → durations)
   need a simulator for honest evaluation.
3. Networks with the FJSP interfaces (see `realworld/routing/network.py`) and a
   factory like `RoutingNets`, so `train.METHODS[...]["build"](config, nets=...)`
   works.
4. A `family` value in its `DatasetSpec`; at that point
   `train_realworld.py` / `eval_realworld.py` get a small family dispatch
   (buffer, tensors, nets, evaluate) instead of their current routing imports.

## Tests

```bash
python -m unittest discover tests
```

`tests/test_realworld.py` checks, on fixtures in the exact raw layouts: the
Amazon score against its recursive definition (and it was validated against the
official scorer: identical on 40 random routes up to 160 stops), conversions
against the raw JSON / parquet values, buffer-rebuilt states and rewards against
env rollouts step by step (static matrix and dynamic distance-based instances),
that per-transition discounts reach the TD target, behaviour validity, and a
train → checkpoint → evaluate run of all six methods.

## Data licenses and citation

Amazon data: CC BY-NC 4.0 (non-commercial), cite Merchán et al., *2021 Amazon
Last Mile Routing Research Challenge: Data Set*, Transportation Science 58(1),
2024. LaDe: Apache-2.0, cite Wu et al., *LaDe: The First Comprehensive
Last-mile Express Dataset from Industry*, KDD 2024. Converted episode sets are
derived works and keep these licenses.
