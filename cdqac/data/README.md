# Offline datasets

Every dataset lives in its own directory and is addressed by an id, `<problem>/<name>`:

```
train_datasets/
├── fjsp/
│   ├── sd1_10x5/
│   │   ├── dataset.json        manifest: contents, provenance, checksums
│   │   ├── train.npz           training instances
│   │   ├── val.npz             validation instances
│   │   └── sources/            one file per trajectory source
│   │       ├── pdr.npz
│   │       ├── random.npz
│   │       ├── ga_hof.npz
│   │       ├── ga_pop.npz
│   │       ├── eps_pdr_0.1.npz
│   │       └── eps_pdr_0.2.npz
│   ├── sd1_15x10/
│   └── sd1_20x10/
└── jsp/
    ├── taillard_10x5/
    └── taillard_15x10/
```

Train on one with `python train.py --dataset fjsp/sd1_15x10`. The `use_*` / `n_*` options pick
which of its trajectories go into the replay buffer (e.g. `--use_random true --n_random 100`).

## Released datasets

| id | instances (train / val) | ops per instance | sources (trajectories per instance) |
|---|---|---|---|
| `fjsp/sd1_10x5` | 1000 / 100 | 52 / 44–55 | pdr 16, random 300, ga_hof 100, ga_pop 200, eps_pdr_0.1 100\*, eps_pdr_0.2 100\* |
| `fjsp/sd1_15x10` | 1000 / 100 | 155 / 141–156 | same |
| `fjsp/sd1_20x10` | 500 / 100 | 199 / 184–216 | same (eps for all 500) |
| `jsp/taillard_10x5` | 500 / 100 | 50 / 50 | pdr 4, random 300, ga_hof 100, ga_pop 200 |
| `jsp/taillard_15x10` | 500 / 100 | 150 / 150 | same |

\* first 500 instances only. `python -m cdqac.data info <id>` prints the full description.

## Commands

```bash
python -m cdqac.data list                               # datasets under ./train_datasets
python -m cdqac.data info fjsp/sd1_10x5 -v              # sources, makespans, provenance
python -m cdqac.data validate fjsp/sd1_10x5 --replay 5  # format, checksums, makespan replay
```

## File format

All files are plain numeric arrays (`np.load` never needs `allow_pickle`). Integer arrays use
the smallest unsigned type that fits; cast them before arithmetic.

**Splits** (`train.npz`, `val.npz`):

| array | shape | meaning |
|---|---|---|
| `job_length` | `[N, n_j]` | operations per job |
| `op_pt` | `[N, O_max, n_m]` | processing time of operation *o* on machine *m*; 0 = *m* cannot process *o*. Rows past `n_ops[i]` are zero padding |
| `n_ops` | `[N]` | operations of each instance (`job_length.sum(1)`) |

Operations are numbered job by job: job 0's operations first, in order, then job 1's, and so on.

**Sources** (`sources/<name>.npz`) cover the leading `N_s ≤ N` training instances:

| array | shape | meaning |
|---|---|---|
| `actions` | `[N_s, K, O]` | trajectories as action sequences, action = `job * n_m + machine` |
| `makespan` | `[N_s, K]` | makespan each trajectory reaches |
| `counts` | `[N_s]` | *optional*: trajectories per instance when it varies (rows past it are padding) |
| `names` | `[K]` | *optional*: label of each slot, e.g. the dispatching rule |

**Manifest** (`dataset.json`): `id`, `problem`, `n_j`, `n_m`, `action_encoding`,
`action_mask` (always `true`, see below), `generator`
(how the instances were made: name, seeds, parameters), `provenance` (what the data was
converted or derived from), `splits` and `sources` (file, number of instances, trajectories per
instance, description, `order`), and the `sha256` of every file.

## Sources and what their order means

| source | contents | `order` |
|---|---|---|
| `pdr` | priority dispatching rules named `<job>_<machine>`: job rule {MWR, LWR, LOPNR, MOPNR} × machine rule {EST, LST, LPT, SPT} (`env/dispatching.py`); JSP keeps the 4 job rules with SPT | `rule`: slot *k* is `names[k]`, e.g. `MWR_EST` |
| `random` | a uniformly random feasible action at every step | `generation` |
| `ga_hof` | genetic-algorithm hall of fame | `makespan_asc`: `--n_ga_hof 10` keeps the 10 best |
| `ga_pop` | final genetic-algorithm population, with many duplicates | `generation`: `--n_ga_pop` keeps the first *n*, `--ga_pop_random` samples by 1/makespan |
| `eps_pdr_<eps>` | PDR (4 job rules × SPT, cycled) taking a random feasible action with probability *eps* | `generation` |

`--remove_duplicate true` (default) drops repeated trajectories per instance after selection.
Enabling a source the dataset lacks, or one that covers fewer instances than
`--num_instances`, is an error.

## The action mask

All data is generated, replayed and evaluated with the environment's action mask: at each
decision only operation–machine pairs that can start at the current decision time are allowed.
It is not optional:

- the environments reject `mask_actions=False`, and train.py rejects `--use_mask false` (the
  option only remains so old configs still parse);
- every manifest declares `"action_mask": true`, which `validate` requires;
- replaying a trajectory that takes a forbidden action is an error, both in
  `python -m cdqac.data validate --replay N` and when train.py fills the replay buffer.

Trajectories generated with `env/dispatching.py` or the scripts in `data_generation/` satisfy
the mask by construction. Check your own with `validate --replay`.

## Bring your own data

Write instances and trajectories in one call, then check and train:

```python
from cdqac.data.writer import write_dataset

write_dataset(
    "./train_datasets", "fjsp/my_10x5",
    train=(job_lengths, op_pts),          # per instance: [n_j] ints, [n_ops, n_m] ints
    val=(val_job_lengths, val_op_pts),
    sources={"random": {"actions": actions, "makespans": makespans}},  # actions[i]: [K, n_ops]
)
```

```bash
python -m cdqac.data validate fjsp/my_10x5 --replay 5
python train.py --dataset fjsp/my_10x5
```

train.py replays the training data in an environment that needs every training instance to
have the same number of operations (validation instances may differ). Sources can be added
later with `cdqac.data.writer.add_source`.

## Generating datasets

```bash
# instances + PDR + random (+ eps-PDR); --seed 101 reproduces the released training instances
python -m data_generation.generate_datasets --dataset fjsp/sd1_10x5 --generator sd1 \
    --n_j 10 --n_m 5 --num_train 1000 --seed 101 \
    --sources pdr random:300 eps_pdr:0.1:100:500 eps_pdr:0.2:100:500

# add sources to an existing dataset
python -m data_generation.add_random_trajectories --dataset fjsp/sd1_10x5 --num_random 300
python -m data_generation.add_eps_pdr_rules --dataset fjsp/sd1_10x5 --eps_random 0.1 --num_runs 100

# GA: run data_generation/run_GA_custom.py inside the Job Shop Scheduling Benchmark repo on the
# dataset directory, then add its JSON output as ga_hof / ga_pop
python -m data_generation.add_ga_data --dataset fjsp/sd1_10x5 --ga_folder ./GA_data/sd1_10x5
```

Generators: `sd1` (FJSP, processing times 1–20), `sd2` (FJSP, 1–99), `taillard` (JSP). The
released training instances (`--seed 101`), PDR trajectories and random trajectories
(`--source_seed 100`) are reproduced exactly. The JSP validation instances are reproduced
too (`--val_seed 1101`). The FJSP validation instances come from
[songwenas12/fjsp-drl](https://github.com/songwenas12/fjsp-drl), and the released eps-PDR
and GA trajectories cannot be regenerated bit for bit.

## Legacy single-file datasets

Before this format, a dataset was one pickled `.npy` file per size (`train_10_5.npy`,
`val_10_5.npy`, or the paper-era `SD1_train_10_5_1000.npy`). They hold the same data as the
dataset directories and are kept on the Hub under `train_datasets/legacy/{fjsp,jsp}/`
(`python download_datasets.py --legacy --no_datasets` puts them there locally). They still work:

- `python train.py --train_instance train_15_10.npy` loads
  `./train_datasets/legacy/fjsp/train_15_10.npy` (another folder with `--data_path`, e.g.
  `--data_path ./train_datasets/legacy/jsp`) if it exists, validating on `val_15_10.npy` next
  to it, and otherwise uses the converted dataset (`fjsp/sd1_15x10`).
- The analysis scripts accept a legacy path wherever they take a dataset id.
- Both paths keep only the 16 PDR rules listed above, renamed from the legacy
  `<job>_<machine>_masked` to `<job>_<machine>`; the legacy files also hold LCT/MCT/FAM rules,
  which are ignored.
- `python -m data_generation.convert_legacy --all` converts the released files. Every array
  is compared with the original, and `--replay_check N` replays trajectories to check the
  makespans.

Loading a legacy file unpickles it, which can execute code: only load files you trust.
