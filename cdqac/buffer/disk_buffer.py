"""Disk-backed replay buffer + PyTorch Dataset/DataLoader pipeline for TD3-AWR.

The big per-transition state arrays (fea_j, comp_idx, fea_pairs, ...) are stored
in numpy memmaps on disk instead of RAM, so dataset size is bounded by disk, not
memory. Small per-transition scalars (reward, done, action, next_idx, mc_return)
stay in RAM (a few bytes per transition).

Binary-valued fields (op_mask, mch_mask, comp_idx -- the env only ever writes
0/1 into them) are stored as uint8 on disk and expanded back to float32 at read
time, cutting disk footprint and I/O by ~30%.

Sampling goes through ``TD3TransitionDataset``: each DataLoader "item" is a whole
batch of indices, fetched with a single fancy-index read per field (sorted for
disk locality) and normalized on the fly with the stats from
``DiskBuffer.normalize_state``. Combined with ``num_workers``/``prefetch_factor``
and pinned memory, batch preparation overlaps with GPU compute.

IMPORTANT for datasets larger than RAM: uniformly-random single-transition reads
degrade to one disk seek per transition once the memmaps no longer fit the OS
page cache -- catastrophic on network filesystems (NFS/Lustre). For that regime:
  * put the buffer on node-local scratch (``$SLURM_TMPDIR``), and
  * set ``chunk_len`` > 1 on the samplers (e.g. 32) so batches are built from
    contiguous index runs (sequential I/O). Transitions are laid out env-major
    (index = t * n_envs + e), so a contiguous chunk holds the same timestep of
    DIFFERENT trajectories -- batches stay diverse.

Batches have the exact ``Buffer.sample_td3`` format:
    state(8-tuple), next_state(8-tuple), actions, next_actions, rewards, dones,
    mc_returns
"""

import mmap
import os
import shutil

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler


class DiskBuffer:
    """Drop-in replacement for ``Buffer`` whose state tensors live on disk."""

    def __init__(self, size: int, n_j: int, n_m: int, n_op: int, f_j: int = 10,
                 f_m: int = 8, device: str = 'cpu', dir_path: str = './buffer_cache'):
        self.float_type = torch.float32
        self.size = size
        self.ptr = 0
        self.curr_size = 0
        self.device = device
        self.n_j = n_j
        self.n_m = n_m
        self.n_op = n_op
        self.f_j = f_j
        self.f_m = f_m
        self.dir_path = os.path.abspath(dir_path)
        os.makedirs(self.dir_path, exist_ok=True)

        # name -> (per-transition shape, on-disk dtype, dtype handed to the model)
        self.field_specs = {
            'fea_j': ((n_op, f_j), np.float32, np.float32),
            'op_mask': ((n_op, 3), np.uint8, np.float32),
            'fea_m': ((n_m, f_m), np.float32, np.float32),
            'mch_mask': ((n_m, n_m), np.uint8, np.float32),
            'dynamic_pair_mask': ((n_j, n_m), np.bool_, np.bool_),
            'comp_idx': ((n_m, n_m, n_j), np.uint8, np.float32),
            'candidate': ((n_j,), np.int32, np.int64),
            'fea_pairs': ((n_j, n_m, f_m), np.float32, np.float32),
        }
        self.fields = {}
        total_bytes = 0
        for name, (shape, disk_dtype, _) in self.field_specs.items():
            path = os.path.join(self.dir_path, name + '.dat')
            self.fields[name] = np.memmap(path, dtype=disk_dtype, mode='w+',
                                          shape=(size,) + shape)
            total_bytes += self.fields[name].nbytes
        print(f"[DiskBuffer] {size:,} transitions -> "
              f"{total_bytes / 1024 ** 3:.1f} GB of memmap files in {self.dir_path}")

        # Small per-transition data stays in RAM.
        self.reward = np.zeros((size, 1), dtype=np.float32)
        self.mc_return = np.zeros((size, 1), dtype=np.float32)
        self.done = np.zeros((size, 1), dtype=np.float32)
        self.action = np.zeros((size, 1), dtype=np.int64)
        self.next_idx = np.zeros(size, dtype=np.int64)

        self._checked_binary = False

        # Filled by normalize_state(); identity until then.
        self.stats = {
            'mean_fea_j': np.zeros((1, 1, f_j), dtype=np.float32),
            'std_fea_j': np.ones((1, 1, f_j), dtype=np.float32),
            'mean_fea_m': np.zeros((1, 1, f_m), dtype=np.float32),
            'std_fea_m': np.ones((1, 1, f_m), dtype=np.float32),
            'mean_fea_pairs': np.zeros((1, 1, 1, f_m), dtype=np.float32),
            'std_fea_pairs': np.ones((1, 1, 1, f_m), dtype=np.float32),
        }

    def validate_action(self, job_idx, machine_idx, dynamic_pair_mask):
        """Validate if the action respects the dynamic pair mask"""
        return ~dynamic_pair_mask[:, job_idx, machine_idx]

    def __len__(self):
        return self.curr_size

    def set_mc_return(self, mc_return: np.ndarray, start_idx: int, end_idx: int):
        self.mc_return[start_idx:end_idx] = np.asarray(
            mc_return, dtype=np.float32).reshape(-1, 1)

    @staticmethod
    def _to_np(tensor, dtype):
        return np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype=dtype)

    def push(self, state, next_state, action, reward, done):
        n_envs = state.fea_j_tensor.shape[0]
        indices = np.arange(self.ptr, self.ptr + n_envs, dtype=np.int64) % self.size
        self.curr_size = min(self.size, self.curr_size + n_envs)

        if not self._checked_binary:
            # uint8 storage silently truncates fractional values -- guard the
            # assumption that these masks are strictly 0/1 once per buffer.
            for label, t in [('op_mask', state.op_mask_tensor),
                             ('mch_mask', state.mch_mask_tensor),
                             ('comp_idx', state.comp_idx_tensor)]:
                vals = t.detach().cpu().numpy()
                if not np.isin(vals, (0, 1)).all():
                    raise ValueError(
                        f"DiskBuffer stores {label} as uint8 but the env produced "
                        f"non-binary values; change its dtype in field_specs.")
            self._checked_binary = True

        specs = self.field_specs
        self.fields['fea_j'][indices] = self._to_np(state.fea_j_tensor, specs['fea_j'][1])
        self.fields['op_mask'][indices] = self._to_np(state.op_mask_tensor, specs['op_mask'][1])
        self.fields['fea_m'][indices] = self._to_np(state.fea_m_tensor, specs['fea_m'][1])
        self.fields['mch_mask'][indices] = self._to_np(state.mch_mask_tensor, specs['mch_mask'][1])
        self.fields['dynamic_pair_mask'][indices] = self._to_np(
            state.dynamic_pair_mask_tensor, specs['dynamic_pair_mask'][1])
        self.fields['comp_idx'][indices] = self._to_np(state.comp_idx_tensor, specs['comp_idx'][1])
        self.fields['candidate'][indices] = self._to_np(state.candidate_tensor, specs['candidate'][1])
        self.fields['fea_pairs'][indices] = self._to_np(state.fea_pairs_tensor, specs['fea_pairs'][1])

        self.reward[indices] = np.asarray(reward, dtype=np.float32).reshape(-1, 1)
        self.done[indices] = np.asarray(done, dtype=np.float32).reshape(-1, 1)
        self.action[indices] = np.asarray(action, dtype=np.int64).reshape(-1, 1)

        self.ptr = int(indices[-1]) + 1
        self.next_idx[indices] = indices + n_envs

    def flush(self):
        for mm in self.fields.values():
            mm.flush()

    def close(self, delete: bool = False):
        """Release memmap handles; optionally delete the on-disk files."""
        for name in list(self.fields):
            mm = self.fields.pop(name)
            # np.memmap keeps the file open through its _mmap attribute
            if hasattr(mm, '_mmap') and mm._mmap is not None:
                mm._mmap.close()
        if delete:
            shutil.rmtree(self.dir_path, ignore_errors=True)

    # ------------------------------------------------------------------ #
    def _streaming_stats(self, name: str, reduce_ndim: int, chunk: int = 4096):
        """Mean/std (unbiased, matching torch .std) over the filled part of a
        memmap field, reducing over the first ``reduce_ndim`` axes, computed in
        chunks so peak RAM stays at ~chunk transitions."""
        mm = self.fields[name]
        feat_shape = mm.shape[reduce_ndim:]
        s = np.zeros(feat_shape, dtype=np.float64)
        ss = np.zeros(feat_shape, dtype=np.float64)
        n = 0
        axes = tuple(range(reduce_ndim))
        for start in range(0, self.ptr, chunk):
            block = np.asarray(mm[start:min(start + chunk, self.ptr)],
                               dtype=np.float64)
            s += block.sum(axis=axes)
            ss += (block * block).sum(axis=axes)
            n += int(np.prod(block.shape[:reduce_ndim]))
        mean = s / n
        var = (ss - s * s / n) / max(n - 1, 1)
        std = np.sqrt(np.maximum(var, 0.0))
        std = np.maximum(std, 1e-8)
        keepdim = (1,) * reduce_ndim + feat_shape
        return (mean.reshape(keepdim).astype(np.float32),
                std.reshape(keepdim).astype(np.float32))

    def normalize_state(self):
        """Compute normalization stats over the filled part of the buffer.

        Unlike ``Buffer.normalize_state`` this does NOT rewrite the data: the
        raw values stay on disk and ``TD3TransitionDataset`` applies
        (x - mean) / std on the fly at read time. Returns the same 6-tuple of
        torch tensors as ``Buffer.normalize_state``.
        """
        self.flush()
        mean_fj, std_fj = self._streaming_stats('fea_j', reduce_ndim=2)
        mean_fm, std_fm = self._streaming_stats('fea_m', reduce_ndim=2)
        mean_fp, std_fp = self._streaming_stats('fea_pairs', reduce_ndim=3)
        self.stats = {
            'mean_fea_j': mean_fj, 'std_fea_j': std_fj,
            'mean_fea_m': mean_fm, 'std_fea_m': std_fm,
            'mean_fea_pairs': mean_fp, 'std_fea_pairs': std_fp,
        }
        return (torch.from_numpy(mean_fj), torch.from_numpy(mean_fm),
                torch.from_numpy(std_fj), torch.from_numpy(std_fm),
                torch.from_numpy(mean_fp), torch.from_numpy(std_fp))


class TD3TransitionDataset(Dataset):
    """Batch-indexed dataset over a (finalized) DiskBuffer.

    ``__getitem__`` takes an ARRAY of transition indices (yielded by the batch
    samplers below) and returns a full collated batch in ``sample_td3`` format,
    so no per-sample collate is needed. Memmaps are opened lazily per process,
    which keeps the dataset cheap to pickle to DataLoader workers.

    ``madvise_random=True`` disables kernel readahead on the memmaps (Linux
    only) -- use it with fully-random sampling (chunk_len == 1) on datasets
    larger than RAM, where readahead only wastes disk bandwidth. Leave it off
    for chunked/sequential access.
    """

    def __init__(self, buffer: DiskBuffer, madvise_random: bool = False):
        self.dir_path = buffer.dir_path
        self.field_specs = dict(buffer.field_specs)
        self.n = buffer.curr_size
        self.total_size = buffer.size
        self.madvise_random = madvise_random
        self.reward = buffer.reward[:].copy()
        self.done = buffer.done[:].copy()
        self.action = buffer.action[:].copy()
        self.mc_return = buffer.mc_return[:].copy()
        self.next_idx = np.minimum(buffer.next_idx[:].copy(), self.n - 1)
        self.stats = {k: v.copy() for k, v in buffer.stats.items()}
        self._mm = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_mm'] = None  # never pickle memmap handles (would inline the data)
        return state

    def _ensure_open(self):
        if self._mm is None:
            self._mm = {
                name: np.memmap(os.path.join(self.dir_path, name + '.dat'),
                                dtype=disk_dtype, mode='r',
                                shape=(self.total_size,) + shape)
                for name, (shape, disk_dtype, _) in self.field_specs.items()
            }
            if self.madvise_random and hasattr(mmap, 'MADV_RANDOM'):
                for arr in self._mm.values():
                    try:
                        arr._mmap.madvise(mmap.MADV_RANDOM)
                    except (AttributeError, OSError, ValueError):
                        pass

    def __len__(self):
        return self.n

    def _take(self, name, idx_sorted, inv):
        # Read in sorted order for disk locality, then restore batch order and
        # expand compact on-disk dtypes to what the model expects.
        out_dtype = self.field_specs[name][2]
        return np.ascontiguousarray(self._mm[name][idx_sorted][inv], dtype=out_dtype)

    def _read_state(self, idx):
        order = np.argsort(idx, kind='stable')
        inv = np.empty_like(order)
        inv[order] = np.arange(order.size)
        idx_sorted = idx[order]
        st = self.stats

        fea_j = self._take('fea_j', idx_sorted, inv)
        fea_j -= st['mean_fea_j']
        fea_j /= st['std_fea_j']
        fea_m = self._take('fea_m', idx_sorted, inv)
        fea_m -= st['mean_fea_m']
        fea_m /= st['std_fea_m']
        fea_pairs = self._take('fea_pairs', idx_sorted, inv)
        fea_pairs -= st['mean_fea_pairs']
        fea_pairs /= st['std_fea_pairs']

        return (
            torch.from_numpy(fea_j),
            torch.from_numpy(self._take('op_mask', idx_sorted, inv)),
            torch.from_numpy(self._take('candidate', idx_sorted, inv)),
            torch.from_numpy(fea_m),
            torch.from_numpy(self._take('mch_mask', idx_sorted, inv)),
            torch.from_numpy(self._take('comp_idx', idx_sorted, inv)),
            torch.from_numpy(self._take('dynamic_pair_mask', idx_sorted, inv)),
            torch.from_numpy(fea_pairs),
        )

    def __getitem__(self, batch_idx):
        self._ensure_open()
        idx = np.asarray(batch_idx, dtype=np.int64).reshape(-1)
        nxt = self.next_idx[idx]

        state = self._read_state(idx)
        next_state = self._read_state(nxt)
        actions = torch.from_numpy(self.action[idx])
        next_actions = torch.from_numpy(self.action[nxt])
        rewards = torch.from_numpy(self.reward[idx])
        dones = torch.from_numpy(self.done[idx])
        mc_returns = torch.from_numpy(self.mc_return[idx])

        return state, next_state, actions, next_actions, rewards, dones, mc_returns


class RandomBatchSampler(Sampler):
    """Yields ``num_batches`` arrays of random indices, matching
    ``Buffer.sample_td3`` semantics. Re-iterating continues the random stream
    instead of repeating it.

    ``chunk_len`` > 1 builds each batch from ``batch_size // chunk_len``
    CONTIGUOUS index runs starting at random offsets. Disk reads become
    sequential (essential once the buffer exceeds the OS page cache,
    especially on network filesystems). Transitions are stored env-major, so a
    run of length <= n_envs spans different trajectories at one timestep.
    """

    def __init__(self, data_size: int, batch_size: int, num_batches: int,
                 seed: int = 0, chunk_len: int = 1):
        if chunk_len < 1 or batch_size % chunk_len != 0:
            raise ValueError("chunk_len must be >= 1 and divide batch_size")
        self.data_size = data_size
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.seed = seed
        self.chunk_len = min(chunk_len, data_size)
        self._epoch = 0

    def __iter__(self):
        rng = np.random.default_rng((self.seed, self._epoch))
        self._epoch += 1
        if self.chunk_len == 1:
            for _ in range(self.num_batches):
                yield rng.integers(0, self.data_size, size=self.batch_size,
                                   dtype=np.int64)
        else:
            n_chunks = self.batch_size // self.chunk_len
            offsets = np.arange(self.chunk_len, dtype=np.int64)
            for _ in range(self.num_batches):
                starts = rng.integers(0, self.data_size - self.chunk_len + 1,
                                      size=n_chunks, dtype=np.int64)
                yield (starts[:, None] + offsets).reshape(-1)

    def __len__(self):
        return self.num_batches


class EpochBatchSampler(Sampler):
    """Yields every transition exactly once per iteration in shuffled order,
    matching ``Buffer.epoch_generator_td3`` semantics; reshuffles per epoch.

    ``chunk_len`` > 1 shuffles at the granularity of contiguous chunks instead
    of single transitions, keeping disk reads sequential (see
    ``RandomBatchSampler``).
    """

    def __init__(self, data_size: int, batch_size: int, seed: int = 0,
                 chunk_len: int = 1):
        if chunk_len < 1:
            raise ValueError("chunk_len must be >= 1")
        self.data_size = data_size
        self.batch_size = batch_size
        self.seed = seed
        self.chunk_len = min(chunk_len, data_size)
        self._epoch = 0

    def __iter__(self):
        rng = np.random.default_rng((self.seed, self._epoch))
        self._epoch += 1
        if self.chunk_len == 1:
            perm = rng.permutation(self.data_size).astype(np.int64)
        else:
            starts = np.arange(0, self.data_size, self.chunk_len, dtype=np.int64)
            rng.shuffle(starts)
            perm = np.concatenate([
                np.arange(s, min(s + self.chunk_len, self.data_size), dtype=np.int64)
                for s in starts])
        for start in range(0, self.data_size, self.batch_size):
            yield perm[start:start + self.batch_size]

    def __len__(self):
        return (self.data_size + self.batch_size - 1) // self.batch_size


def make_td3_dataloader(dataset: TD3TransitionDataset, sampler: Sampler,
                        num_workers: int = 4, prefetch_factor: int = 4,
                        pin_memory: bool = True, persistent_workers: bool = True):
    """DataLoader whose sampler yields whole index-batches; batch_size=None
    disables automatic batching so dataset[idx_array] passes through as-is."""
    kwargs = dict(batch_size=None, sampler=sampler, num_workers=num_workers,
                  pin_memory=pin_memory)
    if num_workers > 0:
        kwargs['prefetch_factor'] = prefetch_factor
        kwargs['persistent_workers'] = persistent_workers
    return DataLoader(dataset, **kwargs)


def batch_to_device(batch, device: str, non_blocking: bool = True):
    state, next_state, actions, next_actions, rewards, dones, mc_returns = batch

    def mv(t):
        return t.to(device, non_blocking=non_blocking)

    return (tuple(mv(t) for t in state), tuple(mv(t) for t in next_state),
            mv(actions), mv(next_actions), mv(rewards), mv(dones), mv(mc_returns))
