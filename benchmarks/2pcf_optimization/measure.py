"""Isolated, fresh-process 2PCF ablation worker. No full-size result reuse."""
import argparse
import gc
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type=Path, required=True)
    args = parser.parse_args()
    case = json.loads(args.case.read_text())
    folder = args.case.parent
    root = Path(__file__).resolve().parent
    rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
    group = case['rank_groups'][rank]
    os.sched_setaffinity(0, group)
    sys.path.insert(0, str(root / case['source']))
    import numba
    import numpy as np
    from mpi4py import MPI
    from window_kernels import CENTRES, WINDOW

    import pyhermes
    from pyhermes.base.sfc_projection import SFCProjection
    from pyhermes.io import SFCField, WindowFunc
    from pyhermes.theory.corr2pcf import Corr_2PCF
    from pyhermes.utils import convolution
    assert Path(pyhermes.__file__).resolve().is_relative_to((root / case['source']).resolve())
    comm = MPI.COMM_WORLD
    assert comm.rank == rank and comm.size == case.get('ranks', 8)
    threads = case.get('threads', 4)
    numba.set_num_threads(threads)
    counters = {}
    active = False

    def instrument(owner, name, label):
        original = getattr(owner, name)

        def wrapped(*values, **options):
            if not active:
                return original(*values, **options)
            started = time.perf_counter()
            try:
                return original(*values, **options)
            finally:
                entry = counters.setdefault(label, dict(count=0, seconds=0.0))
                entry['count'] += 1
                entry['seconds'] += time.perf_counter() - started
        setattr(owner, name, wrapped)

    instrument(convolution, 'rfftn', 'forward_fft')
    instrument(convolution, 'irfftn', 'inverse_fft')
    instrument(SFCField, 'copy', 'field_copy')
    instrument(Corr_2PCF, '_broadcast_field', 'broadcast_field')
    instrument(Corr_2PCF, '_delta_field', 'delta_field')
    instrument(WindowFunc, 'as_array', 'window_as_array_including_cached_calls')
    if case['source'] == 'candidate':
        from pyhermes.utils import corr2pcf_spectral
        instrument(corr2pcf_spectral, 'rfftn', 'forward_fft')

    def thread_masks():
        masks = {}
        for task in Path('/proc/self/task').iterdir():
            try:
                mask = sorted(os.sched_getaffinity(int(task.name)))
            except ProcessLookupError:
                continue
            assert set(mask) <= set(group), (rank, task.name, mask, group)
            masks[task.name] = mask
        return masks

    def calculation(positions, resolution, measured=False):
        nonlocal active
        comm.Barrier()
        started = time.perf_counter()
        projection = SFCProjection({'SFCProjection': dict(box_size=1000., J=resolution,
            wavelet_mode='db2', wavelet_level=10, phi_resolution=1024, threads=threads,
            weight_normalization='catalog', save_particle_data=False, fout_path='')})
        if rank == 0:
            projection.particle_pos = np.asarray(positions, dtype=np.float32)
        field = projection.run(save_result=False)
        comm.Barrier()
        projected = time.perf_counter()
        active = measured
        options = dict(random='uniform', binning_window=WINDOW,
            sampling=dict(s=CENTRES.tolist()), products=['xi'], threads=threads,
            weight_normalization='catalog', binning_window_cache=False,
            memory_strategy='speed', fout_path='')
        if case['source'] == 'candidate':
            options.update(pair_reduction=case['pair_reduction'], reuse_input_fields=case['reuse_input_fields'])
        task = Corr_2PCF({'Corr_2PCF': options})
        created = time.perf_counter()
        messages = {}
        original_info = task.logger.info

        def log_info(message, *args, **kwargs):
            for label in ['Pre-2PCF setup time:', '2PCF main loop time:']:
                if str(message).startswith(label):
                    messages[label] = float(str(message).split(':', 1)[1].split()[0])
            return original_info(message, *args, **kwargs)
        task.logger.info = log_info
        task.prepare_input_fields(sfc_field1=field, sfc_field2=field)
        prepared = time.perf_counter()
        output = task.run(save_result=False)
        comm.Barrier()
        ended = time.perf_counter()
        task.logger.info = original_info
        active = False
        phases = dict(total_seconds=ended-started, projection_seconds=projected-started,
            task_construction_seconds=created-projected, explicit_input_preparation_seconds=prepared-created,
            run_seconds=ended-prepared, correlation_seconds=ended-projected)
        if rank == 0:
            metrics = getattr(task, 'runtime_metrics', {})
            phases['main_loop_seconds'] = metrics.get('main_loop_seconds', messages['2PCF main loop time:'])
            phases['pre_2pcf_setup_seconds'] = metrics.get('pre_2pcf_setup_seconds', messages['Pre-2PCF setup time:'])
            phases['compute_including_spectral_setup_seconds'] = metrics.get('compute_including_spectral_setup_seconds', phases['main_loop_seconds'])
            return np.asarray(output.xi).reshape(-1), field, phases, metrics
        return None, None, phases, {}

    positions = None
    if rank == 0:
        if case['gate']:
            positions = np.random.default_rng(731).uniform(0., 1000., (256, 3)).astype(np.float32).astype(np.float64)
        else:
            with np.load(case['catalog']) as data:
                positions = np.ascontiguousarray(data['pos'], dtype=np.float64)
            assert positions.shape == (406728, 3)
        input_sha = hashlib.sha256(memoryview(positions).cast('B')).hexdigest()
        if not case['gate']:
            assert input_sha == '077c096962fd042a8dfc4defa25c43a0ea29764c1b419bcf27798b489b138389'
    warm_start = time.perf_counter()
    calculation(positions[:64] if rank == 0 else None, 5)
    gc.collect()
    comm.Barrier()
    warmup_seconds = time.perf_counter() - warm_start
    before_masks = thread_masks()
    if rank == 0:
        save(folder / 'measurement_started.json', dict(started=time.time(), warmup_seconds=warmup_seconds))
    comm.Barrier()
    vector, field, phases, metrics = calculation(positions, 5 if case['gate'] else case.get('J', 8), measured=True)
    if rank == 0:
        save(folder / 'measurement_ended.json', dict(ended=time.time()))
    record = dict(rank=rank, counters=counters, affinity=sorted(os.sched_getaffinity(0)),
                  before_thread_masks=before_masks, after_thread_masks=thread_masks(), phases=phases)
    save(folder / f'rank_{rank}.json', record)
    all_records = comm.gather(record, root=0)
    if rank == 0:
        assert vector.shape == (31,) and np.all(np.isfinite(vector))
        if not case['gate']:
            with np.load(root / ('reference_J'+str(case.get('J', 8))+'.npz')) as data:
                np.testing.assert_allclose(vector, data['xi'], rtol=1e-8, atol=1e-11)
        np.savez(folder / 'measurement.npz', xi=vector, s=CENTRES)
        field_sha = hashlib.sha256(memoryview(np.ascontiguousarray(field.epsilon)).cast('B')).hexdigest()
        save(folder / 'run.json', dict(status='complete', case=case, phases=phases,
            runtime_metrics=metrics, input_positions_sha256=input_sha, field_sha256=field_sha,
            output_sha256=digest(folder / 'measurement.npz'), ranks=all_records,
            warmup_seconds=warmup_seconds, package_file=pyhermes.__file__,
            note='Full timing excludes imports, small-J warmup, catalog/output IO, SHA and thread audits. Nested per-function CPU wall durations overlap and must not be summed. Main baseline loop log rounded to 0.0001s.'))


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        traceback.print_exc()
        try:
            from mpi4py import MPI
            MPI.COMM_WORLD.Abort(1)
        except Exception:
            pass
        raise
