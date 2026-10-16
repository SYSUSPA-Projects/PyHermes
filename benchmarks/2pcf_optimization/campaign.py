"""Single Slurm allocation: five ablations, then 3 x J8 at fixed 8 x 4."""
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parent
OUT = Path(os.environ['FIG13_OPT_OUTPUT'])


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def topology():
    raw = subprocess.check_output(['lscpu', '-p=CPU,CORE,SOCKET,NODE'], text=True)
    allocation = subprocess.check_output(['scontrol', 'show', 'job', '-d', os.environ['SLURM_JOB_ID'], '--oneliner'], text=True)
    (OUT / 'allocation.txt').write_text(allocation)
    # Slurm CPU_IDs use topology numbering; they are not Linux CPU IDs.
    # The batch affinity is Slurm's actual Linux mask after --hint=nomultithread.
    allowed = set(os.sched_getaffinity(0))
    save(OUT / 'batch_affinity.json', dict(allowed=sorted(allowed)))
    (OUT / 'topology.txt').write_text(raw)
    assert len(allowed) == 32, ('Expected an allocation-bound32CPU batch mask', sorted(allowed))
    nodes, seen = {}, set()
    for line in raw.splitlines():
        if line.startswith('#'):
            continue
        cpu, core, socket, node = map(int, line.split(','))
        if cpu in allowed and (socket, core) not in seen:
            nodes.setdefault(node, []).append(cpu)
            seen.add((socket, core))
    # Build only four-core groups wholly within one NUMA node, balanced if possible.
    chunks = {node: [sorted(values)[i:i+4] for i in range(0, len(values)-3, 4)]
              for node, values in nodes.items()}
    assert len(seen) == 32, ('Allocation must contain32 distinct physical cores', nodes)
    leftovers = [cpu for values in nodes.values() for cpu in sorted(values)[len(values)//4*4:]]
    groups = []
    while len(groups) < 8:
        progress = False
        for node in sorted(chunks):
            if chunks[node] and len(groups) < 8:
                groups.append(chunks[node].pop(0))
                progress = True
        if not progress:
            assert len(leftovers) >= 4, ('Cannot form32-core rank groups', nodes)
            groups.append(leftovers[:4])
            leftovers = leftovers[4:]
    return groups, raw, sorted(allowed)


def launch(case, manifest_sha):
    folder = OUT / ('gates' if case['gate'] else 'runs') / case['name']
    assert not folder.exists(), ('Refuse duplicate/partial case; inspect first', folder)
    folder.mkdir(parents=True)
    case = dict(case, source_manifest_sha256=manifest_sha)
    save(folder / 'case.json', case)
    env = dict(os.environ, PYTHONPATH=str(ROOT / case['source']) + ':' + str(ROOT),
        OMP_NUM_THREADS='4', NUMBA_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        NUMEXPR_NUM_THREADS='1', OMP_DYNAMIC='FALSE', OMP_MAX_ACTIVE_LEVELS='1', OMP_PROC_BIND='false',
        NUMBA_CACHE_DIR=str(OUT / 'numba_cache'), PYHERMES_MPI_LOG_MODE='root')
    command = [str(Path(sys.executable).parent / 'mpirun'), '-np', '8', '--map-by', 'slot',
               '--bind-to', 'none', sys.executable, str(ROOT / 'measure.py'), '--case', str(folder / 'case.json')]
    started, peak, samples, overhead = time.time(), 0, 0, 0.
    with (folder / 'worker.log').open('w') as stream:
        process = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        launcher = psutil.Process(process.pid)
        try:
            while process.poll() is None:
                if time.time() - started > 600:
                    raise TimeoutError('Worker exceeded600s; preserve failure scene')
                if (folder / 'measurement_started.json').exists() and not (folder / 'measurement_ended.json').exists():
                    sample_start = time.perf_counter()
                    rss = 0
                    try:
                        for child in [launcher] + launcher.children(recursive=True):
                            try:
                                rss += child.memory_info().rss
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                pass
                    except psutil.NoSuchProcess:
                        pass
                    peak, samples = max(peak, rss), samples+1
                    overhead += time.perf_counter()-sample_start
                time.sleep(.1)
            assert process.returncode == 0, (case['name'], process.returncode)
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=10)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                pass
            raise
    run = read(folder / 'run.json')
    assert sha(folder / 'measurement.npz') == run['output_sha256']
    assert case['gate'] or (samples > 0 and peak > 0)
    save(folder / 'timing.json', dict(status='complete', process_tree_RSS_bytes=peak,
        sample_count=samples, sampler_interval_seconds=.1, sampler_wall_seconds=overhead,
        launch_seconds=time.time()-started, launcher_command=command,
        note='Measured interval launcher+descendants RSS; shared pages can repeat. SlurmMaxRSS separate.'))
    print(json.dumps(dict(completed=case['name'], seconds=run['phases']['total_seconds'], RSS_GiB=peak/1024**3)), flush=True)


def main():
    assert os.environ.get('SLURM_JOB_ID')
    assert int(os.environ['SLURM_NTASKS']) == 8 and int(os.environ['SLURM_CPUS_PER_TASK']) == 4
    manifest = read(ROOT / 'source_manifest.json')
    for name, expected in manifest['files'].items():
        assert sha(ROOT / name) == expected, name
    config = read(ROOT / 'config.json')
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'campaign.json').exists(), 'No blind resume or duplicate campaign'
    groups, raw, allowed = topology()
    (OUT / 'topology.txt').write_text(raw)
    save(OUT / 'hardware.json', dict(host=os.uname().nodename, rank_groups=groups, allowed=allowed,
        lscpu=json.loads(subprocess.check_output(['lscpu', '-J'], text=True)),
        slurm={key: value for key, value in os.environ.items() if key.startswith('SLURM_')}))
    save(OUT / 'environment.json', dict(python=sys.version, executable=sys.executable,
        mpi=subprocess.check_output([str(Path(sys.executable).parent/'mpirun'), '--version'], text=True),
        versions={name: __import__(name).__version__ for name in ['numpy', 'scipy', 'numba', 'psutil']},
        cache='Numba disk cache allowed; each sample new process/field/window; OS cache not cleared; noGPU'))
    state = dict(status='running', job=os.environ['SLURM_JOB_ID'], stage='gates', completed_runs=[],
        started_utc=datetime.now(timezone.utc).isoformat(), source_manifest_sha256=sha(ROOT/'source_manifest.json'))
    save(OUT / 'campaign.json', state)
    arms = config['arms']
    for arm in arms:
        launch(dict(arm, name='gate_'+arm['arm'], gate=True, rank_groups=groups), state['source_manifest_sha256'])
    reference = np.load(OUT/'gates/gate_main/measurement.npz')['xi']
    for arm in arms:
        np.testing.assert_allclose(np.load(OUT/'gates'/('gate_'+arm['arm'])/'measurement.npz')['xi'], reference, rtol=1e-8, atol=1e-11)
    save(OUT/'gate_verification.json', dict(status='passed', arms=5, bins=31, rtol=1e-8, atol=1e-11))
    for case in config['schedule']:
        state.update(stage='production', current=case['name'])
        save(OUT/'campaign.json', state)
        launch(dict(case, gate=False, rank_groups=groups, catalog=config['catalog']), state['source_manifest_sha256'])
        state['completed_runs'].append(case['name'])
        save(OUT/'campaign.json', state)
    state.update(status='complete', stage='measurements_complete', current=None,
                 completed_utc=datetime.now(timezone.utc).isoformat())
    save(OUT/'campaign.json', state)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        import traceback
        traceback.print_exc()
        if (OUT/'campaign.json').exists():
            state = read(OUT/'campaign.json')
            state.update(status='failed', error=repr(error))
            save(OUT/'campaign.json', state)
        raise
