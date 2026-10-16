"""Minimal follow-on: J9/8x4 and J8/4x8, main versus combined only."""

import json
import os
import subprocess
from datetime import datetime, timezone

import campaign as common
import numpy as np


def main():
    root, out = common.ROOT, common.OUT
    assert os.environ.get("SLURM_JOB_ID")
    assert (
        int(os.environ["SLURM_NTASKS"]) == 8
        and int(os.environ["SLURM_CPUS_PER_TASK"]) == 4
    )
    manifest = common.read(root / "source_manifest.json")
    for name, expected in manifest["files"].items():
        assert common.sha(root / name) == expected, name
    config = common.read(root / "config.json")
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / "campaign.json").exists()
    groups, raw, allowed = common.topology()
    topology = {}
    for line in raw.splitlines():
        if not line.startswith("#"):
            cpu, core, socket, node = map(int, line.split(","))
            topology[cpu] = (socket, core, node)
    cpus = [cpu for group in groups for cpu in group]
    groups_by_threads = {4: groups}
    node_cpus = {}
    for cpu in cpus:
        node_cpus.setdefault(topology[cpu][2], []).append(cpu)
    whole, leftovers = [], []
    for node in sorted(node_cpus):
        values = sorted(node_cpus[node])
        whole.extend([values[i : i + 8] for i in range(0, len(values) - 7, 8)])
        leftovers.extend(values[len(values) // 8 * 8 :])
    whole.extend([leftovers[i : i + 8] for i in range(0, len(leftovers), 8)])
    assert len(whole) == 4 and all(len(group) == 8 for group in whole)
    groups_by_threads[8] = whole
    common.save(
        out / "hardware.json",
        dict(
            host=os.uname().nodename,
            rank_groups_by_threads=groups_by_threads,
            allowed=allowed,
            lscpu=json.loads(subprocess.check_output(["lscpu", "-J"], text=True)),
            slurm={
                key: value
                for key, value in os.environ.items()
                if key.startswith("SLURM_")
            },
        ),
    )
    common.save(
        out / "environment.json",
        dict(
            python=common.sys.version,
            executable=common.sys.executable,
            versions={
                name: __import__(name).__version__
                for name in ["numpy", "scipy", "numba", "psutil"]
            },
            cache="Numba disk cache allowed; no full-size field/window/result reuse; OS not cleared; noGPU",
        ),
    )
    state = dict(
        status="running",
        job=os.environ["SLURM_JOB_ID"],
        stage="gates",
        completed_runs=[],
        started_utc=datetime.now(timezone.utc).isoformat(),
        source_manifest_sha256=common.sha(root / "source_manifest.json"),
    )
    common.save(out / "campaign.json", state)
    reference = None
    for case in config["gates"]:
        common.launch(
            dict(case, gate=True, rank_groups=groups_by_threads[case["threads"]]),
            state["source_manifest_sha256"],
        )
        curve = np.load(out / "gates" / case["name"] / "measurement.npz")["xi"]
        if reference is None:
            reference = curve
        np.testing.assert_allclose(curve, reference, rtol=1e-8, atol=1e-11)
    common.save(
        out / "gate_verification.json",
        dict(status="passed", cases=len(config["gates"]), rtol=1e-8, atol=1e-11),
    )
    for case in config["schedule"]:
        state.update(stage="production", current=case["name"])
        common.save(out / "campaign.json", state)
        common.launch(
            dict(
                case,
                gate=False,
                rank_groups=groups_by_threads[case["threads"]],
                catalog=config["catalog"],
            ),
            state["source_manifest_sha256"],
        )
        state["completed_runs"].append(case["name"])
        common.save(out / "campaign.json", state)
    state.update(
        status="complete",
        stage="measurements_complete",
        current=None,
        completed_utc=datetime.now(timezone.utc).isoformat(),
    )
    common.save(out / "campaign.json", state)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        import traceback

        traceback.print_exc()
        if (common.OUT / "campaign.json").exists():
            state = common.read(common.OUT / "campaign.json")
            state.update(status="failed", error=repr(error))
            common.save(common.OUT / "campaign.json", state)
        raise
