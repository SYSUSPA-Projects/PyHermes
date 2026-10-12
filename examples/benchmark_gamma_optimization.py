"""Interleave baseline/candidate microbenchmarks; never label them halo timings."""

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numba
import numpy as np

from pyhermes.io import SFCField
from pyhermes.utils import corr3pcf_multipoles as candidate
from pyhermes.utils.wavelet_grid import sample_scaling_function


def summary(values):
    return {name: float(fn(values)) for name, fn in
            [("median", np.median), ("min", np.min), ("max", np.max)]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-module", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    numba.set_num_threads(args.threads)
    spec = importlib.util.spec_from_file_location("gamma_opt_baseline", args.baseline_module)
    baseline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = baseline
    spec.loader.exec_module(baseline)
    phi = sample_scaling_function("db2", 10)
    records = []
    for length in args.lengths:
        rng = np.random.default_rng(432 + length)
        source = SimpleNamespace(epsilon=rng.normal(size=(length,) * 3), phi_array=phi,
                                 phi_resolution=1024, phi_support=3, L=length)
        first, second = rng.normal(size=(2, length, length, length)) + 0.3j * rng.normal(
            size=(2, length, length, length))
        contexts = {name: module._prepare_multipole_cpu_context(source)
                    for name, module in [("baseline", baseline), ("candidate", candidate)]}
        functions = {name: lambda module=module, context=contexts[name]:
                     module.compute_multipole_m_summand(first, second, context)[0]
                     for name, module in [("baseline", baseline), ("candidate", candidate)]}
        expected = functions["baseline"]()
        actual = functions["candidate"]()
        np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-11)
        samples = {"baseline": [], "candidate": []}
        for repeat in range(3):
            for name in (["baseline", "candidate"] if repeat % 2 == 0 else ["candidate", "baseline"]):
                start = time.perf_counter()
                value = functions[name]()
                samples[name].append(time.perf_counter() - start)
                np.testing.assert_allclose(value, expected, rtol=1e-8, atol=1e-11)
        records.append({"kind": "dense_complex_gamma_kernel", "length": length,
                        "normalized_absolute_difference": float(abs(actual - expected)),
                        "seconds": samples, "summary": {k: summary(v) for k, v in samples.items()},
                        "speedup": summary(samples["baseline"])["median"] /
                        summary(samples["candidate"])["median"]})

        field = SFCField(threads=args.threads)
        field.sfc_info = {"J": int(np.log2(length)), "L": length, "V": length**3,
                          "box_size": 1000.0, "scale_factor": length / 1000.0,
                          "phi_resolution": 1024, "wavelet_mode": "db2", "wavelet_level": 10,
                          "phi_array": phi, "phi_support": 3}
        field.format_sfc_params()
        field.epsilon = source.epsilon
        modes = [(ell, m) for ell in range(3) for m in range(ell + 1)]
        window = {"type": "shell", "len_args": {"R": 40.0}}
        modules = {"baseline": baseline, "candidate": candidate}
        def convolve(name):
            module = modules[name]
            context = module._prepare_legendre_convolution_context(field)
            return [module._stream_convolution_fields(field, window, ell, args.threads,
                                                     [m], conv_context=context)[0]
                    for ell, m in modes]
        previous = convolve("baseline")
        current = convolve("candidate")
        difference = max(float(np.max(abs(a - b))) for a, b in zip(current, previous))
        for a, b in zip(current, previous):
            np.testing.assert_allclose(a, b, rtol=1e-8, atol=1e-11)
        del current, previous
        samples = {"baseline": [], "candidate": []}
        for repeat in range(3):
            for name in (["baseline", "candidate"] if repeat % 2 == 0 else ["candidate", "baseline"]):
                start = time.perf_counter()
                fields = convolve(name)
                samples[name].append(time.perf_counter() - start)
                del fields
        records.append({"kind": "six_mode_convolution_with_source_fft_setup", "length": length,
                        "max_absolute_difference": difference, "seconds": samples,
                        "summary": {k: summary(v) for k, v in samples.items()},
                        "speedup": summary(samples["baseline"])["median"] /
                        summary(samples["candidate"])["median"]})
    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    result = {"status": "verified", "scope": "Synthetic dense fields, warmed Numba; not halo end-to-end speedup.",
              "threads": args.threads, "baseline_source_sha256": sha(args.baseline_module),
              "candidate_source_sha256": sha(candidate.__file__), "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
