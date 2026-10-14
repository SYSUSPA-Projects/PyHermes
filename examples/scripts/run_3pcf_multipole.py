"""Run Gamma or particle multipoles from a YAML configuration."""

import argparse
import os
from pathlib import Path


def main():
    examples = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", default=str(examples / "configs/param_3pcf_multipole_lmax14.yaml"))
    parser.add_argument("--summation-method", choices=["gamma", "particle"])
    parser.add_argument("--summation-backend", choices=["cpu", "gpu"])
    parser.add_argument("--execution-mode", choices=["serial", "pair_mpi", "sample_mpi"])
    parser.add_argument("--threads", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = Path(args.config).resolve()
    if "OMPI_COMM_WORLD_SIZE" not in os.environ:
        os.environ.setdefault("OMPI_MCA_btl", "self")
    from pyhermes.param.parambase import read_param
    from pyhermes.theory import Corr_3PCF_Multipole

    os.chdir(examples)
    params = read_param(config_path=str(config))
    for name in ["summation_method", "summation_backend", "execution_mode", "threads"]:
        value = getattr(args, name)
        if value is not None:
            params["Corr_3PCF_Multipole"][name] = value
    Corr_3PCF_Multipole(params).run(overwrite=args.overwrite)


if __name__ == "__main__":
    main()
