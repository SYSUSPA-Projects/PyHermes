import logging
import os
import sys

from termcolor import colored


_RANK_ENV_KEYS = ("OMPI_COMM_WORLD_RANK", "PMI_RANK", "PMIX_RANK", "SLURM_PROCID")
_SIZE_ENV_KEYS = ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMIX_SIZE", "SLURM_NTASKS")
_VALID_MPI_LOG_MODES = {"root", "all"}


def _first_int_environment(keys, default):
    for key in keys:
        value = os.environ.get(key)
        if value not in (None, ""):
            try:
                return int(value)
            except ValueError:
                continue
    return int(default)


_mpi_rank = _first_int_environment(_RANK_ENV_KEYS, 0)
_mpi_size = max(1, _first_int_environment(_SIZE_ENV_KEYS, 1))
_mpi_log_mode = os.environ.get("PYHERMES_MPI_LOG_MODE", "root").strip().lower()
if _mpi_log_mode not in _VALID_MPI_LOG_MODES:
    _mpi_log_mode = "root"


def configure_mpi_logging(rank=None, size=None, mode=None):
    """Configure process-wide MPI log filtering for PyHermes loggers."""
    global _mpi_rank, _mpi_size, _mpi_log_mode

    if rank is not None:
        _mpi_rank = int(rank)
    if size is not None:
        _mpi_size = max(1, int(size))
    if mode is None:
        mode = os.environ.get("PYHERMES_MPI_LOG_MODE", _mpi_log_mode)
    normalized_mode = str(mode).strip().lower()
    if normalized_mode not in _VALID_MPI_LOG_MODES:
        raise ValueError(
            "PYHERMES_MPI_LOG_MODE must be either 'root' or 'all', "
            f"got {mode!r}."
        )
    _mpi_log_mode = normalized_mode


class MPIRankFilter(logging.Filter):
    """Keep routine logs on rank 0 while preserving errors from every rank."""

    def filter(self, record):
        record.mpi_rank_prefix = ""
        if _mpi_size > 1 and (_mpi_log_mode == "all" or _mpi_rank != 0):
            record.mpi_rank_prefix = f"[rank {_mpi_rank}/{_mpi_size}] "
        if _mpi_size <= 1 or _mpi_log_mode == "all" or _mpi_rank == 0:
            return True
        return record.levelno >= logging.ERROR



def setup_logger(module_name, class_name=None, level=logging.INFO, stream_handler=True):
    class ColoredFormatter(logging.Formatter):
        COLOR_MAP = {
            logging.INFO: 'green',
            logging.WARNING: 'yellow',
            logging.ERROR: 'red',
            logging.DEBUG: 'blue',
        }
        def format(self, record):
            levelname = record.levelname
            if record.levelno in self.COLOR_MAP:
                record.levelname = colored(levelname, self.COLOR_MAP[record.levelno])
            try:
                return super().format(record)
            finally:
                record.levelname = levelname
    name = module_name
    if class_name:
        name += f':{class_name}'
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if stream_handler and not logger.handlers:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.addFilter(MPIRankFilter())
        formatter = ColoredFormatter(
            '%(asctime)s - %(levelname)s - %(name)s - %(mpi_rank_prefix)s%(message)s',
            datefmt='%H:%M:%S',
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
