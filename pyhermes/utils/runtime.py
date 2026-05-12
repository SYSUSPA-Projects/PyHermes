"""Runtime configuration helpers for PyHermes."""

_NUMBA_CONFIGURED = False
_NUMBA_THREADS = None


def configure(threads=1):
    """
    Configure Numba threads for this process.

    Re-applying with the same value is a no-op; changing the value updates
    the current runtime setting.
    """
    global _NUMBA_CONFIGURED, _NUMBA_THREADS
    requested_threads = max(1, int(threads))
    if _NUMBA_CONFIGURED and _NUMBA_THREADS == requested_threads:
        return
    from numba import get_num_threads, set_num_threads
    set_num_threads(requested_threads)
    _NUMBA_THREADS = int(get_num_threads())
    _NUMBA_CONFIGURED = True
