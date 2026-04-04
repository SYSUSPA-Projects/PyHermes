__all__ = ["TaskBase"]


def __getattr__(name):
    if name == "TaskBase":
        from .pipeline import TaskBase
        return TaskBase
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
