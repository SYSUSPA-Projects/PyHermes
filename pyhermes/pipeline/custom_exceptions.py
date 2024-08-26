# --------------------------------------------------------
# Exceptions inherited from kiopy

"""Custom Exceptions.
These mostly don't do anything special, but are defined such that the
exceptions I raise don't conflict with generic exceptions raised by built-ins.
"""
class DataError(Exception) :
    """Exception to raise if data of some sort is invalid or does not have
    expected properties."""
    pass

class NextIteration(Exception) :
    """Exception raised to skip iterations in a nested loop in a controled way.
    """
    pass

class FileParameterTypeError(TypeError) :
    """Exception to raise if a parameter read from file should be a certain
    type and is not."""
    pass

class ParameterFileError(Exception) :
    """Exception to raise if reading a parameter file fails."""
    pass



# --------------------------------------------------------
# Exceptions migrated from pipeline(tlpipeline)

class PipelineConfigError(Exception):
    """Raised when there is an error setting up a pipeline."""
    pass

class PipelineRuntimeError(Exception):
    """Raised when there is a pipeline related error at runtime."""
    pass

class PipelineStopIteration(Exception):
    """This stops the iteration of `next()` in pipeline tasks.
    Pipeline tasks should raise this exception in the `next()` method to stop
    the iteration of the task and to proceed to `finish()`.
    Note that if `next()` receives input data as an argument, it is not
    required to ever raise this exception.  The pipeline will proceed to
    `finish()` once the input data has run out.
    """
    pass

class PipelineFinished(Exception):
    """Raised by tasks that have been completed."""
    pass

class PipelineMissingData(Exception):
    """Used for flow control when input data is yet to be produced."""
    pass

# class _PipelineFinished(Exception):
#     """Raised by tasks that have been completed."""
#     pass

# class PipelineFinished(Exception):
#     """Public exception raised when a task is completed."""
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         raise _PipelineFinished(*args, **kwargs)

# class _PipelineMissingData(Exception):
#     """Used for flow control when input data is yet to be produced."""
#     pass