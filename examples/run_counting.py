"""
Example: run counting with PyHermes

This script evaluates the field on many random positions and writes the
resulting counting data product.
"""

import sys

from pyhermes.theory.counting import Counting
from pyhermes.param.parambase import read_param

counting_config = "./configs/param_counting.yaml"
if len(sys.argv) > 1:
    counting_config = sys.argv[1]

counting_params = read_param(config_path=counting_config)
counting = Counting(param_task=counting_params)
counting.run(overwrite=True)
