"""
Example: run Convols with PyHermes

This script reads a particle catalog and computes the multiresolution
coefficient field used by later PyHermes analyses.
"""

import sys

from pyhermes.base.convols import Convols
from pyhermes.param.parambase import read_param

convols_config = "./configs/param_convols.yaml"
if len(sys.argv) > 1:
    convols_config = sys.argv[1]

convols_params = read_param(config_path=convols_config)
convols = Convols(param_task=convols_params)
convols.run(overwrite=True)
