"""
Example: run 3PCF multipole with PyHermes

This script computes sampled 3PCF multipole moments from saved SFCField
inputs using the streamed CPU-convolution + CUDA-summation pipeline.
"""

import sys

from pyhermes.param.parambase import read_param
from pyhermes.theory.corr3pcf_multipole import Corr_3PCF_Multipole

corr3pcf_multipole_config = "./configs/param_3pcf_multipole.yaml"
if len(sys.argv) > 1:
    corr3pcf_multipole_config = sys.argv[1]

corr3pcf_multipole_params = read_param(config_path=corr3pcf_multipole_config)
corr3pcf_multipole = Corr_3PCF_Multipole(param_task=corr3pcf_multipole_params)
corr3pcf_multipole.run(overwrite=True)
