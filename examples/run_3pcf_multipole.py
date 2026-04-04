"""
Example: run 3PCF multipole with PyHermes

This script computes the multipole moments zeta_l(r1, r2) from a saved
ConvolsData product using the streamed CPU-convolution + CUDA-summation
pipeline.
"""

import sys

from pyhermes.theory.corr3pcf_multipole import Corr_3PCF_Multipole
from pyhermes.param.parambase import read_param


corr3pcf_multipole_config = "./configs/param_3pcf_multipole.yaml"
if len(sys.argv) > 1:
    corr3pcf_multipole_config = sys.argv[1]

corr3pcf_multipole_params = read_param(config_path=corr3pcf_multipole_config)
corr3pcf_multipole = Corr_3PCF_Multipole(param_task=corr3pcf_multipole_params)
corr3pcf_multipole.run(overwrite=True)
