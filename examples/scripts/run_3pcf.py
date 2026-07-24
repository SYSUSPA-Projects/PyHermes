"""
Example: run 3PCF with PyHermes

This script computes the 3-point correlation function from a saved
SFCField product using the standard PyHermes 3PCF pipeline.
"""

import sys

from pyhermes.param.parambase import read_param
from pyhermes.theory.corr3pcf import Corr_3PCF

corr3pcf_config = "./configs/param_3pcf_pcenter_nrot20.yaml"
if len(sys.argv) > 1:
    corr3pcf_config = sys.argv[1]

corr3pcf_params = read_param(config_path=corr3pcf_config)
corr3pcf = Corr_3PCF(param_task=corr3pcf_params)
corr3pcf.run(overwrite=True)
