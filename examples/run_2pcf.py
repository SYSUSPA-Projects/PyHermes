"""
Example: run 2PCF with PyHermes

This script computes the 2-point correlation function from a saved
ConvolsData product using the standard PyHermes 2PCF pipeline.
"""

import sys

from pyhermes.param.parambase import read_param
from pyhermes.theory.corr2pcf import Corr_2PCF

corr2pcf_config = "./configs/param_2pcf.yaml"
if len(sys.argv) > 1:
    corr2pcf_config = sys.argv[1]

corr2pcf_params = read_param(config_path=corr2pcf_config)
corr2pcf = Corr_2PCF(param_task=corr2pcf_params)
corr2pcf.run(overwrite=True)
