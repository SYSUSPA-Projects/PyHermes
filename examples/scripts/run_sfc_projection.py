"""
Example: run SFCProjection with PyHermes

This script reads a particle catalog and computes the multiresolution
coefficient field used by later PyHermes analyses.
"""

import sys

from pyhermes.base.sfc_projection import SFCProjection
from pyhermes.param.parambase import read_param

sfc_projection_config = "./configs/param_sfc_projection.yaml"
if len(sys.argv) > 1:
    sfc_projection_config = sys.argv[1]

sfc_params = read_param(config_path=sfc_projection_config)
sfc_projection = SFCProjection(param_task=sfc_params)
sfc_projection.run(overwrite=True)
