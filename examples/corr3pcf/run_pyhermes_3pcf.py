from pyhermes.theory.corr3pcf import Corr_3PCF
from pyhermes.param.parambase import read_param



param_input = read_param()
Corr_3PCF(param_task=param_input).run()
