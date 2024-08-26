from pyhermes.theory.corr2pcf import Corr_2PCF
from pyhermes.param.parambase import read_param



param_input = read_param()
Corr_2PCF(param_task=param_input).run()
