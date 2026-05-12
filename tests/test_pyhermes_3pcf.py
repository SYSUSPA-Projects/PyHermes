from pyhermes.theory.corr3pcf import Corr_3PCF
from pyhermes.param.parambase import read_param



param_input = read_param(config_path='./param_3pcf.json')
# Do not specify argv 'config_path' if you want to specify
#  parameter file in command line (i.e., python ...py -c <config_path>)
# param_input = read_param()
Corr_3PCF(param_task=param_input).run()
