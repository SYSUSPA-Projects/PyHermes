from pyhermes.base.convols import Convols
from pyhermes.param.parambase import read_param



param_input = read_param(config_path='./param_convols.json')
# Do not specify argv 'config_path' if you want to specify
#  parameter file in command line (i.e., python ...py -c <config_path>)
# param_input = read_param()
Convols(param_task=param_input).run()
