from pyhermes.theory.counting import Counting
from pyhermes.param.parambase import read_param



param_input = read_param(config_path='./param_counting.json')
# Do not specify argv 'config_path' if you want to specify
#  parameter file in command line (i.e., python ...py -c <config_path>)
Counting(param_task=param_input).run()
