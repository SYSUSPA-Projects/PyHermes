from pyhermes.theory.counting import Counting
from pyhermes.param.parambase import read_param



param_input = read_param()
Counting(param_task=param_input).run()
