from pyhermes.base.convols import Convols
from pyhermes.param.parambase import read_param



param_input = read_param()
Convols(param_task=param_input).run()
