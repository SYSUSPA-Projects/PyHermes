import os
import sys
import inspect
import argparse
import importlib
import copy

import yaml
import json5

import pyhermes
from pyhermes.utils import func_util
from pyhermes.utils.mpi_util import MPI
from pyhermes.param.logbase import setup_logger 
import pyhermes.pipeline.custom_exceptions as ce


REPLACE_KEYS = {
    "Corr_2PCF.binning_window",
    "Corr_2PCF.sampling",
    "Corr_3PCF_Multipole.binning_window12",
    "Corr_3PCF_Multipole.binning_window13",
    "Corr_3PCF_Multipole.sampling",
    "Corr_3PCF_Multipole.sample_mpi",
}


def print_flush(msg):
    print(msg)
    sys.stdout.flush()

# Common user interface
def read_param(config_path=None):
    '''
    read_param: Common user interface to handle parameters.
    arguments:
        config_path = <YOUR/CONFIG/PATH> , if you set jupyer = True, you need to specify the config file path.
    '''
    if not config_path:
        parser = ParamBase.get_parser()
        args = parser.parse_args()
        config_path = args.config
    param_base = ParamBase(config_file_path=config_path)
    param_user = param_base.read_config()
    return param_user


class ParamBase(object):
    '''
    The class to read parameters in yaml or json(5) format. 
    Include: 
        read default parameters (read the json file at <module>/default_params.json)
        read user parameters (with specified path)
        update default parameters (update default value by user parameters)
    '''
    
    def __init__(self, config_file_path=None):
        self.default_params = {}
        self.logger = setup_logger(__name__, self.__class__.__name__)
        self.config_file_path = config_file_path
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()

    @classmethod
    def get_parser(cls):
        _version = pyhermes.__version__
        _desc = f"Welcome to PyHermes V{_version} \nCheck the document for more details: https://pyhermes.astroslacker.com\nFeel free to ask if you have any questions.\nContact: \n  dingdluan@gmail.com\n  juwj@mail2.sysu.edu.cn"
        parser = argparse.ArgumentParser(
            description=_desc,
            formatter_class=argparse.RawTextHelpFormatter
            )
        parser.add_argument("-c", 
                            "--config", 
                            type=str, 
                            default="",
                            help="path for the parameter file",
                            metavar=""
                            )
        return parser
    
    def _detect_paramfile_type(self, file_path):
        try:
            with open(file_path, 'r') as file:
                content = file.read()
                try:
                    json5.loads(content)
                    self.logger.info('Input parameter file format is JSON(5)')
                    return 'json'
                # except json5.JSONDecodeError:
                except ValueError:
                    pass  # not json
                try:
                    yaml.safe_load(content)
                    self.logger.info('Input parameter file format is YAML')
                    return 'yaml'
                except yaml.YAMLError:
                    pass  # not YAML
                raise ValueError("Unsupported file type or invalid parameter file format")
        except FileNotFoundError:
            self.logger.error(f"Parameter file not found: '{file_path}'. This should not have happened, pipeline stopped!")
            func_util.safe_exit(1)
        except Exception as e:
            self.logger.error(f"Cannot determine file type: {e}")
            self.logger.error("Support parameter file formats: <JSON> and <YAML>")
            self.logger.error(f"Please see the document for details")
            func_util.safe_exit(1)
    
    def read_paramfile(self, file_path):
        file_type = self._detect_paramfile_type(file_path)
        with open(file_path, 'r') as file:
            content = file.read()
            if file_type == 'json':
                param_dict = json5.loads(content)
            elif file_type == 'yaml':
                param_dict = yaml.safe_load(content)
        return param_dict

    def _recursive_update(self, default_dict, new_dict, parent_key='', section=None):
        # If a specific section is provided, update only that section
        if section:
            # Check if the section exists in both, one, or none of the dictionaries
            section_in_default = section in default_dict
            section_in_new = section in new_dict
            if section_in_default and section_in_new:
                # If section is in both dictionaries, update it recursively
                self._recursive_update(default_dict[section], new_dict[section], parent_key=section)
                return
            elif section_in_default:
                self.logger.error(f"Section <{section}> found only in default parameters, but not user-input.")
                return
            elif section_in_new:
                default_dict[section] = new_dict[section]
                self.logger.error(f"Section <{section}> found only in user-input parameters, but not default.")
                return
            else:
                # If section is in neither dictionary, log error and exit
                self.logger.error(f"Section <{section}> found in neither dictionary. This should not have happened, pipeline stopped!")
                func_util.safe_exit(1)
        for key, value in new_dict.items():
            full_key = f'{parent_key}.{key}' if parent_key else key
            if key not in default_dict:
                if isinstance(value, dict):
                    # Add new level
                    if full_key.startswith("Corr_2PCF.sampling."):
                        self.logger.info(f"Adding Corr_2PCF sampling coordinate: '{full_key}'")
                    else:
                        self.logger.warning(f"Adding non-default level: '{full_key}'")
                    default_dict[key] = {}  # Init new level
                    self._recursive_update(default_dict[key], value, full_key)
                else:
                    # Add new key
                    if parent_key == "Corr_2PCF.sampling" or parent_key.startswith("Corr_2PCF.sampling."):
                        self.logger.info(f"Adding Corr_2PCF sampling value: '{full_key}' as '{value}'")
                    elif parent_key != 'SFCProjection.window':
                        # Skip warning for window_args
                        self.logger.warning(f"Adding non-default key: '{full_key}'")
                    else:
                        # ↓ Use special info instead of warning ↑
                        self.logger.info(f"Adding customizable window arg: '{full_key}' as '{value}'")
                    default_dict[key] = value
            elif full_key in REPLACE_KEYS:
                self.logger.info(f"Using user-provided replacement value for '{full_key}'.")
                default_dict[key] = copy.deepcopy(value)
            elif isinstance(value, dict) and isinstance(default_dict[key], dict):
                # Recursively to due the whole dict structure
                self._recursive_update(default_dict[key], value, full_key)
            else:
                if not self._is_type_compatible(full_key, default_dict[key], value):
                    self.logger.warning(f"Type mismatch for key: '{full_key}' !!!")
                else:
                    if default_dict[key] != value:
                        old_value = 'empty' if default_dict[key] == '' or default_dict[key] == [] else default_dict[key]
                        self.logger.info(f"Default '{full_key}' from '{old_value}' to '{value}'")
                default_dict[key] = value

    def _is_type_compatible(self, full_key, default_value, new_value):
        if isinstance(new_value, type(default_value)):
            return True
        if isinstance(default_value, (int, float)) and isinstance(new_value, (int, float)):
            return True
        # Some fields intentionally support either a single string or a list of strings.
        if full_key.endswith(".products"):
            default_ok = isinstance(default_value, (str, list, tuple))
            new_ok = isinstance(new_value, (str, list, tuple))
            return default_ok and new_ok
        # Random inputs intentionally allow a missing default (null/None) to be
        # overridden by a runtime string such as "uniform" or a data path.
        if full_key.endswith(".random"):
            return default_value is None and isinstance(new_value, str)
        # Angle sampling specs accept either dict configs or explicit arrays/lists.
        if full_key.endswith(".theta") or full_key.endswith(".mu") or full_key.endswith(".s"):
            default_ok = isinstance(default_value, (dict, list, tuple))
            new_ok = isinstance(new_value, (dict, list, tuple))
            return default_ok and new_ok
        if full_key.endswith(".binning_window"):
            default_ok = isinstance(default_value, (dict, str))
            new_ok = isinstance(new_value, (dict, str))
            return default_ok and new_ok
        if full_key.endswith(".len_args"):
            default_ok = isinstance(default_value, (dict, list, tuple, str))
            new_ok = isinstance(new_value, (dict, list, tuple, str))
            return default_ok and new_ok
        if full_key.endswith(".los_args"):
            default_ok = isinstance(default_value, (dict, list, tuple))
            new_ok = isinstance(new_value, (dict, list, tuple))
            return default_ok and new_ok
        return False
    
    def recursive_update(self, default_dict, new_dict, parent_key='', section=None):
        return self._recursive_update(default_dict, new_dict, parent_key=parent_key, section=section)

    def _read_config_jsonPre(self, config_fname):
        try:
            with open(config_fname) as f:
                config = json5.load(f)
        except FileNotFoundError:
            self.logger.error(f"Parameter file not found: '{config_fname}'. This should not have happened, pipeline stopped!")
            func_util.safe_exit(1)
        except Exception as e:
            self.logger.error(f"Reading configure file error: {e}. This should not have happened, pipeline stopped!")
            func_util.safe_exit(1)
        return config
    
    def _get_dir_from_path(self, fpath):
        dir_path = os.path.dirname(fpath)
        dir_name = os.path.basename(dir_path)
        return dir_path, dir_name

    def _find_class_dir(self, class_task):
        # Get module name of the task
        module_name = class_task.__module__
        # Dynamic import module using importlib
        module = importlib.import_module(module_name)
        # Get module_path using inspect
        module_path = inspect.getfile(module)
        # Get module_dir
        module_dir, _ = self._get_dir_from_path(fpath=module_path)
        return module_dir

    def _read_default(self, default_fname):
        _ , dir_name = self._get_dir_from_path(fpath=default_fname)
        self.logger.info(f"Set default parameters of module <{dir_name}> ...")
        _default_params = self._read_config_jsonPre(config_fname=default_fname)
        # judge whether key 'enable_default_param' exist
        if "enable_default_param" not in _default_params:
            self.logger.warning(f"No 'enable_default_param' found in {default_fname}, skipping parameter loading for module <{dir_name}>!")
            return
        # judge whether the module should have parameter
        if _default_params.get("enable_default_param") is not True:
            self.logger.warning(f"Key 'enable_default_param' not set to 'True' in {default_fname}, skipping parameter loading for module <{dir_name}>!")
            return
        # to check whether the key is already existed
        for key in _default_params.keys():
            if key == "enable_default_param":
                continue
            # if key in ParamBase.default_params:
            if key in self.default_params:
                self.logger.error(f"Key '{key}' already exists in default parameters, did you add it again? This should not have happened, pipeline stopped!")
                func_util.safe_exit(1)
        # ParamBase.default_params.update(_default_params)
        self.default_params.update(_default_params)
    
    def read_default(self, class_task):
        module_dir=self._find_class_dir(class_task)
        default_param_path=os.path.join(module_dir, "default_params.json")
        self._read_default(default_fname=default_param_path)

    def read_config(self):
        if self.rank == 0:
            if self.config_file_path:
                self.logger.info(f"Reading configure file: '{self.config_file_path}'")
                # user_params = self._read_config_jsonPre(self.config_file_path)
                user_params = self.read_paramfile(self.config_file_path)
                return user_params
            else:
                self.logger.error(f"No configure file specified in pipeline")
                parser = self.get_parser()
                print("")
                print("----------------------------------------------------------------------")
                parser.print_help()
                print("----------------------------------------------------------------------")
                print("")
                self.logger.error(f"Please set configure file path with '-c' then try again")
                func_util.safe_exit(1)
