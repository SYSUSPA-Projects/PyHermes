import sys
import logging

from termcolor import colored



def setup_logger(module_name, class_name=None, level=logging.INFO, stream_handler=True):
    class ColoredFormatter(logging.Formatter):
        COLOR_MAP = {
            logging.INFO: 'green',
            logging.WARNING: 'yellow',
            logging.ERROR: 'red',
            logging.DEBUG: 'blue',
        }
        def format(self, record):
            levelname = record.levelname
            if record.levelno in self.COLOR_MAP:
                levelname_colored = colored(levelname, self.COLOR_MAP[record.levelno])
                record.levelname = levelname_colored
            return super().format(record)
    name = module_name
    if class_name:
        name += f':{class_name}'
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if stream_handler and not logger.handlers:
        handler = logging.StreamHandler(stream=sys.stdout)
        formatter = ColoredFormatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%H:%M:%S')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger