import logging
from logging.config import dictConfig
import sys
from pathlib import Path

from rich.logging import RichHandler


BASE_DIR = Path(__file__).parent.parent.absolute()  # C:\Users\reigHns\mnist

LOGS_DIR = Path(BASE_DIR, "logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR = Path(BASE_DIR, "plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Logger
logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "minimal": {"format": "%(message)s"},
        "detailed": {
            "format": "%(levelname)s %(asctime)s [%(filename)s:%(funcName)s:%(lineno)d]\n%(message)s\n"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "stream": sys.stdout,
            "formatter": "minimal",
            "level": logging.DEBUG,
        },
        "normal_scheduler": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": Path(LOGS_DIR, "normal_scheduler.log"),
            "maxBytes": 10485760,  # 1 MB
            "backupCount": 10,
            "formatter": "detailed",
            "level": logging.INFO,
        },
        "warmup_scheduler": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": Path(LOGS_DIR, "warmup_scheduler.log"),
            "maxBytes": 10485760,  # 1 MB
            "backupCount": 10,
            "formatter": "detailed",
            "level": logging.INFO,
        },
    },
    "loggers": {
        "normal_scheduler": {
            "handlers": ["console", "normal_scheduler"],
            "level": logging.INFO,
            "propagate": True,
        },
        "warmup_scheduler": {
            "handlers": ["console", "warmup_scheduler"],
            "level": logging.INFO,
            "propagate": True,
        },
    },
}
# call this function and set a global variable to set the logger to the configuration dictionary that you have.
dictConfig(logging_config)
# logger -> creates a logger class which corresponds to the settings in the logging_config[loggers[root]]
logger_normal_scheduler = logging.getLogger("normal_scheduler")
logger_warmup_scheduler = logging.getLogger("warmup_scheduler")
# set markup to true for console
logger_normal_scheduler.handlers[0] = RichHandler(markup=True)
logger_warmup_scheduler.handlers[0] = RichHandler(markup=True)
