import logging
import os
import optuna
from multiprocessing import current_process

def configure_logging():
    logging.basicConfig(format='%(asctime)s %(name)-16s %(levelname)-8s %(message)s',
                        level=os.getenv("LOGLEVEL","INFO").upper(),
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=f"logfile_{current_process().name}")
    optuna.logging.enable_propagation()  # Propagate logs to the root logger.
    optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.