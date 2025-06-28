import os
import sys
import logging

from datetime import datetime
from utils.file import init_folder


# Initial the basic logger
def init_basic_logger(verbose: bool = False) -> None:
    FORMAT = '%(asctime)s %(filename)s %(levelname)s: %(message)s'
    logging.basicConfig(
        stream=sys.stdout, level=logging.DEBUG if verbose else logging.INFO, format=FORMAT)


# MakeFileHandler - Create all directories first and then let FileHandler open the files.
class MakeFileHandler(logging.FileHandler):
    """http://stackoverflow.com/a/600612/190597 (tzot)"""

    def __init__(self, filename, mode='a', encoding=None, delay=0):
        init_folder(os.path.dirname(filename), del_if_exist=False)
        logging.FileHandler.__init__(self, filename, mode, encoding, delay)


# Initial the dual output logger, stdout and output to a log file
def init_dual_output_logger(filename: str, verbose: bool = False) -> None:
    # Specify logging format
    FORMAT = '%(asctime)s %(filename)s %(levelname)s: %(message)s'
    formatter = logging.Formatter(FORMAT)

    # Specify the logging file name and create a MakeFileHandler to handle file writing
    log_filename = f'logs/{filename}_{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.log'
    file_handler = MakeFileHandler(
        filename=log_filename, encoding='utf-8')

    # Set the format and logging level of MakeFileHandler
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG if verbose else logging.INFO)

    # Create standard output stream handler
    stream_handler = logging.StreamHandler(sys.stdout)

    # Set the format and logging level of standard output stream handler
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.DEBUG if verbose else logging.INFO)

    # Obtain root logger and disable the propagation
    root = logging.getLogger()
    root.propagate = False

    # Set logging level of root logger
    root.setLevel(logging.DEBUG if verbose else logging.INFO)

    # Attach MakeFileHandler and StreamHandler to root logger
    root.addHandler(file_handler)
    root.addHandler(stream_handler)
