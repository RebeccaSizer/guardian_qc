"""
logger.py: This script configures the logger function used to log
messages that describe Python functionality of scripts used by the
SEA software package.

The create_logger function is responsible for:
    - Providing real-time logs of application activity for Users
      and Developers.
    - Displaying log messages to the terminal output.
    - Committing log messages with a logging level of 'ERROR' or
      higher to a rotating file handler.
    - Supporting auditing, debugging, troubleshooting and root
      cause analysis.

Almost all of the code used in this script derived from the
Peter-J-Freeman/SeqKitSTP2025 github repository.
"""

import os
import logging
from logging.handlers import RotatingFileHandler

# Create a logger instance
def create_logger():
    """
    This function configures the logger module imported into every other script in this software package.
    It creates a parental logger to which a stream-handler and file-handler are assigned to.

    The stream-handler controls the error messages printed to the terminal output. The logger is set to DEBUG level
    which means that all messages are printed to screen. Messages are prefixed with the date, time, and message level.

    The file-handler controls the error messages logged in log files, stored in the 'logs' directory. The logger is set
    to ERROR level, which means that only messages that correspond with errors that directly inhibit this software
    packages functionality are logged. Messages are prefixed with the date, time, and message level. File-handler
    rotates through up to 5 files, storing a maximum of 500 KB of messages before overwriting the next log file, if it
    exists.
    """

    # Make the 'logs' directory to store log files.
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
    logs_dir = os.path.join(base_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # Create logger
    logger = logging.getLogger('SEA_logger')
    logger.setLevel(logging.DEBUG)  # Set the root logger level to DEBUG

    # Stream handler with DEBUG level
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_formatter = logging.Formatter("%(asctime)s [%(levelname)s]: %(message)s", datefmt="%d-%m-%Y %H:%M:%S")
    stream_handler.setFormatter(stream_formatter)

    # File handler with ERROR level and rotating file configuration
    file_handler = RotatingFileHandler(os.path.join(logs_dir, "SEA.log"),
                                       maxBytes=500000,  # 500 KB
                                       backupCount=5)   # 5 files are on rotation
    file_handler.setLevel(logging.ERROR)
    file_formatter = logging.Formatter("'%(asctime)s - %(name)s - [%(levelname)s]: %(message)s'", datefmt="%d-%m-%Y %H:%M:%S")
    file_handler.setFormatter(file_formatter)

    # Add handlers to the logger
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger

# Instantiate the logger
logger = create_logger()