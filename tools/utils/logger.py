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

"""

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)