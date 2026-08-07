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
import config
from logging.handlers import RotatingFileHandler

from pathlib import Path

log_dir = Path("outputs/logs")
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        RotatingFileHandler(
            log_dir / config.LOG_FILE,
            maxBytes=10 * 1024 * 1024,  # Rotate at 10 MB
            backupCount=5               # Keep 5 old log files
        ),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)