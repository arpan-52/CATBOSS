#!/usr/bin/env python3
"""
Colorful logging utility for CATBOSS.

Author: Arpan Pal
Institution: NCRA-TIFR
"""

import logging
import sys


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels."""

    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",
        "BOLD": "\033[1m",
    }

    def format(self, record):
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[levelname]}{self.COLORS['BOLD']}"
                f"{levelname}{self.COLORS['RESET']}"
            )
        return super().format(record)


def setup_logger(name="catboss", level=logging.INFO, verbose=False):
    """
    Setup a colorful logger for CATBOSS.

    Args:
        name: Logger name
        level: Logging level (default: INFO)
        verbose: If True, set level to DEBUG

    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger(name)

    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(level)

    logger.handlers = []

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)

    formatter = ColoredFormatter(
        fmt="%(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.propagate = False

    return logger


def print_banner():
    """Print the CATBOSS ASCII art banner."""
    banner = """
\033[35m\033[1m
                                                     900002       0805   80000000000 00000008        20002      300008     100009
                                                  70000000000    10000   00000088800 0080008800   0708008000  7008008808  008088008
                                                 1800      6     880808      087     008    008  888 7  7 888 000        800
                                                 000           7088  000     8 8     89 000080  000  881  7005 0000009    8000009
                                                 000           8 0   2009     80     9 89998000 000  4080 7005   1800000    1000000
                                                  008     20  26208000008    00       00     008 008 000 7000  9     8007 9     8003
                                                   8000080003 8 0      000   078     00000088001  0000000088  0000000008 0088008000
                                                      9005   669       7666  666     66666965        9006        20081      50003
\033[0m

\033[33m\033[1m                                           CATBOSS - Radio Astronomy RFI Flagging Suite\033[0m
\033[36m                                      Developed by Arpan Pal, NCRA-TIFR | Modern, Fast, Sleek\033[0m
"""
    print(banner)


def print_cat_on_hunt(cat_name):
    """Print the cat story."""
    message = """
\033[36m  Pooh and Nimki are my first two cats. They're excellent hunters and mischievous
  troublemakers, each with a very different personality. I named this package
  after them. Happy RFI hunting!\033[0m
"""
    print(message)


def print_summary_box(title, items):
    """Print a formatted summary."""
    print(f"\n\033[1m  {title}\033[0m")
    for key, value in items:
        print(f"  {key}: {value}")
    print()
