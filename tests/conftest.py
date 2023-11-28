
# std
import sys

# third-party
from loguru import logger


logger.add(sys.stderr, level='DEBUG')
logger.enable('pyshoc')
logger.enable('recipes')
