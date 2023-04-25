
from shoc.pipeline.logging import config

logger = config()

for _ in range(10):
    logger.info('Ping!')

for _ in range(3):
    logger.info('Hi!')
    logger.warning('Yar!')
    logger.critical('BOOM!')
    
logger.success('final')