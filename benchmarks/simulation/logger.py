import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter('%(asctime)s [%(filename)s:%(levelname)s] %(message)s]')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)