
# std
import time
import numbers

# local
from recipes.pprint.nrs import TIME_DIVISORS, ymdhms

# relative
from .logging import logger


def save_fig(fig, filename):
    if filename:
        logger.info('Saving image: {}', filename)
        fig.savefig(filename)


def human_time(age):

    fill = (' ', ' ', ' ', 0, 0, 0)

    if not isinstance(age, numbers.Real):
        return '--'

    mags = 'yMdhms'
    for m, d in zip(mags[::-1], TIME_DIVISORS[::-1]):
        if age < d:
            break

    i = mags.index(m) + 1
    if i < 5:
        return ymdhms(age, mags[i], f'{mags[i+1]}.1', fill=fill)

    return ymdhms(age, 's', 's1?', fill=fill)


def get_file_age(path, dne=''):
    if not path.exists():
        return dne

    now = time.time()
    info = path.stat()
    return now - max(info.st_mtime, info.st_ctime)
