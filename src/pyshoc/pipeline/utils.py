
# std
import time
import numbers
from pathlib import Path

# local
from recipes.pprint.nrs import TIME_DIVISORS, ymdhms


# ---------------------------------------------------------------------------- #
def human_time(age):

    fill = (' ', ' ', ' ', 0, 0, 0)

    if not isinstance(age, numbers.Real):
        # print(type(age))
        return '--'

    off = 0
    mags = 'yMdhms'
    for m, d in zip(mags[::-1], TIME_DIVISORS[::-1]):
        if age < d:
            off = 1
            break

    # n = 5
    i = mags.index(m) + off
    base = mags[min(i, 5)]
    fmt = mags[i + 2] if (i < 4) else 's1?'
    return ymdhms(age, base, fmt, fill=fill)


def get_file_age(path, dne='--', human=False):

    path = Path(path)
    if not path.exists():
        return dne

    now = time.time()
    info = path.stat()
    age = now - min(info.st_mtime, info.st_ctime)
    
    if human:
        return human_time(age)
    
    return age
