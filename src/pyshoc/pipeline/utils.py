
from recipes.pprint.nrs import TIME_DIVISORS, ymdhms


def human_time(age):
    mags = 'yMdhms'
    for m, d in zip(mags[::-1], TIME_DIVISORS[::-1]):
        if age < d:
            break

    i = mags.index(m) + 1
    if i < 5:
        return ymdhms(age, mags[i], f'{mags[i+1]}.1')

    return ymdhms(age, 's', 's1?')


def get_file_age(path, dne=''):
    if not path.exists():
        return dne

    now = time.time()
    info = path.stat()
    return now - max(info.st_mtime, info.st_ctime)
