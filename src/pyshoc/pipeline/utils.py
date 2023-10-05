
# std
from pathlib import Path
import sys
import time
import numbers

# third-party
from matplotlib.figure import Figure

# local
from recipes.pprint.nrs import TIME_DIVISORS, ymdhms

# relative
from .logging import logger


def get_figure(ui=None, *keys, **kws):
    if ui:
        tab = ui.add_tab(*keys, fig=kws)
        return tab.figure

    if plt := sys.modules.get('matplotlib.pyplot'):
        return plt.figure(**kws)

    return Figure(**kws)


def save_figure(fig, filename, overwrite=False):
    if filename:
        filename = Path(filename)
        if not filename.exists() or overwrite:
            logger.info('Saving image: {}', filename)
            fig.savefig(filename)


# alias
save_fig = save_figure


def human_time(age):

    fill = (' ', ' ', ' ', 0, 0, 0)

    if not isinstance(age, numbers.Real):
        # print(type(age))
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
