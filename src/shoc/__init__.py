"""
pyshoc - Data analysis tools for the Sutherland High-Speed Optical Cameras.
"""


# std
import random
import operator as op
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version

# third-party
import yaml
from astropy.io.fits.hdu.base import register_hdu

# local
import motley
from recipes import string
from recipes.misc import get_terminal_size

# relative
from .core import *
from .caldb import CalDB


try:
    __version__ = version('pyshoc')
except PackageNotFoundError:
    __version__ = '?.?.?'

# settings
SRC_ROOT = Path(__file__).parent
CONFIG = yaml.load((SRC_ROOT / 'config.yaml').read_text(),
                   Loader=yaml.FullLoader)


# Banner
LOGO = (SRC_ROOT / 'banner/logo.txt').read_text()


def _partition_indices(text):
    for line in text.splitlines():
        i0 = next(string.where(line[::+1], op.ne, ' '), 0)
        i1 = len(line) - next(string.where(line[::-1], op.ne, ' '), -1)
        yield line, i0, i1


def _partition(text):
    for line, i0, i1 in _partition_indices(text):
        yield line[:i0], line[i0:i1], line[i1:]


def color_logo(**style):
    return '\n'.join(head + motley.apply(mid, **style) + tail
                     for head, mid, tail in _partition(LOGO))


def _over_starfield(text, width, stars, frq=0.5, buffer=2):
    assert frq <= 1
    buffer = int(buffer)
    # stars = stars.rjust(int(width / frq))

    for line, i, j in _partition_indices(text):
        if i > buffer:
            i -= buffer

        i1 = max(len(line), width) - j
        if i1 > buffer:
            i1 -= buffer
            j += buffer

        yield ''.join((*random.choices(*zip(*stars.items()), k=i),
                       line[i:j],
                       *random.choices(*zip(*stars.items()), k=i1),
                       '\n'))


def over_starfield(text, width=None):
    # ✹✵☆ ★☆✶
    stars = {' ': 2000,
             '.': 20,
             '`': 10,
             '+': 10,
             '*': 10,
             '✷': 2,
             '☆': 1,}
            #  '🪐': 1, # NOTE: not monospace...
            #  '🌘': 1} 

    if width is None:
        width = motley.get_width(text)

    return ''.join(_over_starfield(text, width, stars))


def make_banner(subtitle='', width=None, **style):
    width = int(width or get_terminal_size()[0])

    now = datetime.now()
    now = f'{now.strftime("%d/%m/%Y %H:%M:%S")}.{now.microsecond / 1e5:.0f}'

    logo = motley.justify(color_logo(fg=style.pop('fg', '')), '^', width)
    x = logo.rindex('\n')
    y = x - next(string.where(logo[x - 1::-1], op.ne, ' '))
    logo = ''.join((logo[:y], '🪐', logo[y+2:]))
    return motley.banner(
        over_starfield(
            motley.format('{{now:|B,darkgreen}: <{width:}}\n'
                          '{logo}\n'
                          '{{subtitle:|B,purple}: >{width}}\n'
                          '{{version:|Bk}: >{width}}\n',
                          **locals(), version=__version__),
        ),
        width, **style
    ).replace('🪐 ', '🪐') # NOTE: not monospace...

# register HDU classes (order is important!)
register_hdu(shocHDU)


# initialize calibration database
calDB = CalDB(CONFIG['caldb'])
