"""
Console welcome banner for pipeline.
"""

# std
import random
import operator as op
from pathlib import Path
from datetime import datetime

# local
import motley
from recipes import string
from recipes.misc import get_terminal_size


# logo
LOGO = (Path(__file__).parent / 'logo.txt').read_text()

# ✹✵☆ ★☆✶
#  '🪐': 1, # NOTE: not monospace...
#  '🌘': 1} #📸
STARS = {' ': 2000,
         '.': 20,
         '`': 10,
         '+': 10,
         '*': 10,
         '✷': 2,
         '☆': 1, }

# ---------------------------------------------------------------------------- #


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


def over_starfield(text, width=None, stars=None):
    stars = stars or STARS

    
    if width is None:
        width = motley.get_width(text)

    return ''.join(_over_starfield(text, width, stars))


def make_banner(format, subtitle='', width=None, **style):
    from pyshoc import __version__

    width = int(width or get_terminal_size()[0])

    now = datetime.now()
    now = f'{now.strftime("%d/%m/%Y %H:%M:%S")}.{now.microsecond / 1e5:.0f}'

    logo = motley.justify(color_logo(fg=style.pop('fg', '')), '^', width)
    x = logo.rindex('\n')
    y = x - next(string.where(logo[(x - 1)::-1], op.ne, ' '))
    logo = ''.join((logo[:y], '🪐', logo[(y + 2):]))
    return motley.banner(
        over_starfield(motley.format(format, **locals(), version=__version__)),
        width, **style
    ).replace('🪐 ', '🪐')  # NOTE: not monospace...
