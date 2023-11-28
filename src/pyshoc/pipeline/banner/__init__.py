"""
Console welcome banner for pipeline.
"""

# std
import random
import operator as op
import itertools as itt
from pathlib import Path
from datetime import datetime

# third-party
import more_itertools as mit

# local
import motley
from recipes import string
from recipes.shell import terminal


# ---------------------------------------------------------------------------- #
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
         '☆': 1}

# ---------------------------------------------------------------------------- #


def _color_text(text, **style):
    for line in text.splitlines():
        new = ''
        for space, text in string.partition_whitespace(line):
            if text:
                text = motley.apply(text, **style)
            new += space + text

        yield new


def color_logo(**style):
    return '\n'.join(_color_text(LOGO, **style))


def _background_stars(text, stars, buffer=2, threshold=10):
    ws = ' ' * buffer
    population, weights = zip(*stars.items())
    for part in motley.codes.parse(text, True):
        if part.csi:
            yield str(part)
            continue

        for space, text in string.partition_whitespace(part.text, threshold):
            if space:
                nl = (text == '\n')
                yield ws
                yield from random.choices(population, weights,
                                          k=len(space) - (1 + nl) * buffer)
                yield ws * nl
            yield text


def background_stars(text, stars=None):
    return ''.join(_background_stars(text, stars or STARS))


def make_banner(format, subtitle='', width=None, **style):
    from pyshoc import __version__
    from sys import version_info

    width = int(width or terminal.get_size()[0])
    halfwidth = width // 2

    now = datetime.now()
    now = f'{now.strftime("%d/%m/%Y %H:%M:%S")}.{now.microsecond / 1e5:.0f}'
    python = 'Python {}.{}.{}'.format(*version_info)

    logo = motley.justify(color_logo(fg=style.pop('fg', '')), '^', width)
    banner = motley.format(format, **locals(), version=__version__)
    banner = motley.banner(background_stars(banner), width, **style)
    return banner.replace('🪐 ', '🪐')  # NOTE: not monospace...
