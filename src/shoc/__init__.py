"""
pyshoc - Data analysis tools for the Sutherland High-Speed Optical Cameras
"""


# std libs
from importlib.metadata import version, PackageNotFoundError

# third-party libs
from astropy.io.fits.hdu.base import register_hdu

# relative libs
from .core import *
from .colours import COLOURS
from .caldb import CalDB


try:
    __version__ = version('pyshoc')
except PackageNotFoundError:
    __version__ = '?.?.?'


# Banner
LOGO = (Path(__file__).parent / 'banner/banner.txt').read_text()


def make_banner(subtitle='', width=80):
    from motley.box import TextBox, clear_box, underline

    text = f'{subtitle}\nv{__version__}'
    parts = ('',
             clear_box(LOGO, width),
             clear_box(text, width, '>'))
    return TextBox(underline, '')('\n'.join(parts), width)


# register HDU classes (order is important!)
register_hdu(shocHDU)


# initialize calibration database
calDB = CalDB('/media/Oceanus/work/Observing/data/SHOC/calibration/')
