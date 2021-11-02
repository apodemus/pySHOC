"""
pyshoc - Data analysis tools for the Sutherland High-Speed Optical Cameras.
"""


# std
from importlib.metadata import PackageNotFoundError, version

# third-party
import yaml
from astropy.io.fits.hdu.base import register_hdu

# local
from recipes.misc import get_terminal_size
from motley import banner, format, justify

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
LOGO = (SRC_ROOT / 'banner/banner.txt').read_text()


def make_banner(subtitle='', width=None, **style):
    width = int(width or get_terminal_size()[0])
    subtext = format('{:|purple}\n{v{v:}:|k}', subtitle, v=__version__)
    return banner('\n'.join(
        ('', #justify(apply('Welcome to', 'purple'), '<', width),
         justify(LOGO, '^', width),
         justify(subtext, '>',  width))),
        width,
        **style
    )


# register HDU classes (order is important!)
register_hdu(shocHDU)


# initialize calibration database
calDB = CalDB(CONFIG['caldb'])
