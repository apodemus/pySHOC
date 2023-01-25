"""
pyshoc - Data analysis tools for the Sutherland High-Speed Optical Cameras.
"""

# std
from importlib.metadata import PackageNotFoundError, version

# third-party
import yaml
from astropy.io.fits.hdu.base import register_hdu

# local
from recipes.dicts import AttrReadItem, DictNode

# relative
from .core import *
from .caldb import CalDB


# ---------------------------------------------------------------------------- #
try:
    __version__ = version('pyshoc')
except PackageNotFoundError:
    __version__ = '?.?.?'


# ---------------------------------------------------------------------------- #
# settings

class ConfigNode(DictNode, AttrReadItem):
    pass


CONFIG = ConfigNode(
    **yaml.load((Path(__file__).parent / 'config.yaml').read_text(),
                Loader=yaml.FullLoader)
)
# CONFIG.freeze()

# ---------------------------------------------------------------------------- #
# register HDU classes (must happen before CalDB init)
register_hdu(shocHDU)


# initialize calibration database
calDB = CalDB(CONFIG.caldb)
