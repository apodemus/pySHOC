"""
pyshoc - Data analysis tools for the Sutherland High-Speed Optical Cameras.
"""

# std
from importlib.metadata import PackageNotFoundError, version

# third-party

from astropy.io.fits.hdu.base import register_hdu



# relative
from .config import CONFIG
from .core import *
from .caldb import CalDB


# ---------------------------------------------------------------------------- #
try:
    __version__ = version('pyshoc')
except PackageNotFoundError:
    __version__ = '?.?.?'



# ---------------------------------------------------------------------------- #
# register HDU classes (must happen before CalDB init)
register_hdu(shocHDU)


# initialize calibration database
calDB = CalDB(CONFIG.calibration.folder)
