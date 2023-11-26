"""
pyshoc - Data analysis tools for the Sutherland High-Speed Optical Cameras.
"""

# std
from importlib.metadata import PackageNotFoundError, version

# third-party
from astropy.io.fits.hdu.base import register_hdu

# relative
from .core import *
from .caldb import CalDB
from .config import CONFIG


# ---------------------------------------------------------------------------- #
try:
    __version__ = version('pyshoc')
except PackageNotFoundError:
    __version__ = '?.?.?'


# ---------------------------------------------------------------------------- #
# register HDU classes (must happen before CalDB init)
register_hdu(shocHDU)


# initialize calibration database
if not (folder := CONFIG.calibration.get('folder')):
    from platformdirs import user_data_path

    folder = user_data_path('pyshoc') / 'caldb'

# Load calibration database
calDB = CalDB(folder)
