"""
pyshoc - Analysis tools for data from the Sutherland High-Speed Optical Cameras.
"""

# std
from importlib.metadata import PackageNotFoundError, version

# third-party
from astropy.io.fits.hdu.base import register_hdu

# relative
from . import config
from .caldb import CalDB
from .core import (
    HDU, MATCH, Binning, CalibrationHDU, Campaign, DarkHDU, FilenameHelper,
    Filters, FlatHDU, GroupedRuns, Master, MasterDark, MasterFlat, Messenger,
    OldDarkHDU, OldFlatHDU, OldHDU, OutAmpMode, ReadoutMode, RollOverState,
    TableHelper
)


# ---------------------------------------------------------------------------- #
try:
    __version__ = version('pyshoc')
except PackageNotFoundError:
    __version__ = '?.?.?'


# ---------------------------------------------------------------------------- #
# register HDU classes (must happen before CalDB init)
register_hdu(HDU)

# Load calibration database
calDB = CalDB(config.calibration.folder)
