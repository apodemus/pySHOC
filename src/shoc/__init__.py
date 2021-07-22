"""
pyshoc - Data analysis tools for the Sutherland High-Speed Optical Cameras
"""


# std libs
from importlib.metadata import version, PackageNotFoundError

# third-party libs
from astropy.io.fits.hdu.base import register_hdu

# relative libs
from .core import *
from .caldb import CalDB


try:
    __version__ =  version('pyshoc')
except PackageNotFoundError:
    __version__ = '?.?.?'

# register HDU classes (order is important!)
register_hdu(shocHDU)


# initialize calibration database
calDB = CalDB('/media/Oceanus/work/Observing/data/SHOC/calibration/')
# calDB.autovivify(False)

# # Collect named cubes in a dict
# from recipes.iter import flatiter, itersubclasses
#
# namedObsClasses = {cls.kind: cls
#                    for cls in flatiter((itersubclasses(shocObs), shocObs))}
# namedRunClasses = {cls.obsClass.kind: cls
#                    for cls in flatiter((itersubclasses(shocRun), shocRun))}
#
#
# def cubeFactory(kind):  # not really a factory
#     return namedObsClasses.get(kind, shocObs)
#
#
# def runFactory(kind):
#     return namedRunClasses.get(kind, shocRun)
#
#
# def load(filenames, kind='science'):
#     """
#
#     Parameters
#     ----------
#     filenames
#     kind
#
#     Returns
#     -------
#
#     """
#     run = shocRun.load(filenames=filenames, kind=kind)
#     return run
