"""
pyshoc - Data analysis tools for the Sutherland High-Speed Optical Cameras
"""

from astropy.io.fits.hdu.base import register_hdu
from .core import *


# register HDU classes (order is important!)
register_hdu(shocBiasHDU)
register_hdu(shocFlatHDU)
register_hdu(shocNewHDU)
register_hdu(shocOldHDU)


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
