from astropy.io.fits.hdu.base import register_hdu

from recipes.iter import flatiter, itersubclasses

from .core import *

# register HDU classes
register_hdu(shocNewHDU)
register_hdu(shocOldHDU)
# TODO: Consider shocBiasHDU, shocFlatFieldHDU + match headers by looking at obstype keywords


# Collect named cubes in a dict
namedCubeClasses = {cls.name: cls
                    for cls in flatiter((itersubclasses(shocCube), shocCube))}
namedRunClasses = {cls.cubeClass.name: cls
                    for cls in flatiter((itersubclasses(shocRun), shocRun))}

def cubeFactory(name):      # FIXME: not a factory
    return namedCubeClasses.get(name, shocCube)

def runFactory(name):
    return namedRunClasses.get(name, shocRun)


def load(filenames, kind='science'):
    """

    :param filenames:
    :param kind:
    :return:
    """
    cls = runFactory(kind)
    # print(cls, '!')
    run = cls(filenames=filenames, label=kind)
    return run