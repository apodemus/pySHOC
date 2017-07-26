from astropy.io.fits.hdu.base import register_hdu

from recipes.iter import flatiter, itersubclasses

from .core import *

# register HDU classes
register_hdu(shocNewHDU)
register_hdu(shocOldHDU)
# TODO: Consider shocBiasHDU, shocFlatFieldHDU + match headers by looking at obstype keywords


# Collect named cubes in a dict
namedObsClasses = {cls.kind: cls
                    for cls in flatiter((itersubclasses(shocObs), shocObs))}
namedRunClasses = {cls.obsClass.kind: cls
                    for cls in flatiter((itersubclasses(shocRun), shocRun))}

def cubeFactory(kind):      # not really a factory
    return namedObsClasses.get(kind, shocObs)

def runFactory(kind):
    return namedRunClasses.get(kind, shocRun)


def load(filenames, kind='science'):
    """

    Parameters
    ----------
    filenames
    kind

    Returns
    -------

    """
    run = shocCube.load(filenames=filenames, kind=kind)
    return run