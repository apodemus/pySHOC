import inspect
from pathlib import Path

from astropy.coordinates import SkyCoord
from astropy.coordinates.name_resolve import NameResolveError

from recipes.io import warn
from obstools.jparser import Jparser
from decor.misc import persistant_memoizer


# get coordinate cache file
here = inspect.getfile(inspect.currentframe())
moduleDir = Path(here).parent
cooCacheName = '.coordcache'
cooCachePath = moduleDir / cooCacheName

@persistant_memoizer(cooCachePath)
def resolver(name, verbose=True):
    """Get the target coordinates from object name if known"""
    # try extract J coordinates from name.  We do this first, since it is faster than a sesame query
    try:
        return Jparser(name).skycoord()
    except ValueError:
        pass

    # Attempts a SIMBAD Sesame query with the given object name
    if verbose:
        print('\nQuerying SIMBAD database for {}...'.format(repr(name)))
    return SkyCoord.from_name(name)


def retrieve_coords(name, verbose=True):
    """Attempts a SIMBAD Sesame query with the given object name"""
    try:
        coo = resolver(name, verbose)
        if verbose:
            fmt = dict(precision=2, sep=' ', pad=1)
            ra = coo.ra.to_string(unit='h', **fmt)
            dec = coo.dec.to_string(unit='deg', alwayssign=1, **fmt)
            print('The following ICRS J2000.0 coordinates were retrieved:\n'
                  'RA = {}, DEC = {}\n'.format(ra, dec))
        return coo

    except NameResolveError as e:
        if verbose:
            warn('Coordinates for object {!r} could not be retrieved due to the following exception:\n'
                 '{!s}'.format(name, e))       # TODO traceback?


def retrieve_coords_ra_dec(name, verbose=True, **fmt):
    """return SkyCoords and str rep for ra and dec"""
    coords = retrieve_coords(name, verbose)
    if coords is None:
        return None, None, None

    default_fmt = dict(precision=2, sep=' ', pad=1)
    fmt.update(default_fmt)
    ra = coords.ra.to_string(unit='h', **fmt)
    dec = coords.dec.to_string(unit='deg', alwayssign=1, **fmt)

    return coords, ra, dec


def convert_skycooords(ra, dec, verbose=False):
    """Try convert ra dec to SkyCoord"""
    if ra and dec:
        try:
            return SkyCoord(ra=ra, dec=dec, unit=('h', 'deg'))
        except ValueError as err:
            if verbose:
                warn('Could not interpret coordinates: %s; %s' % (ra, dec))


def combine_single_images(ims, func):  # TODO MERGE WITH shocCube.combine????
    """Combine a run consisting of single images."""
    header = copy(ims[0][0].header)
    data = func([im[0].data for im in ims], 0)

    header.remove('NUMKIN')
    header['NCOMBINE'] = (len(ims), 'Number of images combined')
    for i, im in enumerate(ims):
        imnr = '{1:0>{0}}'.format(3, i + 1)  # Image number eg.: 001
        comment = 'Contributors to combined output image' if i == 0 else ''
        header['ICMB' + imnr] = (im.get_filename(), comment)

    # outname = next( ims[0].filename_gen() )  #uses the FilenameGenerator of the first image in the shocRun

    # pyfits.writeto(outname, data, header=header, output_verify='warn', overwrite=True)
    return pyfits.PrimaryHDU(data, header)  # outname