import inspect
import logging
from pathlib import Path

import astropy.io.fits as pyfits
from astropy.coordinates import SkyCoord
from astropy.coordinates.name_resolve import NameResolveError

from recipes.io import warn
from obstools import jparser
# from decor.misc import persistent_memoizer
from recipes.decor import memoize
# from motley.profiling.timers import timer

# get coordinate cache file
here = inspect.getfile(inspect.currentframe())  # this_filename
moduleDir = Path(here).parent
cooCacheName = '.coordcache'
cooCachePath = moduleDir / cooCacheName

dssCacheName = '.dsscache'
dssCachePath = moduleDir / dssCacheName


@memoize.to_file(cooCachePath)
def resolver(name):
    """Get the target coordinates from object name if known"""
    # try extract J coordinates from name.  We do this first, since it is
    # faster than a sesame query
    if jparser.search(name):
        return jparser.to_skycoord(name)

    # Attempts a SIMBAD Sesame query with the given object name
    logging.info('Querying SIMBAD database for %r.', name)
    return SkyCoord.from_name(name)


def convert_skycoords(ra, dec, verbose=False):
    """Try convert ra dec to SkyCoord"""
    if ra and dec:
        try:
            return SkyCoord(ra=ra, dec=dec, unit=('h', 'deg'))
        except ValueError as err:
            if verbose:
                warn('Could not interpret coordinates: %s; %s' % (ra, dec))


def retrieve_coords(name):
    """
    Attempts to retrieve coordinates from name, first by parsing the name, or by
    doing SIMBAD Sesame query for the coordinates associated with name

    Examples
    --------
    coords = retrieve_coords('MASTER J061451.7-272535.5')
    coords = retrieve_coords('UZ For')
    ...
    """
    try:
        coo = resolver(name)
        fmt = dict(precision=2, sep=' ', pad=1)
        ra = coo.ra.to_string(unit='h', **fmt)
        dec = coo.dec.to_string(unit='deg', alwayssign=1, **fmt)
        logging.info(
                'The following ICRS J2000.0 coordinates were retrieved:\n'
                'RA = %s, DEC = %s', ra, dec)
        return coo

    except (NameResolveError, AttributeError) as e:
        logging.exception(
                'Coordinates for object %r could not be retrieved due to the '
                'following exception: ', name)


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


def combine_single_images(ims, func):  # TODO MERGE WITH shocObs.combine????
    """Combine a run consisting of single images."""
    header = copy(ims[0][0].header)
    data = func([im[0].data for im in ims], 0)

    header.remove('NUMKIN')
    header['NCOMBINE'] = (len(ims), 'Number of images combined')
    for i, im in enumerate(ims):
        imnr = '{1:0>{0}}'.format(3, i + 1)  # Image number eg.: 001
        comment = 'Contributors to combined output image' if i == 0 else ''
        header['ICMB' + imnr] = (im.get_filename(), comment)

    # uses the FilenameGenerator of the first image in the shocRun
    # outname = next( ims[0].filename_gen() )

    return pyfits.PrimaryHDU(data, header)  # outname


class STScIServerError(Exception):
    pass


# @timer
@memoize.to_file(dssCachePath)
def get_dss(imserver, ra, dec, size=(10, 10), epoch=2000):
    """
    Grab a image from STScI server and load as HDUList

    Parameters
    ----------
    imserver
    ra
    dec
    size:   Field of view size in 'arcmin'
    epoch

    Returns
    -------

    """

    import textwrap
    from io import BytesIO
    import urllib.request  # , urllib.error, urllib.parse

    known_servers = ('poss2ukstu_blue', 'poss1_blue',
                     'poss2ukstu_red', 'poss1_red',
                     'poss2ukstu_ir',
                     'all')
    if not imserver in known_servers:
        raise ValueError('Unknown server: %s.  Please select from: %s'
                         % (imserver, str(known_servers)))

    # resolve size
    h, w = size  # FIXME: if number

    # make url
    url = textwrap.dedent('''\
            http://archive.stsci.edu/cgi-bin/dss_search?
            v=%s&
            r=%f&d=%f&
            e=J%f&
            h=%f&w=%f&
            f=fits&
            c=none''' % (imserver, ra, dec, epoch, h, w)
                          ).replace('\n', '')
    # log
    logging.info("Retrieving %s'x %s' image for object at J%.1f coordinates "
                 "RA = %.3f; DEC = %.3f from %r", h, w, epoch, ra, dec,
                 imserver)
    # get data from server
    data = urllib.request.urlopen(url).read()
    if b'ERROR' in data[:1000]:
        msg = data[76:194].decode().replace('\n<PRE>\n', ' ')
        raise STScIServerError(msg)

    # load into fits
    fitsData = BytesIO()
    fitsData.write(data)
    fitsData.seek(0)
    return pyfits.open(fitsData, ignore_missing_end=True)


import numpy as np
from recipes.array import ndgrid
# from obstools.fastfits import FitsCube
#
