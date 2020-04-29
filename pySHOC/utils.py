import logging
from pathlib import Path

import astropy.io.fits as pyfits
from astropy.coordinates import SkyCoord
from astropy.coordinates.name_resolve import NameResolveError

from recipes.decor import memoize
# from motley.profiling.timers import timer

from recipes.introspection.utils import get_module_name

# module level logger
logger = logging.getLogger(get_module_name(__file__))

# get coordinate cache file
cachePath = Path.home() / '.cache/pySHOC'  # todo windows / mac ??
cooCachePath = cachePath / 'coords.pkl'
dssCachePath = cachePath / 'dss.pkl'


class STScIServerError(Exception):
    pass


def get_coordinates(name_or_coords):
    """
    Get object coordinates from object name or string of coordinates. If the
    coordinates could not be resolved, return None


    Parameters
    ----------
    name_or_coords

    Returns
    -------
    astropy.coordinates.SkyCoord or None
    """
    if isinstance(name_or_coords, SkyCoord):
        return name_or_coords

    try:
        # first try interpret coords.
        # eg. ImageRegistrationDSS('06:14:51.7 -27:25:35.5', (3, 3))
        return SkyCoord(name_or_coords, unit=('h', 'deg'))
    except ValueError:
        return get_coords_named(name_or_coords)


def get_coords_named(name):
    """
    Attempts to retrieve coordinates from name, first by parsing the name, or by
    doing SIMBAD Sesame query for the coordinates associated with name.

    Examples
    --------
    coords = get_coords_named('MASTER J061451.7-272535.5')
    coords = get_coords_named('UZ For')
    ...
    """
    try:
        coo = resolver(name)
    except NameResolveError as e:  # AttributeError
        logger.warning(
                'Coordinates for object %r could not be retrieved due to the '
                'following exception: \n%s', name, str(e))
    else:
        if isinstance(coo, SkyCoord):
            fmt = dict(precision=2, sep=' ', pad=1)
            logger.info(
                    'The following ICRS J2000.0 coordinates were retrieved:\n'
                    'α = %s; δ = %s',
                    coo.ra.to_string(unit='h', **fmt),
                    coo.dec.to_string(unit='deg', alwayssign=1, **fmt))
        return coo


@memoize.to_file(cooCachePath)
def resolver(name):
    """Get the target coordinates from object name if known"""
    # try extract J coordinates from name.  We do this first, since it is
    # faster than a sesame query

    # Attempts a SIMBAD Sesame query with the given object name
    logger.info('Querying SIMBAD database for %r.', name)
    try:
        return SkyCoord.from_name(name, parse=True)
    except NameResolveError as e:
        # check if the name is bad - something like "FLAT" or "BIAS", we want
        # to cache these bad values also to avoid multiple sesame queries for
        # bad values like these
        if str(e).startswith("Unable to find coordinates for name"):
            return None

        # If we are here, it probably means there is something wrong with the
        # connection:
        # NameResolveError: "All Sesame queries failed."
        raise


def convert_skycoords(ra, dec):
    """Try convert ra dec to SkyCoord"""
    if ra and dec:
        try:
            return SkyCoord(ra=ra, dec=dec, unit=('h', 'deg'))
        except ValueError:
            logger.warning(
                    'Could not interpret coordinates: %s; %s' % (ra, dec))


def retrieve_coords_ra_dec(name, verbose=True, **fmt):
    """return SkyCoords and str rep for ra and dec"""
    coords = get_coords_named(name)
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


# @timer
@memoize.to_file(dssCachePath)  # memoize for performance
def get_dss(server, ra, dec, size=(10, 10), epoch=2000):
    """
    Grab a image from STScI server and load as HDUList.

    Parameters
    ----------
    server
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
    if not server in known_servers:
        raise ValueError('Unknown server: %s.  Please select from: %s'
                         % (server, str(known_servers)))

    # resolve size
    h, w = size  # FIXME: if number

    # make url
    url = textwrap.dedent(f'''\
            http://archive.stsci.edu/cgi-bin/dss_search?
            v={server}&
            r={ra}&d={dec}&
            e=J{epoch}&
            h={h}&w={w}&
            f=fits&
            c=none''').replace('\n', '')
    # log
    logger.info("Retrieving %s'x %s' image for object at J%.1f coordinates "
                "RA = %.3f; DEC = %.3f from %r", h, w, epoch, ra, dec, server)

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
