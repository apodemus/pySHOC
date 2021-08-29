"""
Compare barycentric corrections for 
astropy, 
spice, 
PINT, 
TEMPOS2

barrycorrpy  
    https://github.com/shbhuk/barycorrpy/wiki/08.-JDUTC-to-BJDTDB-converter

utc2bjd

"""

# std
import urllib
import re
import numbers
import itertools as itt
from pathlib import Path

# third-party
import numpy as np
import spiceypy as spice
import more_itertools as mit
from joblib import Parallel, delayed
from joblib._parallel_backends import SequentialBackend
from astropy.time import Time
from astropy.constants import c, G, M_sun
from astropy.coordinates import SkyCoord, EarthLocation, FK5

from barycorrpy.utc_tdb import JDUTC_to_BJDTDB

# import multiprocessing as mp

# noinspection PyUnresolvedReferences

# TODO: decide whether precession is necessary based on required precision
# need a table with magnitude of effects (upper limit, lower limit in s/s) to
# decide this.  look as TEMPOS2 paper for computing these

# spice kernels
SPICE_KERNEL_PATH = Path.home() / 'work/repos/SpiceyPy/kernels'
SPK = str(SPICE_KERNEL_PATH / 'de430.bsp')  # ephemeris kernel:
LSK = str(SPICE_KERNEL_PATH / 'naif0012.tls')  # leap second kernel

# NOTE:
# Despite these corrections, there is still a ~0.02s offset between the
# BJD_TDB computed by this code and the IDL code at
# http://astroutils.astronomy.ohio-state.edu/time/.
# The function below can be used to check this

# some characters don't work in submitting the php form
HTML_TRANSLATIONS = str.maketrans('-:', '  ')
N_LIMIT_WEB = int(1e4)

# parser for returned JDs
JD_REGEX = re.compile(r'(\d{7}\.\d{9})\n')

#
SECONDS_PER_DAY = 86400
# speed of light in km per day
C_KMPD = c.to('km/day').value
# Schwarzchild radius of the sun over c (used for shapiro delay)
RS_C = (2 * G * M_sun / c**3).to('day').value


def load_kernels():
    # load kernels:
    # https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/         (leap second)
    # https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/ (ephemeris)

    spice.furnsh(SPK)
    spice.furnsh(LSK)


def with_initializer(self, f_init):
    # HACK for custom process initializer for joblib.Parallel
    # adapted from:
    # https://github.com/joblib/joblib/issues/381#issuecomment-480910348

    if isinstance(self._backend, SequentialBackend):
        return self

    if not hasattr(self._backend, '_workers'):
        self.__enter__()

    workers = self._backend._workers
    origin_init = workers._initializer

    def new_init():
        origin_init()
        f_init()

    workers._initializer = (new_init if callable(origin_init) else f_init)
    return self


def precess(coords, t, every=0):
    """

    Parameters
    ----------
    coords
    t
    every

    Returns
    -------

    """
    assert isinstance(every, numbers.Integral)
    assert isinstance(t, Time)

    if t.shape == ():
        return coords.transform_to(FK5(equinox=t))

    if every == 0:
        return coords.transform_to(FK5(equinox=t[0]))

    from scipy.interpolate import interp1d

    n = len(t)
    idx = list(range(0, n, every))
    if idx[-1] != n - 1:
        idx += [n - 1]
    sub = t[idx]

    #
    with Parallel(n_jobs=-1) as parallel:
        ra, dec = zip(*parallel(delayed(_precess)(coords, _) for _ in t))

    jd = sub.jd  # ensure we include last point
    return SkyCoord(ra=interp1d(jd, ra)(t.jd),
                    dec=interp1d(jd, dec)(t.jd),
                    unit='deg')


def _precess(coords, t):
    # precess to observation date and time and then transform back to
    # FK5 (J2000)
    cxx = coords.transform_to(FK5(equinox=t))
    return cxx.ra.value, cxx.dec.value


def utc2bjd(t, coords, site):
    """
    Convert to `astropy.Time` times to BJD(TDB) times using the html form at
        http://astroutils.astronomy.ohio-state.edu/time/
    """

    assert isinstance(t, Time)

    if t.isscalar:
        t = t.__class__([t.jd], format='jd', scale=t.scale)

    result = []
    # web form can only handle 1e4 times simultaneously
    # split time array into chunks
    n = t.size
    indices = mit.pairwise(itt.chain(range(0, n, N_LIMIT_WEB), [n]))
    for i0, i1 in indices:
        result.extend(_utc2bjd(t[i0:i1], coords, site))

    result = np.array(result, float)
    if result.size == 1:
        return result.item()
    return result

    # convert back to time object
    # return Time(np.array(result, float),
    #             format='jd',
    #             scale='tdb')


def _utc2bjd(t, coords, site):
    urlbase = 'http://astroutils.astronomy.ohio-state.edu/time/utc2bjd'

    # encode the times to url format str
    # urllib does not success convert newlines etc appropriately,
    # so we manually do the html POST code translations
    newline = '%0D%0A'
    jds = newline.join(t.utc.iso).translate(HTML_TRANSLATIONS)

    # encode payload for the php form
    params = urllib.parse.urlencode(
        dict(observatory=site,
             raunits='hours',
             spaceobs='none',
             ra=coords.ra.to_string('h', sep=' '),
             dec=coords.dec.to_string(sep=' '),
             jds=jds)).encode()

    # submit the form
    req = urllib.request.Request(f'{urlbase}.php?')
    req.add_header('Referer', f'{urlbase}.html')
    raw = urllib.request.urlopen(req, params).read()

    # parse the returned data
    return JD_REGEX.findall(raw.decode())


def light_time_correction_spice(t, coords, origin='barycentric'):
    """
    Convert JD(TDB) to HJD(TDB) or BJD(TDB) by applying the light-time
    corrections.

    Returns
    -------

    """
    ltt = light_time_terms_spice(t, coords, origin)
    return np.sum(ltt, 0).squeeze()


def light_time_terms_spice(t, coords, origin='barycentric'):
    """
    Corrections done for Rømer, Einstein and Shapiro delays are
    computed.

    Params
    ------
    t - `astropy.time.Time` object

    coords - SkyCoord

    """

    assert isinstance(t, Time)

    if origin.startswith('bary'):
        origin = '0'
    elif origin.startswith('helio'):
        origin = '10'
    else:
        raise ValueError('`origin` should be "barycentric" or "heliocentric"')

    # get sun / earth position wrt solar system barycenter
    # with mp.Pool(initializer=load_kernels) as pool:
    n_jobs = -1
    if t.isscalar:
        n_jobs = 1
        t = t.__class__([t.jd], format='jd', scale=t.scale)

    with Parallel(n_jobs=n_jobs) as parallel:
        et = with_initializer(parallel, load_kernels)(
            delayed(spice.utc2et)(_) for _ in t.utc.iso)

    # cartesian position and light travel time from `origin` to earth barycenter
    aberration = 'none'
    frame = 'J2000'
    xyz_earth, _ = get_xyz(et, 'earth', origin, frame, aberration)
    xyz_sun, _ = get_xyz(et, 'sun', origin, frame, aberration)

    # precess -
    #         whether or not to precess coordinates

    # cartesian unit vector pointing to object
    xyz_obj = precess(coords, t, 0).cartesian.xyz.to_value()

    # return xyz_earth, xyz_sun, xyz_obj
    return (romer_delay(xyz_earth, xyz_obj), einstein_delay(t.tt.jd),
            shapiro_delay(xyz_earth, xyz_sun, xyz_obj))


def get_xyz(t, target, observer='SSB', frame='J2000', aberration='none'):
    """
    Return the position of a target body relative to an observing
    body, optionally corrected for light time (planetary aberration)
    and stellar aberration (velocity correction).
    https://en.wikipedia.org/wiki/Aberration_(astronomy)
    https://en.wikipedia.org/wiki/Stellar_aberration_%28derivation_from_Lorentz_transformation%29

    Parameters
    ----------
    t : array
        ephemeris time ET
    target, observer:
        obj - the object for which coordinated are to be calculated
        origin - the origin of the coordinated system for returned values
        Both oj and origin should be one of the following:

          NAIF ID     NAME
          ________    ____________________
          0           'SOLAR_SYSTEM_BARYCENTER'
          0           'SSB'
          0           'SOLAR SYSTEM BARYCENTER'
          1           'MERCURY_BARYCENTER'
          1           'MERCURY BARYCENTER'
          2           'VENUS_BARYCENTER'
          2           'VENUS BARYCENTER'
          3           'EARTH_BARYCENTER'
          3           'EMB'
          3           'EARTH MOON BARYCENTER'
          3           'EARTH-MOON BARYCENTER'
          3           'EARTH BARYCENTER'
          4           'MARS_BARYCENTER'
          4           'MARS BARYCENTER'
          5           'JUPITER_BARYCENTER'
          5           'JUPITER BARYCENTER'
          6           'SATURN_BARYCENTER'
          6           'SATURN BARYCENTER'
          7           'URANUS_BARYCENTER'
          7           'URANUS BARYCENTER'
          8           'NEPTUNE_BARYCENTER'
          8           'NEPTUNE BARYCENTER'
          9           'PLUTO_BARYCENTER'
          9           'PLUTO BARYCENTER'
          10          'SUN'

    frame
    aberration
        Aberration corrections: (from the spice spice.spkpos readme)

        "Reception" case in which photons depart from the target's location at
        the light-time corrected epoch et-lt and *arrive* at the observer's
        location at `et'
        "LT"    Correct for one-way light time (also called "planetary aberration")
                using a Newtonian formulation.
        "LT+S"  Correct for one-way light time and stellar aberration using a
                Newtonian formulation.
        "CN"    Converged Newtonian light time correction.  In solving the light
                time equation, the "CN" correction iterates until the solution
                converges
        "CN+S"  Converged Newtonian light time and stellar aberration corrections.'

        "Transmission" case in which photons *depart* from the observer's location
        at `et' and arrive at the target's location at the light-time corrected
        epoch et+lt ---> prepend 'X' to the description strings as given above.

        Neither special nor general relativistic effects are accounted for in
        the aberration corrections applied by this routine.

    Returns
    -------

    """

    # returned
    # Cartesian 3-vector representing the position of the target body
    # relative to the specified observer in km
    #
    # one-way light time between the observer and target in seconds. If the
    # target position is corrected for aberrations, then `lt' is the one-way
    # light time between the observer and the light time corrected target
    # location.

    t = np.atleast_1d(t)
    n_jobs = -int(t.size > 1) or 1
    with Parallel(n_jobs=n_jobs) as parallel:
        xyz, ltt = zip(*parallel(
            delayed(spice.spkpos)(target, et, frame, aberration, observer)
            for et in t))

    # position, light travel time
    return np.array(xyz), np.array(ltt)  # , np.float128


def romer_delay(xyz_earth, xyz_obj):
    """
    Calculate Rømer delay (classical light travel time correction) in units of
    days. This is the time it takes light to travel between the solar system
    barycenter and the earth-moon barycenter.

    Location of observatory on the surface of the earth is not accounted for,
    but should be <~1e-5 s.

    Notes:
    ------
    https://en.wikipedia.org/wiki/Ole_R%C3%B8mer
    https://en.wikipedia.org/wiki/R%C3%B8mer%27s_determination_of_the_speed_of_light

    """
    # ephemeris units is in km / s .   convert to m / (julian) day
    return (xyz_earth * xyz_obj).sum(1) / C_KMPD


rømer_delay = romer_delay  # :)


def einstein_delay(jd_tt):
    """
    Calculate Einstein delay in units of days.  This is the special
    relativistic component of time dilation due to motion of the observer.
    Equation below taken from astronomical almanac.
    """
    red_jd_tt = jd_tt - 2451545.0
    g = np.radians(357.53 + 0.98560028 * red_jd_tt)  # mean anomaly of Earth
    # Difference in mean ecliptic longitude of the Sun and Jupiter
    l_lj = np.radians(246.11 + 0.90251792 * red_jd_tt)
    delay = 0.001657 * np.sin(g) + 0.000022 * np.sin(l_lj)
    return delay / SECONDS_PER_DAY


def shapiro_delay(xyz_earth, xyz_sun, xyz_obj):
    """
    Calculate Shapiro delay in units of days. This is the gravitational time
    delay effect due to light travelling near a massive body. This represents
    the general relativistic component of time dilation.

    https://en.wikipedia.org/wiki/Shapiro_delay
    """
    # Earth to Sun vector
    x = xyz_earth - xyz_sun
    # Earth to Sun unit vector
    x /= np.sqrt((x * x).sum(-1, keepdims=True))

    # dot product gives cosine of angle between Earth and Sun
    # cosTheta = (x * xyz_obj).sum(1)

    # Approximate for Shapiro delay
    return RS_C * np.log(1 - (x * xyz_obj).sum(1))


def compare(t, coords, site, report=True):

    # ensure time has location set, affects tdb transform (~1 μs)
    t.location = EarthLocation.of_site(site)

    # 
    tb, *msgs = JDUTC_to_BJDTDB(t.utc.jd,
                                ra=coords.ra.deg, dec=coords.dec.deg,
                                obsname=site,)
    

    if tb.size == 1:
        tb = tb.item()                            
    # tb = Time(tb, format='jd', scale='tdb')
    # leap_dir=SPICE_KERNEL_PATH),

    # astropy
    ltta = t.light_travel_time(coords, 'barycentric', t.location)
    ta = t.tdb.jd + ltta.value

    # spice
    ltts = light_time_correction_spice(t, coords)
    ts = t.tdb.jd + ltts

    # Eastman IDL Web service
    te = utc2bjd(t, coords, site)

    tdata = dict(astropy=ta,
                spice=ts,
                barycorrpy=tb,
                eastman=te)

    times = np.array(list(tdata.values()))
    deltas = np.ma.array((times - times[None].T) * SECONDS_PER_DAY)

    if report:
        from motley.table import Table

        deltas[np.diag_indices_from(deltas)] = np.ma.masked
        heads = list(tdata.keys())
        Table(deltas, row_head=heads, col_heads=heads, precision=6)

    return tdata, deltas