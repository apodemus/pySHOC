"""
Functions for time-stamping SHOC data
"""

import logging
from pathlib import Path
from warnings import warn
from urllib.error import URLError

import numpy as np
from astropy.time import Time, TimeDelta
from astropy.table import Table as aTable
from astropy.constants import c, G, M_sun
from astropy.coordinates import EarthLocation, FK5
from astropy.coordinates.angles import Angle
import spiceypy as spice

from obstools.airmass import Young94, altitude

# this should maybe go to __init__.py
# TODO: automate finding latest kernels:
# http://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/          (leap second)
# https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/ (ephemeris)
# load kernels:
spice_kernel_path = Path('/home/hannes/work/repos/SpiceyPy/kernels/')
SPK = str(spice_kernel_path / 'de430.bsp')  # ephemeris kernel:
LSK = str(spice_kernel_path / 'naif0012.tls')  # leap second kernel
spice.furnsh(SPK)
spice.furnsh(LSK)


# from decor.profiler import profiler
# profiler = profile()


def get_updated_iers_table(cache=True, raise_=True):  # TODO: rename
    """Get updated IERS data"""
    from astropy.utils.iers import IERS_A, IERS_A_URL  # import IERS data class
    from astropy.utils.data import download_file

    logging.info('Updating IERS table.')

    # get IERS data tables from cache / download
    try:
        iers_a_file = download_file(IERS_A_URL, cache=cache)
        iers_a = IERS_A.open(iers_a_file)  # load data tables
        logging.info('Done')
        return iers_a
    except URLError as err:
        if raise_:
            raise err
        warn('Unable to update IERS table due to the following exception:\n%s'
             '\nAre you connected to the internet? If not, try re-run with'
             ' cache=True'
             % err)  # TODO with traceback?
        return None


# @profiler.histogram

# TODO: Check the accuracy of these routines against astropy, utc2bjd, etc....

def light_time_corrections(t, coords, precess=False, abcorr=None):
    """
    Barycentric julian day TBD.  Corrections done for Rømer, Einstein and
    Shapiro delays.

    Params
    ------
    coords - SkyCoord

    precess - whether or not to precess coordinates

    abcorr - how (if) aberration corrections should be done

    Aberation corrections: (from the spice spice.spkpos readme)

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

    """

    tabc = ['NONE', 'LT', 'LT+S', 'CN', 'CN+S']
    rabc = ['X' + s for s in tabc]
    allowed_abc = tabc + rabc
    abcorr = str(abcorr).replace(' ', '').upper()
    if abcorr in allowed_abc:
        ABCORR = abcorr
    else:
        warn('Aberration correction specifier {} not understood. Next time use '
             'one of {}.\n No aberration correction(s) will be done. '
             '(Aberration effects should be negligible anyway!)'
             ''.format(abcorr, allowed_abc))

    xyzObj = get_Obj_coords(t, coords, precess=precess)
    xyzSun, xyzEarth = get_Earth_Sun_coords(t[:1], ABCORR)  # HACK for speed!!!!

    rd = romer_delay(xyzEarth, xyzObj)
    ed = einstein_delay(t.tt.jd)
    shd = shapiro_delay(xyzEarth, xyzSun, xyzObj)

    # print('Romer delay', rd)
    # print('Einstein delay', ed)
    # print('Shapiro delya', shd)

    return t.tdb.jd - rd - ed - shd


# @profile()
def get_Earth_Sun_coords(t, ABCORR):
    """
    Get Earth (Geocentric) and solar (Heliocentric) ephemeris (km) relative to solar
    system Barycenter at times t
    """
    FRAME = 'J2000'
    OBSERVER = '0'  # Solar System Barycenter (SSB)

    N = len(t)
    # TODO: vectorize??
    xyzEarth = np.empty((N, 3), np.float128)
    xyzSun = np.empty((N, 3), np.float128)
    ltEarth = np.empty(N)
    ltSun = np.empty(N)
    for i, t in enumerate(t):
        # Ephemeris time (seconds since J2000 TDB)
        et = spice.utc2et(str(t))
        # Earth geocenter wrt SSB in J2000 coordiantes
        xyzEarth[i], ltEarth[i] = spice.spkpos(OBSERVER, et, FRAME, ABCORR,
                                               'earth')
        # Sun heliocenter wrt SSB in J2000 coordiantes
        xyzSun[i], ltSun[i] = spice.spkpos(OBSERVER, et, FRAME, ABCORR, '10')

    return xyzSun, xyzEarth


# def precess(coords, t, every):


def get_Obj_coords(t, coords, precess=False):
    """ """
    if not precess:
        return coords.cartesian.xyz

    if isinstance(precess, str):
        if precess.lower() == 'first':
            precessed = coords.transform_to(FK5(equinox=t[0]))
            return precessed.cartesian.xyz

        # TODO: EVERY
        # TODO: INTERPOLATION?

    N = len(t)
    xyzObj = np.empty((N, 3), np.float128)

    if precess is True:
        # TODO: MULTIPROCESS FOR SPEED!!!!!?????
        for i, t in enumerate(t):
            # bar.progress(i)
            # precess to observation date and time and then transform back to FK5 (J2000)
            coonew = coords.transform_to(FK5(equinox=t))
            xyzObj[i] = coonew.cartesian.xyz

        return xyzObj

    else:
        raise ValueError


def romer_delay(xyzEarth, xyzObj):
    """
    Calculate Rømer delay (classical light travel time correction) in units of
    days

    Notes:
    ------
    https://en.wikipedia.org/wiki/Ole_R%C3%B8mer
    https://en.wikipedia.org/wiki/R%C3%B8mer%27s_determination_of_the_speed_of_light

    """
    # ephemeris units is in km / s .   convert to m / (julian) day
    convf = c.to('km/day').value
    delay = (xyzEarth * xyzObj).sum(1) / convf

    return delay


rømer_delay = romer_delay  # oh yeah!


def einstein_delay(jd_tt):
    """
    Calculate Eistein delay in units of days
    """
    red_jd_tt = jd_tt - 2451545.0
    g = np.radians(357.53 + 0.98560028 * red_jd_tt)  # mean anomaly of Earth
    # Difference in mean ecliptic longitudea of the Sun and Jupiter
    L_Lj = np.radians(246.11 + 0.90251792 * red_jd_tt)
    delay = 0.001657 * np.sin(g) + 0.000022 * np.sin(L_Lj)

    return delay / 86400.


def shapiro_delay(xyzEarth, xyzSun, xyzObj):
    """
    Calculate Shapiro delay in units of days

    https://en.wikipedia.org/wiki/Shapiro_delay
    """
    Earth_Sun = xyzEarth - xyzSun  # Earth to Sun vector
    d_Earth_Sun = np.linalg.norm(Earth_Sun, axis=1)[
        None].T  # Earth to Sun distance
    u_Earth_Sun = Earth_Sun / d_Earth_Sun  # Earth to Sun unit vector

    # dot product gives cosine of angle between Earth and Sun
    cosTheta = (u_Earth_Sun * xyzObj).sum(1)

    # Approximate for Shapiro delay
    delay = (2 * G * M_sun / c ** 3) * np.log(1 - cosTheta)

    # print(cosTheta)
    # print(2*G*M_sun/c**3)

    return delay.to('day').value


# NOTE:
# Despite these corrections, there is still a ~0.02s offset between the
# BJD_TDB computed by this code and the IDL code at
# http://astroutils.astronomy.ohio-state.edu/time/.
# The function below can be used to check this

def utc2bjd(times, coords):  # TODO move
    """
    Use the html form at http://astroutils.astronomy.ohio-state.edu/time/ to
    convert to astropy.Time to BJD_TDB
    """

    import urllib.request, urllib.parse
    import re

    urlbase = 'http://astroutils.astronomy.ohio-state.edu/time/utc2bjd'
    html = urlbase + '.html'
    form = urlbase + '.php?'

    # data for the php form
    payload = dict(observatory='saao',
                   raunits='hours',
                   spaceobs='none')
    payload['ra'] = coords.ra.to_string('h', sep=' ')
    payload['dec'] = coords.dec.to_string(sep=' ')

    # encode form data
    params = urllib.parse.urlencode(payload)

    # web form can only handle 1e4 times simultaneously
    if len(times) > 1e4:
        'TODO'  # split time array into chunks
        raise NotImplementedError

    # encode the times to url format str (urllib does not success convert
    # newlines appropriately)
    newline, spacer, joiner = '%0D%0A', '+', '&'  # html POST code translations
    fixes = ('-', spacer), (':', spacer), (' ', spacer)
    ts = newline.join(times.iso)
    for fix in fixes:
        ts = ts.replace(*fix)

    # append times to url data
    params = (params + '&jds=' + ts).encode()

    # submit the form
    req = urllib.request.Request(form)
    req.add_header('Referer', html)
    raw = urllib.request.urlopen(req, params)

    # parse the returned data
    jdpatr = '(\d{7}\.\d{9})\n'
    matcher = re.compile(jdpatr)
    jds = matcher.findall(raw.decode())

    # convert back to time object
    bjd_tdb = Time(np.array(jds, float), format='jd')
    return bjd_tdb


class HMSrepr(object):
    """
    Mixin class that provided numerical objects with `hms` property for pretty
    representation
    """

    @property
    def hms(self):
        return fmt_hms(self)


class TimeDelta(TimeDelta):
    @property
    def hms(self):
        v = self.value
        precision = 1 if v > 10 else 3
        return fmt_hms(v, precision)


# ******************************************************************************
class Time(Time):
    """
    Extends the astropy.time.core.Time class to include method that returns the
    time in hours.

    The astropy.time package provides functionality for manipulating times and
    dates. Specific emphasis is placed on supporting time scales (e.g. UTC,
    TAI, UT1) and time representations (e.g. JD, MJD, ISO 8601) that are used
    routines. All time scale conversions are done by Cython vectorized
    versions of the ERFA routines and are fast and memory efficient.

    All time manipulations and arithmetic operations are done internally
    using  two 64-bit floats to represent time. Floating point algorithms
    from [1] are used so that the Time object maintains sub-nanosecond
    precision over times spanning the age of the universe.

    [1]     Shewchuk, 1997, Discrete & Computational Geometry 18(3):305-363
    """

    # TODO: HJD:
    # TODO: phase method...

    # @property
    def isosplit(self):
        """Split ISO time between date and time (from midnight)"""
        splitter = lambda x: tuple(x.split('T'))
        dtype = [('utdate', 'U20'), ('utc', 'U20')]
        utdata = np.fromiter(map(splitter,
                                 np.atleast_1d(self.utc.isot)), dtype)
        utdate, utc = utdata['utdate'], utdata['utc']
        if len(utdate) == 1:
            return utdate[0], utc[0]
        return utdate, utc

    @classmethod
    def isomerge(cls, utdate, utc, **kws):
        # construct Time object from utdate and utc string arrays
        a = np.char.add(np.char.add(utdate, 'T'), utc)
        return cls(a, format='isot', scale='utc', **kws)

    def time_from_local_midnight(self, unit='s'):
        """
        get the TimeDelta since local midnight for the date of the first
        time stamp
        """
        utdate0, _ = self[0].isosplit()
        return (self.utc - Time(utdate0)).to(unit)

    # #@property
    # def hours(self):
    #     """Converts time to hours since midnight."""
    #     vals = np.atleast_1d( self.iso )
    #     hours = np.empty(vals.shape)
    #     for j,val in enumerate(vals):
    #         _, hms = val.split()
    #         h,m,s = map( float, hms.split(':') )
    #         hours[j] = h + m/60. + s/3600.
    #
    #     #if len(hours)==1:
    #         #return hours[0]
    #     #else:
    #     return hours

    # @property
    # def sec(self):
    #     return self.hours * 3600.

    #
    # @property
    # def radians(self):
    #     return np.radians( self.hours*15. )

    def check_iers_table(self):
        """
        For the UT1 to UTC offset, one has to interpolate in observed values
        provided by the International Earth Rotation and Reference Systems
        Service. By default, astropy is shipped with the final values provided
        in Bulletin B, which cover the period from 1962 to shortly before an
        astropy release, and these will be used to compute the offset if the
        delta_ut1_utc attribute is not set explicitly. For more recent times,
        one can download an updated version of IERS B or IERS A (which also
        has predictions), and set delta_ut1_utc as described in
        get_delta_ut1_utc
        """
        from astropy.utils.iers import TIME_BEYOND_IERS_RANGE

        delta, status = self.get_delta_ut1_utc(return_status=True)
        beyond = (status == TIME_BEYOND_IERS_RANGE)

        if np.any(beyond):
            warn('{} / {} times are outside of range covered by IERS table.'
                 ''.format(beyond.sum(), len(beyond)))
        return beyond

    # @profile()
    # def bjd(self, coords, precess='first', abcorr=None): #TODO:  OO
    #
    #
    #     xyzObj = get_Obj_coords(self, coords, precess=precess)
    #     xyzSun, xyzEarth = get_Earth_Sun_coords(self)
    #
    #     rd = romer_delay(xyzEarth, xyzObj)
    #     ed = einstein_delay(self.tt.jd)
    #     shd = shapiro_delay(xyzEarth, xyzSun)
    #
    #     #print( 'Romer delay', rd )
    #     #print( 'Einstein delay', ed )
    #     #print( 'Shapiro delya', shd )
    #
    #     return self.tdb.jd - rd - ed - shd

    # def __add__(self, other):
    #     print( 'INTERCEPTING ADD' )
    #     ts = super(Time, self).__add__(other)
    #     format = ts.format
    #     scale = ts.scale
    #     precision = ts.precision
    #
    #     embed()
    #
    #     t = Time(ts.jd1, ts.jd2, format=ts.format, scale=scale,
    #               precision=precision, copy=False, location=self.location)
    #     #gives it the additional methods defined above
    #     return getattr(t.replicate(format=format), scale)

    # def __sub__(self, other):
    #     print( 'INTERCEPTING SUB!' )
    #     ts = super(Time, self).__sub__(other)
    #     format = ts.format
    #     scale = ts.scale
    #     precision = ts.precision
    #     t = Time(ts.jd1, ts.jd2, format=ts.format, scale=scale,
    #              precision=precision, copy=False, location=self.location)
    #     gives it the additional methods defined above
    #     return getattr(t.replicate(format=format), scale)


# ******************************************************************************
def timingFactory(cube):
    if cube.needs_timing:
        return shocTimingOld
    return shocTimingNew


def unknown():
    pass


# ******************************************************************************
class NoGPSTriggerProvided(Exception):
    pass


class Duration(float, HMSrepr):
    pass


class Trigger(object):
    def __init__(self, header):
        self.mode = header['trigger']
        self.start = header.get('gpsstart')
        # self.start is now either None (older SHOC data), or a str representing
        # start time

    def __str__(self):
        return self.mode

    def set(self, t0=None, triggers_in_local_time=True):
        """convert trigger time to seconds from midnight UT"""

        # NOTE: GPS triggers are assumed to be in SAST.  If they are provided in
        # UT pass triggers_in_local_time = False
        # get time zone info
        timezone = -7200 if triggers_in_local_time else 0
        if self.start is None:
            if t0 is None:
                raise ValueError('Please provide the GPS trigger (start) time.')
        else:
            t0 = self.start

        trigsec = Angle(t0, 'h').to('arcsec').value / 15.
        trigsec += timezone
        # trigger.start now in sec UTC from midnight

        # adjust to positive value -- this needs to be done (since tz=-2) so
        # we don't accidentally shift the date. (since we are measuring time
        # from midnight on the previous day DATE/FRAME in header)
        if trigsec < 0:
            trigsec += 86400

        self.start = TimeDelta(trigsec, format='sec')

    def is_internal(self):
        return self.mode == 'Internal'

    def is_external(self):
        return self.mode.startswith('External')

    is_gps = is_external

    def is_external_start(self):
        return self.is_gps() and self.mode.endswith('Start')
    # is_gps_all = is_external_start


# ******************************************************************************
class shocTimingBase(object):
    # set location
    # self.location = EarthLocation.of_site(location)
    # TODO: HOW SLOW IS THIS COMPARED TO ABOVE LINE???
    sutherland = EarthLocation(lat=-32.376006, lon=20.810678, height=1771)

    # TODO:  specify required timing accuracy --> decide which algorithms to use based on this!
    def __init__(self, cube, location=None):
        """
        Do timing corrections on shocObs.

        UTC : Universal Coordinate Time (array)
        LMST : Local mean sidereal time
        JD : Julian Date
        BJD : Barycentric julian day

        Parameters
        ----------
        None

        Returns
        ------
        """

        # NOTE:
        # Older SHOC data (pre 2016)
        # --------------------------
        # Times recorded in FITS header are as follows:
        #   Mode: 'Internal':
        #     * DATE-OBS not recorded
        #     * (FRAME, DATE) Time at the end of the first exposure (file
        #       creation timestamp)
        #     * The time here is rounded to the nearest second of computer clock
        #       ==> uncertainty of +- 0.5 sec (for absolute timing)
        #
        #   Mode: 'External Start':
        #     * EXPOSURE - exposure time (sec)
        #     * KCT, DATE-OBS **not recorded**
        #     * start time **not recorded**
        #
        #   Mode: 'External':
        #     * KCT, DATE-OBS **not recorded**
        #     * EXPOSURE - erroneously stores total accumulated exposure time

        # Recent SHOC data (post software upgrade)
        # ----------------------------------------
        #   Mode: 'Internal':
        #     * DATE-OBS  - start time accurate to microsecond
        #   Mode: 'External [Start]':
        #     * GPSSTART  - GPS start time (UTC; external)
        #     * KCT       - Kinetic Cycle Time
        #   Mode: 'External':
        #     * GPS-INT   - GPS trigger interval (milisecond)

        self.filename = cube.get_filename()
        self.header = header = cube[0].header
        self.data = None  # calculated in `set` method
        self.location = location or self.sutherland

        # Timing trigger mode
        self.trigger = Trigger(header)

        # Date (midnight UT)
        date_str = header['date'].split('T')[0]  # or FRAME
        self.utdate = Time(date_str)
        # this should at least be the correct date!

        # exposure time
        self.texp, self.kct = self.get_kct()
        self.nframes = cube.shape[-1]
        # NOTE:
        # The attributes `exp` and `kct` will be None only if we need the
        # user to provide them explicitly (applicable only to older SHOC data)

        # stamps
        self._t0 = None
        self._t0mid = None
        self._td_kct = None

    def get_kct(self):
        return self.header['exposure'], self.header['kct']

    def get_tdead(self):
        """dead time (readout) between exposures in s"""
        # NOTE:
        # deadtime should always the same value unless the user has (foolishly)
        # changed the vertical clock speed.
        # TODO: MAYBE CHECK stack_header['VSHIFT']
        # WARNING:
        # EDGE CASE: THE DEADTIME MAY BE LARGER IF WE'RE NOT OPERATING IN FRAME
        # TRANSFER MODE!
        return 0.00676

    # NOTE:
    # the following attributes are accessed as properties to account for the
    # case of (old) GPS triggered data in which the frame start time and kct
    # are not available upon initialization (they are missing in the header).
    # If missing they will be calculated upon accessing the attribute.

    @property
    def duration(self):
        """Duration of the observation"""
        if self.kct:
            return Duration(self.nframes * self.kct)

    @property
    def t0(self):
        """time stamp for the 0th frame start"""
        if self._t0 is None:
            self._t0, self._td_kct = self.get_time_data()
        return self._t0

    @property
    def t0repr(self):
        """
        Representative str for `t0`. For printing run info before timestamp
        has been set.

        eg: '2015-02-24 18:15:36.5'
        or: 'unknown' if GPS trigger needs to be set by user (old SHOC data)

        """
        try:
            dt = self.t0.to_datetime()
            # 1 decimal precision for seconds
            s = round(dt.second + dt.microsecond / 1e6, 1)
            ds = ('%.1f' % (s - int(s))).lstrip('0')
            return dt.strftime('%Y-%m-%d %H:%M:%S') + ds

        except NoGPSTriggerProvided as err:
            from motley import codes
            return codes.apply('unknown', 'r')

    @property
    def t0mid(self):
        """time stamp for the 0th frame mid exposure"""
        if self._t0mid is None:
            self._t0mid = self.t0 + 0.5 * self.td_kct
        return self._t0mid

    @property
    def td_kct(self):
        """Kinetic Cycle Time as a astropy.time.TimeDelta object"""
        if self._t0 is None:
            self._t0, self._td_kct = self.get_time_data()
        return self._td_kct

    def get_time_data(self):  # TODO: rename
        """
        Return the mid time of the first frame in UTC and the cycle time
        (exposure + dead time) as TimeDelta object
        """
        # new / fixed data!  Rejoice!
        header = self.header
        tStart = header['DATE-OBS']
        # NOTE: This keyword is confusing (UTC-OBS would be better), but since
        # it is now in common  use, we (reluctantly) do the same.
        # time for start of first frame
        t0 = Time(tStart, format='isot', scale='utc', precision=9,
                  location=self.location)
        # note: TimeDelta has higher precision than Quantity
        td_kct = TimeDelta(self.kct, format='sec')
        return t0, td_kct

    def set(self, t0, iers_a=None, coords=None):  # TODO: corrections):

        # create time stamps
        timeDeltaSequence = self.td_kct * np.arange(self.nframes, dtype=float)
        t = t0 + timeDeltaSequence
        # `t` is an astropy.time.Time instance containing time stamps for all
        # frames in the cube

        # set leap second offset from most recent IERS table
        delta, status = t.get_delta_ut1_utc(iers_a, return_status=True)
        if np.any(status == -2):
            warn('Using predicted leap-second values from IERS.')
        t.delta_ut1_utc = delta

        # initialize array for timing data
        # TODO: external control for which are calculated
        self.data = timeData = np.recarray(len(t), dtype=[('utdate', 'U20'),
                                                          ('uth', float),
                                                          ('utsec', float),
                                                          ('utc', 'U30'),
                                                          ('utstr', 'U20'),
                                                          ('lmst', float),
                                                          ('jd', float),
                                                          ('gjd', float),
                                                          ('bjd', float),
                                                          ('altitude', float),
                                                          ('airmass', float)])
        # compute timestamps for various scales
        # timeData.texp           = texp
        timeData.utc = t.utc.isot
        # UTC in decimal hours for each frame
        uth = (t.utc - self.utdate).to('hour')
        timeData.uth = uth.value
        timeData.utsec = uth.to('s').value

        # split UTDATE and UTC time
        utdata = t.isosplit()
        timeData.utdate = utdata['utdate']
        timeData.utstr = utdata['utc']

        # LMST for each frame
        lmst = t.sidereal_time('mean', longitude=self.location.longitude)
        timeData.lmst = lmst
        # timeData.last          = t.sidereal_time('apparent', longitude=lon)

        # Julian Dates
        timeData.jd = t.jd
        # timeData.ljd           = np.floor(timeData.jd)
        timeData.gjd = t.tcg.jd  # geocentric julian date

        # do barycentrization
        if (coords is not None) and (
                self.location is not None):  # verbose syntax to avoid  #quantity.py:892: FutureWarning
            # barycentric julian date (with light travel time corrections)
            bjd = light_time_corrections(t, coords, precess='first',
                                         abcorr=None)
            timeData.bjd = bjd

            # altitude and airmass
            timeData.altitude = altitude(coords.ra.radian,
                                         coords.dec.radian,
                                         lmst.radian,
                                         self.location.latitude.radian)
            timeData.airmass = Young94(np.pi / 2 - timeData.altitude)

        return timeData

    def export(self, filename, with_slices=False, count=0):  # single_file=True,
        """write the timing data for the stack to file(s)."""

        def make_header_line(info, fmt, delimiter):
            import re
            matcher = re.compile('%-?(\d{1,2})')
            padwidths = [int(matcher.match(f).groups()[0]) for f in fmt]
            padwidths[0] -= 2
            colheads = [s.ljust(p) for s, p in zip(info, padwidths)]
            return delimiter.join(colheads)

        # print( 'Writing timing data to file...' )
        TKW = ['utdate', 'uth', 'utsec', 'lmst', 'altitude', 'airmass',
               'jd', 'gjd', 'bjd']
        fmt = ('%-10s', '%-12.9f', '%-12.6f', '%-12.9f', '%-12.9f', '%-12.9f',
               '%-18.9f', '%-18.9f', '%-18.9f')
        formats = dict(zip(TKW, fmt))

        table = aTable(self.data[TKW])

        # if with_slices:
        #     # TKW     = ['filename'] + TKW
        #     # fmt     = ('%-35s',) + fmt
        #     slices = np.fromiter(map(os.path.basename, self.real_slices), 'U35')
        #     formats['filename'] = '%-35s'
        #     table.add_column(Column(slices, 'filename'), 0)

        delimiter = '\t'
        # timefile = self.get_filename(1, 0, 'time')
        table.write(filename,
                    delimiter=delimiter,
                    format='ascii.commented_header',
                    formats=formats,
                    overwrite=True)

        # if single_file:
        # Write all timing data to a single file
        # delimiter = ' '
        # timefile = self.get_filename(1, 0, 'time')
        # header = make_header_line( TKW, fmt, delimiter )
        # np.savetxt(timefile, T, fmt=fmt, header=header )

        # HACK! BECAUSE IRAF SUX
        # use_iraf = False
        # if use_iraf:
        #     link_to_short_name_because_iraf_sux(timefile, count, 'time')

        # else:
        # for i, tkw in enumerate(TKW):
        ##write each time sequence to a separate file...
        # fn = '{}.{}'.format(self.get_filename(1,0), tkw)
        # if tkw in TKW_sf:
        # if fn.endswith('uth'): fn.replace('uth', 'utc')
        # np.savetxt( fn, T[i], fmt='%.10f' )

    def stamp(self, j, t0=None, coords=None):
        """Timestamp the header"""
        # FIXME: repeat print not necessary
        logging.info('Updating the starting times for datacube %s ...',
                     self.filename)

        header = self.header
        timeData = self.data

        from IPython import embed
        logging.debug('timing.stamp' * 50)
        embed()

        # update timestamp in header
        header['utc-obs'] = (timeData.uth[j], 'Start of frame exposure in UTC')
        header['LMST'] = (timeData.lmst[j], 'Local Mean Sidereal Time')
        header['UTDATE'] = (timeData.utdate[j], 'Universal Time Date')

        header['JD'] = (timeData.jd[j], 'Julian Date (UTC)')
        # header['LJD']      = ( timeData.ljd[j], 'Local Julian Date' )
        header['GJD'] = (timeData.gjd[j], 'Geocentric Julian Date (TCG)')

        if coords:
            header['BJD'] = (timeData.bjd[j], 'Barycentric Julian Date (TDB)')
            header['AIRMASS'] = (timeData.airmass[j], 'Young (1994) model')
            # TODO: set model name dynamically

        # elif j!=0:
        # warn( 'Airmass not yet set for {}!\n'.format( self.get_filename() ) )

        # header['TIMECORR'] = ( True, 'Timing correction done' )        #imutil.hedit(imls[j], 'TIMECORR', True, add=1, ver=0)                                       #Adds the keyword 'TIMECORR' to the image header to indicate that timing correction has been done
        # header.add_history('Timing information corrected at %s' %str(datetime.datetime.now()), before='HEAD' )            #Adds the time of timing correction to header HISTORY

    # def set_airmass( self, coords=None, lat=-32.376006 ):
    # """Airmass"""
    # if coords is None:
    # header = self[0].header
    # ra, dec = header['ra'], header['dec']
    # coords = SkyCoord( ra, dec, unit=('h', 'deg') )     #, system='icrs'

    # ra_r, dec_r = coords.ra.radian, coords.dec.radian
    # lmst_r = self.data.lmst.radian
    # lat_r = np.radians(lat)

    # self.data.altitude = altitude( coords.ra.radian,
    # coords.dec.radian,
    # self.data.lmst.radian,

    # dec_r, lmst_r, lat_r)
    # z = np.pi/2 - self.altitude
    # self.data.airmass = Young94(z)


# ******************************************************************************
class shocTimingNew(shocTimingBase):
    pass


# ******************************************************************************
class shocTimingOld(shocTimingBase):
    def get_kct(self):

        stack_header = self.header

        if self.trigger.is_internal():
            # kinetic cycle time between start of subsequent exposures in sec.  i.e. exposure time + readout time
            t_kct = stack_header['KCT']
            t_exp = stack_header['EXPOSURE']
            # In internal triggering mode EXPOSURE stores the actual correct exposure time.
            #                       and KCT stores the Kinetic cycle time (dead time + exposure time)

        # GPS Triggering (External or External Start)
        elif self.trigger.is_gps():
            if self.trigger.is_external_start():  # External Start
                t_exp = stack_header[
                    'EXPOSURE']  # exposure time in sec as in header
                t_dead = self.get_tdead()
                t_kct = t_dead + t_exp  # Kinetic Cycle Time
            else:
                # trigger mode external - exposure and kct needs to be provided at terminal through -k
                return None, None
                #
                # t_kct = float(args.kct)                     #kct provided by user at terminal through -k
                # t_exp = t_kct - t_dead                      #set the 'EXPOSURE' header keyword

        return t_exp, t_kct

    def get_time_data(self):
        """
        Extract ralavent time data from the FITS header.

        Returns
        ------
        First frame mid time

        """
        header = self.header

        # date_str = header['DATE'].split('T')[0]
        # utdate = Time(date_str)     #this should at least be the correct date!

        t_exp, t_kct = self.texp, self.kct
        td_kct = TimeDelta(t_kct,
                           format='sec')  # NOTE: TimeDelta has higher precision than Quantity
        # td_kct = t_kct * u.sec

        if self.trigger.is_internal():
            # Initial time set to middle of first frame exposure
            # NOTE: this hardly matters for sub-second t_exp, as the time recorded
            #       in header FRAME is rounded to the nearest second
            # time for end of first frame
            utf = Time(header['DATE'],  # or FRAME
                       format='isot', scale='utc',
                       # use nano-second precision internally, although time
                       # stamp in header is only micro-second accuracy
                       precision=9,
                       location=self.location)
            t0mid = utf - 0.5 * td_kct  # mid time of first frame
            # return  tmid, td_kct

        if self.trigger.is_gps():
            if self.trigger.start:
                t0 = self.utdate + self.trigger.start
                t0 = Time(t0.isot, format='isot', scale='utc', precision=9,
                          location=self.location)
                t0mid = t0 + 0.5 * td_kct  # set t0 to mid time of first frame
            else:
                raise NoGPSTriggerProvided(
                        'No GPS triggers provided for %r. Please set '
                        'self.trigger.start' % self.filename)

        # stack_hdu.flush(output_verify='warn', verbose=1)
        # IF TIMECORR --> NO NEED FOR GPS TIMES TO BE GIVEN EXPLICITLY
        logging.debug('%s : TRIGGER is %s. tmid = %s; KCT = %s sec',
                      self.filename, self.trigger.mode.upper(), t0mid, t_kct)

        return t0mid, td_kct

    def stamp(self, j, t0=None, coords=None):

        shocTimingBase.stamp(self, j, t0, coords)

        header = self.header
        header['KCT'] = (self.kct, 'Kinetic Cycle Time')  # set KCT in header
        header['EXPOSURE'] = (self.texp, 'Integration time')  # set in header
        if t0:
            # also set DATE-OBS keyword
            header['DATE-OBS'] = str(t0)
            if self.trigger.is_gps():
                # Set correct (GPS triggered) start time in header
                header['GPSSTART'] = (str(t0), 'GPS start time (UTC; external)')
                # TODO: OR set empty?
