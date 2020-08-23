"""
Functions for time-stamping SHOC data and writing time stamps to FITS headers
"""


import logging
# from pathlib import Path
# from warnings import warn
# from urllib.error import URLError

import numpy as np
from astropy import time
from astropy.table import Table as aTable
from astropy.coordinates import EarthLocation
import astropy.units as u
from astropy.coordinates.angles import Angle
from astropy.utils import lazyproperty
# from astroplan import Observer
# import spiceypy as spice

from obstools.airmass import Young94, altitude
from recipes import pprint, memoize


# TODO: backends = {'astropy', 'astroutils', 'spice'}
# TODO: Q: are gps times in UTC / UT1 ??

# SHOC Exposure dead time in frame transfer mode
DEAD_TIME = 0.00676  # seconds           header['vshift'] * 1024 ????

# set location
# self.location = EarthLocation.of_site('SAAO')
SUTHERLAND = EarthLocation(lat=-32.376006, lon=20.810678, height=1771)
TIMEZONE = +2 * u.hour  # SAST is UTC + 2
# SUTH_OBS = Observer(SUTHERLAND)

# TODO: locations are slightly different for each telescope.  Exact GPS
#  locations listed in Shocnhelp
# Telescope    Latitude          Longitude         Altidude (m)
# 74inch       32 o 27.73’ S     20 o 48.70’ E     1822
# 40inch       32 o 22.78’ S     20 o 48.60’ E     1810
# Lesedi       32 o 22.78’ S     20 o 48.63’ E     1811
# SALT ????


def iso_split(t, dtype=[('date', 'U10'), ('utc', 'U18')]):
    """Split ISO time between date and time (from midnight)"""
    assert isinstance(t, Time)
    return np.array(np.char.split(t.isot, 'T').tolist())


def iso_merge(date, utc, sep='T'):
    """
    Vectorized merging for date and time strings to make isot format strings
    """
    return np.char.add(np.char.add(date, sep), utc)


def time_from_local_midnight(t, unit='s'):
    """
    get the TimeDelta since local midnight for the date of the first
    time stamp
    """
    date0, _ = iso_split(t[0]).T
    return (t.utc - Time(date0)).to(unit)


class _UnknownTime(object):
    # singleton
    def __str__(self):
        return '??'

    def __add__(self, other):
        return UnknownTime

    def __bool__(self):
        return False

# class UnknownStartTime(_UnknownTime):
#     def __add__(self, other):
#         return UnknownTime


# singleton
UnknownTime = _UnknownTime()


class NoGPSTime(Exception):
    pass


class UnknownLocation(Exception):
    pass


class UnknownPointing(Exception):
    pass


class HMSrepr(object):
    """
    Mixin class that provided numerical objects with `hms` property for pretty
    representation
    """

    @property
    def hms(self):
        return pprint.hms(self)


class Duration(float, HMSrepr):
    pass


class Trigger(object):
    def __init__(self, header):
        self.mode = header['trigger']
        self.start = header.get('gpsstart', None) or UnknownTime
        # self.start is either ISO format str, or `UnknownTime` (old SHOC data)
        # FIXME: it's confusing keeping the start time here AND in timing.t0

    def __str__(self):
        return self.mode[:2] + '.'

    def __repr__(self):
        return f'{self.__class__}:{self.mode}'

    def set(self, t0=None, sast=True):
        """
        Set GPS trigger start time for the observation. For most cases, this
        function can be called without arguments. You will only need to explicitly 
        provide the GPS trigger times for older SHOC data where these were not
        recorded in the fits headers.  In this case input values are assumed to
        be in SAST. If you are providing UTC times use the `sast=False` flag.

        Parameters
        ----------
        t0 : str, optional
            A string representing the time: eg: '04:20:00'. If not given, the
            GPS starting time recorded in the fits header will be used. 
        sast : bool, optional
            Flag indicating that the times are given in local SAST time, by
            default True

        Raises
        ------
        ValueError
            If the GPS trigger time GPSSTART is not given and cannot be found in
            the fits header
        """

        #
        if self.start is None:
            if t0 is None:
                raise ValueError(
                    'Please provide the GPS trigger (start) time.')
        else:
            t0 = self.start

        # get time zone info
        tz = 2 if sast else 0
        # Convert trigger time to seconds from midnight
        h = Angle(t0, 'h').h - tz
        # trigger.start now in sec UTC from midnight

        # adjust to positive value -- this needs to be done (since tz=-2) so
        # we don't accidentally shift the date. (since we are measuring time
        # from midnight on the previous day DATE/FRAME in header)
        sec = (h + ((h <= 0) * 24)) * 3600
        self.start = TimeDelta(sec, format='sec')

    def is_internal(self):
        """Check if trigger mode is 'Internal'"""
        return self.mode == 'Internal'

    def is_gps(self):
        """
        Check if GPS was used to trigger the exposure, either the first frame
        only, or each frame
        """
        return self.mode.startswith('External')

    def is_gps_start(self):
        """
        Check if GPS was used to start the exposure sequence.
        ie. mode is 'External Start'
        """
        return self.mode.endswith('Start')

    def is_gps_loop(self):
        """
        Check if GPS was used to trigger every exposure in the stack.
        ie. mode is 'External'
        """
        return self.is_gps() and not self.is_gps_start()

    @property
    def symbol(self):
        # add symbolic rep for trigger
        if self.is_internal():
            return ''
        if self.is_external():
            return '⟲'
        if self.is_external_start():
            return '↓'


# ******************************************************************************
class shocTimingBase(object):
    """
    Time stamps and corrections for SHOC data.

    Below is a summary of the timing information available in the SHOC fits
    headers

    Recent SHOC data (post software upgrade)
    ----------------------------------------
      Mode: 'Internal':
        * DATE-OBS
            - The time the user pushed start button (UTC)
        * FRAME
            - Time at the **end** of the first exposure
        * KCT
            - Kinetic Cycle Time   (exposure time + dead time)
            - **Not recorded** for single frame exposures: ACQMODE == 'Single Shot'

      Mode: 'External Start':
        * DATE-OBS
            - **NB** This is not the time stamp for the first image frame if the
              observations are GPS triggered
        * FRAME
            - as above
        * GPSSTART
            - GPS start time (UTC)
        * KCT
            - **not recorded**

      Mode: 'External':
        * GPSSTART, KCT, DATE-OBS, FRAME
            - as above
        * GPS-INT   - GPS trigger interval (milliseconds)


    Older SHOC data (pre 2015)
    --------------------------
      Mode: 'Internal':
        * KCT
            - Kinetic Cycle Time   (exposure time + dead time)
        * DATE-OBS
            - **not recorded**
        * FRAME, DATE
           - Time at the **end** of the first exposure (file creation timestamp)
           - The time here is rounded to the nearest second of computer clock
             ==> uncertainty of +- 0.5 sec (for absolute timing)

      Mode: 'External Start':
        * EXPOSURE
            - exposure time (sec)
        * KCT, DATE-OBS
            - **not recorded**
        * FRAME, DATE-OBS
            - **not recorded**

      Mode: 'External':
        * KCT, DATE-OBS
            **not recorded**
        * EXPOSURE
            **wrong**  erroneously stores total accumulated exposure time
    """

    # TODO: get example set of all mode + old / new combo + print table
    # TODO Then print with pySHOC.core.print_timing_table(run)

    # TODO:  specify required timing accuracy --> decide which algorithms to
    #  use based on this!

    # TODO: option to do flux weighted time stamps!!

    # def __new__(cls, hdu):
    #     if 'shocOld' in hdu.__class__.__name__:
    #         return super(shocTimingBase, cls).__new__(shocTimingOld)
    #     else:
    #         return super(shocTimingBase, cls).__new__(shocTimingNew)

    def __init__(self, hdu, location=SUTHERLAND):
        """
        Create the timing interface for a shocHDU

        Parameters
        ----------
        hdu : pySHOC.core.shocHDU

        location : astropy.coordinates.EarthLocation, optional
            Location of the observation (used for barycentric corrections), 
            by default SUTHERLAND
        """

        self._hdu = hdu
        self.header = header = hdu.header
        self.data = None  # calculated in `set` method
        self.location = location

        # Timing trigger mode
        self.trigger = Trigger(header)
        # self.delta = None  #
        # self.sigma = 0.001 # # timestamp uncertainty

        # Date
        # use FRAME here since always available in header
        date_str = header['FRAME'].split('T')[0]
        self.ut_date = Time(date_str)
        # FIXME: need this ?  self._hdu.date..
        # this should at least be the correct date!

    def __array__(self):
        return self.t

    @lazyproperty
    def expose(self):
        """
        Exposure time (integration time) for a single image

        """
        if self.trigger.is_gps_loop():
            return self.gps_cycle_time() - self.dead

        return self.header['EXPOSURE']

    @lazyproperty
    def kct(self):
        """
        Kinetic cycle time = Exposure time + Dead time
        """
        if self.trigger.is_internal():
            if self.header['ACQMODE'] == 'Single Scan':
                return self.header['EXPOSURE'] + self.dead
            return self.header['KCT']

        if self.trigger.is_gps_loop():
            return self.gps_cycle_time()

        if self.trigger.is_gps_start():
            return self.header['EXPOSURE'] + self.dead

        # this should never happen!
        raise ValueError('Unknown frame cycle time')

    exp = expose
    """exposure time"""
    cycle = kct
    """kinetic cycle time"""
    dead = DEAD_TIME
    """dead time (readout) between exposures in s"""
    # NOTE:
    # deadtime should always the same value unless the user has (foolishly)
    # changed the vertical clock speed.
    # TODO: MAYBE CHECK stack_header['VSHIFT']
    # EDGE CASE: THE DEADTIME MAY BE LARGER IF WE'RE NOT OPERATING IN FRAME
    # TRANSFER MODE!

    def gps_cycle_time(self):
        """
        For trigger mode 'External': Get repreat time (in seconds) for gps
        triggers from fits header
        For trigger mode 'Internal' or 'External Start' this will return None
        """
        if self.trigger.is_gps_loop():
            return int(self.header['GPS-INT']) / 1000

    @lazyproperty
    def t(self):
        """
        Create time stamps for all images in the stack. This returns the
        mid-exposure time stamps

        Returns
        -------
        Time
            Time object that derives from `astropy.time.Time` and holds all
            timestamps for the image stack
        """
        deltas = self.delta * np.arange(self._hdu.nframes, dtype=float)
        t0mid = self.t0 + 0.5 * self.delta
        return t0mid + deltas

    # NOTE:
    # the following attributes are accessed as properties to account for the
    # case of (old) GPS triggered data in which the frame start time and kct
    # are not available upon initialization (they are missing in the header).
    # If missing they will be calculated upon accessing the attribute.

    @property
    def duration(self):
        """Duration of the observation"""
        if self.kct:
            return Duration(self._hdu.nframes * self.kct)

    @lazyproperty
    def delta(self):
        """Kinetic Cycle Time (δt) as a astropy.time.TimeDelta object"""
        self.t0, delta = self.get_t0_kct()
        return delta

    @lazyproperty
    def t0(self):
        """time stamp for the 0th frame start"""
        # noinspection PyAttributeOutsideInit
        t0, self.delta = self.get_t0_kct()
        return t0

    @property
    def _t0_repr(self):
        """
        Representative str for `t0` the start time of an observation. For old
        SHOC data with External GPS triggering, this needs to be set by the
        user since it is not recorded in the headers. However, we want to be
        able to print info about printing run info before timestamp has been
        set.

        Examples
        --------
        Internal Trigger
            '2015-02-24 18:15:36.5'
        External Trigger
            '2015-02-24 18:15:36.5*'
        External Trigger (Old SHOC)
            '??'
        """
        # FIXME: can you eliminate this function by making
        #   t0 = UnknownTime()
        if self.trigger.start is None:
            import motley
            return motley.red('??')
        else:
            return self.t0.iso

    def get_t0_kct(self):  # TODO: rename
        """
        Return the start time of the first frame in UTC and the cycle time
        (exposure + dead time) as TimeDelta object
        """
        # new / fixed data!  Rejoice!
        header = self.header
        if self.trigger.is_gps():
            # GPS triggered
            t_start = self.trigger.start
        else:
            # internal triggered
            t_start = header['DATE-OBS']
            # NOTE: For new shoc data, DATE-OBS key is always present, and
            # a more accurate time stamp than FRAME, so we always prefer to use
            # that

        # DATE-OBS
        # This keyword is confusing (UTC-OBS would be better), but since
        #  it is now in common  use, we (reluctantly) do the same.

        # time for start of first frame
        t0 = Time(t_start, format='isot', scale='utc',
                  # NOTE output format precision not numerical precision
                  precision=1, location=self.location)
        # TimeDelta has higher numberical precision than Quantity
        delta = TimeDelta(self.kct, format='sec')
        return t0, delta

    @lazyproperty
    def lmst(self):
        """LMST for at frame mid exposure times"""
        return self.t.sidereal_time('mean', longitude=self.location.lon)
    
    @lazyproperty
    def hour(self):
        """UTC in units of hours"""
        return (self.t - self.ut_date).to('hour').value

    def _check_coords_loc(self):
        if self._hdu.coords is None:
            raise UnknownPointing

        if self.location is None:
            raise UnknownLocation

    @lazyproperty
    def bjd(self):
        """
        Barycentric julian date [BJD(TDB)] at frame mid exposure times
        (includes light travel time corrections)
        """
        return self.t.bjd(self._hdu.coords).jd

    @lazyproperty
    def hjd(self):
        """
        Heliocentric julian date [HJD(TCG)] at frame mid exposure times
        (includes light travel time corrections)
        """
        return self.t.hjd(self._hdu.coords).jd

    def airmass(self):
        """airmass of object at frame mid exposure times via Young 94"""
        return Young94(np.pi / 2 - self.altitude)

    def altitude(self):
        """altitude of object at frame mid exposure times"""
        self._check_coords_loc()
        coords = self._hdu.coords
        return altitude(coords.ra.radian,
                        coords.dec.radian,
                        self.lmst.radian,
                        self.location.lat.radian)

    def is_during_twilight(self):
        """
        Check if the entire observation takes place during twilight. This is
        helpful in determining if an observation is for flat fields.
        """
        from obstools.plan import Sun

        t0, t1 = self.t[[0, -1]]
        # pass a string 'sutherland' below instead of the module variable
        # SUTHERLAND since EarthLocation is not hashable and will therefore not
        # cache the result of the call below
        sun = Sun('SAAO', str(self._hdu.date))
        return (  # entire observation occurs during evening twilight
            np.all((sun.set < t0) & (t1 < sun.dusk.astronomical))
            or  # entire observation occurs during morning twilight
            np.all((sun.rise > t1) & (t0 > sun.dawn.astronomical))
        )

    def export(self, filename,  delimiter=' '):
        """write the timing data for the stack to file(s)."""

        def make_header_line(info, fmt, delimiter):
            import re
            matcher = re.compile(r'%-?(\d{1,2})')
            padwidths = [int(matcher.match(f).groups()[0]) for f in fmt]
            padwidths[0] -= 2
            colheads = [s.ljust(p) for s, p in zip(info, padwidths)]
            return delimiter.join(colheads)

        # print( 'Writing timing data to file...' )
        formats = {'utdate': '%-10s',
                   'uth': '%-12.9f',
                   'utsec': '%-12.6f',
                   'lmst': '%-12.9f',
                   'altitude': '%-12.9f',
                   'airmass': '%-12.9f',
                   'jd': '%-18.9f',
                   'gjd': '%-18.9f',
                   'bjd': '%-18.9f'}

        #
        table = aTable(self.data[tuple(formats.keys())])
        table.write(filename,
                    delimiter=delimiter,
                    format='ascii.commented_header',
                    formats=formats,
                    overwrite=True)

    def stamp(self, j):
        """
        Timestamp the header

        Parameters
        ----------
        j
        t0
        coords

        Returns
        -------

        """

        # FIXME: repeat print not necessary
        logging.info('Time stamping %s', self._hdu.filepath.name)

        header = self.header
        t = self.t[j]

        # update timestamp in header
        header['UTC-OBS'] = (t.utc.isot, 'Start of frame exposure in UTC')
        header['LMST'] = (self.lmst[j], 'Local Mean Sidereal Time')
        # header['UTDATE'] = (times.utdate[j], 'Universal Time Date')

        header['JD'] = (t.jd, 'Julian Date (UTC)')
        # header['HJD'] = (times.gjd[j], 'Geocentric Julian Date (TCG)')
        # header['LJD']      = ( times.ljd[j], 'Local Julian Date' )

        if not ((self._hdu.coords is None) or (self.location is None)):
            header['HJD'] = (self.hjd[j], 'Heliocentric Julian Date (TDB)')
            header['BJD'] = (self.bjd[j], 'Barycentric Julian Date (TDB)')
            header['AIRMASS'] = (self.airmass[j], 'Young (1994) model')
            # TODO: set model name dynamically

        # Add info to header HISTORY
        header.add_history(f'pySHOC: added time stamps at {Time.now()}',
                           before='HEAD')


class shocTimingNew(shocTimingBase):
    pass


class shocTimingOld(shocTimingBase):
    @lazyproperty
    def kct(self):
        return self.header.get('KCT', self.dead + self.exp)

    @lazyproperty
    def expose(self):
        return self.header.get('EXPOSURE', UnknownTime)

        # if self.trigger.is_internal():
        #     # kinetic cycle time between start of subsequent exposures in sec.
        #     # i.e. exposure time + readout time
        #     kct = header['KCT']
        #     self.expose = header['EXPOSURE']
        #     # In internal triggering mode EXPOSURE stores the actual correct
        #     # exposure time. KCT stores the Kinetic cycle time
        #     # (dead time + exposure time)
        #
        #
        # # GPS Triggering (External or External Start)
        # elif self.trigger.is_gps():
        #     if self.trigger.is_external_start():  # External Start
        #         # exposure time in sec as in header
        #         self.expose = header['EXPOSURE']
        #         kct = self.dead + self.expose  # Kinetic Cycle Time
        #     else:
        #         # trigger mode external - exposure and kct needs to be provided
        #         # at terminal
        #         t_exp, t_kct = None, None
        #
        # return t_exp, t_kct

    def get_t0_kct(self):
        """
        Extract ralavent time data from the FITS header.

        Returns
        ------
        First frame mid time

        """

        # TimeDelta has higher precision than Quantity
        delta = TimeDelta(self.kct, format='sec')

        # NOTE `precision` here this is the decimal significant figures for
        # formatting and does not represent absolute precision
        options = dict(format='isot', scale='utc',
                       location=self.location,
                       precision=1)

        if self.trigger.is_internal():
            # Initial time set to middle of first frame exposure
            # this hardly matters for sub-second t_exp, as the time
            # recorded in header FRAME is rounded to the nearest second
            # time for end of first frame

            # start time of first frame
            t0 = Time(self.header['DATE'], **options) - delta
            #               or FRAME (equivalent for OLD SHOC data)

        else:  # self.trigger.is_gps():
            if self.trigger.start:
                t0 = self.ut_date + self.trigger.start
                t0 = Time(t0.isot, **options)
            else:
                raise NoGPSTime(
                    'No GPS triggers available for %r. Please set '
                    'self.trigger.start' % self._hdu.filepath.name)

        # stack_hdu.flush(output_verify='warn', verbose=1)
        logging.debug('%s : TRIGGER is %s. tmid = %s; KCT = %s sec',
                      self._hdu.filepath.name, self.trigger.mode.upper(),
                      t0, self.kct)

        return t0, delta

    def stamp(self, j, t0=None, coords=None):
        #
        shocTimingBase.stamp(self, j)

        # set KCT / EXPOSURE in header
        header = self.header
        header['KCT'] = (self.kct, 'Kinetic Cycle Time')
        header['EXPOSURE'] = (self.exp, 'Integration time')

        if t0:
            # also set DATE-OBS keyword
            header['DATE-OBS'] = str(t0)
            if self.trigger.is_gps():
                # Set correct (GPS triggered) start time in header
                header['GPSSTART'] = (
                    str(t0), 'GPS start time (UTC; external)')
                # TODO: OR set empty?


class TimeDelta(time.TimeDelta):
    @property
    def hms(self):
        v = self.value
        precision = 1 if v > 10 else 3
        return pprint.hms(v, precision)


class Time(time.Time):
    """
    Extends the `astropy.time.core.Time` class to include a few more convenience
    methods
    """

    # TODO: HJD:

    def bjd(self, coords, location=None):
        """BJD(TDB)"""
        return self.tdb + self.light_travel_time(coords, 'barycentric',
                                                 location)

    def hjd(self, coords, location=None):
        """HJD(TCB)"""
        return self.tcb + self.light_travel_time(coords, 'heliocentric',
                                                 location)

    # @property
    # def tjd(self):

    #     return np.floor(times.jd)

    # TODO: rjd,

    @property
    def gjd(self):
        # geocentric julian date
        return self.tcg.jd

