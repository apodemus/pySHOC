"""
Functions for time-stamping SHOC data and writing time stamps to FITS headers 

NOTE:
Older SHOC data (pre 2016)
--------------------------
Times recorded in FITS header are as follows:
  Mode: 'Internal':
    * KCT       - Kinetic Cycle Time   (exposure time + dead time)
    * DATE-OBS **not recorded**
    * (FRAME, DATE) Time at the end of the first exposure (file creation
      timestamp)
    * The time here is rounded to the nearest second of computer clock
      ==> uncertainty of +- 0.5 sec (for absolute timing)

  Mode: 'External Start':
    * EXPOSURE - exposure time (sec)
    * KCT, DATE-OBS **not recorded**
    * start time **not recorded**

  Mode: 'External':
    * KCT, DATE-OBS **not recorded**
    * EXPOSURE - **wrong**  erroneously stores total accumulated exposure time

Recent SHOC data (post software upgrade)
----------------------------------------
  Mode: 'Internal':
    * DATE-OBS  - start time accurate to microsecond
  Mode: 'External [Start]':
    * GPSSTART  - GPS start time (UTC; external)
    * KCT       - Kinetic Cycle Time
  Mode: 'External':
    * GPS-INT   - GPS trigger interval (milliseconds)

"""

from astropy.utils import lazyproperty
import logging
from pathlib import Path
from warnings import warn
from urllib.error import URLError

import numpy as np
from astropy import time
from astropy.table import Table as aTable
from astropy.constants import c, G, M_sun
from astropy.coordinates import EarthLocation, FK5
from astropy.coordinates.angles import Angle
import spiceypy as spice

from obstools.airmass import Young94, altitude
from recipes import pprint

# TODO: backends = {'astropy', 'astroutils', 'spice'}
# TODO: Q: are gps times in UTC / UT1 ??

# SHOC Exposure dead time in frame transfer mode
DEAD_TIME = 0.00676  # seconds
SECONDS_PER_DAY = 86400.

# set location
# self.location = EarthLocation.of_site('sutherland')
SUTHERLAND = EarthLocation(lat=-32.376006, lon=20.810678, height=1771)


# TODO: locations are slightly different for each telescope.  Exact GPS
#  locations listed in Shocnhelp
# Telescope    Latitude          Longitude         Altidude (m)
# 74inch       32 o 27.73’ S     20 o 48.70’ E     1822
# 40inch       32 o 22.78’ S     20 o 48.60’ E     1810
# 30inch       32 o 22.78’ S     20 o 48.63’ E     1811
# SALT ????

def iso_splitter(x):
    """worker for splitting date from time in ISOT time format string"""
    return tuple(x.split('T'))


def iso_split(t, dtype=[('date', 'U10'), ('utc', 'U18')]):
    """Split ISO time between date and time (from midnight)"""
    date_utc = np.fromiter(map(iso_splitter, np.atleast_1d(t.utc.isot)),
                           dtype)
    return date_utc


def iso_merge(date, utc, ):
    """Vectorized merging for date and time strings to make isot format strings"""
    return np.char.add(np.char.add(date, 'T'), utc)


def time_from_local_midnight(t, unit='s'):
    """
    get the TimeDelta since local midnight for the date of the first
    time stamp
    """
    date0, _ = iso_split(t[0])
    return (t.utc - Time(date0)).to(unit)


# def check_iers_table(self):
    # this no longer needed since astropy does this automatically now
#     from astropy.utils.iers import TIME_BEYOND_IERS_RANGE

#     delta, status = self.get_delta_ut1_utc(return_status=True)
#     beyond = (status == TIME_BEYOND_IERS_RANGE)

#     if np.any(beyond):
#         warn('{} / {} times are outside of range covered by IERS table.'
#                 ''.format(beyond.sum(), len(beyond)))
#     return beyond

class _UnknownTime(object):
    # singleton
    def __str__(self):
        return '??'

    def __add__(self, other):
        return UnknownTime

    def __bool__(self):
        return False


# singleton
UnknownTime = _UnknownTime()


class NoGPSTime(Exception):
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

    def __str__(self):
        return self.mode[:2] + '.'

    def set(self, t0=None, triggers_in_local_time=True):
        """
        convert trigger time to seconds from midnight UT

        GPS triggers are assumed to be in SAST.  If they are provided in
        UT use `triggers_in_local_time=False`


        Parameters
        ----------
        t0
        triggers_in_local_time

        Returns
        -------

        """

        #
        # get time zone info
        timezone = -7200 if triggers_in_local_time else 0
        if self.start is None:
            if t0 is None:
                raise ValueError(
                    'Please provide the GPS trigger (start) time.')
        else:
            t0 = self.start

        # Convert trigger time to seconds from midnight
        sec = Angle(t0, 'h').to('arcsec').value / 15.
        sec += timezone
        # trigger.start now in sec UTC from midnight

        # adjust to positive value -- this needs to be done (since tz=-2) so
        # we don't accidentally shift the date. (since we are measuring time
        # from midnight on the previous day DATE/FRAME in header)
        if sec < 0:
            sec += SECONDS_PER_DAY

        self.start = TimeDelta(sec, format='sec')

    def is_internal(self):
        return self.mode == 'Internal'

    def is_gps(self):
        return self.mode.startswith('External')

    def is_gps_start(self):
        return self.mode.endswith('Start')

    def is_gps_loop(self):
        return self.is_gps() and not self.is_gps_start()


# ******************************************************************************
class shocTimingBase(object):
    """
    Timestamps and corrections for SHOC data.
    """

    # TODO:  specify required timing accuracy --> decide which algorithms to
    #  use based on this!

    # def __new__(cls, hdu):
    #     if 'shocOld' in hdu.__class__.__name__:
    #         return super(shocTimingBase, cls).__new__(shocTimingOld)
    #     else:
    #         return super(shocTimingBase, cls).__new__(shocTimingNew)

    def __init__(self, hdu, location=SUTHERLAND):
        # NOTE:
        # Older SHOC data (pre 2016)
        # --------------------------
        # Times recorded in FITS header are as follows:
        #   Mode: 'Internal':
        #     * KCT       - Kinetic Cycle Time   (exposure time + dead time
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
        #     * GPS-INT   - GPS trigger interval (milliseconds)

        self._hdu = hdu
        self.header = header = hdu.header
        self.data = None  # calculated in `set` method
        self.location = location

        # Timing trigger mode
        self.trigger = Trigger(header)

        # Date
        # use FRAME here since always available in header
        date_str = header['FRAME'].split('T')[0]  # TODO: iso_split?
        self.ut_date = Time(date_str)
        # this should at least be the correct date!

        # exposure time
        # self.n_frames = hdu.shape[0]
        # NOTE:
        # The attributes `exp` and `kct` will be None only if we need the
        # user to provide them explicitly (applicable only to older SHOC data)

        # # stamps
        # self._t0 = None
        # self._t0mid = None
        # self._td_kct = None

    def gps_cycle_time(self):
        """
        For trigger mode 'External': Get repreat time (in seconds) for gps
        triggers from fits header
        """
        if self.trigger.is_gps_loop():
            return int(self.header['GPS-INT']) / 1000

    @lazyproperty
    def t_cycle(self):
        """
        Kinetic cycle time = Exposure time + Dead time
        """
        return self.gps_cycle_time() or self.header['KCT']

    @lazyproperty
    def t_expose(self):
        """
        Exposure time (integration time) for a single image

        """
        if self.trigger.is_gps_loop():
            return self.gps_cycle_time() - self.t_dead

        return self.header['EXPOSURE']

    t_exp = t_expose
    t_cyc = t_cycle

    @lazyproperty
    def t_dead(self):
        """dead time (readout) between exposures in s"""
        # NOTE:
        # deadtime should always the same value unless the user has (foolishly)
        # changed the vertical clock speed.
        # TODO: MAYBE CHECK stack_header['VSHIFT']
        # WARNING:
        # EDGE CASE: THE DEADTIME MAY BE LARGER IF WE'RE NOT OPERATING IN FRAME
        # TRANSFER MODE!
        return DEAD_TIME

    # @property
    # def gps(self):
    #     # add symbolic rep for trigger
    #     if self.trigger.is_internal():
    #         return ''
    #     if self.trigger.is_external():
    #         return ' →'
    #     if self.trigger.is_external_start():
    #         return ' ⟲'

    # NOTE:
    # the following attributes are accessed as properties to account for the
    # case of (old) GPS triggered data in which the frame start time and kct
    # are not available upon initialization (they are missing in the header).
    # If missing they will be calculated upon accessing the attribute.

    @property
    def duration(self):
        """Duration of the observation"""
        if self.t_cycle:
            return Duration(self._hdu.nframes * self.t_cycle)

    @lazyproperty
    def t0(self):
        """time stamp for the 0th frame start"""
        # noinspection PyAttributeOutsideInit
        t0, self.td_kct = self.get_time_data()
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
        if self.trigger.start is None:
            import motley
            return motley.red('??')
        else:
            dt = self.t0.to_datetime()
            # 1 point decimal precision for seconds
            s = round(dt.second + dt.microsecond / 1e6, 1)
            ds = ('%.1f' % (s - int(s))).lstrip('0')
            t = dt.strftime('%Y-%m-%d %H:%M:%S') + ds
            return t

    @lazyproperty
    def t0mid(self):
        """time stamp for the 0th frame mid exposure"""
        return self.t0 + 0.5 * self.td_kct

    @lazyproperty
    def td_kct(self):
        """Kinetic Cycle Time as a astropy.time.TimeDelta object"""
        # noinspection PyAttributeOutsideInit
        self.t0, td_kct = self.get_time_data()
        return td_kct

    def get_time_data(self):  # TODO: rename
        """
        Return the mid time of the first frame in UTC and the cycle time
        (exposure + dead time) as TimeDelta object
        """
        # new / fixed data!  Rejoice!
        header = self.header
        t_start = header['FRAME']

        # DATE-OBS
        # This keyword is confusing (UTC-OBS would be better), but since
        #  it is now in common  use, we (reluctantly) do the same.

        # time for start of first frame
        t0 = Time(t_start, format='isot', scale='utc', precision=9,
                  location=self.location)
        # TimeDelta has higher precision than Quantity
        td_kct = TimeDelta(self.t_cycle, format='sec')
        return t0, td_kct

    def get(self, t0, coords=None):
        """
        Get time stamps per frame
        """

        # TODO: corrections):

        # create time stamps
        # `t` is an `astropy.time.Time` instance containing time stamps for all
        # frames in the cube
        time_deltas = self.td_kct * np.arange(self._hdu.nframes, dtype=float)
        t = t0 + time_deltas

        # UTC : Universal Coordinate Time (array)
        # LMST : Local mean sidereal time
        # JD : Julian Date
        # BJD : Barycentric julian day

        # initialize array for timing data
        # TODO: external control for which are calculated
        self.data = times = np.recarray(len(t), dtype=[
            # ('utdate', 'U20'),
            # ('uth', float),
            # ('utsec', float),
            ('utc', 'U30'),
            # ('utstr', 'U20'),
            ('lmst', float),
            ('jd', float),
            # FIXME: will not compute if coordinates not set
            ('hjd', float),
            ('bjd', float),
            ('altitude', float),
            ('airmass', float)])
        # compute timestamps for various scales
        # times.texp           = texp
        times.utc = t.utc.isot
        # UTC in decimal hours for each frame
        # uth = (t.utc - self.utdate).to('hour')
        # times.uth = uth.value
        # times.utsec = uth.to('s').value

        # split UTDATE and UTC time
        # times.utdate, times.utstr = t.isosplit()

        # LMST for each frame
        lmst = t.sidereal_time('mean', longitude=self.location.lon)
        times.lmst = lmst
        # times.last          = t.sidereal_time('apparent', longitude=lon)

        # Julian Dates
        times.jd = t.jd
        # times.ljd           = np.floor(times.jd)
        # times.gjd = t.tcg.jd  # geocentric julian date

        # do barycentrization
        if (coords is not None) and (self.location is not None):
            # barycentric julian date [BJD(TDB)]
            #   (includes light travel time corrections)
            times.bjd = t.bjd(coords).jd

            # heliocentric julian date [HJD(TCB)]
            #   (includes light travel time corrections)
            times.hjd = t.hjd(coords).jd

            # altitude and airmass
            times.altitude = altitude(coords.ra.radian,
                                      coords.dec.radian,
                                      lmst.radian,
                                      self.location.lat.radian)
            times.airmass = Young94(np.pi / 2 - times.altitude)

        return times

    def export(self, filename,  delimiter=' '):
        """write the timing data for the stack to file(s)."""

        def make_header_line(info, fmt, delimiter):
            import re
            matcher = re.compile('%-?(\d{1,2})')
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

    def stamp(self, j, t0=None, coords=None):
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
        times = self.data

        # update timestamp in header
        header['UTC-OBS'] = (times.utc[j], 'Start of frame exposure in UTC')
        header['LMST'] = (times.lmst[j], 'Local Mean Sidereal Time')
        # header['UTDATE'] = (times.utdate[j], 'Universal Time Date')

        header['JD'] = (times.jd[j], 'Julian Date (UTC)')
        # header['HJD'] = (times.gjd[j], 'Geocentric Julian Date (TCG)')
        # header['LJD']      = ( times.ljd[j], 'Local Julian Date' )

        if coords:
            header['HJD'] = (times.hjd[j], 'Heliocentric Julian Date (TDB)')
            header['BJD'] = (times.bjd[j], 'Barycentric Julian Date (TDB)')
            header['AIRMASS'] = (times.airmass[j], 'Young (1994) model')
            # TODO: set model name dynamically

        # Add info to header HISTORY
        header.add_history(f'pySHOC: added time stamps at {Time.now()}',
                           before='HEAD')


class shocTimingNew(shocTimingBase):
    pass


class shocTimingOld(shocTimingBase):
    @lazyproperty
    def t_cycle(self):
        return self.header.get('KCT', self.t_dead + self.t_expose)

    @lazyproperty
    def t_expose(self):
        return self.header.get('EXPOSURE', UnknownTime)

        # if self.trigger.is_internal():
        #     # kinetic cycle time between start of subsequent exposures in sec.
        #     # i.e. exposure time + readout time
        #     t_cycle = header['KCT']
        #     self.t_expose = header['EXPOSURE']
        #     # In internal triggering mode EXPOSURE stores the actual correct
        #     # exposure time. KCT stores the Kinetic cycle time
        #     # (dead time + exposure time)
        #
        #
        # # GPS Triggering (External or External Start)
        # elif self.trigger.is_gps():
        #     if self.trigger.is_external_start():  # External Start
        #         # exposure time in sec as in header
        #         self.t_expose = header['EXPOSURE']
        #         t_cycle = self.t_dead + self.t_expose  # Kinetic Cycle Time
        #     else:
        #         # trigger mode external - exposure and kct needs to be provided
        #         # at terminal
        #         t_exp, t_kct = None, None
        #
        # return t_exp, t_kct

    def get_time_data(self):
        """
        Extract ralavent time data from the FITS header.

        Returns
        ------
        First frame mid time

        """

        # TODO: option to do flux weighted time stamps!!

        header = self.header

        # date_str = header['DATE'].split('T')[0]
        # utdate = Time(date_str)     #this should at least be the correct date!

        td_kct = TimeDelta(self.t_cycle, format='sec')
        # TimeDelta has higher precision than Quantity

        if self.trigger.is_internal():
            # Initial time set to middle of first frame exposure
            #  this hardly matters for sub-second t_exp, as the time
            #  recorded in header FRAME is rounded to the nearest second
            # time for end of first frame
            utf = Time(header['DATE'],  # or FRAME
                       format='isot', scale='utc',
                       precision=6,
                       # this is the decimal significant figures for formatting
                       # and does not represent absolute precision
                       location=self.location)
            t0mid = utf - 0.5 * td_kct  # mid time of first frame
            # return  tmid, td_kct

        else:  # self.trigger.is_gps():
            if self.trigger.start:
                t0 = self.ut_date + self.trigger.start
                t0 = Time(t0.isot, format='isot', scale='utc', precision=9,
                          location=self.location)
                t0mid = t0 + 0.5 * td_kct  # set t0 to mid time of first frame
            else:
                raise NoGPSTime(
                    'No GPS triggers available for %r. Please set '
                    'self.trigger.start' % self._hdu.filepath.name)

        # stack_hdu.flush(output_verify='warn', verbose=1)
        logging.debug('%s : TRIGGER is %s. tmid = %s; KCT = %s sec',
                      self._hdu.filepath.name, self.trigger.mode.upper(),
                      t0mid, self.t_cycle)

        return t0mid, td_kct

    def stamp(self, j, t0=None, coords=None):

        #
        shocTimingBase.stamp(self, j, t0, coords)

        # set KCT / EXPOSURE in header
        header = self.header
        header['KCT'] = (self.t_cycle, 'Kinetic Cycle Time')
        header['EXPOSURE'] = (self.t_expose, 'Integration time')

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
