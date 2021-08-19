"""
Functions for time-stamping SHOC data and writing time stamps to FITS headers
"""


# std libs
import inspect
import logging
import functools as ftl

# third-party libs
import numpy as np
import astropy.units as u
from astropy import time
from astropy.utils import lazyproperty
from astropy.table import Table as aTable
from astropy.coordinates import AltAz
from astropy.coordinates.angles import Angle

# local libs
import motley
from obstools.airmass import Young94, altitude
from recipes.oo import Null
from recipes.io import read_lines
from recipes.logging import LoggingMixin, logging, get_module_logger


# from pathlib import Path
# from warnings import warn
# import spiceypy as spice

# TODO: backends = {'astropy', 'astroutils', 'spice'}
# TODO: Q: are gps times in UTC / UT1 ??


# module level logger
logger = get_module_logger()
logging.basicConfig()
logger.setLevel(logging.INFO)


# --------------------------------- constants -------------------------------- #

# SHOC Exposure dead time in frame transfer mode
DEAD_TIME = 0.00676  # seconds           header['vshift'] * 1024 ????
TIMEZONE = +2 * u.hour  # SAST is UTC + 2

# ----------------------------- Helper Functions ----------------------------- #


def is_lazy(_):
    return isinstance(_, lazyproperty)


def iso_split(t, dtype=[('date', 'U10'), ('utc', 'U18')]):
    """Split ISO time between date and time (from midnight)"""
    assert isinstance(t, Time)
    return np.array(np.char.split(t.isot, 'T').tolist())


def iso_merge(date, utc, sep='T'):
    """
    Vectorize merging for date and time strings to make isot format strings
    """
    return np.char.add(np.char.add(date, sep), utc)


def time_from_local_midnight(t, unit='s'):
    """
    Get the elapsed time in seconds since local midnight on the date of the first
    time stamp in the sequence
    """
    date0, _ = iso_split(t[0]).T
    return (t.utc - Time(date0)).to(unit)


def timing_info_table(run):
    """prints the various timing keys in the headers"""
    from motley.table import Table

    keys = ['TRIGGER', 'DATE', 'DATE-OBS', 'FRAME',
            'GPSSTART', 'GPS-INT', 'KCT', 'EXPOSURE']
    tbl = [[type(obs).__name__, *(obs.header.get(key, '--') for key in keys)]
                for obs in run]
    return Table(tbl, chead=['TYPE'] + keys)


# -------------------------------- Exceptions -------------------------------- #


class UnknownTimeException(Exception):
    pass


class UnknownLocation(Exception):
    pass


class UnknownPointing(Exception):
    pass

# ------------------------------ Helper Classes ------------------------------ #


class _UnknownTime(Null):
    
    
    def __str__(self):
        return motley.red('??')

    def __add__(self, _):
        return self

    def __radd__(self, _):
        return self

    def __sub__(self, _):
        return self

    def __mul__(self, _):
        return self

    def __rmul__(self, _):
        return self
    
    def __neg__(self):
        return self


# singleton
UnknownTime = _UnknownTime()


# class HMSrepr:
#     """
#     Mixin class that provided numerical objects with `hms` property for pretty
#     representation
#     """

#     @property
#     def hms(self):
#         return ppr.hms(self, unicode=True, precision=1)


# class Duration(float, HMSrepr):``
#     pass

class GPSTimeStampMissing:
    def __str__(self):
        return motley.red('\N{WARNING SIGN}')


NO_GPS = GPSTimeStampMissing()


class Trigger:
    """
    Simple class representing the trigger mechanism that started the CCD
    exposure sequence
    """

    FLAG_INFO = {
        'flag': {
            '*':        'GPS Trigger',
            NO_GPS:     'GPSSTART missing - timestamp may be inaccurate'
        },
        'loop_flag': {
            '*':        'GPS Repeat'
        }
    }
    # SYMBOLS =     ''
    #       loop = '⟲'
    #       start = '↓'

    def __init__(self, header):
        self.mode = header['trigger']

        self.flag = '*' * int(self.is_gps)
        if self.is_gps and ('GPSSTART' not in header):
            self.flag = NO_GPS

        self.loop_flag = ''
        if self.is_gps_loop and ('GPS-INT' in header):
            self.loop_flag = '*'

    def __str__(self):
        return self.mode[:2] + '.'

    def __repr__(self):
        # {self.__class__.__name__}:
        # return f'{self.mode}: t0={self.start}'
        return f'{self.__class__.__name__}: {self.mode}'

    @lazyproperty
    def is_internal(self):
        """Check if trigger mode is 'Internal'"""
        return self.mode == 'Internal'

    @lazyproperty
    def is_gps(self):
        """
        Check if GPS was used to trigger the exposure, either the first frame
        only, or each frame
        """
        return self.mode.startswith('External')

    @lazyproperty
    def is_gps_start(self):
        """
        Check if GPS was used to start the exposure sequence.
        ie. mode is 'External Start'
        """
        return self.mode.endswith('Start')

    @lazyproperty
    def is_gps_loop(self):
        """
        Check if GPS was used to trigger every exposure in the stack.
        ie. mode is 'External'
        """
        return self.is_gps and not self.is_gps_start




class Time(time.Time):
    """
    Extends the `astropy.time.core.Time` class to include a few more convenience
    methods
    """

    _flag = ''     # implement representation flags for , GPSTime

    # def __new__(cls, val, *args, **kws):
    #     if val in (None, UnknownTime):
    #         return UnknownTime
    #     return super().__new__(cls, val, *args, **kws)

    # def to_value(self, fmt, subfmt='*'):
    #     # handle .iso, .utc, etc appending `_flag`
    #     return super().to_value(fmt, subfmt) #+ self._flag

    def lmst(self, longitude):
        """LMST for at frame mid exposure times"""
        return self.sidereal_time('mean', longitude=longitude)

    def zd(self, coords, location):
        # zenith distance
        frames = AltAz(obstime=self, location=location)
        return coords.transform_to(frames).altaz.zen

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


# TODO: using this properly requires full implementation on Time and TimeDelta
# subclasses with type preservation, which is tricky
# class InaccurateTimeStamp(Time):
#     _flag = NO_GPS_FLAG


# class GPSTime(time.Time):
#     _flag = '*'


class TimeDelta(time.TimeDelta):
    # pass
#     # _flag = ''

    def __new__(cls, val, **kws):
        if val is UnknownTime:
            return UnknownTime
        return super().__new__(cls, val, **kws)

#         # obj = super().__new__(cls, val, **kws)
#         # print('NEW', cls, val.__class__, obj.__class__ )
#         # return obj
#         # from IPython import embed
#         # embed(header="Embedded interpreter at 'timing.py':271")

#         # if isinstance(val, cls):
#         #     opts = dict(format=kws.get('format', None),
#         #                          copy=kws.get('copy', False))
#         #     return val.replicate(**opts)

#         # if issubclass(cls, val.__class__):
#         #     print('*'*88)
#         #     return super().__new__(cls, val.jd1, val.jd2, val.format, val.scale)

#         # return super().__new__(TimeDelta, val, **kws)
#         # return obj
#         # print('init super with', val, kws)
#     #     # return

#     # def __init__(self, val, val2=None, format=None, scale=None, copy=False):
#         # print('INIT', val, self.__class__, val.__class__)

#         # if issubclass(cls, val.__class__):
#         #     print('HAJAJAJDIIWY!!!!')
#             # cls()


#         # super().__init__(val, val2, format, scale, copy)

#         # if not hasattr(self, '_time'):
#         #     self._init_from_vals(val, val2, format, scale, copy)

#         # print(self)


#     # def replicate(self, format, copy):
#     #     print('REPLICATE', format, copy)
#     #     return super().replicate(format, copy)

#     @property
#     def hms(self):
#         return ppr.hms(self.to('s').value, unicode=True, precision=1) #+ self._flag

#     # def __str__(self):
#     #     return super().__str__() + self._flag

#     def __mul__(self, other):
#         return self.__class__(super().__mul__(other))

#     def __truediv__(self, other):
#         return self.__class__(super().__truediv__(other))


# class GPSTimeDelta(TimeDelta):
#     _flag = '*'


class Date(time.Time):
    """
    We need this so the Time instances print in date format instead
    of the class representation format, when print is called on, for eg. a tuple
    containing a date_time object.
    """

    def __repr__(self):
        return self.strftime('%Y-%m-%d')

    def __format__(self, spec):
        if self.shape == () and ('d' in spec):
            return self.strftime('%Y%m%d')
        return super().__format__(spec)

    # def __add__(self, other):
    #     t = super().__add__(other)
    #     kls = other.__class__
    #     if issubclass(kls, TimeDelta):
    #         kls = self.__class__
    #     return kls()


# ******************************************************************************


class shocTiming(LoggingMixin):
    """
    Time stamps and corrections for SHOC data.

    Below is a summary of the timing information available in the SHOC fits
    headers

    Recent SHOC data (post software upgrade)
    ----------------------------------------
      TRIGGER: 'Internal':
        * DATE-OBS
            - The time the user pushed start button (UTC)
        * FRAME
            - Time at the **end** of the first exposure (file creation time)
        * KCT
            - Kinetic Cycle Time   (exposure time + dead time)
            - **Not recorded** for single frame exposures:
                ACQMODE == 'Single Shot'
        * EXPOSURE
            - exposure time (sec)

      TRIGGER: 'External Start':
        * DATE-OBS
            - **NB** This is not the time stamp for the first image frame if the
              observations are GPS triggered
        * FRAME
            - as above
        * KCT
            - **not recorded**
        * EXPOSURE
            - exposure time (sec)
        * GPSSTART
            - GPS start time (UTC)


      TRIGGER: 'External':
        * GPSSTART, KCT, DATE-OBS, FRAME
            - as above
        * GPS-INT
            - GPS trigger interval (milliseconds)
        * EXPOSURE
            - exposure time (sec)

    Older SHOC data (pre 2015)
    --------------------------
      TRIGGER: 'Internal':
        * DATE-OBS
            - **not recorded**
        * FRAME, DATE
           - Time at the **end** of the first exposure (file creation timestamp)
           - The time here is rounded to the nearest second of computer clock
             ==> uncertainty of +- 0.5 sec (for absolute timing)
        * KCT
            - Kinetic Cycle Time   (exposure time + dead time)
        * EXPOSURE
            - exposure time (sec)

      TRIGGER: 'External Start':
        * FRAME, DATE-OBS, KCT, GPSSTART, GPS-INT
            - **not recorded**
        * EXPOSURE
            - exposure time (sec)

      TRIGGER: 'External':
        * KCT, DATE-OBS, GPSSTART, GPS-INT
            **not recorded**
        * EXPOSURE
            **wrong**  erroneously stores total accumulated exposure time
    """

    # TODO: get example set of all mode + old / new combo + print table
    #   Then print with  `timing_info_table(run)`
    # TODO:  specify required timing accuracy --> decide which algorithms to
    #     use based on this!
    # TODO: option to do flux weighted time stamps!!

    def __new__(cls, hdu):
        kls = shocTiming
        if 'Old' in hdu.__class__.__name__:
            kls = shocTimingOld

        return super().__new__(kls)

    def __init__(self, hdu, **kws):
        """
        Create the timing interface for a shocHDU

        Parameters
        ----------
        hdu : pyshoc.core.shocHDU

        location : astropy.coordinates.EarthLocation, optional
            Location of the observation (used for barycentric corrections),

        """

        self.hdu = hdu
        self.header = header = hdu.header
        self.location = hdu.location
        self.options = {**dict(format='isot',
                               scale='utc',
                               location=self.location,
                               precision=1),
                        **kws}
        # NOTE `precision` here this is the decimal significant figures for
        # formatting and does not represent absolute precision

        # Timing trigger mode
        self.trigger = Trigger(header)

        # Date from header
        # use FRAME here since it is most often available in header
        date, _ = header.get('FRAME', 'T').split('T')
        self.date = Date(date) if date else UnknownTime
        

        # search for external gps time file
        # if self.trigger.flag is NO_GPS:
        #     file = hdu.file.path
        #     extern = next(file.parent.glob('gps.*'), None)
        #     if extern:
        #         t0 = read_gps(extern)[file.name]
        #         self.t0 = Time(str(self.date) + t0)
        #         self.trigger.flag = '*'

    # def __array__(self): 
    #     return self.t

    def __getitem__(self, key):
        return self.t[key]

    def from_local(self, t0, tz=2):
        """
        Set start time of the observation. You will only need to explicitly
        provide times for older SHOC data with external GPS triggering where
        these timestamps were not recorded in the fits headers. Input value for
        `t0` is assumed to be SAST. If you are providing UTC times use the
        `sast=False` flag.

        Parameters
        ----------
        t0 : str, optional A string representing the time: eg: '04:20:00'. If
            not given, the GPS starting time recorded in the fits header will be
            used. sast : bool, optional Flag indicating that the times are given
            in local SAST time, by default True

        Raises
        ------
        ValueError If the GPS trigger time GPSSTART is not given and cannot be
            found in the fits header
        """

        # Convert trigger time to seconds from midnight
        h = Angle(t0, 'h').hour - tz

        # adjust to positive value -- this needs to be done for non-zero tz so
        # we don't accidentally shift the date. Since DATE/FRAME in header is
        # UT.
        sec = (h + ((h <= 0) * 24)) * 3600
        # `h` now in sec UTC from midnight
        return Time(self.date) + TimeDelta(sec, format='sec')

    def _reset_cache(self):
        """Reset all lazy properties.  Will work for subclasses"""
        for key, _ in inspect.getmembers(self.__class__, is_lazy):
            self.__dict__.pop(key, None)

    @lazyproperty
    def t0(self):
        """
        Timestamp for the starting time of the 0th frame in UTC as a
        `shoc.timing.Time` object.
        """

        # `t0` can be used as a representative str for the start time of an
        # observation. For old SHOC data with External GPS triggering, this will
        # be an `UnknowTime` object which (formats as '??') since this info is
        # not recorded in the headers. This allows pprinting info about the
        # run before timestamps have been set.

        hdu = self.hdu
        if hdu.rollover:  # FIXME: only for old data.
            # compute timestamp for files that rolled over 2Gb limit on old
            # server
            self.logger.info('Computing timestamps for %s, rolled over from %s',
                             hdu.file.name, hdu.rollover.parent)
            from .core import shocHDU

            assert isinstance(self.hdu.rollover.parent, shocHDU)
            return self.hdu.rollover.parent.t[-1] + self.delta

        # Timing keys in order of accuracy
        # if not GPS triggered skip GPSSTART key
        offset = 0
        # is_gps = self.trigger.is_gps()
        internal = self.trigger.is_internal  # int(not is_gps)
        #self.t0_flag = '*' * int(is_gps)
        search_keys = ['GPSSTART', 'DATE-OBS', 'DATE', 'FRAME'][internal:]
        for i, key in enumerate(search_keys):
            if key in self.header:
                return Time(self.header[key], **self.options) + offset

            # if self.trigger.is_gps:
                # Time = InaccurateTimeStamp

            if i > 1:
                offset = -self.delta

        return UnknownTime

        # NOTE:
        # For NEW (>2015) data, DATE-OBS key is always present, and a more
        # accurate time stamp (time the start button was pressed) than FRAME
        # (nearest second after first frame readout), so we always prefer to
        # use the former
        # if 'DATE-OBS' in self.header:
        #     return self.header['DATE-OBS']

        # For OLD (<2015) data
        # FRAME, DATE is time at the **end** of the first exposure (file
        # creation timestamp).
        # Need to adjust to start of first frame.
        # This relevant for exposure times greater than 0.5s only since the time
        # recorded in header FRAME is rounded to the nearest second of
        # file creation (approx end of first frame), timestamp uncertainty is
        # 0.5s

        # if 'DATE' in self.header:
        #     return Time(self.header['DATE'], **self.options) - self.delta

        # return UnknownTime

    @t0.setter
    def t0(self, t0):
        # reset lazyproperties
        self._reset_cache()
        return Time(t0, **self.options)

    @property
    def t0_flagged(self):
        return f'{self.t0.iso}{self.trigger.flag}'

    @lazyproperty
    def expose(self):
        """
        Exposure time (integration time) for a single image
        """
        if self.trigger.is_gps_loop:
            # GPS triggered
            # self.t0_flag = '*'
            delta = self.header.get('GPS-INT', None)
            if delta:
                return int(delta) / 1000 - self.dead

        else:
            # For TRIGGER 'Internal' or 'External Start' EXPOSURE stores the actual
            # correct exposure time
            exp = self.header.get('EXPOSURE', None)
            if exp:
                return exp

        return UnknownTime

    @lazyproperty
    def delta(self):
        """
        Frame to frame interval in seconds = Exposure time + Dead time
        """
        return TimeDelta(self.expose + self.dead, format='sec')

    # @lazyproperty
    # def delta(self):
    #     """
    #     Frame to frame interval (δt) in seconds as a `astropy.time.TimeDelta`
    #     object
    #     """
    #     # TimeDelta has higher precision than Quantity
    #     return TimeDelta(self.interval, format='sec')

    exp = t_exp = expose
    """exposure time"""

    dead = DEAD_TIME
    """dead time (readout) between exposures in s"""

    # NOTE:
    # deadtime should always the same value unless the user has (foolishly)
    # changed the vertical clock speed.
    # TODO: MAYBE CHECK stack_header['VSHIFT']
    # EDGE CASE: THE DEADTIME MAY BE LARGER IF WE'RE NOT OPERATING IN FRAME
    # TRANSFER MODE!

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
        #
        self.check_info()
        deltas = self.delta * np.arange(self.hdu.nframes, dtype=float)
        t0mid = self.t0 + 0.5 * self.delta
        return t0mid + deltas

    def check_info(self):
        # Need some info from the headers to compute the time stamps. check if
        # this is available
        mia = [name for name, i in {'EXPOSURE': self.expose,
                                    'DATE-OBS': self.t0}.items()
               if not i]
        if mia:
            plural = (len(mia) > 1)
            raise UnknownTimeException(
                f'No timestamps available for {self.hdu.file.name}. '
                f'Please set{" the" * (not plural)} {" and ".join(mia)} header'
                f' keyword{"s" * plural}.'
            )

    # NOTE:
    # the following attributes are accessed as properties to account for the
    # case of (old) GPS triggered data in which the frame start time and kct
    # are not available upon initialization (they are missing in the header).
    # If missing, UnknownTime object will be returned

    @property
    def duration(self):
        """Duration of the observation"""
        return self.hdu.nframes * self.delta
        # return UnknownTime

    @lazyproperty
    def lmst(self):
        """LMST for at frame mid exposure times"""
        return self.t.lmst(self.location.lon)

    @lazyproperty
    def ha(self):
        return self.lmst - self.hdu.coords.ra

    @lazyproperty
    def zd(self):
        # zenith distance
        return self.t.zd(self.hdu.coords, self.location)

    @lazyproperty
    def hour(self):
        """UTC in units of hours since midnight"""
        return (self.t - self.date).to('hour').value

    def _check_coords_loc(self):
        if self.hdu.coords is None:
            raise UnknownPointing

        if self.location is None:
            raise UnknownLocation

    @lazyproperty
    def bjd(self):
        """
        Barycentric julian date [BJD(TDB)] at frame mid exposure times
        (includes light travel time corrections)
        """
        return self.t.bjd(self.hdu.coords, self.hdu.location).jd

    @lazyproperty
    def hjd(self):
        """
        Heliocentric julian date [HJD(TCG)] at frame mid exposure times
        (includes light travel time corrections)
        """
        return self.t.hjd(self.hdu.coords).jd

    def airmass(self):
        """airmass of object at frame mid exposure times via Young 94"""
        return Young94(np.pi / 2 - self.altitude)

    def altitude(self):
        """altitude of object at frame mid exposure times"""
        self._check_coords_loc()
        coords = self.hdu.coords
        return altitude(coords.ra.radian,
                        coords.dec.radian,
                        self.lmst.radian,
                        self.location.lat.radian)

    def is_during_twilight(self):
        """
        Check if the entire observation takes place during twilight. This is
        helpful in determining if an observation is for flat fields.
        """
        from obstools.plan.skytracks import Sun

        t0, t1 = self.t[[0, -1]]
        # pass a string 'sutherland' below instead of the module variable
        # SUTHERLAND since EarthLocation is not hashable and will therefore not
        # cache the result of the call below
        sun = Sun('SAAO', str(self.hdu.date))
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
        table = aTable(self.t[tuple(formats.keys())])
        table.write(filename,
                    delimiter=delimiter,
                    format='ascii.commented_header',
                    formats=formats,
                    overwrite=True)

    def stamp(self, j=0):
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
        logging.debug('Time stamping %s', self.hdu.file.path.name)

        header = self.header
        t = self.t[j]

        # DATE-OBS
        # This keyword is confusing (UTC-OBS would be better), but since
        #  it is now in common  use, we (reluctantly) do the same.

        # update timestamp in header
        header['UTC-OBS'] = (t.utc.isot, 'Start of frame exposure in UTC')
        header['LMST'] = (self.lmst[j], 'Local Mean Sidereal Time')
        # header['UTDATE'] = (times.utdate[j], 'Universal Time Date')

        header['JD'] = (t.jd, 'Julian Date (UTC)')
        # header['HJD'] = (times.gjd[j], 'Geocentric Julian Date (TCG)')
        # header['LJD']      = ( times.ljd[j], 'Local Julian Date' )

        if None not in (self.hdu.coords, self.location):
            header['HJD'] = (self.hjd[j], 'Heliocentric Julian Date (TDB)')
            header['BJD'] = (self.bjd[j], 'Barycentric Julian Date (TDB)')
            header['AIRMASS'] = (self.t.airmass[j], 'Young (1994) model')
            # TODO: set model name dynamically

        # Add info to header HISTORY
        header.add_history(f'pyshoc: added time stamps at {Time.now()}',
                           before='HEAD')


class shocTimingOld(shocTiming):

    def stamp(self, j, t0=None, coords=None):
        #
        shocTiming.stamp(self, j)

        # set KCT / EXPOSURE in header
        header = self.header
        header['KCT'] = (self.interval, 'Kinetic Cycle Time')
        header['EXPOSURE'] = (self.exp, 'Integration time')

        if t0:
            # also set DATE-OBS keyword
            header['DATE-OBS'] = str(t0)
            if self.trigger.is_gps():
                # Set correct (GPS triggered) start time in header
                header['GPSSTART'] = (
                    str(t0), 'GPS start time (UTC; external)')
                # TODO: OR set empty?
