# __version__ = '3.14'


# std
import re
import operator as op
import warnings as wrn
import itertools as itt
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field

# third-party
import numpy as np
from scipy import stats
from astropy.time import Time
from astropy.utils import lazyproperty
from astropy.io.fits.hdu import PrimaryHDU
from astropy.coordinates import SkyCoord, EarthLocation

# local
from scrawl.imagine import plot_image_grid
from motley.table import AttrTable
from motley.utils import vstack_groups
from pyxides import Groups, OfType
from pyxides.vectorize import MethodVectorizer, repeat
from recipes import pprint
from recipes.dicts import pformat
from recipes.string import sub, remove_prefix
from recipes.introspect import get_caller_name
from obstools.image.calibration import keep
from obstools.stats import median_scaled_median
from obstools.utils import get_coords_named, convert_skycoords
from obstools.campaign import FilenameHelper, PhotCampaign, HDUExtra

# relative
from .utils import str2tup
from .printing import BraceContract
from .readnoise import readNoiseTables
from .timing import shocTiming, Trigger
from .convert_keywords import KEYWORDS as KWS_OLD_TO_NEW
from .header import headers_intersect, HEADER_KEYS_MISSING_OLD


# emit these warnings once only!
wrn.filterwarnings('once', 'Using telescope pointing coordinates')

# TODO
# __all__ = ['']

# TODO: can you pickle these classes


# ------------------------------ Instrument info ----------------------------- #

#            SHOC1, SHOC2
SERIAL_NRS = [5982, 6448]

# ------------------------------ Telescope info ------------------------------ #

# Field of view of telescopes in arcmin
FOV = {'74in':     (1.29, 1.29),
       '40in':     (2.85, 2.85),
       'lesedi': (5.7, 5.7)}
FOVr = {'74':    (2.79, 2.79)}  # with focal reducer
# fov30 = (3.73, 3.73) # decommissioned
# '30': fov30, '0.75': fov30, # decommissioned

# Exact GPS locations of telescopes
TEL_GEO_LOC = {'74in':   (20.81167,  -32.462167, 1822),
               '40in':   (20.81,     -32.379667, 1810),
               'lesedi': (20.8105,   -32.379667, 1811),
               'salt':   (20.810808, -32.375823, 1798)}
LOCATIONS = {tel: EarthLocation.from_geodetic(*geo)
             for tel, geo in TEL_GEO_LOC.items()}

# names
TEL_NAME_EQUIVALENT = {'74': 74, '1.9': 74,
                       '40': 40, '1.0': 40, '1': 40}
KNOWN_TEL_NAMES = [*LOCATIONS, *TEL_NAME_EQUIVALENT]


# ----------------------------- module constants ----------------------------- #

CALIBRATION_NAMES = ('bias', 'flat', 'dark')
OBSTYPE_EQUIVALENT = {'bias': 'dark',
                      'skyflat': 'flat'}
KNOWN_OBSTYPE = {*CALIBRATION_NAMES, *OBSTYPE_EQUIVALENT, 'object'}

EMPTY_FILTER_NAME = '∅'

# Attributes for matching calibration frames
ATT_EQUAL_DARK = ('camera', 'binning',  # subrect
                  'readout.frq', 'readout.preAmpGain', 'readout.outAmp.mode')  # <-- 'readout'
ATT_CLOSE_DARK = ('readout.outAmp.emGain', )  # 'timing.exp'

ATT_EQUAL_FLAT = ('telescope', 'camera', 'binning', 'filters')
ATT_CLOSE_FLAT = ('timing.date', )

MATCH = {}
MATCH['flats'] = MATCH['flat'] = MATCH_FLATS = (ATT_EQUAL_FLAT, ATT_CLOSE_FLAT)
MATCH['darks'] = MATCH['dark'] = MATCH_DARKS = (ATT_EQUAL_DARK, ATT_CLOSE_DARK)

# regex to find the roll-over number (auto_split counter value in filename)
REGEX_ROLLED = re.compile(r'\._X([0-9]+)')


# ----------------------------- helper functions ----------------------------- #


def _3d(array):
    """Add axis in 0th position to make 3d"""
    if array.ndim == 2:
        return array.reshape((1, *array.shape))
    if array.ndim == 3:
        return array
    raise ValueError('Not 3D!')


def apply_stack(func, *args, **kws):  # TODO: move to proc
    # TODO:  MULTIPROCESS HERE!
    return func(*args, **kws)


def get_tel(name):
    """
    Get standardized telescope name from description.

    Parameters
    ----------
    name : str or int
        Telescope name (see Examples).

    Returns
    -------
    str
        Standardized telescope name

    Examples
    --------
    >>> get_tel(74)
    '74in'
    >>> get_tel(1.9)
    '74in'
    >>> get_tel('1.9 m')
    '74in'
    >>> get_tel(1)
    '40in'
    >>> get_tel('40     in')
    '40in'
    >>> get_tel('LESEDI')
    'lesedi'

    Raises
    ------
    ValueError
        If name is unrecognised.
    """

    # sanitize name:  strip "units" (in,m), lower case
    name = str(name).rstrip('inm ').lower()
    if name not in KNOWN_TEL_NAMES:
        raise ValueError(f'Telescope name {name!r} not recognised. Please '
                         f'use one of the following\n: {KNOWN_TEL_NAMES}')
    if name in TEL_NAME_EQUIVALENT:
        return f'{TEL_NAME_EQUIVALENT[name]}in'
    return name


def get_fov(telescope, unit='arcmin', with_focal_reducer=False):
    """
    Get telescope field of view

    Parameters
    ----------
    telescope
    with_focal_reducer
    unit

    Returns
    -------

    Examples
    --------
    >>> get_fov(1)               # 1.0m telescope
    >>> get_fov(1.9)             # 1.9m telescope
    >>> get_fov(74)              # 1.9m
    >>> get_fov('74in')          # 1.9m
    >>> get_fov('40 in')         # 1.0m
    """

    telescope = get_tel(telescope)
    fov = (FOVr if with_focal_reducer else FOV).get(telescope)

    # at this point we have the FoV in arcmin
    # resolve units
    if unit in ('arcmin', "'"):
        factor = 1
    elif unit in ('arcsec', '"'):
        factor = 60
    elif unit.startswith('deg'):
        factor = 1 / 60
    else:
        raise ValueError('Unknown unit %s' % unit)

    return np.multiply(fov, factor)

# ------------------------------ Helper classes ------------------------------ #

# class yxTuple(tuple):
#     def __init__(self, *args):
#         assert len(self) == 2
#         self.y, self.x = self


class Binning:
    """Simple class to represent CCD pixel binning"""

    def __init__(self, args):
        # assert len(args) == 2
        self.y, self.x = args

    def __repr__(self):
        return f'{self.y}x{self.x}'
        # return f'{self.__class__.__name__}{self.y, self.x}'

    def __iter__(self):
        yield from (self.y, self.x)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return (self.y == other.y) and (self.x == other.x)
        return False

    def __hash__(self):
        return hash((self.y, self.x))

    def __lt__(self, other):
        if isinstance(other, self.__class__):
            return (self.y, self.x) < (other.y, other.x)

        raise TypeError(
            f"'<' not supported between instances of {self.__class__} and "
            f"{type(other)}")


class Filters:
    """Simple class to represent positions of the filter wheels"""

    def __init__(self, a, b=None):
        A = self.get(a)
        # sloan filters usually in wheel B. Keep consistency here when assigning
        if (b is None) and A.islower():
            self.A = EMPTY_FILTER_NAME
            self.B = A
        else:
            self.A = A
            self.B = self.get(b)

    def get(self, long):
        # get short description like "U",  "z'", or "∅"
        if long == 'Empty':
            return EMPTY_FILTER_NAME
        if long:
            return long.split(' - ')[0]
        return (long or EMPTY_FILTER_NAME)

    def __members(self):
        return self.A, self.B

    def __repr__(self):
        return f'{self.__class__.__name__}{self.__members()}'

    # def __format__(self, spec):
    #     return next(filter(EMPTY_FILTER_NAME.__ne__, self), '')

    def __str__(self):
        return self.name

    def __iter__(self):
        yield from self.__members()

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__members() == other.__members()
        return False

    def __hash__(self):
        return hash(self.__members())

    @property
    def name(self):
        """Name of the non-empty filter in either position A or B, else ∅"""
        return next(filter(EMPTY_FILTER_NAME.strip, self), EMPTY_FILTER_NAME)

    def to_header(self, header):
        _remap = {EMPTY_FILTER_NAME: 'Empty'}
        for name, val in self.__dict__.items():
            header[f'FILTER{name}'] = _remap.get(val, val)


@dataclass()
class OutAmpMode:
    """
    Class to encapsulate the CCD output amplifier settings
    """
    mode_long: str
    emGain: int = 0

    # note the gain is sometimes erroneously recorded in the header as having a
    #  non-zero value even though the pre-amp mode is CON.

    def __post_init__(self):
        # EM gain is sometimes erroneously recorded as some non-zero value
        # even though the outamp mode is CON.  I think this happens if you
        # switch  from EM mode back to CON mode.  Here we make sure `emGain`
        # is alwalys 0 when the outamp mode is CON. This is
        # necessary fo matching to work correctly
        if self.mode_long.startswith('C'):
            self.mode = 'CON'
            self.emGain = 0
        else:
            self.mode = 'EM'

    def __repr__(self):
        return ': '.join(map(str, filter(None, (self.mode, self.emGain))))


@dataclass(order=True)  # (frozen=True)
class ReadoutMode:
    """
    Helper class for working with different modes of the CCD output amplifier
    and pre-amplifier.
    """
    frq: float
    preAmpGain: float
    outAmp: OutAmpMode
    ccdMode: str = field(repr=False)
    serial: int = field(repr=False)

    @classmethod
    def from_header(cls, header):
        return cls(int(round(1.e-6 / header['READTIME'])),  # readout freq MHz
                   header['PREAMP'],
                   OutAmpMode(header['OUTPTAMP'], header.get('GAIN', '')),
                   header['ACQMODE'],
                   header['SERNO'])

    def __post_init__(self):
        #
        self.isEM = (self.outAmp.mode == 'EM')
        self.isCON = not self.isEM
        self.mode = self  # hack for verbose access `readout.mode.isEM`

        # Readout noise
        # set the correct values here as attributes of the instance. These
        # values are absent in the headers
        (self.bit_depth,
         self.sensitivity,
         self.noise,
         self.time,
         self.saturation,
         self.bias_level
         ) = readNoiseTables[self.serial][
            (self.frq, self.outAmp.mode, self.preAmpGain)]

    def __iter__(self):
        yield from (self.frq, self.preAmpGain, self.outAmp.mode, self.outAmp.emGain)

    def __repr__(self):
        return (f'{self.frq}MHz {self.preAmpGain} {self.outAmp}')

    # @property
    # def short(self):
    #     return (f'{self.frq}MHz {self.outAmp}')

    def __hash__(self):
        return hash((self.frq,
                     self.preAmpGain,
                     self.outAmp.mode,
                     self.outAmp.emGain))

    # def _repr_short(self, with_units=True):
    #     """short representation for tables etc"""
    #     if with_units:
    #         units = (' e⁻/ADU', ' MHz')
    #     else:
    #         units = ('', '')
    #     return 'γ={0:.1f}{2:s}; f={1:.1f}{3:s}'.format(self.gain,
    #                                                    self.freq, *units)

    # def fn_repr(self):
    #     return 'γ{:g}@{:g}MHz.{:s}'.format(
    #             self.preAmpGain, self.frq, self.outAmp.mode)


class RollOver():

    nr = 0
    parent = None

    def __init__(self, hdu):
        # Check whether the filenames contain '._X' an indicator for whether the
        # datacube reached the 2GB windows file size limit on the shoc server,
        # and was consequently split into a sequence of fits cubes. The
        # timestamps of these need special treatment

        path = hdu.file.path
        if not path:
            return

        checked = map(REGEX_ROLLED.match, path.suffixes)
        match = next(filter(None, checked), None)
        self.nr = n = int(match[1]) if match else 0
        self.parent = path.name.replace(f'._X{n}', f'._X{n-1}' if n > 2 else '')

    def __bool__(self):
        return bool(self.nr)

    @property
    def state(self):
        return bool(self)


class shocFilenameHelper(FilenameHelper):
    @property
    def nr(self):
        """
        File sequence number
            eg. '0010' in 'SHA_20200729.0010.fits'
        """
        return self.path.suffixes[0].lstrip('.')


class Messenger:
    # TODO: control through logger and Handlers etc
    def message(self, message, sep='.'):
        """Make a message tagged with class and function name"""
        return (f'{self.__class__.__name__}{sep}{get_caller_name(2)} {Time.now()}: '
                f'{message}')

# ------------------------------ HDU Subclasses ------------------------------ #


class shocHDU(HDUExtra, Messenger):

    _FilenameHelperClass = shocFilenameHelper
    __shoc_hdu_types = {}
    filename_format = None

    @classmethod
    def match_header(cls, header):
        return ('SERNO' in header)

    def __new__(cls, data, header, obstype=None, *args, **kws):
        # Choose subtypes of `shocHDU` here - simpler than using `match_header`
        # NOTE:`_BaseHDU` creates a `_BasicHeader`, which does not contain
        # hierarch keywords, so for SHOC we cannot tell if it's the old format
        # header by checking those.

        # handle direct init here eg:   shocDarkHDU(data, header)
        if cls in cls.__shoc_hdu_types.values():
            return super().__new__(cls)

        # assign class based on obstype and age
        obstype = (obstype or '').lower()
        if not obstype:
            obstype = header.get('OBSTYPE', '')

        # check if all the new keywords are present
        if any(kw not in header for kw in HEADER_KEYS_MISSING_OLD):
            age = 'Old'

        # identify calibration files
        age, kind, suffix = '', '', 'HDU'
        obstype = OBSTYPE_EQUIVALENT.get(obstype, obstype)
        if obstype in CALIBRATION_NAMES:
            kind = obstype.title()
            if (header['NAXIS'] == 2 and header.get('MASTER')):
                suffix = 'Master'
                age = ''

        #
        class_name = f'shoc{age}{kind}{suffix}'
        cls.logger.debug(f'{obstype=}, {class_name=}')
        if class_name not in ['shocHDU', *cls.__shoc_hdu_types]:
            # pylint: disable=no-member
            cls.logger.warning('Unknown OBSTYPE: %r', obstype)

        # cls = cls.__shoc_hdu_types.get(class_name, cls)
        # print(f'{class_name=}; {cls=}')
        return super().__new__(cls.__shoc_hdu_types.get(class_name, cls))

    def __init_subclass__(cls):
        cls.__shoc_hdu_types[cls.__name__] = cls

    def __init__(self, data=None, header=None, obstype=None, *args, **kws):
        # init PrimaryHDU
        super().__init__(data=data, header=header, *args, **kws)

        #
        self.camera = f'SHOC{SERIAL_NRS.index(header["SERNO"]) + 1}'
        self.location = LOCATIONS.get(self.telescope)
        self._coords = None     # placeholder
        # # field of view
        # self.fov = self.get_fov()

        # image binning
        self.binning = Binning(header[f'{_}BIN'] for _ in 'VH')

        # sub-framing
        self.subrect = np.array(header['SUBRECT'].split(','), int)
        # subrect stores the sub-region of the full CCD array captured for this
        #  observation
        # xsub, ysub = (xb, xe), (yb, ye) = \
        xsub, ysub = self.subrect.reshape(-1, 2) // tuple(self.binning)
        # for some reason the ysub order is reversed
        self.sub = tuple(xsub), tuple(ysub[::-1])
        self.sub_slices = list(map(slice, *np.transpose(self.sub)))

        # CCD mode
        self.readout = ReadoutMode.from_header(header)

        # orientation
        # self.flip_state = yxTuple(header['FLIP%s' % _] for _ in 'YX')
        # WARNING: flip state wrong for EM!!
        # NOTE: IMAGE ORIENTATION reversed for EM mode data since it gets read
        #  out in opposite direction in the EM register to CON readout.

        # # manage on-the-fly image orientation
        # self.oriented = ImageOrienter(self, x=self.readout.isEM)

        # # manage on-the-fly calibration for large files
        # self.calibrated = ImageCalibration(self)

        # filters
        self._filters = Filters(*(header.get(f'FILTER{_}', 'Empty')
                                  for _ in 'AB'))

        # observation type
        if obstype:
            self.obstype = obstype
        else:
            self._obstype = self.header.get('OBSTYPE')

    def __str__(self):
        attrs = ('t.t0_flagged', 'binning', 'readout.mode', 'filters')
        info = ('', *op.attrgetter(*attrs)(self), '')
        sep = ' | '
        return f'<{self.__class__.__name__}:{sep.join(map(str, info)).strip()}>'

    def __repr__(self):
        return str(self)

    @property
    def nframes(self):
        """Total number of images in observation"""
        return 1 if (self.ndim == 2) else self.shape[0]

    @property
    def obstype(self):
        return self._obstype

    @obstype.setter
    def obstype(self, obstype):
        if obstype not in KNOWN_OBSTYPE:
            raise ValueError(f'Unrecognised {obstype=}')

        self._obstype = self.header['OBSTYPE'] = obstype

    @property
    def filters(self):
        return self._filters

    @filters.setter
    def filters(self, filters):
        self._filters = Filters(*str2tup(filters))
        self._filters.to_header(self.header)

    @property
    def telescope(self):
        return self.header.get('TELESCOP')

    @telescope.setter
    def telescope(self, telescope):
        tel = self.header['TELESCOP'] = get_tel(telescope)
        self.location = LOCATIONS.get(tel)

    @property
    def target(self):
        return self.header.get('OBJECT')

    @target.setter
    def target(self, name):
        ot = self.target
        trg = self.header['OBJECT'] = str(name)
        if ot != trg:
            del self.coords

    def _get_coords(self):
        """
        The target coordinates.  This function will look in multiple places
        to find the coordinates.
        1. header 'OBJRA' 'OBJDEC'
        2. use coordinates pointed to in SIMBAD if target name in header under
           'OBJECT' key available and connection available.
        3. header 'TELRA' 'TELDEC'.  warning emit

        Returns
        -------
        astropy.coordinates.SkyCoord

        """
        # target coordinates
        header = self.header

        ra, dec = header.get('OBJRA'), header.get('OBJDEC')
        coords = convert_skycoords(ra, dec)
        if coords:
            return coords

        if self.target:
            # No / bad coordinates in header, but object name available - try
            # resolve
            coords = get_coords_named(self.target)

        if coords:
            return coords

        # No header coordinates / Name resolve failed / No object name available
        # LAST resort use TELRA, TELDEC. This will only work for newer SHOC
        # data for which these keywords are available in the header
        # note: These will lead to slightly less accurate timing solutions,
        #  so emit warning
        ra, dec = header.get('TELRA'), header.get('TELDEC')
        coords = convert_skycoords(ra, dec)

        if coords:
            wrn.warn('Using telescope pointing coordinates')

        return coords

    def _set_coords(self, coords):
        from astropy.coordinates import SkyCoord

        self._coords = coo = SkyCoord(coords, unit=('h', 'deg'))
        self.header.update(
            OBJRA=coo.ra.to_string('hourangle', sep=':', precision=1),
            OBJDEC=coo.dec.to_string('deg', sep=':', precision=1),
        )

    coords = lazyproperty(_get_coords, _set_coords)

    def pointing_zenith(self, tol=2):
        """
        Whether the telescope was pointing at the zenith at the start of the
        observation. Useful for helping to discern if the observation is a sky
        flat.

        Parameters
        ----------
        tol : int, optional
            Tolerance in degrees, by default 2

        Returns
        -------
        bool
            True if point to zenith, False otherwise
        """
        if self.coords is None:
            raise ValueError(
                'No coordinates available for observation HDU. Please assign '
                'the `target` attribute with the source name, or set the '
                '`coord` directly.'
            )
        return self.t[0].zd(self.coords, self.location).deg < tol

        # def pointing_park(obs, tol=1):
        #     if obs.telescope == '40in':
        #         'ha' approx 1
        #     return pointing_zenith(obs, tol)

        # NOTE: for the moment, the methods below are duplicated while migration
        #  to this class in progress

    # ------------------------------------------------------------------------ #
    # Timing

    @lazyproperty
    def timing(self):
        # Initialise timing
        # this is delayed from init on the hdu above since this class may
        # initially be created with a `_BasicHeader` only, in which case we
        # will not yet have all the correct keywords available yet to identify
        # old vs new shoc data at init.
        # try:
        return shocTiming(self)
        # except:
        #     return UnknownTime

    # alias
    t = timing

    @property
    def needs_timing(self):
        """
        check for date-obs keyword to determine if header information needs
        updating
        """
        return ('date-obs' not in self.header)
        # TODO: is this good enough???

    @property
    def date(self):
        return self.t.date

    @lazyproperty
    def rollover(self):
        return RollOver(self)

    # ------------------------------------------------------------------------ #

    @lazyproperty
    def oriented(self):
        """
        Use this to get images with the correct orientation:
        North up, East left.

        This is necessary since EM mode images have opposite x-axis
        orientation since  pixels are read out in the opposite direction
        through the readout register.

        Images taken in CON mode are flipped left-right.
        Images taken in EM  mode are left unmodified

        """
        from obstools.image.calibration import ImageOrienter

        # will flip EM images x axis (2nd axis)
        return ImageOrienter(self, x=self.readout.isCON)

    def get_fov(self):
        return get_fov(self.telescope)

    def get_rotation(self):
        """
        Get the instrument rotation (position angle) wrt the sky in radians.
        This value seems to be constant for most SHOC data from various
        telescopes that I've checked. You can measure this yourself by using the
        `obstools.campaign.PhotCampaign.coalign_dss` method.
        """
        return 0#-0.04527

    def guess_obstype(self, sample_size=10, subset=(0, 100),
                      return_stats=False):
        # FIXME: sample_size misleading name since it is really number of
        #  frames used to get median image
        """
        Guess the observation type based on statistics of a sample image as well
        as observation time.  This function is experimental comes with no
        guarantees whatsoever.

        This implements a very basic decision tree classifier based on 3
        features: mean, stddev, skewness (the first 3 statistical moments) of
        the sampled median image. These values are normalized to the saturation
        value of the CCD given the instrumental setup in order to compare fairly
        between diverse datasets.

        Since for SHOC 0-time readout is impossible, the label 'bias' is
        technically erroneous, we prefer to the label 'dark'. Moreover, it's
        impossible to distinguish here between 'bias' and 'dark' frames based on
        the data values alone - the distinction comes in how they are used in
        the reduction - and since the dark current for SHOC is so low, dark
        frames are routinely used for bias subtraction. We choose to use the
        label 'dark' since it is a physical description instead of a procedural
        one and therefore more meaningful in this context.


        Parameters
        ----------
        sample_size : int, optional
            Size of the sample from which the median image will be constructed,
            by default 10
        subset : tuple, optional
            Subset of the image stack that will be sampled, by default (0, 100)
        return_stats : bool, optional
            Whether or not to return the computed sample statistics,
            by default False

        Returns
        -------
        str: {'object', 'flat', 'dark', 'bad'}
            label for observation type
        """

        #
        #     return 'flat'

        img = self.sampler.median(sample_size, subset)
        n_obs, min_max, *moments = stats.describe(img.ravel())
        m, v, s, k = np.divide(moments, self.readout.saturation)

        # s = 0 implies all constant pixel values.  These frames are sometimes
        # created erroneously by the SHOC GUI
        if v == 0 or m >= 1.5:
            o = 'bad'

        if self.coords and self.pointing_zenith():
            # either bias or flat
            # Flat fields are usually about halfway to the saturation value
            if 0.15 <= m < 1.5:
                o = 'flat'

            # dark images have roughly symmetric pixel distributions whereas
            # objects images have asymmetric distributions since the contain stars
            elif s < 2.5e-4:
                o = 'dark'

        # what remains must be on sky!
        else:
            o = 'object'

        if return_stats:
            return o, (m, v, s, k)

        return o

    # ------------------------------------------------------------------------ #
    # image arithmetic

    def combine(self, func=None, args=(), **kws):
        """
        Combine images in the stack by applying the function `func` along the
        0th dimension.

        Parameters
        ----------
        func
        args
        kws

        Returns
        -------

        """

        if not callable(func):
            raise TypeError(
                f'Parameter `func` should be callable. Received {type(func)}'
            )

        # check if single image
        if (self.ndim == 2) or ((self.ndim == 3) and len(self.data) == 1):
            return self

        # log some info
        msg = f'{func.__name__} of {self.nframes} images from {self.file.name}'
        msg = self.message(msg, sep='')
        self.logger.info(remove_prefix(msg, self.__class__.__name__))

        # combine across images
        kws.setdefault('axis', 0)
        # TODO: self.calibrated ???
        hdu = shocHDU(func(self.data, *args, **kws), self.header)

        # FIXME: MasterHDU not used here :(((

        # update header
        hdu.header['MASTER'] = True
        hdu.header['NCOMBINE'] = self.nframes
        hdu.header.add_history(msg)

        return hdu

    def subtract(self, hdu):
        """
        Subtract the image

        Parameters
        ----------
        hdu

        Returns
        -------

        """
        # This may change the data type.
        self.data = self.data - hdu.data

        # update history
        msg = self.message(f'Subtracted image {hdu.file.name}')
        self.header.add_history(msg)
        return self

    # ------------------------------------------------------------------------ #

    def get_save_name(self, name_format=None, ext='fits'):
        # TODO: at filename helper?
        name_format = name_format or self.filename_format

        if self.file.path:
            return self.file.path

        if name_format:
            # get attribute values and replace unwnated characters for filename
            fmt = name_format.replace('{', '{0.')
            name = sub(fmt.format(self), {' ': '-', "'": '', ': ': ''})
            return name + f'.{ext.lstrip(".")}'

    # @expose.args()
    def save(self, filename=None, folder=None, name_format=None,
             overwrite=False):

        filename = filename or self.get_save_name(name_format)
        if filename is None:
            raise ValueError('Please provide a filename for saving.')

        path = Path(filename)
        if folder is not None:
            folder = Path(folder)

        if path.parent == Path():  # cwd
            if folder is None:
                raise ValueError('Please provide a folder location, or specify '
                                 'a absolute path as `filename`.')
            path = folder / path

        if not path.parent.exists():
            self.logger.info('Creating directory: %r', str(path.parent))
            path.parent.mkdir()

        action = 'Saving to'
        if path.exists():
            action = 'Overwriting'

        self.logger.info('%s %r', action, str(path))
        self.writeto(path, overwrite=overwrite)

        # reload
        return self.__class__.readfrom(path)


# FILENAME_TRANS = str.maketrans({'-': '', ' ': '-'})

class shocOldHDU(shocHDU):
    def __init__(self, data, header, *args, **kws):

        super().__init__(data, header, *args, **kws)

        # fix keywords
        for old, new in KWS_OLD_TO_NEW:
            if old in header:
                self.header.rename_keyword(old, new)

    # @lazyproperty
    # def timing(self):
    #     return shocTimingOld(self)


class shocCalibrationHDU(shocHDU):
    _combine_func = None  # place-holder

    # TODO: set target=''

    def get_coords(self):
        return

    def combine(self, func=_combine_func, args=(), **kws):
        return super().combine(func or self._combine_func, args, **kws)


class shocDarkHDU(shocCalibrationHDU):
    # TODO: don't pprint filters since irrelevant

    filename_format = '{obstype}-{camera}-{binning}-{readout}'
    _combine_func = staticmethod(np.median)

    # "Median combining can completely remove cosmic ray hits and radioactive
    # decay trails from the result, which cannot be done by standard mean
    # combining. However, median combining achieves an ultimate signal to noise
    # ratio about 80% that of mean combining the same number of frames. The
    # difference in signal to noise ratio can be by median combining 57% more
    # frames than if mean combining were used. In addition, variants on mean
    # combining, such as sigma clipping, can remove deviant pixels while
    # improving the S/N somewhere between that of median combining and ordinary
    # mean combining. In a nutshell, if all images are "clean", use mean
    # combining. If the images have mild to severe contamination by radiation
    # events such as cosmic rays, use the median or sigma clipping method."
    # - Newberry

    def __init__(self, data, header, obstype='dark', *args, **kws):
        super().__init__(data, header, obstype, *args, **kws)


class shocFlatHDU(shocCalibrationHDU):

    filename_format = '{obstype}-{t.date:d}-{telescope}-{camera}-{binning}-{filters}'
    _combine_func = staticmethod(median_scaled_median)
    # default combine algorithm first median scales each image, then takes
    # median across images

    def __init__(self, data, header, obstype='flat', *args, **kws):
        super().__init__(data, header, obstype, *args, **kws)


class shocOldDarkHDU(shocOldHDU, shocDarkHDU):
    pass


class shocOldFlatHDU(shocOldHDU, shocDarkHDU):
    pass

# class shocBiasHDU(shocDarkHDU):
#     # alias
#     pass


class shocMaster(shocCalibrationHDU):
    pass


class shocDarkMaster(shocMaster, shocDarkHDU):
    pass


class shocFlatMaster(shocMaster, shocFlatHDU):
    pass

# -------------------------- Pretty printing helpers ------------------------- #


def hms(t):
    """sexagesimal formatter"""
    return pprint.hms(t.to('s').value, unicode=True, precision=1)


class TableHelper(AttrTable):

    def get_table(self, run, attrs=None, **kws):
        # Add '*' flag to times that are gps triggered
        flags = {}
        postscript = []
        for key, attr in [('timing.t0', 'timing.trigger.flag', ),
                          ('timing.exp', 'timing.trigger.loop_flag')]:
            flg = flags[self.aliases[key]] = run.attrs(attr)
            for flag in set(flg):
                if flag:
                    info = Trigger.FLAG_INFO[self.get_header(attr)]
                    postscript.append(f'{key}{flag}: {info[flag]}')

        # get table
        postscript = '\n'.join(postscript)
        table = super().get_table(run, attrs, flags=flags,
                                  footnotes=postscript, **kws)

        # compacted `filter` displays 'A = ∅' which is not very clear. Go more
        # verbose again for clarity
        # HACK:
        if table.compact_items:
            replace = {'A': 'filter.A',
                       'B': 'filter.B'}
            table.compact_items = {replace.get(name, name): item
                                   for name, item in table.compact_items.items()
                                   }
            table._compact_table = table._get_compact_table()
        return table


# ------------------------------------- ~ ------------------------------------ #

class shocCampaign(PhotCampaign, OfType(shocHDU), Messenger):
    #
    pretty = BraceContract(brackets='', per_line=1, indent=4,  hang=True)

    # controls which attributes will be printed
    tabulate = TableHelper(
        # attrs =
        # todo:
        # 'file.stem' : Column('filename', unit='', fmt='', total=True )
        ['file.stem',
         'telescope', 'camera',
         'target', 'obstype',
         'filters.A', 'filters.B',
         'nframes', 'ishape', 'binning',
         #  'readout.preAmpGain',
         'readout.mode',
         #  'readout.mode.frq',
         #  'readout.mode.outAmp',

         'timing.t0',
         'timing.exp',
         'timing.duration',
         ],
        aliases={
            'file.stem': 'filename',
            'telescope': 'tel',
            'camera': 'camera',
            'nframes': 'n',
            'binning': 'bin',
            # 'readout.mode.frq': 'mode',
            'readout.preAmpGain': 'preAmp',  # 'γₚᵣₑ',
            'timing.exp': 'tExp',
            # 'timing.t0': 't0',
            # 'timing.duration': 'duration'
        },
        formatters={
            'duration': hms,
            't0': op.attrgetter('iso'),
            'tExp': str  # so that UnknownTime formats as ??
        },
        units={
            # 'readout.preAmpGain': 'e⁻/ADU',
            # 'readout.mode.frq': 'MHz',
            # 'readout.mode': ''
            'tExp': 's',
            't0': 'UTC',
            'ishape': 'y, x',
            'duration': 'hms'
        },

        compact=True,
        title_props=dict(txt=('underline', 'bold'), bg='b'),
        too_wide=False,
        totals=['n', 'duration'])

    @classmethod
    def new_groups(cls, *keys, **kws):
        return shocObsGroups(cls, *keys, **kws)

    # def map(self, func, *args, **kws):
    #     """Map and arbitrary function onto the data of each observation"""
    #     shocHDU.data.get

    # @expose.args
    @classmethod
    def load_files(cls, filenames, *args, **kws):
        run = super().load_files(filenames, *args, **kws)

        rolled = run.group_by('rollover.state')
        ok = rolled.pop(False)
        rolled = rolled.pop(True, ())
        for hdu in rolled:
            hdu.rollover.parent = parent = ok[hdu.rollover.parent]
            if not parent:
                wrn.warn(f'Expected parent files {parent} to be loaded with '
                         f'this file. Timestamps for {hdu.file.name} will be '
                         f'wrong!!')

        # Check for external GPS timing file
        need_gps = run.missing_gps()
        if need_gps:
            gps = run.search_gps_file()
            if gps:
                cls.logger.info('Using gps timestamps from %r', str(gps))
                run.provide_gps(gps)

        return run

    def save(self, filenames=(), folder=None, name_format=None, overwrite=False):
        if filenames:
            assert len(filenames) == len(self)
        else:
            filenames = self.calls.get_save_name(name_format)

        if len(set(filenames)) < len(self):
            from recipes.lists import tally
            dup = [fn for fn, cnt in tally(filenames).items() if cnt > 1]
            self.logger.warning('Duplicate filenames: %s', dup)

        hdus = self.calls.save(None, folder, name_format, overwrite)
        # reload
        self[:] = self.__class__(hdus)
        return self

    def thumbnails(self, statistic='mean', depth=10, subset=None,
                   title='file.name', calibrated=False, figsize=None, **kws):
        """
        Display a sample image from each of the observations laid out in a grid.
        Image thumbnails are all the same size, even if they are shaped
        differently, so the size of displayed pixels may be different

        Parameters
        ----------
        statistic: str
            statistic for the sample
        depth: int
            number of images in each observation to combine
        subset: int or tuple
            index interval for subset of the full observation to draw from, by
            default use (0, depth)
        title: str
            key to attribute of item that will be used as title
        calibrated: bool
            calibrate the image if bias / flat field available
        figsize:  tuple
               size of the figure in inches

        Returns
        -------

        """

        # get axes title strings
        titles = []
        if isinstance(title, str) and ('{' in title):
            # convert to proper format string
            title = title.replace('{', '{0.')
            titles = list(map(title.format, self))
        else:
            for ats in self.attrs(title):
                if not isinstance(ats, tuple):
                    ats = (ats, )
                titles.append('\n'.join(map(str, ats)))

        # get sample images
        if subset is None:
            subset = (0, depth)

        # get sample images
        sample_images = self.calls(f'sampler.{statistic}', depth, subset)

        if calibrated:
            # print(list(map(np.mean, sample_images)))
            sample_images = [hdu.calibrated(im) for hdu, im in
                             zip(self, sample_images)]
            # print(list(map(np.mean, sample_images)))

        return plot_image_grid(sample_images, titles=titles, figsize=figsize,
                               **kws)

    def guess_obstype(self, plot=False):
        """
        Identify what type of observation each dataset represents by running
        'schoHDU.guess_obstype' on each.

        Parameters
        ----------
        plot

        Returns
        -------

        """

        obstypes, stats = zip(*self.calls.guess_obstype(return_stats=True))

        if plot:
            self.plot_image_stats(stats, obstypes,
                                  ['mean', 'var', 'skewness', 'kurtosis'])

        groups = self.new_groups()
        for kind in ['object', 'flat', 'dark', 'bad']:
            # this makes the order of the groups nice
            selected = np.array(obstypes) == kind
            if selected.any():
                groups[kind] = self[selected]

        return groups

    def match(self, other, exact, closest=(), cutoffs=(), keep_nulls=False):
        from .match import MatchedObservations

        return MatchedObservations(self, other)(
            exact, closest, cutoffs, keep_nulls
        )

    def combine(self, func=None, args=(), **kws):
        """
        Combine each `shocHDU` in the campaign into a 2D image by calling `func`
        on each data stack.  Can be used to compute image statistics or compute 
        master flat / dark image for calibration.

        Parameters
        ----------
        func

        Returns
        -------

        """
        # if func is None:
        #     types = set(map(type, self))
        #     if len(types) > 1:
        #         raise ValueError(
        #             'Please provide a function to combine. Combine function '
        #             'will only be automatically selected if the Campaign has '
        #             'observations of uniform obstype.'
        #         )

        #     func = types.pop()._combine_func
        # #
        # assert callable(func), f'`func` should be a callable not {type(func)}'
        return self.__class__(self.calls.combine(func, args, **kws))

    def stack(self):
        """
        Stack data

        Returns
        -------

        """
        if len(self) == 1:
            return self[0]

        shapes = set(self.attrs('ishape'))
        if len(shapes) > 1:
            raise ValueError('Cannot stack images with different shapes')

        # keep only header keywords that are the same in all headers
        header = headers_intersect(self)
        header['FRAME'] = self[0].header['date']  # HACK: need date for flats!
        header['NAXIS'] = 3                  # avoid misidentification as master
        hdu = shocHDU(np.concatenate([_3d(hdu.data) for hdu in self]),
                      header)
        msg = self.message(f'Stacked {len(self)} files.')
        hdu.header.add_history(msg)
        return hdu

    def merge_combine(self, func, *args, **kws):
        return self.combine(func, *args, **kws).stack().combine(func, *args, **kws)

    def subtract(self, other):
        if isinstance(other, shocCampaign) and len(other) == 1:
            other = other[0]

        if not isinstance(other, PrimaryHDU):
            raise TypeError(f'Expected shocHDU, got {type(other)}')

        if len(other.shape) > 2:
            raise TypeError(
                'The input hdu contains multiple images instead of a single '
                'image. Do `combine` to compute the master bias.'
            )

        return self.__class__([hdu.subtract(other) for hdu in self])

    def set_calibrators(self, darks=None, flats=None):
        if darks:
            mo = self.match(darks, *MATCH_DARKS)
            mo.left.set_calibrators(mo.right)

        if flats:
            mo = self.match(flats, *MATCH_FLATS)
            mo.left.set_calibrators(flats=mo.right)

    def plot_image_stats(self, stats, labels, titles, figsize=None):
        """
        Compare image statistics across observations in the run.  This serves as
        a diagnostic for guessing image obstypes


        Parameters
        ----------
        stats
        labels
        titles
        figsize

        Returns
        -------

        """
        from matplotlib import pyplot as plt

        stats = np.array(stats)
        assert stats.size

        idx = defaultdict(list)
        for i, lbl in enumerate(labels):
            idx[lbl].append(i)

        n = stats.shape[1]
        fig, axes = plt.subplots(1, n, figsize=figsize, sharey=True)
        for j, ax in enumerate(axes):
            ax.set_title(titles[j])
            m = 0
            for lbl, i in idx.items():
                ax.plot(stats[i, j], range(m, m + len(i)), 'o', label=lbl)
                m += len(i)
            ax.grid()

        # noinspection PyUnboundLocalVariable
        ax.invert_yaxis()
        ax.legend()

        # filenames as ticks
        z = []
        list(map(z.extend, idx.values()))

        ax = axes[0]
        ax.set_yticklabels(np.take(self.files.stems, z))
        ax.set_yticks(np.arange(len(self) + 1) - 0.5)
        for tick in ax.yaxis.get_ticklabels():
            tick.set_va('top')
        # plt.yticks(np.arange(len(self) + 1) - 0.5,
        #            np.take(names, z),
        #            va='top')

        fig.tight_layout()
        return fig, axes

    def set_names(self, named_coords, tolerance=1.):
        """
        Set the object names and coordinates according to the mapping in
        `named_coords`. This can be used to correct mistakes in the recording of
        the header information (these commonly occur due to human error). This
        will loop through the obserrvations, matching the telescope pointing
        cooridnates to the closest matching coordinates amongst those given. The
        `coords` attribute of the observations will be set to these coordinates,
        and the `target` attributes to the corresponding name key. The `obstype`
        attribute will also be set to 'objecct'. If there are no observations
        within `tolerance` degrees from any targets, these attributes will be
        left unchanged.

        CRITICAL: This function uses the telescope pointing coordinates as
        listed in the hdu headers. Only use this function if you are sure that
        these are correct. They are sometimes wrong, in which case this function
        will fail or assign objects incorrectly files.

        Parameters
        ----------
        named_coords : dict name, coordinate mapping tolerance : float
            coordinate distance tolerance in degrees
        """
        # is_flat = run.calls('pointing_zenith')
        # run = self[np.logical_not(is_flat)]

        names, coords = zip(*named_coords.items())
        coords = SkyCoord(coords)

        # intentionally ignore coordinates pointed to by the name in the header
        # since it may be wrong:
        cxx = SkyCoord(*(self.calls('header.get', x)
                         for x in ('telra',  'teldec')), unit=('h', 'deg'))

        for obs, cx in zip(self, cxx):
            i = coords.separation(cx).argmin()
            obj_coords = coords[i]
            selected = self[cxx.separation(obj_coords).deg < tolerance]
            if selected:
                selected.attrs.set(repeat(coords=obj_coords,
                                          target=names[i],
                                          obstype='object'))

    def required_calibration(self, kind):
        """
        Given the set of image stacks, get the set of unique instrumental setups
        (camera, telescope, binning etc.), that lack calibration ('dark',
        'flat') observations.

        Note: OBSTYPE need to be accurate for this to work
        """

        kind = kind.lower()
        assert kind in CALIBRATION_NAMES
        kind = OBSTYPE_EQUIVALENT.get(kind, kind)

        g = self.group_by('obstype')
        attx, _ = MATCH[kind]

        atrset = set(g['object'].attrs(*attx))
        if kind == 'dark':
            atrset |= set(g['flat'].attrs(*attx))

        atrset -= set(g[kind].attrs(*attx))
        return sorted(atrset, key=str)

    def missing_calibration(self, report=False):
        missing = {kind: self.required_calibration(kind)
                   for kind in ('flat', 'dark')}

        if report:
            s = ''
            for key, modes in missing.items():
                s += key.upper()
                s += (':\t' + '\n\t'.join(map(str, modes)))
                s += '\n'
            print(s)

        return missing

    def missing_gps(self):
        return self.select_by(**{'t.trigger.flag': bool}).sort_by('t.t0')

    def search_gps_file(self):
        # get common root folder
        folder = self.files.common_root()
        if folder:
            gps = folder / 'gps.sast'
            if gps.exists():
                return gps

    def provide_gps(self, filename):
        # read file with gps triggers
        path = Path(filename)
        names, times = np.loadtxt(str(filename), str, unpack=True)
        need_gps = self.select_by(**{'t.trigger.flag': bool}).sort_by('t.t0')
        assert need_gps == self[names].sort_by('t.t0')

        # assert len(times) == len(self)
        tz = 2 * path.suffix.endswith('sast')
        for obs, t0 in zip(need_gps, times):
            t = obs.t
            t.t0 = t.from_local(t0, tz)
            t.trigger.flag = '*'

        # print('TIMEDELTA')
        # print((Time(t0) - Time(need_gps.attrs('t.t0'))).to('s'))

        # TODO: header keyword


class shocObsGroups(Groups):
    """
    Emulates dict to hold multiple shocRun instances keyed by their shared
    common attributes. The attribute names given in groupId are the ones by
    which the run is separated into unique segments (which are also shocRun
    instances). This class attempts to eliminate the tedium of computing
    calibration frames for different observational setups by enabling loops
    over various such groupings.
    """

    def __init__(self, factory=shocCampaign, *args, **kws):
        # set default default factory ;)
        super().__init__(factory, *args, **kws)

    def __repr__(self):
        w = len(str(len(self)))
        i = itt.count()
        return pformat(self, lhs=lambda s: f'{next(i):<{w}}: {s}', hang=True)

    def get_tables(self, attrs=None, titled=True, **kws):
        """
        Get a dictionary of tables (`motley.table.Table` objects) for this
        grouping. This method assists pretty printing groups of observation
        sets.

        Parameters
        ----------
        kws

        Returns
        -------

        """
        return shocCampaign.tabulate(self, attrs, titled=titled,
                                     filler_text='NO MATCH', **kws)

    def pformat(self, titled=True, braces=False, vspace=1, **kws):
        """
        Run pprint on each group
        """
        tables = self.get_tables(titled=titled, **kws)
        return vstack_groups(tables, not bool(titled),  braces, vspace)

    def pprint(self, titled=True, headers=False, braces=False, vspace=0, **kws):
        print(self.pformat(titled, braces, vspace, **kws))

    # def combine(self, func=None, args=(), **kws):
    #     return self.calls.combine(func, args, **kws)

    combine = MethodVectorizer('combine')  # , convert=shocCampaign
    stack = MethodVectorizer('stack')  # , convert=shocCampaign

    def merge_combine(self, func=None, *args, **kws):
        return self.combine(func, *args, **kws).stack().combine(
                func, *args, **kws)


    def select_by(self, **kws):
        out = self.__class__()
        out.update({key: obs
                    for key, obs in self.calls.select_by(**kws).items()
                    if len(obs)})
        return out

    # TODO: move methods below to MatchedObservations ???
    def comap(self, func, other, *args, **kws):
        out = self.__class__()
        for key, run in self.items():
            co = other[key]
            if run is None:
                continue

            out[key] = None if co is None else func(run, co, *args, **kws)
        return out

    def _comap_method(self, other, name, *args, **kws):

        missing = set(self.keys()) - set(other.keys())
        if missing:
            raise ValueError(
                'Can\'t map method {name} over group. Right group is missing '
                'values for the following keys: ' + '\n'.join(map(str, missing))
            )

        out = self.__class__()
        for key, run in self.items():
            co = other[key]
            if run is None:
                continue

            out[key] = (None if co is None else
                        getattr(run, name)(co, *args, **kws))
        return out

    def subtract(self, other):
        """
        Group by group subtraction

        Parameters
        ----------
        other :
            Dictionary with key-value pairs. Values are HDUs.

        Returns
        ------
        Bias subtracted shocObsGroups
        """

        return self._comap_method(other, 'subtract')

    def set_calibrators(self, darks=None, flats=None):
        """

        Parameters
        ----------
        darks
        flats

        Returns
        -------

        """
        darks = darks or {}
        flats = flats or {}
        for key, run in self.items():
            if run is None:
                continue

            bias = darks.get(key, keep)
            flat = flats.get(key, keep)
            for hdu in run:
                hdu.set_calibrators(bias, flat)

    def save(self, folder=None, name_format=None, overwrite=False):
        # since this calls `save` on polymorphic class HDU / Campaign and 'save'
        # method in each of those have different signature, we have to unpack
        # the keyword

        for key, obj in self.items():
            self[key] = obj.save(folder=folder,
                                 name_format=name_format,
                                 overwrite=overwrite)
