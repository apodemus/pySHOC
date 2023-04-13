

# std
import re
import operator as op
import warnings as wrn
import textwrap as txw
import itertools as itt
from pathlib import Path
from string import Template
from collections import defaultdict
from dataclasses import dataclass, field

# third-party
import numpy as np
from scipy import stats
from astropy.io import fits
from astropy.time import Time
from astropy.utils import lazyproperty
from astropy.coordinates import SkyCoord

# local
from scrawl.image import plot_image_grid
from pyxides import Groups, OfType
from pyxides.vectorize import MethodVectorizer, repeat
from motley.table import Table
from motley.utils import ALIGNMENT_MAP_INV, vstack
from motley.table.attrs import AttrTable, AttrColumn as Column
from obstools.image.noise import StdDev
from obstools.stats import median_scaled_median
from obstools.utils import convert_skycoords, get_coords_named
from obstools.campaign import (ImageHDU, PhotCampaign,
                               FilenameHelper as _FilenameHelper)
from recipes import pprint
from recipes.dicts import pformat
from recipes.oo.temp import temporary
from recipes.functionals import raises
from recipes.pprint.formatters import Decimal
from recipes.introspect import get_caller_name
from recipes.string import named_items, strings, sub, indent as indented

# relative
from .config import CONFIG
from .utils import str2tup
from .tel_info import tel_info, get_tel
from .printing import BraceContract
from .readnoise import readNoiseTables
from .timing import Trigger, shocTiming
from .convert_keywords import KEYWORDS as KWS_OLD_TO_NEW
from .header import HEADER_KEYS_MISSING_OLD, headers_intersect


# emit these warnings once only!
wrn.filterwarnings('once', 'Using telescope pointing coordinates')

# TODO
# __all__ = ['']

# TODO: can you pickle these classes


# ------------------------------ Instrument info ----------------------------- #

#            SHOC1, SHOC2
SERIAL_NRS = [5982, 6448]
SERVER_NAMES = {'SHOC2': 'shd',
                'SHOC1': 'sha'}

PIXEL_SIZE = 13e-6  # m

# Diffraction limits for shoc in pixels assuming
# green light:  λ = 5e-7

# F_RATIOS = {'1.9m': 4.85 }
DIFF_LIMS = {'1.9m':   1.22 * 5e-7 * 4.85 / PIXEL_SIZE}
#    '1.0m':  ,
#    'lesedi':,
#    'salt':  }

# ----------------------------- module constants ----------------------------- #

CALIBRATION_NAMES = ('bias', 'flat', 'dark')
OBSTYPE_EQUIVALENT = {'bias':    'dark',
                      'skyflat': 'flat'}
KNOWN_OBSTYPE = {*CALIBRATION_NAMES, *OBSTYPE_EQUIVALENT, 'object'}


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


def hms_latex(x, precision=0):
    return R'\hms{{{:02.0f}}}{{{:02.0f}}}{{{:04.1f}}}'.format(
        *pprint.nrs.sexagesimal(x.value, precision=precision)
    )


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


def slice_size(s):
    if isinstance(s, slice):
        return s.stop - s.start

    if isinstance(s, tuple):
        return tuple(map(slice_size, s))

    raise TypeError(f'Invalid type: {type(s)}')


def slice_sizes(l):
    return list(map(slice_size, l))

# ------------------------------ Helper classes ------------------------------ #


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
        elif isinstance(other, (tuple, list)):
            return (self.y, self.x) == tuple(other)

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
            self.A = CONFIG.empty_filter_string
            self.B = A
        else:
            self.A = A
            self.B = self.get(b)

    def get(self, long):
        # get short description like "U",  "z'", or "∅"
        if long == 'Empty':
            return CONFIG.empty_filter_string
        if long:
            return long.split(' - ')[0]
        return (long or CONFIG.empty_filter_string)

    def __members(self):
        return self.A, self.B

    def __repr__(self):
        return f'{self.__class__.__name__}{self.__members()}'

    # def __format__(self, spec):
    #     return next(filter(CONFIG.empty_filter_string.__ne__, self), '')

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
        return next(filter(CONFIG.empty_filter_string.strip, self), CONFIG.empty_filter_string)

    def to_header(self, header):
        _remap = {CONFIG.empty_filter_string: 'Empty'}
        for name, val in self.__dict__.items():
            header[f'FILTER{name}'] = _remap.get(val, val)


@dataclass()
class OutAmpMode:
    """
    Class to encapsulate the CCD output amplifier setting
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
                   OutAmpMode(header['OUTPTAMP'], header.get('GAIN', 0)),
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


class RollOverState:

    nr = 0
    parent = None

    def __init__(self, hdu):
        # Check whether the filenames contain '._X' an indicator for whether the
        # datacube reached the 2GB windows file size limit on the old shoc
        # server, and was consequently split into a sequence of fits cubes. The
        # timestamps of these need special treatment.

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


class FilenameHelper(_FilenameHelper):
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
        """Make a message tagged with class and function name."""
        return (f'{self.__class__.__name__}{sep}{get_caller_name(2)} '
                f'{Time.now()}: {message}')


# ------------------------------ HDU Subclasses ------------------------------ #


class shocHDU(ImageHDU, Messenger):

    __shoc_hdu_types = {}
    filename_format = None
    _FilenameHelperClass = FilenameHelper

    @classmethod
    def match_header(cls, header):
        return ('SERNO' in header)

    def __getnewargs__(self):
        self.logger.trace('unpickling: {}', self)
        return (None, self.header, self.obstype)

    def __new__(cls, data, header, obstype=None, *args, **kws):

        # handle direct init here eg:   shocDarkHDU(data, header)
        if cls in cls.__shoc_hdu_types.values():
            return super().__new__(cls)

        # assign class based on obstype and age
        obstype = (obstype or '').lower()
        if not obstype:
            obstype = header.get('OBSTYPE', '')

        # check if all the new keywords are present
        # NOTE:`_BaseHDU` creates a `_BasicHeader`, which does not contain
        # hierarch keywords, so for SHOC we cannot tell if it's the old format
        # header by checking those.
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

        # Choose subtypes of `shocHDU` here - simpler than using `match_header`
        class_name = f'shoc{age}{kind}{suffix}'
        cls.logger.trace(f'{class_name=}, {obstype=}')
        if class_name not in ['shocHDU', *cls.__shoc_hdu_types]:
            # pylint: disable=no-member
            cls.logger.warning('Unknown OBSTYPE: {!r:}', obstype)

        # cls = cls.__shoc_hdu_types.get(class_name, cls)
        return super().__new__(cls.__shoc_hdu_types.get(class_name, cls))

    def __init_subclass__(cls):
        cls.__shoc_hdu_types[cls.__name__] = cls

    def __init__(self, data=None, header=None, obstype=None, *args, **kws):

        # init PrimaryHDU
        super().__init__(data, header, *args, **kws)

        #
        self.camera = f'SHOC{SERIAL_NRS.index(header["SERNO"]) + 1}'
        self.location = getattr(tel_info.get(self.telescope), 'loc', None)
        self._coords = None     # placeholder
        self._wcs = None
        # # field of view
        # self.fov = self.get_fov()

        # image binning
        self.binning = Binning(header[f'{_}BIN'] for _ in 'VH')

        # sub-framing
        # `SUBRECT` stores the sub-region of the full CCD array captured for the
        # observation. NOTE that for some reason the ysub order is reversed in
        # the header
        subrect = np.array(header['SUBRECT'].split(','), int)
        xsub, ysub = subrect.reshape(-1, 2) // tuple(self.binning)
        self.subrect = tuple(map(slice, *zip(ysub[::-1], xsub)))

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
        tel = tel_info[telescope]
        self.header['TELESCOP'] = tel.name
        self.location = tel.loc

    @property
    def target(self):
        return self.header.get('OBJECT')

    @target.setter
    def target(self, name):
        ot = self.target
        trg = self.header['OBJECT'] = str(name)
        if ot != trg:
            del self.coords

    @property
    def diffraction_limit(self):
        # diffraction limit 1.9m
        return DIFF_LIMS.get(self.telescope, None)

    def get_server_path(self, server=CONFIG.remote.server):
        if None in (self.file.path, self.telescope):
            return

        year, month, day, *_ = tuple(self.t.t0.ymdhms)
        tel = get_tel(self.telescope, metric=False)

        path = (f'/data/{tel}/{SERVER_NAMES[self.camera]}/'
                f'{year}/{month:0>2d}{day:0>2d}/'
                f'{self.file.name}')

        if server:
            path = f'{server}{path}'

        return Path(path)

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
        return RollOverState(self)

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

        # will flip CON images x axis (2nd axis)
        return ImageOrienter(self, x=self.readout.isCON)

    # def set_calibrators(self)

    def get_fov(self):
        fov = tel_info[self.telescope].fov
        # scale by fractional size of subframe image
        full_frame = np.floor_divide(1028, tuple(self.binning))
        return fov * (slice_size(self.subrect) / full_frame)

    def get_rotation(self):
        """
        Get the instrument rotation (position angle) wrt the sky in radians.
        This value seems to be constant for most SHOC data from various
        telescopes that I've checked. You can measure this yourself by using the
        `obstools.campaign.PhotCampaign.coalign_dss` method.
        """
        return 0  # -0.04527

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

    def combine(self, func=None, sigma_func=None, args=(), **kws):
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
                f'Parameter `func` should be callable. Received {type(func)}.'
            )

        # check if single image
        if (self.ndim == 2) or ((self.ndim == 3) and len(self.data) == 1):
            return self

        # log some info
        msg = f'{func.__name__} of {self.nframes} images from {self.file.name}.'
        self.logger.info(msg)

        # combine across images
        kws.setdefault('axis', 0)
        # TODO: self.calibrated ???
        combined = func(self.data, *args, **kws)

        hdu = shocHDU(combined, self.header)

        if sigma_func is None:
            if func is np.mean:
                # hdu.noise_model valid
                pass
            elif func is np.median:
                # std = self.noise_model.std_of_median(self.data)
                hdu.noise_model = StdDev(self.noise_model.std_of_median(self.data))
            else:
                raise ValueError('Please provide function to compute uncertainty.')

        # FIXME: MasterHDU not used here :(((

        # update header
        hdu.header['MASTER'] = True
        hdu.header['NCOMBINE'] = self.nframes
        hdu.header.add_history(self.message(msg, sep=''))
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
            return f"{name}.{ext.lstrip('.')}"

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
            self.logger.info('Creating directory: {!r:}', str(path.parent))
            path.parent.mkdir()

        action = ('Saving to  ', 'Overwriting')[path.exists()]
        self.logger.info('{:s} {!r:}', action, str(path))
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
    # - Newberry 91

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

    # def get_table(self, run, attrs=None, **kws):
    #     #
    #     self._foot_fmt = ' {flag}  {info}'
    #     table = super().get_table(run, attrs, **kws)

    #     # HACK: compacted `filter` displays 'A = ∅' which is not very clear. Go more
    #     # verbose again for clarity
    #     if table.summary.items and table.summary.loc != -1:
    #         replace = {'A': 'filter.A',
    #                    'B': 'filter.B'}
    #         table.summary.items = {replace.get(name, name): item
    #                                for name, item in table.summary.items.items()}
    #         table.inset = table.summary()
    #     return table

    def to_latex(self, style='table', indent=2, **kws):
        fname = f'_to_{style}'
        if not hasattr(LatexWriter, fname):
            raise NotImplementedError(fname)

        writer = LatexWriter(self.parent)
        worker = getattr(writer, fname)
        table = worker(**kws).replace('\n    ', f'\n{" " * indent}')
        return '\n'.join(map(str.rstrip,  table.splitlines()))

    def to_xlsx(self, path):
        tabulate = AttrTable.from_columns({
            'file.name':          Column('filename', align='<'),
            'timing.t0.datetime': Column('Time', '[UTC]', fmt='YYYY-MM-DD HH:MM:SS'),
            'timing.exp':         Column('Exposure', '[s]', fmt='0.???'),
            'timing.duration':    Column(convert=lambda t: t.value / 86400,
                                         fmt='[HH]"ʰ"MM"ᵐ"SS"ˢ"', unit='[hms]',
                                         total=True),
            'telescope':          ...,
            'filters.name':       Column('Filter'),
            'camera':             ...,
            'readout.mode':       Column(convert=str),
            # 'nframes':            Column('n', total=True),
            # 'binning':            Column('bin', unit='y, x', header_level=1),
            'binning.y':          ...,
            'binning.x':          ...,

        },
            header_levels={'binning': 1},
            show_groups=False,
            title='Observational Setup'
        )

        tabulate.parent = self.parent
        return tabulate.to_xlsx(path,
                                align={'filename': '<',
                                       'mode': '<',
                                       'binning': '>',
                                       ...: '^'},
                                header_formatter=str.title,
                                )
        # widths={'binning': 5})

        # tabulate = AttrTable.from_spec({
        #     'Filename':        'file.name',
        #     'Time [UTC]':      'timing.t0.datetime:YYYY-MM-DD HH:MM:SS',
        #     'Exp [s]':         'timing.exp',
        #     'Duration [s]':    'timing.duration',
        #     'camera':          '',
        #     'telescope':       '',
        #     'Filter':           'filters.name',
        #     'binning.y':       '',
        #     'binning.x':       '',
        #     'Mode':            'readout.mode!s'
        # },
        #     converters={'timing.duration': lambda t: t.value / 86400},
        #     header_levels={'binning', 1},
        #     show_groups=False,
        #     totals='timing.duration'
        # )


class LatexWriter:
    def __init__(self, container):
        self.parent = container

    def _tabular_body(self,
                      booktabs=True, unicodemath=False,
                      flag_fmt='$\:^{{{flag}}}$',
                      foot_fmt=R'\hspace{{1eM}}$^{{{flag}}}$\ {info}\\',
                      summary_fmt=R'\hspace{{1eM}}{{{key} = {val}¿ [{unit}]?}}\\',
                      timing_flags=None,
                      **kws):

        if timing_flags is None:
            # †  ‡  §
            if unicodemath:
                timing_flags = {-1: '!',  # ⚠ not commonly available in all fonts
                                0:  '↓',
                                1:  '⟳'}
            else:
                timing_flags = {-1: '!',
                                0:  '*',
                                1:  '\dagger'}  # '

        # change flags symbols temporarily
        with temporary(Trigger, FLAG_SYMBOLS=timing_flags):
            # controls which attributes will be printed
            tabulate = TableHelper.from_columns({
                # FIXME: these columns just change the titles from tabulate
                'timing.t0':       Column('$t_0$', fmt='{.iso}'.format, unit='UTC',
                                          flags=op.attrgetter('t.trigger.t0_flag')),
                'telescope':       Column('Telescope'),
                'camera':          Column('Camera'),
                'filters.name':    Column('Filter'),
                'nframes':         Column('n', total=True),
                'readout.mode':    Column('Readout Mode'),
                'binning':         Column('Binning', unit='y, x', align='^',
                                          fmt=R'{0.y}\times{0.x}'),
                'timing.exp':      Column('$t_{\mathrm{exp}}$', fmt=str, unit='s',
                                          flags=op.attrgetter('t.trigger.texp_flag')),
                'timing.duration': Column('Duration', fmt=hms_latex, unit='hh:mm:ss',
                                          total=True)},
                row_nrs=1,
                frame=False,
                hlines=False,
                col_head_style=None,
                borders={...: '& ', -1: r'\\'},
                summary=dict(footer=True, n_cols=1, bullets='', align='<',
                             pillars=['t0'], fmt=summary_fmt),
                too_wide=False,
                insert=({
                    -2:                         R'\toprule',
                    0:                          R'\midrule',
                    (n := len(self.parent)):    R'\midrule',
                    n + 1:                      R'\bottomrule'
                } if booktabs else {}),
                flag_fmt=flag_fmt,
                footnotes=(footnotes := Trigger.get_flags()),
                foot_fmt=foot_fmt
            )
            tabulate.parent = self.parent
            with temporary(Table, _nrs_header='\#'):  # HACK for latex symbol
                tbl = tabulate(title=False, col_groups=None, **kws)

        # HAck out the footnotes for formatting downstream
        footnotes = tbl.footnotes[:]
        tbl.footnotes = []
        return tbl, footnotes

    def _tabular_colspec(self, tbl, indent=4):
        col_spec = list(map(ALIGNMENT_MAP_INV.get, tbl.align[tbl._idx_shown]))
        # letters = list(map(chr, range(65, 65 + len(col_spec))))
        letters = [f'{i:c}' for i in range(65, 65 + len(col_spec))]
        letters[0] = f'% {letters[0]}'
        col_spec = Table([letters, col_spec],
                         col_borders=[*['& '] * (len(letters) - 1), ''],
                         frame=False, too_wide=False)
        # col_spec._idx_shown = tbl._idx_shown
        widths = tbl.col_widths[tbl._idx_shown]
        col_spec.max_width = 1000  # HACK since split is happening here... # FIXME
        col_spec.col_widths = np.array([(l, w)[l < w]
                                        for l, w in zip(map(len, letters), widths)])
        col_spec.truncate_cells(widths)
        letters, spec = indented(str(col_spec).rstrip(), indent).splitlines()

        return '\n'.join((letters, spec.replace('&', ' ')))

    def _to_table(self, star='*', pos='ht!',
                  options=R'\centering',
                  caption=None, cap=None,
                  label=None, env='tabular',
                  booktabs=True, unicodemath=False):
        indent = 4
        # options
        star = '*' if star else ''
        cap = f'[{{{cap!s}}}]' if cap else ''
        caption = rf'\caption{cap}{{{caption}}}' if caption else ''
        label = f'\\label{{{label}}}' if label else ''

        #
        tbl, footnotes = self._tabular_body(booktabs, unicodemath)

        # get column spec
        return Template(txw.dedent(R'''
            \begin{table$star}[$pos]
                $options
                $caption
                $label
                %
                \begin{$env}{%
                $colspec
                }
                $body
                \end{$env}

                \footnotesize
                \raggedright
                $footnotes

            \end{table$star}
            ''')).substitute(
            locals(),
            colspec=self._tabular_colspec(tbl),
            body=indented(str(tbl), indent),
            footnotes=f'\n{" " * indent}'.join(np.char.strip(footnotes))
        )

    def _to_ctable(self, options='star, nosuper', pos='ht!',
                   caption=None, cap=None, label=None,
                   hspace_body='-1.5cm', hspace_footer='-1.2cm',
                   booktabs=True, unicodemath=True):
        indent = 4
        options = f',\n{" " * indent}'.join(
            filter(None, (f'{options}',
                          f'{pos     = !s}',
                          f'caption = {{{caption!s}}}' if caption else '',
                          f'cap     = {{{cap!s}}}' if cap else '',
                          f'{label   = !s}' if label else ''))
        )

        tbl, footnotes = self._tabular_body(
            booktabs, unicodemath,
            flag_fmt=R'\tmark[$\:{{{flag}}}$]',
            foot_fmt=R'\tnote[$^{{{flag}}}\:$]{{{info}}}',
            summary_fmt=R'\tnote[{{}}]{{{key} = {val}¿ {unit}?}}'
        )

        hack = '% ' if (len(tbl._idx_shown) < 7) else ''
        return Template(txw.dedent(R'''
            \ctable[
                $options,
                % hack to move table into left margin
                ${hack}doinside= {\hspace*{$hspace_body}}
            ]{
                % column spec
                $colspec
            }{
                % footnotes
                % also move footnotes to keep alignment consistent
                $hack\hspace*{$hspace_footer}
                $footnotes
            }{
                % tabular body
                $body
            }
            ''')).substitute(
            locals(),
            body=indented(tbl, indent),
            colspec=self._tabular_colspec(tbl),
            footnotes=f'\n{" " * indent}'.join(footnotes)
        )

    def _to_tabularray(self, options='',  # pos='ht!',
                       caption=None, cap=None, label=None,
                       hspace_body='-1.5cm', hspace_footer='-1.2cm',
                       booktabs=True, unicodemath=True):

        indent = 4  # since Template is hard coded indent 4
        options = f',\n{" " * indent}'.join(
            filter(None, (f'{options}',
                          #   f'{pos     = !s}',
                          f'caption = {{{caption!s}}}' if caption else '',
                          f'entry   = {{{cap!s}}}' if cap else '',
                          f'{label   = !s},' if label else ''))
        )

        tbl, footnotes = self._tabular_body(
            booktabs, unicodemath,
            flag_fmt=R'\TblrNote{{{flag}}}',
            foot_fmt=R'note{{${{{flag}}}$}} = {{{info}}},',
            summary_fmt=R'remark{{{key}}} = {{{val}{unit}}},'
        )

        return Template(txw.dedent(R'''
            \begin{tblr}[
                % outer spec
                $options
                % footnotes
                % \hspace*{$hspace_footer}
                $footnotes
            ]{% inner spec
                % Column headers (to appear on every page for page-split tables)
                row{1-2} = {c, m, font=\bfseries},
                rowhead = 2,
                % rowfoot = 1,
                colspec={
                $colspec
                },
            }
            $body
            \end{tblr}
            ''')).substitute(
            locals(),
            colspec=self._tabular_colspec(tbl, indent),
            body=indented(tbl, indent),
            footnotes=f'\n{" " * indent}'.join(footnotes)
        )


# ------------------------------------- ~ ------------------------------------ #


class shocCampaign(PhotCampaign, OfType(shocHDU), Messenger):

    # Pretty repr
    pretty = BraceContract(brackets='', per_line=1, indent=4,  hang=True)

    # This controls which attributes will be tabulated for `pprint` method
    tabulate = TableHelper.from_columns(
        {'file.stem':          Column('filename'),
         'telescope':          Column('tel'),
         'camera':             ...,
         'target':             ...,
         'obstype':            ...,
         'filters.name':       Column('filter'),
         'nframes':            Column('n', total=True,
                                      fmt=Decimal(0, thousands=' ')),
         'binning':            Column('bin', unit='y, x', align='^'),
         'readout.mode':       ...,
         'timing.t0':          Column('t0', unit='UTC',
                                      flags=op.attrgetter('t.trigger.t0_flag')),
         'timing.exp':         Column('tExp', fmt=str, unit='s',
                                      flags=op.attrgetter('t.trigger.texp_flag')),
         'timing.duration':    Column(fmt=hms, unit='hms', total=True)
         #   'ishape':     Column(unit='y, x'),
         #   'readout.preAmpGain':     Column('preAmp', unit='y, x'),
         #   'readout.mode':  ...,
         #  'readout.mode.frq':   ...,
         #  'readout.mode.outAmp':    ...,
         #   'filters.B':          ...,
         },
        row_nrs=1,
        summary=True,
        title_style=dict(fg=('underline', 'bold'), bg='b'),
        too_wide=False,
        footnotes=Trigger.get_flags()
    )

    @classmethod
    def new_groups(cls, *keys, **kws):
        return shocObsGroups(cls, *keys, **kws)

    # def map(self, func, *args, **kws):
    #     """Map and arbitrary function onto the data of each observation"""
    #     shocHDU.data.get

    # @expose.args
    @classmethod
    def load(cls, files_or_dir, recurse=False, extensions=('fits', 'FITS'), **kws):

        run = super().load(files_or_dir, recurse, extensions, allow_empty=False,
                           **kws)

        # look for gps timesstamps if needed
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
        if run.missing_gps() and (gps := run.search_gps_file()):
            cls.logger.opt(colors=True).info(
                "Using gps timestamps from <b><g>'{!s:}'</></>",
                # motley.stylize(
                #     "Using gps timestamps from '{!r:|darkgreen,B}'"),
                gps.relative_to(gps.parent.parent)
            )
            run.provide_gps(gps)

        # Standardize naming convention for telscopes
        said = False
        for hdu in run:
            if hdu.telescope and (tel := tel_info[hdu.telescope].name) != hdu.telescope:
                if not said:
                    cls.logger.info('Switching to metric names for telescopes.')
                    said = True
                hdu.telescope = tel

        return run

    def save(self, filenames=(), folder=None, name_format=None, overwrite=False):
        if filenames:
            assert len(filenames) == len(self)
        else:
            filenames = self.calls.get_save_name(name_format)

        if len(set(filenames)) < len(self):
            from recipes.lists import tally
            dup = [fn for fn, cnt in tally(filenames).items() if cnt > 1]
            self.logger.warning('Duplicate filenames: {:s}', dup)

        hdus = self.calls.save(None, folder, name_format, overwrite)
        # reload
        self[:] = self.__class__(hdus)
        return self

    def thumbnails(self, statistic='mean', size=5, subset=None, depth=None,
                   title='file.name', calibrated=False, **kws):
        """
        Display a sample image from each of the observations laid out in a grid.
        Image thumbnails are all the same size, even if they are shaped
        differently, so the size of displayed pixels may be different

        Parameters
        ----------
        statistic: str
            statistic for the sample
        size: int
            Number of images in each observation to combine.
        subset: int or tuple or slice or ellipsis or None
            Index interval for subset of the full observation to draw from, by
            default use (0, size).
        title: str
            Key mapping to attribute of item that will be used as title.
        calibrated: bool
            Calibrate the image if bias / flat field available.

        Other keyword parameters are passed to `plot_image_grid`.

        Returns
        -------
        ImageGrid
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
            subset = (0, size)

        # get sample images
        if depth is None:
            func, args = f'sampler.{statistic}', (size, subset)
        else:
            func, args = 'get_sample_image', (statistic, depth, subset)

        sample_images = self.calls(func, *args)

        if calibrated:
            # print(list(map(np.mean, sample_images)))
            sample_images = [hdu.calibrated(im) for hdu, im in
                             zip(self, sample_images)]
            # print(list(map(np.mean, sample_images)))

        return plot_image_grid(sample_images, titles=titles, **kws)

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

        return MatchedObservations(self, other)(exact, closest, cutoffs, keep_nulls)

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

        if not isinstance(other, fits.PrimaryHDU):
            raise TypeError(f'Expected shocHDU, got {type(other)}')

        if len(other.shape) > 2:
            raise TypeError(
                'The input hdu contains multiple images instead of a single '
                'image. Do `combine` to compute the master bias.'
            )

        return self.__class__([hdu.subtract(other) for hdu in self])

    def set_calibrators(self, darks=None, flats=None):
        if darks:
            matched = self.match(darks, *MATCH_DARKS)
            matched.left.set_calibrators(matched.right)

        if flats:
            matched = self.match(flats, *MATCH_FLATS)
            matched.left.set_calibrators(flats=matched.right)

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
            if selected := self[cxx.separation(obj_coords).deg < tolerance]:
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
        return self.select_by(**{'t.trigger.t0_flag': bool}).sort_by('t.t0')

    def search_gps_file(self):
        # get common root folder
        if folder := self.files.common_root():
            gps = folder / 'gps.sast'
            if gps.exists():
                return gps

    def provide_gps(self, filename):
        # read file with gps triggers
        path = Path(filename)
        names, times = np.loadtxt(str(filename), str, unpack=True)
        need_gps = self.select_by(**{'t.trigger.t0_flag': bool}).sort_by('t.t0')
        assert need_gps == self[names].sort_by('t.t0')

        # assert len(times) == len(self)
        tz = 2 * path.suffix.endswith('sast')
        for obs, t0 in zip(need_gps, times):
            t = obs.t
            t.t0 = t.from_local(t0, tz)
            t.trigger._header_gps_info_missing['t0'] = False

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

    def tabulate(self, attrs=None, titled=True, **kws):
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
        return shocCampaign.tabulate.get_tables(
            self, attrs, titled=titled, filler_text='NO MATCH', **kws
        )

    def pformat(self, titled=True, braces=False, vspace=1, **kws):
        """
        Run pprint on each group
        """
        tables = self.tabulate(titled=titled, **kws)
        return vstack.from_dict(tables, not bool(titled), braces, vspace)

    def pprint(self, titled=True, headers=False, braces=False, vspace=0, **kws):
        print(self.pformat(titled, braces, vspace, **kws))

    #
    combine = MethodVectorizer('combine')  # , convert=shocCampaign
    stack = MethodVectorizer('stack')  # , convert=shocCampaign

    def merge_combine(self, func=None, *args, **kws):
        #
        return self.combine(func, *args, **kws).stack().combine(func, *args, **kws)

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

    def _comap_method(self, other, name, handle_missing=raises(ValueError),
                      *args, **kws):
        if missing := set(self.keys()) - set(other.keys()):
            handle_missing(
                f"Can't map method {name!r} over groups. Right group is "
                f'missing values for the following '
                + named_items(strings(missing), 'key', fmt='\n'.join)
            )

        out = self.__class__()
        for key, run in self.items():
            co = other[key]
            if not run:
                continue

            out[key] = (getattr(run, name)(co, *args, **kws)
                        if co else None)

        return out

    def subtract(self, other, handle_missing=raises(ValueError)):
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

        return self._comap_method(other, 'subtract', handle_missing)

    def set_calibrators(self, darks=None, flats=None):
        """

        Parameters
        ----------
        darks
        flats

        Returns
        -------

        """
        from obstools.image.calibration import keep

        darks = darks or {}
        flats = flats or {}

        for key, run in self.items():
            if run is None:
                continue

            dark = darks.get(key, keep)
            flat = flats.get(key, keep)
            for hdu in run:
                hdu.set_calibrators(dark, flat)

    def save(self, folder=None, name_format=None, overwrite=False):
        # since this calls `save` on polymorphic class HDU / Campaign and 'save'
        # method in each of those have different signature, we have to unpack
        # the keyword

        for key, obj in self.items():
            self[key] = obj.save(folder=folder,
                                 name_format=name_format,
                                 overwrite=overwrite)
