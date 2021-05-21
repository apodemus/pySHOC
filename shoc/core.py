# __version__ = '3.14'


# std libs
import operator as op
from obstools.phot.campaign import FnHelp
import functools as ftl
import itertools as itt
from collections import defaultdict
import warnings as wrn
from pathlib import Path

# third-party libs
import numpy as np
import more_itertools as mit
from dataclasses import dataclass, field
from astropy.utils import lazyproperty
from astropy.io.fits.hdu import PrimaryHDU
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.time import Time

from scipy import stats

# local libs
from motley import codes
from motley.utils import overlay
from motley.table import AttrTable
from obstools.phot.campaign import PhotCampaign, HDUExtra
from obstools.stats import median_scaled_median
from obstools.image.calibration import keep
from obstools.utils import get_coords_named, convert_skycoords
from recipes import pprint
from recipes.sets import OrderedSet
from recipes.containers import Grouped, OfType
from recipes.introspect import get_caller_name
from recipes.string import sub, remove_prefix
from recipes import bash
from scrawl.imagine import plot_image_grid

# relative libs
# from shoc.image.sample import ResampleFlip
from .timing import shocTiming, shocTimingOld, Trigger
from .readnoise import readNoiseTables
from .convert_keywords import KEYWORDS as KWS_OLD_TO_NEW
from .header import headers_intersect, HEADER_KEYS_MISSING_OLD
# from recipes.decor import expose
from .pprint import BraceContract

from motley.table import Table

# only emit this warning once!
wrn.filterwarnings('once', 'Using telescope pointing coordinates')


# TODO
# __all__ = ['']

# TODO: can you pickle these classes
# FIXME: shocHDU classes are assigned based on header keywords which are often
#   incorrect!!!  Think of a better way of re-assigning the HDU classes

# ------------------------------ Instrument info ----------------------------- #

#            SHOC1, SHOC2
SERIAL_NRS = [5982, 6448]

# ------------------------------ Telescope info ------------------------------ #

# Field of view of telescopes in arcmin
FOV74 = (1.29, 1.29)
FOV74r = (2.79, 2.79)  # with focal reducer
FOV40 = (2.85, 2.85)
FOV_LESEDI = (5.7, 5.7)
# fov30 = (3.73, 3.73) # decommissioned

FOV = {'74': FOV74, '1.9': FOV74,
       '40': FOV40, '1.0': FOV40, '1': FOV40,
       # '30': fov30, '0.75': fov30, # decommissioned
       }
FOVr = {'74': FOV74r,
        '1.9': FOV74r}

# Exact GPS locations of telescopes
TEL_GEO_LOC = {'74in': (20.81167, -32.462167, 1822),
               '40in': (20.81, -32.379667, 1810),
               'lesedi': (20.8105, -32.379667, 1811),
               'salt': (20.810808, 32.375823, 1798)}
LOCATIONS = {tel: EarthLocation.from_geodetic(*geo)
             for tel, geo in TEL_GEO_LOC.items()}


# ----------------------------- module constants ----------------------------- #
CALIBRATION_NAMES = ('bias', 'flat', 'dark')
OBSTYPE_EQUIVALENT = {'bias': 'dark'}
EMPTY_FILTER_NAME = '∅'

# Attributes for matching calibration frames
ATT_EQUAL_DARK = ('camera', 'binning',  # subrect
                  'readout.frq', 'readout.preAmpGain', 'readout.outAmp.mode')  # <-- 'readout'
ATT_CLOSE_DARK = ('readout.outAmp.emGain', 'timing.exp')

ATT_EQUAL_FLAT = ('telescope', 'camera', 'binning', 'filters')
ATT_CLOSE_FLAT = ('timing.date', )

MATCH = {}
MATCH['flats'] = MATCH['flat'] = MATCH_FLATS = (ATT_EQUAL_FLAT, ATT_CLOSE_FLAT)
MATCH['darks'] = MATCH['dark'] = MATCH_DARKS = (ATT_EQUAL_DARK, ATT_CLOSE_DARK)


# ----------------------------- helper functions ----------------------------- #

def str2tup(keys):
    if isinstance(keys, str):
        keys = keys,  # a tuple
    return keys


def apply_stack(func, *args, **kws):  # TODO: move to proc
    # TODO:  MULTIPROCESS HERE!
    return func(*args, **kws)


def split_dist(v0, v1):
    # split sub groups for closest match. closeness of runs is measured by
    # the sum of the relative distance between attribute values.
    # todo: use cutoff
    
    v0 = np.c_[v0][:, None]
    v1 = np.c_[v1][None]
    delta_mtx = np.abs(v0 - v1)

    # with wrn.catch_warnings():
    #     wrn.filterwarnings('ignore', 'divide by zero', RuntimeWarning)
    # scale = delta_mtx.max(1, keepdims=True))
    dist = delta_mtx.sum(-1)
    selection = (dist == dist.min(1, keepdims=True))
    
    for l1 in selection:
        # there may be multiple HDUs that are equidistant from the selected set
        # group these together
        l0 = (l1 == selection).all(1)
        # values are the same for this group (selected by l0), so we can just
        # take the first row of attribute values
        # vals array to tuple for hashing
        yield tuple(v0[l0][0, 0]), l0, l1, delta_mtx[l0][:, l1]


# def get_id(hdu):
#     """
#     Unique identifier (hash) for hdus
#     """
#     # use the hashed header as identifier for the file in HISTORY
#     fid = string_to_int(str(hdu.header))
#     if hdu._file:
#         fid = " ".join((str(fid), hdu.filename))
#     return fid


# def string_to_int(s):
#     # persistent hash https://stackoverflow.com/a/2511232
#     return int(''.join(('%.3d' % ord(x) for x in s)))


# def int_to_string(n):
#     s = str(n)
#     return ''.join((chr(int(s[i:i + 3])) for i in range(0, len(s), 3)))


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
    """Simple class to represent position of the filter wheels"""

    def __init__(self, a, b):
        self.A = self.get(a)
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

    def __format__(self, spec):
        return next(filter(EMPTY_FILTER_NAME.__ne__, self), '')

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


class shocFnHelp(FnHelp):
    @property
    def nr(self):
        """
        File sequence number 
            eg. '0010' in 'SHA_20200729.0010.fits'
        """
        return self.path.suffixes[0].lstrip('.')


class Messenger:
    def message(self, message, cls_name=True):
        """Make a message tagged with class and function name"""
        return (f'{self.__class__.__name__}.{get_caller_name(2)} {Time.now()}: '
                f'{message}')

# ------------------------------ HDU Subclasses ------------------------------ #


class shocHDU(HDUExtra, Messenger):
    _FnHelper = shocFnHelp
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

        obstype = (obstype or '').lower()
        if not obstype:
            obstype = header.get('OBSTYPE', '')

        # check if all the new keywords are present
        age, kind, suffix = '', '', 'HDU'
        if any((kw not in header for kw in HEADER_KEYS_MISSING_OLD)):
            age = 'Old'

        # calibration stacks
        obstype = OBSTYPE_EQUIVALENT.get(obstype, obstype)
        if obstype in CALIBRATION_NAMES:
            kind = obstype.title()
            if (header['NAXIS'] == 2 and header['MASTER']):
                suffix = 'Master'
        elif obstype not in ('', 'object'):
            cls.logger.warning('Unknown OBSTYPE: %r', obstype)

        #
        class_name = f'shoc{age}{kind}{suffix}'
        # cls = cls.__shoc_hdu_types.get(class_name, cls)
        # print(f'{class_name=:}; {cls=:}')
        return super().__new__(cls.__shoc_hdu_types.get(class_name, cls))

    def __init_subclass__(cls):
        cls.__shoc_hdu_types[cls.__name__] = cls

    def __init__(self, data=None, header=None, obstype=None, *args, **kws):
        # init PrimaryHDU
        super().__init__(data=data, header=header, *args, **kws)

        serial = header['SERNO']
        self.camera = f'SHOC{SERIAL_NRS.index(serial) + 1}'
        self.telescope = header.get('TELESCOP')
        self.location = LOCATIONS.get(self.telescope)

        # # field of view
        # self.fov = self.get_fov()

        # image binning
        self.binning = Binning(header['%sBIN' % _] for _ in 'VH')

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

        # filters
        self.filters = Filters(*(header.get(f'FILTER{_}', 'Empty')
                                 for _ in 'AB'))

        # object name
        self.target = header.get('OBJECT')
        self.obstype = header.get('OBSTYPE')

        # # manage on-the-fly image orientation
        # self.oriented = ImageOrienter(self, x=self.readout.isEM)

        # # manage on-the-fly calibration for large files
        # self.calibrated = ImageCalibration(self)

    def __str__(self):
        attrs = ('t.t0_flagged', 'binning', 'readout.mode', 'filters')
        info = ('', ) + op.attrgetter(*attrs)(self) + ('',)
        sep = ' | '
        return f'<{self.__class__.__name__}:{sep.join(map(str, info))}>'

    # def __repr__(self):

    @property
    def nframes(self):
        """Total number of images in observation"""
        if self.ndim == 2:
            return 1
        return self.shape[0]  #

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
        from obstools.image.orient import ImageOrienter
        # will flip EM images x axis (2nd axis)
        return ImageOrienter(self, x=self.readout.isCON)

    @lazyproperty
    def timing(self):
        # Initialise timing
        # this is delayed from init on the hdu above since this class may
        # initially be created with a `_BasicHeader` only, in which case we
        # will not yet have all the correct keywords available yet to identify
        # old vs new shoc data
        return shocTiming(self)

    # alias
    t = timing

    @lazyproperty
    def coords(self):
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

        # TODO: optionally query for named sources in this location
        if coords:
            warnings.warn('Using telescope pointing coordinates')

        return coords

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

    def get_fov(self, telescope=None, unit='arcmin', with_focal_reducer=False):
        """
        Get image field of view

        Parameters
        ----------
        telescope
        with_focal_reducer
        unit

        Returns
        -------

        Examples
        --------
        >>> cube = shocObs.load(filename)
        >>> cube.get_fov(1)               # 1.0m telescope
        >>> cube.get_fov(1.9)             # 1.9m telescope
        >>> cube.get_fov(74)              # 1.9m
        >>> cube.get_fov('74in')          # 1.9m
        >>> cube.get_fov('40 in')         # 1.0m
        """

        # PS. welcome to the new millennium, we use the metric system now
        if telescope is None:
            telescope = getattr(self, 'telescope', self.header.get('telescop'))

        if telescope is None:
            raise ValueError('Please specify telescope to get field of view.')

        telescope = str(telescope)
        telescope = telescope.rstrip('inm ')  # strip "units" in name
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

    def get_rotation(self):
        """
        Get the instrument rotation (position angle) wrt the sky in radians.
        This value seems to be constant for most SHOC data from various
        telescopes that I've tested. You can measure this yourself by using the

        `obstools.phot.campaign.PhotCampaign.coalign_dss` method.
        """
        return -0.04527

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

    @property
    def needs_timing(self):
        """
        check for date-obs keyword to determine if header information needs
        updating
        """
        return ('date-obs' not in self.header)
        # TODO: is this good enough???

    # ------------------------------------------------------------------------ #
    # image arithmetic

    def combine(self, func, *args, **kws):
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

        assert callable(func), 'Func needs to be callable'

        # check if single image
        if (self.ndim == 2) or ((self.ndim == 3) and len(self.data) == 1):
            return self

        # log some info
        msg = f'{func.__name__} of {self.nframes} images from {self.file.name}'
        msg = self.message(msg)
        self.logger.info(remove_prefix(msg, self.__class__.__name__))

        # combine across images
        kws.setdefault('axis', 0)
        # TODO: self.calibrated ???
        hdu = self.__class__(func(self.data, *args, **kws), self.header)

        # update header
        hdu.header['MASTER'] = True
        hdu.header['NCOMBINE'] = self.nframes
        hdu.header.add_history(msg)
        return hdu

    def subtract(self, bias):
        """
        Subtract the image

        Parameters
        ----------
        bias

        Returns
        -------

        """
        # This may change the data type.
        self.data = self.data - bias.data

        # update history
        msg = self.message(f'Subtracted image {bias.file.name}')
        self.header.add_history(msg)
        return self

    def auto_update_header(self, **kws):
        # FIXME: better to handle these as property and update header there ?
        new_kws = {'obstype': self.obstype}
        if self.target:
            new_kws['object'] = self.target

        if 'coords' in self.__dict__ and self.coords:
            ra, dec = self.coords.ra, self.coords.dec
            new_kws.update(
                objra=ra.to_string('hourangle', sep=':', precision=1),
                objdec=dec.to_string('deg', sep=':', precision=1),
            )

        self.header.update({**new_kws, **kws})

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

        # any changes to attrs maps to fits header before save
        self.auto_update_header()

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
        return path


# FILENAME_TRANS = str.maketrans({'-': '', ' ': '-'})

class shocOldHDU(shocHDU):
    def __init__(self, data, header, *args, **kws):

        super().__init__(data, header, *args, **kws)

        # fix keywords
        for old, new in KWS_OLD_TO_NEW:
            if old in header:
                self.header.rename_keyword(old, new)

    @lazyproperty
    def timing(self):
        return shocTimingOld(self)


class shocCalibrationHDU(shocHDU):
    _combine_func = None  # place-holder

    # TODO: set target=''

    def get_coords(self):
        return

    def combine(self, func=_combine_func, *args, **kws):
        return super().combine(func or self._combine_func, *args, **kws)


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


class shocFlatHDU(shocCalibrationHDU):

    filename_format = '{obstype}-{t.date:d}-{telescope}-{camera}-{binning}-{filters}'
    _combine_func = staticmethod(median_scaled_median)
    # default combine algorithm first median scales each image, then takes
    # median across images


class shocOldDarkHDU(shocOldHDU, shocDarkHDU):
    pass


class shocOldFlatHDU(shocOldHDU, shocDarkHDU):
    pass

# class shocBiasHDU(shocDarkHDU):
#     # alias
#     pass


class shocDarkMaster(shocDarkHDU):
    pass


class shocFlatMaster(shocFlatHDU):
    pass

# -------------------------- Pretty printing helpers ------------------------- #


def hms(t):
    """sexagesimal formatter"""
    return pprint.hms(t.to('s').value, unicode=True, precision=1)


def get_table(r, attrs=None, **kws):
    return r.table.get_table(r, attrs, **kws)


class TableHelper(AttrTable):

    def get_table(self, run, attrs=None, **kws):
        # Add '*' flag to times that are gps triggered
        flags = {}
        postscript = []
        for key, attr in [('timing.t0', 'timing.trigger.flag', ),
                          ('timing.exp', 'timing.trigger.loop_flag')]:
            head = self.aliases[key]
            flg = flags[head] = run.attrs(attr)
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
    table = TableHelper(
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
            # 'camera': 'camera',
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
            't0': lambda t: t.iso
        },
        units={
            # 'readout.preAmpGain': 'e⁻/ADU',
            # 'readout.mode.frq': 'MHz',
            # 'readout.mode': ''
            'tExp': 's',
            't0': 'UTC',
            'ishape': 'y, x',
        },

        compact=True,
        title_props=dict(txt=('underline', 'bold'), bg='g'),
        too_wide=False,
        totals=['n', 'duration'])

    @classmethod
    def new_groups(cls, *keys, **kws):
        return shocObsGroups(cls, *keys, **kws)

    # def map(self, func, *args, **kws):
    #     """Map and arbitrary function onto the data of each observation"""
    #     shocHDU.data.get

    # @expose.args

    def save(self, folder=None, name_format=None,  overwrite=False):
        filenames = self.calls('save', None, folder, name_format, overwrite)
        return self.__class__.load(filenames)

    def update_headers(self, **kws):
        self.calls('auto_update_header', **kws)

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
            for ats in self.attrs_gen(*str2tup(title)):
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

        #
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

        obstypes, stats = zip(*self.calls('guess_obstype', return_stats=True))

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

    def match(self, other, exact, closest=(), cutoffs=(), keep_nulls=False,
              return_deltas=False, report=False, threshold_warn=None, ):
        """
        Match these observations with those in `other` according to their
        attribute values. Matches exactly the attributes given in `exact`,
        and as closely as possible to those in `closest`. Group both
        campaigns by the values at attributes.

        Parameters
        ----------
        other: shocCampaign
            shocCampaign instance from which observations will be picked to
            match those in this list
        exact: tuple or str
            single or multiple attribute names to check for equality between
            the two runs. For null matches, None is returned.
        closest: tuple or str, optional, default=()
            single or multiple keywords to match as closely as possible between
            the two runs. The attributes which are pointed to by these should
            support item subtraction since closeness is taken to mean the
            absolute difference between the two attribute values.
        keep_nulls: bool, optional, default=False
            Whether to keep the empty matches. ie. if there are observations in
            `other` that have no match in this observation set, keep those
            observations in the grouping and substitute `None` as the value for
            the corresponding key in the resulting dict. This parameter affects
            only matches in the grouping of the `other` shocCampaign.
            Observations without matches in `self` (this run) are always kept so
            that full set of observations are always accounted for in the
            resultant grouping. A consequence of setting this to False (the
            default) is therefore that the two groupings returned by this
            function will have different keys, which may or may not be desired
            for further
            analysis.
        return_deltas: bool, default=False
            return a dict of distance matrices between 'closest' attributes
        report: bool, optional, default=False
            whether to print the resulting matches in a table
        threshold_warn: int, optional, default=None
            If the difference in attribute values for attributes in `closest`
            are greater than `threshold_warn`, a warning is emitted

        Returns
        -------
        out0, out1: shocObsGroups
            a dict-like object keyed on the attribute values of `keys` and
            mapping to unique `shocCampaign` instances
        """

        # self.logger.info(
        #         'Matching %s frames to %s frames by:\tExact %s;\t Closest %r',
        #         other.kind.upper(), self.kind.upper(), exact, closest)

        # create the GroupedRun for science frame and calibration frames
        exact, closest = str2tup(exact), str2tup(closest)
        keys = OrderedSet(filter(None, mit.flatten([exact, closest])))

        if not keys:
            raise ValueError('Need at least one `key` (attribute name) by which'
                             ' to match')
        # assert len(other), 'Need at least one other observation to match'
        if threshold_warn is not None:
            threshold_warn = np.atleast_1d(threshold_warn)
            assert threshold_warn.size == len(closest)

        g0 = self.group_by(*exact)
        g1 = other.group_by(*exact)
        out0 = self.new_groups()
        out0.group_id = keys, {}
        out1 = other.new_groups()
        out1.group_id = keys, {}

        # iterate through all group keys. There may be unmatched groups in both
        deltas = {}
        for key, obs0 in g0.items():
            obs1 = g1.get(key)
            for id_, sub0, sub1, delta in obs0._group_close(obs1, closest):
                gid = (*key, *id_)
                # group
                out0[gid] = sub0
                out1[gid] = sub1
                # delta matrix
                deltas[gid] = delta

        if report:
            pprint_match(out0, out1, deltas, closest, threshold_warn)

        # filter null matches. If you do better with `pprint_match` you can
        # remove this block
        if not keep_nulls:
            for key in list(out1.keys()):
                if out1[key] is None:
                    out1.pop(key)

        if closest and return_deltas:
            return out0, out1, deltas

        return out0, out1

    # def delta_matrix(self, other, keys):
    #     # get delta matrix.  keys are attributes of the HDUs
    #     v0 = self.attrs(*keys)
    #     v1 = other.attrs(*keys)
    #     return np.abs(v0[:, None] - v1)

    def _group_close(self, other, keys):
        # Do the matching - map observation to those in `other` with attribute
        # values matching most closely
        if None in (self, other) or not keys:
            yield (), self, other, None
            return

        # split sub groups for closest match
        for vals, l0, l1, deltas in split_dist(self.attrs(*keys),
                                               other.attrs(*keys)):
            yield vals, self[l0], other[l1], deltas

    def combine(self, func=None, *args, **kws):
        """
        Combine each `shocHDU` in the campaign into a 2D image by calling `func`
        on each data stack.  Can be used to compute image statistics.

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
        return self.__class__(self.calls('combine', func, *args, **kws))

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
        self.calls('auto_update_header')
        header = headers_intersect(self)
        #     header['DATE'] = self[0].header['date']  # HACK
        hdu = shocHDU(np.dstack([hdu.data for hdu in self]),
                      header)
        msg = self.message(f'Stacked {len(self)} files.')
        hdu.header.add_history(msg)
        return hdu

    def merge_combine(self, func, *args, **kws):
        return self.combine(func, *args, **kws).stack().combine(func, *args, **kws)

    def subtract(self, master_bias):

        if not isinstance(master_bias, PrimaryHDU):
            raise TypeError(f'Expected shocHDU, got {type(master_bias)}')

        if len(master_bias.shape) > 2:
            raise TypeError(
                'The input hdu contains multiple images instead of a single '
                'image. Do `combine` to compute the master bias.'
            )

        return self.__class__([hdu.subtract(master_bias) for hdu in self])

    # def set_calibrators(self, bias, flat):

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
        ax.set_yticklabels(np.take(run.files.stems, z))
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

        CRITICAL: Only use this function if you are sure that the telescope
        pointing coordinates are correct in the headers. Turns out they are
        frequently wrong!

        NOTE: The FITS headers are left unchanged by this function which only
        alters the attributed of the `shocHDU` instances.  To update the headers
        you should do `hdu.auto_update_header()` afterwards for each observation,
        or alternatively `run.calls('auto_update_header')` on the shocCampaign.

        Parameters
        ----------
        named_coords : dict
            name, coordinate mapping
        tolerance : float
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
            sel = self[cxx.separation(obj_coords).deg < tolerance]
            if len(sel):
                sel.set_attrs(coords=obj_coords,
                              target=names[i],
                              obstype='object')

    def missing(self, kind):
        kind = kind.lower()
        assert kind in CALIBRATION_NAMES
        kind = OBSTYPE_EQUIVALENT.get(kind, kind)

        g = self.group_by('obstype')
        attr = MATCH[kind][0]

        atrset = set(g['object'].attrs(*attr))
        if kind == 'dark':
            atrset |= set(g['flat'].attrs(*attr))

        atrset -= set(g[kind].attrs(*attr))
        return sorted(atrset, key=str)

    def missing_calibration(self, report=False):
        missing = {kind: self.missing(kind) for kind in ('flat', 'dark')}

        if report:
            s = ''
            for key, modes in missing.items():
                s += key.upper()
                s += (':\t' + '\n\t'.join(map(str, modes)))
                s += '\n'
            print(s)

        return missing

    def gps_start_(self):
        gps_start_missing = np.array(self.attrs('t.trigger.flag'), bool)
        return self[gps_start_missing]

    def no_gps_interval(self):
        return self[self.calls('t.has_gps_interval')]

    # def set_t0_sast(self, times):
    #     assert len(times) == len(self)
    #     for obs, t0 in zip(self, times):
    #         obs.t.t0 = obs.t.from_local(t0)

    def provide_gps(self, filename):

        # read file with gps triggers
        names, sast = np.loadtxt(str(filename), str, unpack=True)

        gps_provided = run[names]
        need_gps = run.select_by(**{'t.trigger.flag': bool})
        assert gps_provided == need_gps

        # t0 = need_gps.attrs('t.t0')
        # need_gps.set_t0_sast(sast)

        # assert len(times) == len(self)
        for obs, t0 in zip(need_gps, sast):
            obs.t.t0 = obs.t.from_local(t0)
            obs.trigger.flag = ''

        # print('TIMEDELTA')
        # print((Time(t0) - Time(need_gps.attrs('t.t0'))).to('s'))

        # TODO: header keyword

    # def partition_by_source():


class GroupTitle:
    width = None
    # formater = 'group {}'

    def __init__(self, i, keys, props):
        self.g = f'group {i}:'
        self.s = self.format_key(keys)
        # self.s = "; ".join(map('{:s} = {:s}'.format, *zip(*info.items())))
        self.props = props

    # @staticmethod
    def format_key(self, keys):
        if isinstance(keys, str):
            return keys
        return "; ".join(map(str, keys))

    def __str__(self):
        return '\n' + overlay(codes.apply(self.s, self.props),
                              self.g.ljust(self.width))


def make_title(keys):
    if isinstance(keys, str):
        return keys
    return "; ".join(map(str, keys))


class shocObsGroups(Grouped):
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

    def get_tables(self, titled=make_title, headers=False, **kws):
        # TODO: move to motley.table
        # FIXME: headers param not used
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

        # TODO: consider merging this functionality into  motley.table
        #       Table.group_rows(), or hstack or some somesuch

        if titled is True:
            titled = make_title

        title = kws.pop('title', self.__class__.__name__)
        ncc = kws.pop('compact', False)  # number of columns in compact part
        kws['compact'] = False

        pp = shocCampaign.table
        attrs = OrderedSet(pp.attrs)
        attrs_grouped_by = ()
        compactable = set()
        # multiple = (len(self) > 1)
        if len(self) > 1:
            if self.group_id != ((), {}):
                keys, _ = self.group_id
                key_types = dict()
                for gid, grp in itt.groupby(keys, type):
                    key_types[gid] = list(grp)
                attrs_grouped_by = key_types.get(str, ())
                attrs -= set(attrs_grouped_by)

            # check which columns are compactable
            attrs_varies = set(key for key in attrs if self.varies_by(key))
            compactable = attrs - attrs_varies
            attrs -= compactable

        # column headers
        headers = pp.get_headers(attrs)

        # handle column totals
        totals = kws.pop('totals', pp.kws['totals'])
        if totals:
            # don't print totals for columns used for grouping since they will
            # not be displayed
            totals = set(totals) - set(attrs_grouped_by) - compactable
            # convert totals to numeric since we remove column headers for
            # lower tables
            totals = list(map(headers.index, pp.convert_aliases(list(totals))))

        units = kws.pop('units', pp.units)
        if units:
            want_units = set(units.keys())
            nope = set(units.keys()) - set(headers)
            units = {k: units[k] for k in (want_units - nope - compactable)}

        tables = {}
        empty = []
        footnotes = OrderedSet()
        for i, (gid, run) in enumerate(self.items()):
            if run is None:
                empty.append(gid)
                continue

            # get table
            if titled:
                # FIXME: problem with dynamically formatted group title.
                # Table wants to know width at runtime....
                title = titled(gid)
                # title = titled(i, gid, kws.get('title_props'))

            tables[gid] = tbl = get_table(run, attrs,
                                          title=title,
                                          totals=totals,
                                          units=units,
                                          # compact=False,
                                          **kws)

            # only first table gets title / headers
            if not titled:
                kws['title'] = None
            if not headers:
                kws['col_headers'] = kws['col_groups'] = None

            # only last table gets footnote
            footnotes |= set(tbl.footnotes)
            tbl.footnotes = []
        #
        tbl.footnotes = list(footnotes)

        # deal with null matches
        first = next(iter(tables.values()))
        if len(empty):
            filler = [''] * first.shape[1]
            filler[1] = 'NO MATCH'
            filler = Table([filler])
            for gid in empty:
                tables[gid] = filler

        # HACK compact repr
        if ncc and first.compactable():
            first.compact = ncc
            first.compact_items = dict(zip(
                list(compactable),
                get_table(run[:1], compactable,
                          chead=None, cgroups=None,
                          row_nrs=False, **kws).pre_table[0]
            ))
            first._compact_table = first._get_compact_table()

        # put empty tables at the end
        # tables.update(empty)
        return tables

    def pformat(self, titled=make_title, headers=False, braces=False, vspace=0,
                **kws):
        """
        Run pprint on each group
        """

        # ΤΟDO: could accomplish the same effect by colour coding...

        from motley.utils import vstack, hstack

        tables = self.get_tables(titled, headers, **kws)
        ordered_keys = list(tables.keys())  # key=sort
        stack = [tables[key] for key in ordered_keys]

        if not braces:
            return vstack(stack, bool(titled), vspace)

        braces = ''
        for i, gid in enumerate(ordered_keys):
            tbl = tables[gid]
            braces += ('\n' * bool(i) +
                       hbrace(tbl.data.shape[0], gid) +
                       '\n' * (tbl.has_totals + vspace))

        # vertical offset
        offset = stack[0].n_head_lines
        return hstack([vstack(stack, True, vspace), braces],
                      spacing=1, offset=offset)

    def pprint(self, titled=make_title, headers=False, braces=False, vspace=0,
               **kws):
        print(self.pformat(titled, headers, braces, vspace, **kws))

    def map(self, func, *args, **kws):
        # runs an arbitrary function on each shocCampaign in the group
        out = self.__class__()
        out.group_id = self.group_id

        for key, obj in self.items():
            if obj is None:
                out[key] = None
            else:
                out[key] = func(obj, *args, **kws)
        return out

    def calls(self, name, *args, **kws):
        """
        For each group of observations (shocCampaign), call the
        method with name `name`  passing  `args` and `kws`.

        Parameters
        ----------
        name
        args
        kws

        Returns
        -------

        """

        def run_method(obj, *args, **kws):
            return getattr(obj, name)(*args, **kws)

        return self.map(run_method, *args, **kws)

    def attrs(self, *keys):
        out = {}
        for key, obs in self.items():
            if obj is None:
                out[key] = None
            elif isinstance(key, shocCampaign):
                out[key] = obs.attrs(*keys)
            elif isinstance(obj, shocHDU):
                out[key] = op.attrgetter(*keys)(obj)
        return out

    def combine(self, func=None, *args, **kws):
        return self.calls('combine', func, *args, **kws)

    def stack(self):
        return self.calls('stack')

    def merge_combine(self, func=None, *args, **kws):
        return self.combine(func, *args, **kws).stack().combine(func, *args, **kws)

    def select_by(self, **kws):
        out = self.__class__()
        out.update({key: obs
                    for key, obs in self.calls('select_by', **kws).items()
                    if len(obs)})
        return out

    def comap(self, func, other, *args, **kws):
        out = self.__class__()
        for key, run in self.items():
            co = other[key]
            if run is None:
                continue

            if co is None:
                out[key] = None
            else:
                out[key] = func(run, co, *args, **kws)

        return out

    def _comap_method(self, other, name, *args, **kws):

        if not set(other.keys()) == set(self.keys()):
            raise ValueError('GroupId mismatch')

        out = self.__class__()
        for key, run in self.items():
            co = other[key]
            if run is None:
                continue

            if co is None:
                out[key] = None
            else:
                out[key] = getattr(run, name)(co, *args, **kws)

        return out

    def subtract(self, biases):
        """
        Do the bias reductions on science / flat field data

        Parameters
        ----------
        biases :
            Dictionary with key-value pairs for master biases

        Returns
        ------
        Bias subtracted shocObsGroups
        """

        return self._comap_method(biases, 'subtract')

    def set_calibrators(self, biases=None, flats=None):
        """

        Parameters
        ----------
        biases
        flats

        Returns
        -------

        """
        biases = biases or {}
        flats = flats or {}
        for key, run in self.items():
            if run is None:
                continue

            bias = biases.get(key, keep)
            flat = flats.get(key, keep)
            for hdu in run:
                hdu.set_calibrators(bias, flat)

    def save(self, folder=None, name_format=None, overwrite=False):
        # since this calls `save` on polymorphic class HDU / Campaign and 'save'
        # method in each of those have different signature, we have to unpack
        # the keyword
        return self.calls('save',
                          folder=folder,
                          name_format=name_format,
                          overwrite=overwrite)


class Filler:
    s = 'NO MATCH'
    table = None

    def __init__(self, style):
        self.style = style

    def __str__(self):
        self.table.pre_table[0, 0] = codes.apply(self.s, self.style)
        return str(self.table)

    @classmethod
    def make(cls, table):
        cls.table = table.empty_like(1, frame=False)


def pprint_match(g0, g1, deltas, closest=(), threshold_warn=(),
                 group_header_style='bold', no_match_style='r', g1_style='c',
                 ):

    # create tmp shocCampaign so we can use the builtin pprint machinery
    tmp = shocCampaign()
    size = sum(sum(map(len, filter(None, g.values()))) for g in (g0, g1))
    depth = np.product(
        np.array(list(map(np.shape, deltas.values()))).max(0)[[0, -1]])
    dtmp = np.ma.empty((size, depth), 'O')  # len(closest)
    dtmp[:] = np.ma.masked

    # remove group-by keys that are same for all
    group_id = tuple(g0.group_id[0])
    varies = [(g0.varies_by(key) | g1.varies_by(key)) for key in group_id]
    unvarying = np.where(~np.array(varies))[0]
    # use_key, = np.where(varies)

    # remove keys that grouping is done by
    attrs = OrderedSet(tmp.table.attrs)
    for g in group_id:
        for a in list(attrs):
            if a.startswith(g):
                attrs.remove(a)

    #
    insert = defaultdict(list)
    highlight = {}
    hlines = []
    n = 0
    for i, (key, obs) in enumerate(g0.items()):
        other = g1[key]
        use = varies[:len(key)]
        display_keys = np.array(key, 'O')[use]
        # headers = tmp.table.get_headers(np.array(group_id)[varies])
        # info = dict(zip(headers, display_keys))

        # insert group headers
        group_header = GroupTitle(i, display_keys, group_header_style)
        insert[n].append((group_header, '<', 'underline'))

        # populate delta table
        s0 = n + np.size(other)
        delta_mtx = np.ma.hstack(deltas.get(key, [np.ma.masked]))
        dtmp[s0:s0 + np.size(obs), :delta_mtx.shape[-1]] = delta_mtx

        #
        for j, (run, c) in enumerate(zip([other, obs], [no_match_style, ''])):
            if run is None:
                insert[n].append(Filler(c))
            else:
                tmp.extend(run or ())
                size = len(run)
                end = n + size
                # highlight other
                if j == 0:
                    for m in range(n, end):
                        highlight[m] = g1_style
                n = end

        # separate groups by horizontal lines
        # hlines.append(n - 1)

    # get title
    lc = len(closest)
    title = 'Matched Observations \nexact: {}'.format(group_id[:-lc])
    if lc:
        title += ' \nclosest: {}'.format(group_id[-lc:])

    # get attribute table
    tbl = get_table(tmp, attrs, hlines=hlines, title=title, title_align='<',
                    insert=insert, row_nrs=False, totals=False)

    # filler lines
    Filler.make(tbl)
    GroupTitle.width = tbl.get_width() - 1

    # fix for final run null match
    if run is None:
        tbl.hlines.pop(-1)
        tbl.insert[n][-1] = (tbl.insert[n][-1], '', 'underline')

    # highlight `other`
    tbl.highlight = highlight

    # hack compact repr
    tbl.compact_items = dict(zip(np.take(group_id, unvarying),
                                 np.take(key, unvarying)))

    # create delta table
    if False:
        import operator as op
        import itertools as itt

        from motley import codes
        from motley.utils import hstack, overlay, ConditionalFormatter

        headers = list(map('Δ({})'.format, tmp.table.get_headers(closest)))
        formatters = []
        fmt_db = {'date': lambda d: d.days}
        deltas0 = next(iter(deltas.values())).squeeze()
        for d, w in zip(deltas0, threshold_warn):
            fmt = ConditionalFormatter('yellow', op.gt,
                                       type(d)(w.item()), fmt_db.get(kw))
            formatters.append(fmt)
        #
        insert = {ln: [('\n', '>', 'underline')] + ([''] * (len(v) - 2))
                  for ln, v in tbl.insert.items()}
        formatters = formatters or None
        headers = headers * (depth // len(closest))
        d_tbl = Table(dtmp, col_headers=headers, formatters=formatters,
                      insert=insert, hlines=hlines)
        print(hstack((tbl, d_tbl)))
    else:
        print(tbl)
        print()
        return tbl
