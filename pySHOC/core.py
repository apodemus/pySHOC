# __version__ = '2.13'


# std libs
from obstools.phot.campaign import FnHelp
import datetime
import functools as ftl
import itertools as itt
from collections import defaultdict
import warnings

# third-party libs
import numpy as np
import more_itertools as mit
from dataclasses import dataclass, field
from astropy.utils import lazyproperty
from astropy.io.fits.hdu import PrimaryHDU
from astropy.coordinates import SkyCoord

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
from recipes.containers.sets import OrderedSet
from recipes.containers import Grouped, OfType

# relative libs
from graphing.imagine import plot_image_grid

# from pySHOC.image.sample import ResampleFlip
from .readnoise import readNoiseTables
from .timing import shocTimingOld, shocTimingNew
from .convert_keywords import KEYWORDS as kw_old_to_new
from .header import HEADER_KEYS_MISSING_OLD, headers_intersect

# only emit this warning once!!
warnings.filterwarnings('once', 'Using telescope pointing coordinates')

# TODO
# __all__ = ['']

# TODO: can you pickle these classes
# FIXME: shocHDU classes are assigned based on header keywords which are often
#   incorrect!!!  Think of a better way of re-assigning the HDU classes

#            SHOC1, SHOC2
SERIAL_NRS = [5982, 6448]

# Field of view of telescopes in arcmin
FOV74 = (1.29, 1.29)
FOV74r = (2.79, 2.79)  # with focal reducer
FOV40 = (2.85, 2.85)
# fov30 = (3.73, 3.73) # decommissioned

FOV = {'74': FOV74, '1.9': FOV74,
       '40': FOV40, '1.0': FOV40, '1': FOV40,
       # '30': fov30, '0.75': fov30, # decommissioned
       }
FOVr = {'74': FOV74r,
        '1.9': FOV74r}

EMPTY_FILTER_STR = '∅'

# Attributes for matching calibration frames
ATT_EQUAL_DARK = ('instrument', 'binning',  # subrect
                  'readout.preAmpGain', 'readout.outAmp.mode', 'readout.frq')  # <-- 'readout'
ATT_CLOSE_DARK = ('readout.outAmp.emGain', 'timing.exp')
MATCH_DARKS = ATT_EQUAL_DARK, ATT_CLOSE_DARK

ATT_EQUAL_FLAT = ('telescope', 'instrument', 'binning', 'filters')
ATT_CLOSE_FLAT = ('date', )
MATCH_FLATS = ATT_EQUAL_FLAT, ATT_CLOSE_FLAT


def str2tup(keys):
    if isinstance(keys, str):
        keys = keys,  # a tuple
    return keys


def hbrace(size, name=''):
    #
    if size < 3:
        return '← ' + str(name) + '\n' * (int(size) // 2)

    d, r = divmod(int(size) - 3, 2)
    return '\n'.join(['⎫'] +
                     ['⎪'] * d +
                     ['⎬ %s' % str(name)] +
                     ['⎪'] * (d + r) +
                     ['⎭'])


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


def apply_stack(func, *args, **kws):  # TODO: move to proc
    # TODO:  MULTIPROCESS HERE!
    return func(*args, **kws)


class Date(datetime.date):
    """
    We need this so the datetime.date instances print in date format instead
    of the class representation format, when print is called on, for eg. a tuple
    containing a date_time object.
    """

    def __repr__(self):
        return str(self)


# class yxTuple(tuple):
#     def __init__(self, *args):
#         assert len(self) == 2
#         self.y, self.x = self


class Binning:
    def __init__(self, args):
        # assert len(args) == 2
        self.y, self.x = args

    def __repr__(self):
        return f'{self.y}x{self.x}'

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


@dataclass()
class OutAmpMode(object):
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
    frq: float
    preAmpGain: float
    outAmp: OutAmpMode
    ccdMode: str = field(repr=False)
    serial: int = field(repr=False)

    @classmethod
    def from_header(cls, header):
        # readout speed
        frq = 1. / header['READTIME']
        frq = int(round(frq / 1.e6))  # MHz
        #
        outAmp = OutAmpMode(header['OUTPTAMP'], header.get('GAIN', ''))
        return cls(frq, header['PREAMP'], outAmp, header['ACQMODE'],
                   header['SERNO'])

    def __post_init__(self):
        self.isEM = (self.outAmp.mode == 'EM')
        self.isCON = not self.isEM
        self.mode = self  # hack for verbose access `readout.mode.isEM`
        # Readout noise
        # set the correct values here as attributes of the instance. These
        # values are absent in the headers
        (self.bit_depth, self.sensitivity, self.noise, self.time,
         self.saturation, self.bias_level) = \
            readNoiseTables[self.serial][
                (self.frq, self.outAmp.mode, self.preAmpGain)]

    def __repr__(self):
        return '{} MHz {}'.format(self.frq, self.outAmp)

    def __hash__(self):
        return hash((self.frq, self.preAmpGain, self.outAmp.mode,
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

    # def adc_bit_depth(self):
    #     if self.frq > 1:
    #         return 14
    #     else:
    #         return 16


# def __lt__(self, other):


class Filters(object):
    """Simple class to represent position of the filter wheels"""

    def __init__(self, a, b):
        self.A = self.get(a)
        self.B = self.get(b)

    def get(self, long):
        # get short description like UVBRI
        if long == 'Empty':
            return EMPTY_FILTER_STR
        if long:
            return long.split(' - ')[0]
        return (long or EMPTY_FILTER_STR)

    def __members(self):
        return self.A, self.B

    def __repr__(self):
        return f'{self.__class__.__name__}{self.__members()}'

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__members() == other.__members()
        return False

    def __hash__(self):
        return hash(self.__members())


class shocFnHelp(FnHelp):
    @property
    def nr(self):
        """
        File sequence number 
            eg. '0010' in 'SHA_20200729.0010.fits'
        """
        return self.path.suffixes[0].lstrip('.')


# HDU Subclasses

class shocHDU(HDUExtra):
    _FnHelper = shocFnHelp

    def __init__(self, data=None, header=None, do_not_scale_image_data=False,
                 ignore_blank=False, uint=True, scale_back=None):
        # init PrimaryHDU
        super().__init__(data=data, header=header,
                         do_not_scale_image_data=do_not_scale_image_data,
                         uint=uint,
                         ignore_blank=ignore_blank,
                         scale_back=scale_back)

        # ImageSamplerHDUMixin.__init__(self)

        serial = header['SERNO']
        shocNr = SERIAL_NRS.index(serial) + 1
        self.instrument = 'SHOC %i' % shocNr
        self.telescope = header.get('TELESCOP')

        # date from header
        self.date = self.nameDate = None  # FIXME: self.t.date ??
        if 'DATE' in header:
            date, time = header['DATE'].split('T')
            self.date = Date(*map(int, date.split('-')))
            # oldSHOC: file creation date
            # starting date of the observing run: used for naming
            h = int(time.split(':')[0])
            nameDate = self.date - datetime.timedelta(int(h < 12))
            self.nameDate = str(nameDate).replace('-', '')

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
        self.target = header.get('OBJECT')  # objName
        self.obstype = header.get('OBSTYPE')

        # # manage on-the-fly image orientation
        # self.oriented = ImageOrienter(self, x=self.readout.isEM)

        # # manage on-the-fly calibration for large files
        # self.calibrated = ImageCalibration(self)

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
        # initially be created with a _`BasicHeader` only, in which case we
        # will not yet have all the correct keywords available yet to identify
        # old vs new.
        # return shocTimingBase(self)
        if 'shocOld' in self.__class__.__name__:
            return shocTimingOld(self)
        else:
            return shocTimingNew(self)

    # alias
    t = timing

    @property
    def nframes(self):
        """Total number of images in observation"""
        return self.shape[0]  #

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
        Whether the telescope was pointing at the zenith during the observation.
        Useful for helping to discern if the observation is a sky flat.

        Parameters
        ----------
        tol : int, optional
            Tolerance in degrees, by default 2

        Returns
        -------
        bool
            True i point to zenith, False otherwise
        """
        lmst = self.t.t[[0, -1]].sidereal_time('mean')
        dec_z = abs(self.coords.dec - self.t.location.lat).deg < tol
        ha_z = all(abs(self.coords.ra - lmst).deg < tol)
        return dec_z & ha_z

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
        cube = shocObs.load(filename)
        cube.get_field_of_view(1)               # 1.0m telescope
        cube.get_field_of_view(1.9)             # 1.9m telescope
        cube.get_field_of_view(74)              # 1.9m
        cube.get_field_of_view('74in')          # 1.9m
        cube.get_field_of_view('40 in')         # 1.0m
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
        # created erroneously by SHOC
        if v == 0 or m >= 1.5:
            o = 'bad'

        elif self.pointing_zenith():
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
        return not ('date-obs' in self.header)
        # TODO: is this good enough???

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

        # check if single image
        if (self.ndim == 2) or ((self.ndim == 3) and len(self.data) == 1):
            return self

        # combine across images
        kws.setdefault('axis', 0)
        hdu = self.__class__(func(self.data, *args, **kws), self.header)

        # update history
        # hdu.header.add_history(
        # print(
        #         f'{Time.now()}: Combined {len(self.data)} images '
        #         f'from file {get_id(self)} with {func.__name__}')
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
        # self.header.add_history
        # print(
        #         f'{Time.now()}: Subtracted image {get_id(bias)}'
        # )
        return self

    def update_header(self, **kws):
        new_kws = dict(
            object=self.target,
            objra=self.coords.ra.to_string('hourangle', sep=':', precision=1),
            objdec=self.coords.dec.to_string('deg', sep=':', precision=1),
            obstype=self.obstype
        )
        self.header.update({**new_kws, **kws})


class shocOldHDU(shocHDU):
    # NOTE:_BaseHDU creates a _BasicHeader, which does not contain hierarch
    # keywords, so for SHOC we cannot tell if it's the old format header by
    # checking those

    @classmethod
    def match_header(cls, header):
        if 'SERNO' not in header:
            return False           # not SHOC!

        # check if any of the new keywords are missing in the header
        return any((kw not in header for kw in HEADER_KEYS_MISSING_OLD))


class shocNewHDU(shocHDU):
    @classmethod
    def match_header(cls, header):
        if 'SERNO' not in header:
            return False        # not SHOC!

        # first check not calibration stack
        for c in [shocBiasHDU, shocFlatHDU]:
            if c.match_header(header):
                return False

        # check if all the new keywords are present
        old, new = zip(*kw_old_to_new)
        return all([kw in header for kw in new])


class shocBiasHDU(shocHDU):
    # self.filters = should be empty always?
    @classmethod
    def match_header(cls, header):
        if 'SERNO' not in header:
            return False
        return 'bias' in header.get('OBSTYPE', '')

    def get_coords(self):
        return

    def combine(self, func=np.median):
        # "Median combining can completely remove cosmic ray hits and
        # radioactive decay trails from the result, which cannot be done by
        # standard mean combining. However, median combining achieves an
        # ultimate signal to noise ratio about 80% that of mean combining the
        # same number of frames. The difference in signal to noise ratio can
        # be by median combining 57% more frames than if mean combining were
        # used. In addition, variants on mean combining, such as sigma
        # clipping, can remove deviant pixels while improving the S/N
        # somewhere between that of median combining and ordinary mean
        # combining. In a nutshell, if all images are "clean", use mean
        # combining. If the images have mild to severe contamination by
        # radiation events such as cosmic rays, use the median or sigma
        # clipping method." - Newberry
        return super().combine(func)


class shocFlatHDU(shocBiasHDU):
    @classmethod
    def match_header(cls, header):
        if 'SERNO' not in header:
            return False
        return 'flat' in header.get('OBSTYPE', '')

    def combine(self, func=median_scaled_median):
        # default combine algorithm first median scales each image,
        # then takes median across images
        return super().combine(func)


class PPrintHelper(AttrTable):
    # def __call__(self, container, attrs=None, **kws):

    def get_table(self, run, attrs=None, **kws):
        # Add '*' flag to times that are gps triggered
        flags = {}
        attrs = attrs or self.attrs
        # index_of = attrs.index
        postscript = []
        for key, fun in [('timing._t0_repr', 'timing.trigger.is_gps'),
                         ('timing.exp', 'timing.trigger.is_gps_loop')]:
            if key in attrs:
                # type needed below so empty arrays work with `np.choose`
                _flags = np.array(run.calls(fun), bool)
                if _flags.any():
                    head = self.headers[key]
                    flags[head] = np.choose(_flags, [' ' * any(_flags), '*'])
                    postscript.append(f'* {head}: {fun}')

        units = kws.pop('units', self.kws['units'])
        if units and kws.get('col_headers'):
            ok = set(map(self.headers.get, attrs)) - {None}
            units = {k: u for k, u in units.items() if k in ok}
        else:
            units = None

        # compacted `filter` displays 'A = ∅' which is not very clear. Go more
        # verbose again for clarity

        table = super().get_table(run, attrs, flags=flags, units=units,
                                  footnotes=postscript, **kws)

        replace = {'A': 'filter.A',
                   'B': 'filter.B'}
        if len(table.compacted):
            table.compacted[0] = [replace.get(name, name)
                                  for name in table.compacted[0]]
        return table


def get_table(r, attrs=None, **kws):
    return r.pprinter.get_table(r, attrs, **kws)


class shocCampaign(PhotCampaign, OfType(shocHDU)):
    # pprinter controls which attributes will be printed
    pprinter = PPrintHelper(
        ['file.stem',
         'telescope', 'instrument',
         'target', 'obstype',
         'filters.A', 'filters.B',
         'nframes', 'ishape', 'binning',
         'readout.preAmpGain',
         'readout.mode',

         'timing._t0_repr',
         'timing.exp',
         'timing.duration',
         ],
        column_headers={
            'file.stem': 'filename',
            'telescope': 'tel',
            'instrument': 'camera',
            'nframes': 'n',
            'binning': 'bin',
            'readout.preAmpGain': 'preAmp',  # 'γₚᵣₑ',
            'timing.exp': 'tExp',
            'timing._t0_repr': 't0',
        },
        formatters={
            'timing.duration': ftl.partial(pprint.hms,
                                           unicode=True,
                                           precision=1)
        },
        units={'preAmp': 'e⁻/ADU',
               'tExp': 's',
               't0': 'UTC'},

        compact=True,
        title_props=dict(txt=('underline', 'bold'), bg='g'),
        too_wide=False,
        totals=['n', 'duration'])

    def new_groups(self, *keys, **kws):
        return shocObsGroups(self.__class__, *keys, **kws)

    # def map(self, func, *args, **kws):
    #     """Map and arbitrary function onto the data of each observation"""
    #     shocHDU.data.get

    # def group_by_obstype(self, *keys, return_index=False, **kws):

    def thumbnails(self, statistic='mean', depth=10, subset=(0, 10),
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
            index interval for subset of the full observation to draw from
        title: str
            key to attribute of item that will be used as title
        calibrated: bool
            calibrate the image if bias / flat field available
        figsize:  tuple
               size of the figure in inches
        Returns
        -------

        """

        # get sample images
        sample_images = self.calls(f'sampler.{statistic}', depth, subset)
        if calibrated:
            # print(list(map(np.mean, sample_images)))
            sample_images = [hdu.calibrated(im) for hdu, im in
                             zip(self, sample_images)]
            # print(list(map(np.mean, sample_images)))

        # get axes title strings
        titles = []
        for ats in self.attrs_gen(*str2tup(title)):
            if not isinstance(ats, tuple):
                ats = (ats, )
            titles.append('\n'.join(map(str, ats)))

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
        closest: tuple or str, optional
            single or multiple keywords to match as closely as possible between
            the two runs. The attributes which are pointed to by these should
            support item subtraction since closeness is taken to mean the
            absolute difference between the two attribute values.
        keep_nulls: bool, optional
            Whether to keep the empty matches. ie. if there are observations in
            `other` that have no match in this observation set, keep those
            observations in the grouping and substitute `None` as the value for
            the corresponding key in the resulting dict. This parameter affects
            only matches in the grouping of the `other` shocCampaign.
            Observations without matches in this campaign are always kept so
            that full set of observations are always accounted for in the
            grouping. A consequence of setting this to False (the default) is
            therefore that the two groupings returned by this function will have
            different keys, which may or may not be desired.
        return_deltas: bool
            return a dict of distance matrices between 'closest' attributes
        threshold_warn: int, optional
            If the difference in attribute values for attributes in `closest`
            are greater than `threshold_warn`, a warning is emitted
        report: bool
            whether to print the resulting matches in a table

        Returns
        -------
        g0: shocObsGroups
            a dict-like object keyed on the attribute values at `keys` and
            mapping to unique shocRun instances
        out_sr
        """

        # self.logger.info(
        #         'Matching %s frames to %s frames by:\tExact %s;\t Closest %r',
        #         other.kind.upper(), self.kind.upper(), exact, closest)

        # create the GroupedRun for science frame and calibration frames
        exact, closest = str2tup(exact), str2tup(closest)
        keys = OrderedSet(filter(None, mit.flatten([exact, closest])))

        assert len(keys), ('Need at least one key (attribute name) by which '
                           'to match')
        assert len(other), 'Need at least one key observation to match'
        if threshold_warn is not None:
            threshold_warn = np.atleast_1d(threshold_warn)
            assert threshold_warn.size == len(closest)

        g0, idx0 = self.group_by(*exact, return_index=True)
        g1, idx1 = other.group_by(*exact, return_index=True)

        # Do the matching - map observation to those in `other` with attribute
        # values matching most closely

        # keys are attributes of the HDUs
        vals0 = np.array(self.attrs(*keys), object)
        vals1 = np.array(other.attrs(*keys), object)

        #
        out0 = self.new_groups()
        out0.group_id = keys, {}
        out1 = other.new_groups()
        out1.group_id = keys, {}
        lme = len(exact)

        # iterate through all group keys. Their may be unmatched groups in both
        deltas = {}
        # all_keys = set(g1.keys()) | set(g0.keys())
        tmp = other.new_groups()
        for key in g0.keys():
            # array to tuple for hashing
            obs0 = g0.get(key)
            obs1 = g1.get(key)

            # todo: function here
            # here we have some exact matches. tiebreak with closest match
            if len(closest) and (None not in (obs0, obs1)):
                # get delta matrix
                v1 = vals1[idx1[key], lme:]
                v0 = vals0[idx0[key], lme:]
                delta_mtx = np.abs(v1[:, None] - v0)
                # split sub groups for closest match
                # if more than one attribute key provided for `closest`,
                # we take 'closest' to mean overall closeness as measured
                # by the sum of absolute differences between these.
                closeness = delta_mtx.sum(-1)
                # todo: use cutoff
                idx = closeness.argmin(0)
                for i in np.unique(idx):
                    l = (idx == i)
                    # check if more than one closest matching
                    ll = (closeness[:, l] == closeness[i, l]).all(-1)
                    # subgroup deltas are the same, use first only
                    gid = key + tuple(v1[ll][0])
                    out0[gid] = obs0[l]
                    out1[gid] = obs1[ll]
                    # delta matrix
                    deltas[gid] = delta_mtx[ll][:, l]
            else:
                if (obs0 is not None) or keep_nulls:
                    out0[key] = obs0

                out1[key] = obs1

        if report:
            pprint_match(out0, out1, deltas, closest, threshold_warn)

        # filter null matches. If you do better with `pprint_match` you can
        # remove this block
        if not keep_nulls:
            for key in list(out1.keys()):
                if out1[key] is None:
                    out1.pop(key)

        if len(closest) and return_deltas:
            return out0, out1, deltas

        return out0, out1

    def combine(self, func, *args, **kws):
        """
        Combine each stack in the run

        Parameters
        ----------
        func

        Returns
        -------

        """
        return self.__class__([hdu.combine(func, *args, **kws)
                               for hdu in self])

    def stack(self):
        """
        Stack data

        Returns
        -------

        """
        # keep only header keywords that are the same in all headers
        hdu = shocHDU(np.dstack([hdu.data for hdu in self]),
                      headers_intersect(self))

        msg = f'Stacked {len(self)} files'
        if None not in self.attrs('_file'):
            names = ", ".join(map("{!r}".format, self.files.names))
            msg = ': '.join((msg, names))
        hdu.header.add_history(msg)
        return hdu

    def merge_combine(self, func, *args, **kws):
        """
        Combines all of the stacks in the run into a single image

        Parameters
        ----------
        func: Callable
           function used to combine
        args:
           extra arguments to *func*
        kws:
            extra keyword arguments to func

        Returns
        ------
        shocObs instance
        """
        combined = self.combine(func, *args, **kws)
        kws.setdefault('axis', 0)
        image = func([hdu.data for hdu in combined], *args, **kws)

        # keep only header keywords that are the same in all headers
        header = headers_intersect(self)
        header['DATE'] = combined[0].header['date']  # HACK
        hdu = shocHDU(image, header)

        names = ", ".join(map("{!r}".format, self.files.names))
        hdu.header.add_history(
            f'Combined {len(self)} files with {func.__name__}: {names} '
        )
        return hdu

    def subtract(self, master_bias):

        if not isinstance(master_bias, PrimaryHDU):
            raise TypeError('Not a shocHDU.')

        if len(master_bias.shape) > 2:
            raise TypeError('The input hdu contains multiple images instead '
                            'of a single image. Do `combine` to compute the '
                            'master bias.')

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
        withing `tolerance` degrees from any targets, these attributes will be
        left unchanged.

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

    def missing_flats(self):
        g = self.group_by('obstype')
        return (set(g['object'].attrs(*MATCH_FLATS[0])) -
                set(g['flat'].attrs(*MATCH_FLATS[0])))

    def missing_darks(self):
        g = self.group_by('obstype')
        return (set(g['object'].join(g['flat']).attrs(*MATCH_DARKS[0])) -
                set(g['flat'].attrs(*MATCH_DARKS[0])))

    def missing_calibration(self, report=False):
        missing = {cal: sorted(getattr(self, f'missing_{cal}s')())
                   for cal in ('flat', 'dark')}

        if report:
            s = ''
            for key, modes in missing.items():
                s += key.upper()
                s += (':\t' + '\n\t'.join(map(str, modes)))
                s += '\n'
            print(s)

        return missing

    # def partition_by_source():


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

    def get_tables(self, titles=False, headers=False, **kws):
        """

        Parameters
        ----------
        kws

        Returns
        -------

        """
        from motley.table import Table
        # TODO: consider merging this functionality into  motley.table
        #       Table.group_rows(), or hstack or some somesuch

        #
        kws['compact'] = False
        kws['title'] = self.__class__.__name__

        pp = shocCampaign.pprinter
        attrs = OrderedSet(pp.attrs)
        attrs_grouped_by = ()
        if self.group_id is not None:
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
        headers = pp.get_headers(attrs)

        # handle column totals
        totals = kws.pop('totals', pp.kws['totals'])
        if totals:
            # don't print totals for columns used for grouping since they will
            # not be displayed
            totals = set(totals) - set(attrs_grouped_by) - compactable
            # convert totals to numeric since we remove column headers for
            # lower tables
            totals = list(map(headers.index, totals))

        units = kws.pop('units', pp.kws['units'])
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
            tables[gid] = tbl = get_table(run, attrs,
                                          totals=totals,
                                          units=units,
                                          # compact=False,
                                          **kws)
            sample = run

            # only first table gets title / headers
            if titles:
                'make title!'  # TODO
            else:
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
        first.compact = True
        r = sample[:1]

        first.compacted = (
            list(compactable),
            get_table(r, compactable,
                      chead=None, cgroups=None,
                      row_nrs=False, **kws).pre_table[0]
        )
        first._compact_table = first._get_compact_table()

        # put empty tables at the end
        # tables.update(empty)
        return tables

    def pprint(self, titles=False, headers=False, braces=True, **kws):
        """
        Run pprint on each group
        """

        # ΤΟDO: could accomplish the same effect by colour coding...

        from motley.utils import vstack, hstack

        tables = self.get_tables(titles, headers, **kws)
        ordered_keys = list(tables.keys())  # key=sort
        stack = [tables[key] for key in ordered_keys]

        if braces:
            braces = ''
            for i, gid in enumerate(ordered_keys):
                tbl = tables[gid]
                braces += ('\n' * bool(i) +
                           hbrace(tbl.data.shape[0], gid) +
                           '\n' * tbl.has_totals)

            # # vertical offset

            offset = stack[0].n_head_lines
            final = hstack([vstack(stack), braces], spacing=1, offset=offset)
        else:
            final = vstack(stack)

        print(final)
        return tables

    def map(self, func, *args, **kws):
        # runs an arbitrary function on each shocCampaign
        out = self.__class__()
        
        for key, obj in self.items():
            if obj is None:
                out[key] = None
            else:
                out[key] = func(obj, *args, **kws)
        return out

    # TODO: multiprocess!

    def _map_method(self, name, *args, **kws):
        def _runner(run, *args, **kws):
            return getattr(run, name)(*args, **kws)

        return self.map(_runner, *args, **kws)

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
        return self._map_method('calls', name, *args, **kws)

    def attrs(self, *keys):
        return self._map_method('attrs', *keys)

    def combine(self, func, *args, **kws):
        return self._map_method('combine', func, *args, **kws)

    def merge_combine(self, func, *args, **kws):
        return self._map_method('merge_combine', func, *args, **kws)

    def select_by(self, **kws):
        out = self.__class__()
        out.update({key: obs
                    for key, obs in self._map_method('select_by', **kws).items()
                    if len(obs)})
        return out

    def co_map(self, func, other, *args, **kws):
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

    def _co_map_func(self, other, name, *args, **kws):

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

        return self._co_map_func(biases, 'subtract')

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


class Filler(object):
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


class GroupHeaderLine(object):
    width = None

    def __init__(self, i, keys, props):
        self.g = f'group {i}:'
        self.s = "; ".join(map(str, keys))
        # self.s = "; ".join(map('{:s} = {:s}'.format, *zip(*info.items())))
        self.props = props

    def __str__(self):
        return '\n' + overlay(codes.apply(self.s, self.props),
                              self.g.ljust(self.width))


def pprint_match(g0, g1, deltas, closest=(), threshold_warn=(),
                 group_header_style='bold', no_match_style='r', g1_style='c',
                 ):
    from collections import defaultdict
    from recipes.containers.sets import OrderedSet

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
    attrs = OrderedSet(tmp.pprinter.attrs)
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
        # headers = tmp.pprinter.get_headers(np.array(group_id)[varies])
        # info = dict(zip(headers, display_keys))

        # insert group headers
        group_header = GroupHeaderLine(i, display_keys, group_header_style)
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
    GroupHeaderLine.width = tbl.get_width() - 1

    # fix for final run null match
    if run is None:
        tbl.hlines.pop(-1)
        tbl.insert[n][-1] = (tbl.insert[n][-1], '', 'underline')

    # highlight `other`
    tbl.highlight = highlight

    # hack compact repr
    tbl.compacted = np.take(group_id, unvarying), np.take(key, unvarying)

    # create delta table
    if False:
        import operator as op
        import itertools as itt

        from motley.table import Table
        from motley import codes
        from motley.utils import hstack, overlay, ConditionalFormatter

        headers = list(map('Δ({})'.format, tmp.pprinter.get_headers(closest)))
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


# class shocObsBase()

# class PhotHelper:
#     """helper class for photometry interface"""
