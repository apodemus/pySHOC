# __version__ = '2.13'


# std libs
import time
import datetime
import functools as ftl
import itertools as itt
from pathlib import Path
from collections import defaultdict

# third-party libs
import numpy as np
import more_itertools as mit
from dataclasses import dataclass, field
from astropy.utils import lazyproperty
from astropy.io.fits.hdu import PrimaryHDU

# local libs
from motley import codes
from motley.utils import overlay
from obstools.phot.utils import Resample
from obstools.phot.campaign import PhotCampaign
from obstools.stats import median_scaled_median
from recipes import pprint
from recipes.logging import LoggingMixin
from recipes.containers.dict_ import pformat
from recipes.containers.set_ import OrderedSet
from recipes.containers import Grouped, AttrTable

# relative libs
from .readnoise import readNoiseTables
from .utils import retrieve_coords, convert_skycoords
from .timing import Time, shocTimingOld, shocTimingNew
from .convert_keywords import KEYWORDS as kw_old_to_new
from .header import HEADER_KEYS_MISSING_OLD, headers_intersect

# import logging


# import matplotlib.pyplot as plt

# from recipes.io import warn

# TODO: do really want pySHOC to depend on obstools ?????

# TODO: choose which to use for timing: spice or astropy
# from .io import InputCallbackLoop

# TODO: maybe from .specs import readNoiseTables, SERIAL_NRS

#            SHOC1, SHOC2
SERIAL_NRS = [5982, 6448]

EMPTY_FILTER_STR = '∅'

# noinspection PyPep8Naming


# TODO
# __all__ = ['']

# TODO: can you pickle these classes
#


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


def str2tup(keys):
    if isinstance(keys, str):
        keys = keys,  # a tuple
    return keys


def get_id(hdu):
    """
    Unique identifier (hash) for hdus
    """
    # use the hashed header as identifier for the file in HISTORY
    fid = string_to_int(str(hdu.header))
    if hdu._file:
        fid = " ".join((str(fid), hdu.filename))
    return fid


def string_to_int(s):
    # persistent hash https://stackoverflow.com/a/2511232
    return int(''.join(('%.3d' % ord(x) for x in s)))


def int_to_string(n):
    s = str(n)
    return ''.join((chr(int(s[i:i + 3])) for i in range(0, len(s), 3)))


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


class yxTuple(tuple):
    def __init__(self, *args):
        assert len(self) == 2
        self.y, self.x = self


class Binning(yxTuple):
    def __repr__(self):
        return '%ix%i' % self


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
        self.mode = self  # cheat!!
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
        return long or EMPTY_FILTER_STR

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


def get_table(r, attrs=None, **kws):
    return r.pprinter.get_table(r, attrs, **kws)


# TODO: add these keywords to old SHOC headers:


# TODO: remove from headers
#  ACT
#  KCT


class DataOrientBase(object):
    def __init__(self, flip):
        """
        Ensure all shoc images have the same orientation relative to the sky
        """
        orient = [..., slice(None), slice(None)]
        if 'y' in flip:
            orient[-2] = slice(None, None, -1)
        if 'x' in flip:
            orient[-1] = slice(None, None, -1)
        #
        self.orient = tuple(orient)


# class shocImageSampler(object):
#     _sampler = None


class ResampleFlip(Resample, DataOrientBase):
    def __init__(self, data, sample_size=None, subset=None, axis=0, flip=''):
        Resample.__init__(self, data, sample_size, subset, axis)
        DataOrientBase.__init__(self, flip)

    def draw(self, n=None, subset=None):
        return Resample.draw(self, n, subset)[self.orient]


class DataOrienter(DataOrientBase):
    def __init__(self, hdu):
        """
        Class that ensures all shoc images have the same orientation relative to
        each other to assist doing image arithmetic
        """
        self.hdu = hdu
        flip = '' if hdu.readout.isEM else 'x'
        super().__init__(flip)

        if hdu.ndim == 3:
            self.data = hdu.section
        elif hdu.ndim == 2:
            self.data = hdu.data

    def __getitem__(self, item):
        return self.data[item][self.orient]


# TODO: move to calibration ..?
class keep:
    pass


class CalibrationImage:
    """Descriptor class for calibration images"""

    def __init__(self, name):
        self.name = f'_{name}'

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return getattr(instance, self.name)

    def __set__(self, instance, value):
        if value is keep:
            return
        if value is not None:
            value = DataOrienter(value)[:]
        setattr(instance, self.name, value)

    def __delete__(self, instance):
        setattr(instance, self.name, None)


class ImageCalibration(DataOrienter):
    """
    Do calibration arithmetic for CCD images upon item access
    """
    bias = CalibrationImage('bias')
    flat = CalibrationImage('flat')

    def __init__(self, hdu, bias=keep, flat=keep):
        super().__init__(hdu)
        self._bias = self._flat = None
        self.bias = bias
        self.flat = flat

    def __str__(self):
        return pformat(dict(bias=self.bias,
                            flat=self.flat),
                       self.__class__.__name__)

    def __repr__(self):
        return str(self)

    def __call__(self, data):
        """
        Do calibration arithmetic on `data` ignoring orientation

        Parameters
        ----------
        data

        Returns
        -------

        """
        # debias
        if self.bias is not None:
            data = data - self.bias

        # flat field
        if self.flat is not None:
            data = data / self.flat

        return data

    def __getitem__(self, item):
        return self(super().__getitem__(item))


# HDU Subclasses
class shocHDU(PrimaryHDU, LoggingMixin):
    def __init__(self, data=None, header=None, do_not_scale_image_data=False,
                 ignore_blank=False, uint=True, scale_back=None):
        PrimaryHDU.__init__(self,
                            data=data, header=header,
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
        self.date = self.nameDate = None
        if 'DATE' in header:
            date, time = header['DATE'].split('T')
            self.date = Date(*map(int, date.split('-')))
            # oldSHOC: file creation date
            # starting date of the observing run: used for naming
            h = int(time.split(':')[0])
            nameDate = self.date - datetime.timedelta(int(h < 12))
            self.nameDate = str(nameDate).replace('-', '')

        # image binning
        self.binning = Binning(header['%sBIN' % _] for _ in 'VH')

        # sub-framing
        self.subrect = np.array(header['SUBRECT'].split(','), int)
        # subrect stores the sub-region of the full CCD array captured for this
        #  observation
        # xsub, ysub = (xb, xe), (yb, ye) = \
        xsub, ysub = self.subrect.reshape(-1, 2) // self.binning
        # for some reason the ysub order is reversed
        self.sub = tuple(xsub), tuple(ysub[::-1])
        self.subSlices = list(map(slice, *np.transpose(self.sub)))

        # CCD mode
        self.readout = ReadoutMode.from_header(header)

        # orientation
        self.flip_state = yxTuple(header['FLIP%s' % _] for _ in 'YX')
        # WARNING: flip state wrong for EM!!
        # NOTE: IMAGE ORIENTATION reversed for EM mode data since it gets read
        #  out in opposite direction in the EM register to CON readout.

        # filters
        self.filters = Filters(*(header.get(f'FILTER{_}', 'Empty')
                                 for _ in 'AB'))

        # object name
        self.target = header.get('OBJECT')  # objName
        self.obstype = header.get('OBSTYPE')

        # manage on-the-fly calibration for large files
        self.calibrated = ImageCalibration(self)

    @property
    def sampler(self):
        """
        An image sampler mixin for SHOC data that automatically corrects the
        orientation of images so they are oriented North up, East left.

        Images taken in CON mode are flipped left-right.
        Images taken in EM  mode are left unmodified
        """
        # use `section` for performance
        flip = '' if self.readout.isEM else 'x'
        # make sure we pass 3d data to sampler. This is a hack so we can use
        # the sampler to get thumbnails from data that is a 2d image,
        # eg. master flats.  The 'sample' will just be the image itself.
        if self.ndim == 3:
            data = self.section
        elif self.ndim == 2:
            data = self.data[None]  # insert axis in front
        else:
            raise ValueError(f'Cannot create image sampler for data with '
                             f'{self.ndim} dimensions.')
        return ResampleFlip(data, flip=flip)

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

    # @property
    # def kct(self):
    #     return self.timing.kct

    @property
    def filepath(self):
        """file name as a Path object"""
        return Path(self._file.name)

    @property
    def filename(self):
        if self._file is not None:
            return self.filepath.stem
        return 'None'

    @property
    def nframes(self):
        """Total number of images in observation"""
        return self.shape[0]  #

    @property
    def ishape(self):
        """Image frame shape"""
        return self.shape[-2:]

    @property
    def ndim(self):
        return len(self.shape)

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
        astoropy.coordinates.SkyCoord

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
            coords = retrieve_coords(self.target)

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
            self.logger.warning('Using telescope pointing coordinates. This '
                                'may lead to barycentric timing correction '
                                'being less accurate.')

        return coords

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
        Get the instrument rotation wrt the sky in radians.  This value seems
        to be constant for most SHOC data from various telescopes that I've
        tested.
        You can measure this yourself by using the
        `obstools.phot.campaign.PhotCampaign.coalign_dss` method.
        """
        return -0.04527

    def guess_obstype(self, sample_size=10, subset=(0, 100),
                      return_stats=False):
        # FIXME: sample_size misleading name since it is really number of
        #  frames used to get median image

        """
        Guess the observation type based on statistics of a sample image.
        Very basic decision tree classifier based on 3 features:
        mean, stddev, skewness (the first 3 statistical moments) of the sampled
        median image.
        These values are normalized to the saturation value of the CCD given the
        instrumental setup in order to compare fairly between diverse datasets.

        This function comes with no guarantees whatsoever.

        Since for SHOC 0-time readout is impossible, the label 'bias' is
        technically erroneous, we use the label 'dark'.

        Returns
        -------
        label: {'object', 'flat', 'dark', 'bad'}

        """
        from scipy import stats

        img = self.sampler.median(sample_size, subset)
        nobs, minmax, *moments = stats.describe(img.ravel())
        m, v, s, k = np.divide(moments, self.readout.saturation)

        # s = 0 implies all constant pixel values.  These frames are sometimes
        # created erroneously by SHOC
        if v == 0 or m >= 1.5:
            o = 'bad'

        # Flat fields are usually about halfway to the saturation value
        elif 0.15 <= m < 1.5:
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

    def set_calibrators(self, bias=keep, flat=keep):
        """
        Set calibration images for this observation. Default it to keep
        previously set image if none are provided here.  To remove a
        previously set calibration image pass a value of `None` to this
        function, or simply delete the attribute `self.calibrated.bias` or
        `self.calibrated.flat`

        Parameters
        ----------
        bias
        flat

        Returns
        -------

        """
        self.calibrated.bias = bias
        self.calibrated.flat = flat

    # plotting
    def display(self, **kws):
        """Display the data"""
        if self.ndim == 2:
            from graphing.imagine import ImageDisplay
            im = ImageDisplay(self.data, **kws)

        elif self.ndim == 3:
            from graphing.imagine import VideoDisplay
            # FIXME: this will load entire data array which might be a tarpit
            #  trap
            im = VideoDisplay(self.data, **kws)

        else:
            raise TypeError('Not an image!! WTF?')

        im.figure.canvas.set_window_title(self.filepath.name)
        return im


def plot_image_grid(images, titles=()):
    """

    Parameters
    ----------
    images
    titles

    Returns
    -------

    """
    n = len(images)
    assert n, 'No images to plot!'

    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from graphing.imagine import ImageDisplay

    # get grid layout
    n_rows, n_cols = auto_grid(n)
    cbar_size = 3

    # create figure
    fig = plt.figure(figsize=(10.5, 9))
    # todo: guess fig size
    # Use gridspec rather than ImageGrid since the latter tends to resize
    # the axes
    gs = GridSpec(n_rows, n_cols * (100 + cbar_size),
                  hspace=0.005,
                  wspace=0.005,
                  left=0.03,
                  right=0.97,
                  bottom=0.03,
                  top=0.98
                  )  # todo: maybe better with tight layout.

    art = []
    indices = np.ndindex(n_rows, n_cols)
    axes = np.empty((n_rows, n_cols), 'O')
    for i, (j, k) in enumerate(indices):
        if i >= n:
            break

        axes[j, k] = ax = fig.add_subplot(gs[j:j + 1,
                                          (100 * k):(100 * (k + 1))])
        imd = ImageDisplay(images[i], ax=ax,
                           cbar=False, hist=False, sliders=False,
                           origin='lower left')
        art.append(imd.imagePlot)

        top = (j == 0)
        bot = (j == n_rows - 1)
        # right = (j == n_cols - 1)
        if k != 0:  # not leftmost
            ax.set_yticklabels([])
        if not (bot or top):
            ax.set_xticklabels([])
        # if right:
        #     ax.yaxis.tick_right()
        if top:
            ax.xaxis.tick_top()

    # labels (names
    for i, (ax, title) in enumerate(
            itt.zip_longest(axes.ravel(), titles, fillvalue='')):
        if ax is None:
            continue
        # add title text
        title = title.replace("\n", "\n     ")
        ax.text(0.025, 0.95, f'{i: <2}: {title}',
                color='w', va='top', fontweight='bold', transform=ax.transAxes)

    # colorbar
    cax = fig.add_subplot(gs[:, -cbar_size * n_cols:])
    # noinspection PyUnboundLocalVariable
    fig.colorbar(imd.imagePlot, cax)

    # https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/multi_image.html
    # Make images respond to changes in the norm of other images (e.g. via
    # the "edit axis, curves and images parameters" GUI on Qt), but be
    # careful not to recurse infinitely!
    def update(changed_image):
        for im in art:
            if (changed_image.get_cmap() != im.get_cmap()
                    or changed_image.get_clim() != im.get_clim()):
                im.set_cmap(changed_image.get_cmap())
                im.set_clim(changed_image.get_clim())

    for im in art:
        im.callbacksSM.connect('changed', update)

    return fig, axes


def plot_stats(run, stats, labels, titles):
    from matplotlib import pyplot as plt

    stats = np.array(stats)
    idx = defaultdict(list)
    for i, lbl in enumerate(labels):
        idx[lbl].append(i)

    n = stats.shape[1]
    fig, axes = plt.subplots(1, n, figsize=(12.5, 8), sharey=True)
    for j, ax in enumerate(axes):
        ax.set_title(titles[j])
        m = 0
        for lbl, i in idx.items():
            ax.plot(stats[i, j], range(m, m + len(i)), 'o', label=lbl)
            m += len(i)
        ax.grid()

    ax.invert_yaxis()
    ax.legend()

    # filenames as ticks
    z = []
    list(map(z.extend, idx.values()))
    names = run.attrs('name')
    ax = axes[0]
    ax.set_yticklabels(np.take(names, z))
    ax.set_yticks(np.arange(len(run) + 1) - 0.5)
    for tick in ax.yaxis.get_ticklabels():
        tick.set_va('top')
    # plt.yticks(np.arange(len(self) + 1) - 0.5,
    #            np.take(names, z),
    #            va='top')

    fig.tight_layout()


# _BaseHDU creates a _BasicHeader, which does not contain hierarch keywords,
# so for SHOC we cannot tell if it's the old format header by checking those

class shocOldHDU(shocHDU):
    @classmethod
    def match_header(cls, header):
        if 'SERNO' not in header:
            return False
        return any((kw not in header for kw in HEADER_KEYS_MISSING_OLD))

    #
    # header['OBJEPOCH'] = (2000, 'Object coordinate epoch')


class shocNewHDU(shocHDU):
    @classmethod
    def match_header(cls, header):
        if 'SERNO' not in header:
            return False

        # first check not calibration stack
        for c in [shocBiasHDU, shocFlatHDU]:
            if c.match_header(header):
                return False

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
    def get_table(self, run, attrs=None, **kws):
        # Add '*' flag to times that are gps triggered
        flags = {}
        index_of = list(self.headers.keys()).index
        for key, fun in [('timing._t0_repr', 'timing.trigger.is_gps'),
                         ('timing.t_expose', 'timing.trigger.is_gps_loop')]:
            # type needed below so empty arrays work with `np.choose`
            _flags = np.array(run.calls(fun), bool)
            flags[index_of(key)] = np.choose(_flags, [' ' * any(_flags), '*'])

        units = kws.pop('units', self.kws['units'])
        if units and kws.get('col_headers'):
            ok = set(map(self.headers.get, attrs or self.attrs)) - {None}
            units = {k: u for k, u in units.items() if k in ok}
        else:
            units = None

        # compacted `filter` displays 'A = ∅' which is not very clear. Go more
        # verbose again for clarity
        table = super().get_table(run, attrs, flags=flags, units=units, **kws)

        replace = {'A': 'filter.A',
                   'B': 'filter.B'}
        if len(table.compacted):
            table.compacted[0] = [replace.get(name, name)
                                  for name in table.compacted[0]]
        return table


class shocCampaign(PhotCampaign):
    pprinter = PPrintHelper(
            ['filename',
             'telescope', 'instrument',
             'target', 'obstype',
             'filters.A', 'filters.B',
             'nframes', 'ishape', 'binning',
             'readout.preAmpGain',
             'readout.mode',

             'timing._t0_repr',
             'timing.t_expose',
             'timing.duration',
             ],
            column_headers={
                'telescope': 'tel',
                'instrument': 'camera',
                'nframes': 'n',
                'binning': 'bin',
                'readout.preAmpGain': 'γₚᵣₑ',
                'timing.t_expose': 'tExp',
                'timing._t0_repr': 't0',
            },
            formatters={
                'timing.duration': ftl.partial(pprint.hms, unicode=True,
                                               precision=1)
            },
            units={'γₚᵣₑ': 'e⁻/ADU',
                   'tExp': 's',
                   't0': 'UTC'},

            compact=1,
            title_props=dict(txt=('underline', 'bold'), bg='g'),
            too_wide=False,
            totals=['n', 'duration'])

    def new_groups(self, *keys, **kws):
        return shocObsGroups(self.__class__, *keys, **kws)

    # def map(self, func, *args, **kws):
    #     """Map and arbitrary function onto the data of each observation"""
    #     shocHDU.data.get

    def thumbnails(self, statistic='mean', depth=10, subset=(0, 10),
                   title='filename', calibrated=False):
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
                ats = ats,
            titles.append('\n'.join(map(str, ats)))

        #
        return plot_image_grid(sample_images, titles)

    def guess_obstype(self, plot=False):
        """
        Identify what type of observation each dataset represents by running
        'guess_obstype' on each.

        Parameters
        ----------
        plot

        Returns
        -------

        """

        obstypes, stats = zip(*self.calls('guess_obstype', return_stats=True))

        if plot:
            plot_stats(self, stats, obstypes, ['mean', 'std', 'ptp'])

        return obstypes

    def match(self, other, exact, closest=(), cutoffs=(), keep_nulls=True,
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
        closest: tuple or str
            single or multiple keywords to match as closely as possible between
            the two runs. The attributes which are pointed to by these should
            support item subtraction since closeness is taken to mean the
            absolute difference between the two attribute values.
        keep_nulls: bool
            keep the empty matches. ie. if there are observations in `other`
            that have no match in this observation set, keep those observations
            and substitute `None` as the value for the corresponding key in
            the grouping.  This parameter affects only matches in the backwards
            direction. Empty matches in the forward direction are always kept so
            that full set of observations here are always accounted for in
            the grouping.  A consequence of setting this to False is
            therefore that the two groupings returned by this function will
            have different keys, which may or may not be desired.
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
        all_keys = set(g1.keys()) | set(g0.keys())
        for i, key in enumerate(all_keys):
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
                    sub = g0[key][l]

                    # check if more than one closest matching
                    ll = (closeness[:, l] == closeness[i, l]).all(-1)
                    # subgroup deltas are the same, use first only
                    gid = key + tuple(v1[ll][0])
                    out0[gid] = sub
                    out1[gid] = obs1[ll]
                    # delta matrix
                    deltas[gid] = delta_mtx[ll][:, l]
            else:
                if (obs0 is not None) or keep_nulls:
                    out0[key] = obs0

                out1[key] = obs1

        if report:
            pprint_match(out0, out1, deltas, closest, threshold_warn)

        if len(closest) and return_deltas:
            return out0, out1, deltas

        return out0, out1

    def select_by(self, **kws):
        keys = tuple(kws.keys())
        seek = tuple(kws.values())
        if len(kws) == 1:
            seek = seek[0]

        vals = self.attrs(*keys)
        selection = (np.array(vals) == seek)
        if len(keys) > 1:
            selection = selection.all(1)

        return self[selection]

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
    from recipes.containers.set_ import OrderedSet

    # create tmp shocCampaign so we can use the builtin pprint machinery
    tmp = shocCampaign()
    size = sum(sum(map(len, filter(None, g.values()))) for g in (g0, g1))
    depth = np.product(
            np.array(list(map(np.shape, deltas.values()))).max(0)[[0, -1]])
    dtmp = np.ma.empty((size, depth), 'O')  # len(closest)
    dtmp[:] = np.ma.masked

    #  remove group-by keys that are same for all
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
        display_keys = np.array(key)[use]
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

        from motley.table import Table, hstack
        from motley import codes
        from motley.utils import overlay, ConditionalFormatter

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


def auto_grid(n):
    x = int(np.floor(np.sqrt(n)))
    y = int(np.ceil(n / x))
    return x, y


class shocObsGroups(Grouped):
    """
    Emulates dict to hold multiple shocRun instances keyed by their shared
    common attributes. The attribute names given in groupId are the ones by
    which the run is separated into unique segments (which are also shocRun
    instances). This class attempts to eliminate the tedium of computing
    calibration frames for different  observational setups by enabling loops
    over various such groupings.
    """

    def __init__(self, default_factory=shocCampaign, *a, **kw):
        # set default default factory ;)
        super().__init__(default_factory, *a, **kw)

    def to_list(self):
        out = self.default_factory()
        for item in self.values():
            if item is None:
                continue
            if isinstance(item, PrimaryHDU):
                out.append(item)
            elif isinstance(item, out.__class__):
                out.extend(item)
            else:
                raise TypeError(f'{item.__class__}')
        return out

    def get_tables(self, **kws):
        """

        Parameters
        ----------
        kws

        Returns
        -------

        """
        from motley.table import Table

        #
        kws['compact'] = False
        kws['title'] = self.__class__.__name__

        pp = shocCampaign.pprinter
        attrs = set(pp.attrs)
        attrs_grouped_by = ()
        if self.group_id is not None:  #
            keys, _ = self.group_id
            key_types = dict()
            for gid, grp in itt.groupby(keys, type):
                key_types[gid] = list(grp)
            attrs_grouped_by = key_types.get(str, ())
            attrs = OrderedSet(pp.attrs) - set(attrs_grouped_by)

        # check compactable
        attrs_varies = set([key for key in attrs if self.varies_by(key)])
        compactable = attrs - attrs_varies
        attrs -= compactable
        headers = pp.get_headers(attrs)

        totals = kws.pop('totals', pp.kws['totals'])
        if totals:
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
        for i, (gid, run) in enumerate(self.items()):
            if run is None:
                empty.append(gid)
                continue

            # get table
            tables[gid] = run.pprinter.get_table(run, attrs,
                                                 totals=totals,
                                                 units=units,
                                                 # compact=False,
                                                 **kws)
            sample = run
            # only first table gets header
            kws['title'] = kws['col_headers'] = kws['col_groups'] = None

        # deal with null matches
        first = next(iter(tables.values()))
        if len(empty):
            filler = [''] * first.shape[1]
            filler[1] = 'NO MATCH'
            filler = Table([filler])
            for gid in empty:
                tables[gid] = filler

        # HACK compact repr
        first.compact = 1
        first.compacted = (list(compactable),
                           np.atleast_1d(sample.attrs(*compactable)[0]))

        # put empty tables at the end
        # tables.update(empty)

        return tables

    def pprint(self, **kws):
        """
        Run pprint on each
        """

        # ΤΟDO: could accomplish the same effect by colour coding...

        from motley.table import vstack, hstack

        tables = self.get_tables(**kws)
        braces = ''
        for i, (gid, tbl) in enumerate(tables.items()):
            braces += ('\n' * bool(i) +
                       hbrace(tbl.data.shape[0], gid) +
                       '\n' * tbl.has_totals)

        # # vertical offset
        stack = list(tables.values())
        offset = stack[0].n_head_nl
        print(hstack([vstack(stack), braces], spacing=1, offset=offset))
        return tables, braces

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

# class shocObsBase()

# class PhotHelper:
#     """helper class for photometry interface"""
