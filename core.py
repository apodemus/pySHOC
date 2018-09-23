# __version__ = '2.13'

import os
import re
import mmap
import time
import datetime
# import logging
import operator
from pathlib import Path
from copy import copy
import itertools as itt
from collections import namedtuple, OrderedDict
from typing import Union, ClassVar

from dataclasses import dataclass, field

# note: THESE IMPORT ARE MEGA SLOW!! ~10s  (localize to mitigate?)
import numpy as np
# import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
from astropy.io.fits.hdu import HDUList, PrimaryHDU
import more_itertools as mit

import recipes.iter as itr
from recipes.io import warn
from recipes.set import OrderedSet
from recipes.list import sorter
from recipes.logging import LoggingMixin
from recipes.dict import AttrReadItem as AttrRepr
from motley.table import Table as sTable
from obstools.utils import fmt_hms

# TODO: choose which to use for timing: spice or astropy
# from .io import InputCallbackLoop
from .utils import retrieve_coords, convert_skycooords
from .timing import timingFactory, Time, get_updated_iers_table
from .header import shocHeader
from .convert_keywords import KEYWORDS as kw_old_to_new
from .filenaming import NamingConvention


# debugging
from IPython import embed
from motley.profiler.timers import timer  # , profiler

# def warn(message, category=None, stacklevel=1):
# return warnings.warn('\n'+message, category=None, stacklevel=1)


# FIXME:
# Many of the functions here have verbosity argument. replace with logging
# FIXME:
# Autodetect corrupt files?  eg.: single exposure files (new) sometimes don't
# contain KCT

# TODO
# __all__ = ['']

# TODO: can you pickle these classes
#

# Regex to match end of FITS header
MATCH_END = re.compile(b'END {77}\s*')
# Regex to identify type of observation bias / dark / flats / science
kindRegex = b"""
(?:OBSTYPE)     # keywords (non-capture group if `capture_keywords` False)
\s*?=\s+        # (optional) whitespace surrounding value indicator '='
'?([^'\s/]*)'?  # value associated with any of the keys (un-quoted)
"""
kindMatcher = re.compile(kindRegex, re.VERBOSE)


def get_obstype(filename):
    filesize = os.path.getsize(filename)
    with open(filename, 'rb') as fileobj:
        filemap = mmap.mmap(
                fileobj.fileno(), filesize, access=mmap.ACCESS_READ
        )
        mo = MATCH_END.search(filemap)
        if mo is None:
            raise IOError('Could not match header')

        m = kindMatcher.search(filemap[:mo.end()])
        if m is None:
            return  # obstype unknown!

        return m.group(1).decode()


# def attrgetter_default(*attrs, **kw):
#     """
#     Like attrgetter, but defaults to None (or specified default) if attr
#     doesn't exist.
#     Always returns tuple, even if only one in attrs (unlike attrgetter)
#     """
#     default = kw.pop('default', None)
#     if kw:
#         raise TypeError("attrgetter_default() got unexpected "
#             "keyword argument(s): %r" % sorted(kw))
#     def fn(obj):
#         getter = lambda attr: getattr(obj, attr, default)
#         return tuple(map(getter, attrs))
#     return fn


def median_scaled_median(data, axis):  # TODO: move to utils
    """"""
    frame_med = np.median(data, (1, 2))[:, None, None]
    scaled = data / frame_med
    return np.median(scaled, axis)


def mad(data, data_median=None, axis=None):      # TODO: move to utils
    """
    Median absolute deviation

    Parameters
    ----------
    data
    data_median

    Returns
    -------

    """
    if data_median is None:
        data_median = np.median(data, axis)
    return np.median(np.abs(data - data_median), axis)


def apply_stack(func, *args, **kws):         # TODO: move to proc
    # TODO:  MULTIPROCESS HERE!
    return func(*args, **kws)


################################################################################
# class definitions
################################################################################
# HDU Subclasses
class shocHDU(PrimaryHDU):
    def __init__(self, data=None, header=None, do_not_scale_image_data=False,
                 ignore_blank=False, uint=True, scale_back=None):
        PrimaryHDU.__init__(self,
                            data=data, header=header,
                            do_not_scale_image_data=do_not_scale_image_data,
                            uint=uint,
                            ignore_blank=ignore_blank,
                            scale_back=scale_back)

        # convert to shocHeader. Passing to the `PrimaryHDU` init does not
        # create the right type of object
        self.header = shocHeader(header)


class shocNewHDU(shocHDU):
    @classmethod
    def match_header(cls, header):
        old, new = zip(*kw_old_to_new)
        return all([kw in header for kw in new])


class shocOldHDU(shocHDU):
    @classmethod
    def match_header(cls, header):
        old, new = zip(*kw_old_to_new)
        return any((kw in header for kw in old))


# # TODO:
# Consider shocBiasHDU, shocFlatFieldHDU + match headers by looking at obstype
# keywords
#
# register_hdu(shocNewHDU)
# register_hdu(shocOldHDU)


class Date(datetime.date):
    """
    We need this so the datetime.date instances print in date format instead
    of the class representation format, when print is called on, for eg. a tuple
    containing a date_time object.
    """

    def __repr__(self):
        return str(self)

    # def fn_repr(self):


# class AttrRepr(AttrReadItem):
#     pass


class yxTuple(tuple):
    def __init__(self, *args):
        assert len(self) == 2
        self.y, self.x = self


class Binning(yxTuple):
    def __str__(self):
        return '%ix%i' % self


@dataclass(order=True)
class Mode:
    readoutFrq: float
    preAmpGain: float
    outAmpMode: str
    nullGainStr: ClassVar = '--'
    emGain: Union[int, str] = nullGainStr
    acqMode: str = field(default='', repr=False)

    def __post_init__(self):
        self.isEM = (self.outAmpMode == 'EM')
        # the gain is sometimes erroneously recorded in the header as having a
        # non-zero value even though the pre-amp mode is CON. We wrap the
        # attribute as a property to retrieve the correct value
        if not self.isEM:
            self.emGain = self.nullGainStr

    def fn_repr(self):
        return 'γ{:g}@{:g}MHz.{:s}'.format(self.preAmpGain, self.readoutFrq,
                                           self.outAmpMode)

    def __hash__(self):
        return hash((self.readoutFrq, self.preAmpGain,
                     self.outAmpMode, self.emGain))

    # def __lt__(self, other):


class PPrintHelper(object):
    # handle default attributes for `pprint` and `get_instrumental_setup`
    _default_attrs = [('shape',),
                      ('binning',),
                      ('preAmpGain',),
                      ('outAmpMode',),
                      ('mode.emGain', 'emGain'),
                      ('timing.trigger.mode', 'trigger.mode'),
                      ('timing.t0repr', 't0 (UTC)'),
                      # TODO: separate date and t0 for even more compact repr
                      ('timing.duration', 'duration'),
                      ('timing.texp', 'tExp (s)'), ]  # 'ron'

    # for each tuple in list _default_attrs, item 0 is attr name, item 1
    # (optional) is the display name

    def __init__(self, attrs=None):

        if attrs is None:
            attrs = self._default_attrs

        self.attrs, self.headers = [], []
        for attr in attrs:
            self.add_attr(*attr)

    def add_attr(self, attr, header=None):

        if not isinstance(attr, str):
            raise ValueError('Attribute must be a str')

        if header is None:
            header = attr

        # avoid duplication
        if attr in self.attrs:
            return

        self.attrs.append(attr)
        self.headers.append(header)


# class shocObsBase()

# class PhotHelper:
#     """helper class for photometry interface"""


class shocObs(HDUList, LoggingMixin):
    """
    Extend the hdu.hdulist.HDUList class to perform simple tasks on the image
    stack.
    """

    # TODO: interface FitsCube for optimized read access ?

    kind = 'science'
    location = 'sutherland'

    pprinter = PPrintHelper()

    @classmethod
    def load(cls, fileobj, kind=None, mode='update', memmap=False,
             save_backup=False, **kwargs):
        """
        Load `shocObs` instance from file
        """
        # This method acts as a factory to create the cube class

        # convert Path to str
        fileobj = str(fileobj)

        # discover observation type
        if kind is None:
            kind = get_obstype(fileobj)

        # make sure we create the right kind of Run
        if kind == 'flat':
            kls = shocFlatFieldObs
        elif kind == 'bias':
            kls = shocBiasObs
        else:
            kls = cls

        return kls._readfrom(fileobj=fileobj, mode=mode, memmap=memmap,
                             save_backup=save_backup, ignore_missing_end=True,
                             # do_not_scale_image_data=True,
                             **kwargs)

    def __init__(self, hdus=None, file=None):

        if hdus is None:
            hdus = []

        # initialize HDUList
        HDUList.__init__(self, hdus, file)

        #
        self.path, self.basename = os.path.split(self.filename())
        if self.basename:
            self.basename = self.basename.replace('.fits', '')

        # if len(self):
        # FIXME:
        # either remove these lines from init or detect new file write. or ??
        # Load the important header information as attributes
        self.instrumental_setup()  # fixme: fails on self.writeto

        # Initialise timing class
        timingClass = timingFactory(self)
        self.timing = timingClass(self)
        # this is a hack that allows us to set the timing associated methods
        # dynamically
        self.kct = self.timing.kct
        # Set this attribute here so the cross matching algorithm works.
        # todo: maybe inherit from the timing classes directly to avoid the
        # previous line

        # except ValueError as err:
        #     if str(err).startswith('No GPS triggers provided'):
        #         pass
        #     else:
        #         raise err
        # else:
        #     warn('Corrupted file: %s' % self.basename)

        # except Exception as err:
        # import traceback
        # warnings.warn('%s: %s.__init__ failed with: %s' %
        #               (self.basename, self.__class__.__name__, str(err)))
        # pass
        # print('FAIL!! '*10)
        # print(err)
        # embed()

        # self.filename_gen = FilenameGenerator(self.basename)
        # self.trigger = None

    def __str__(self):
        filename, dattrs, values = self.get_instrumental_setup()
        attrRep = map('%s = %s'.__mod__, zip(dattrs, values))
        clsname = self.__class__.__name__
        sep = '; '
        return '%s (%s): %s' % (clsname, filename, sep.join(attrRep))

    def instrumental_setup(self):
        """
        Retrieve the relevant information about the observational setup from
        header and set them as attributes.
        """
        # todo: move to __init__

        header = self.header

        # instrument
        serno = header['SERNO']
        serialIds = [5982, 6448]
        shocNr = serialIds.index(serno) + 1
        self.instrument = 'SHOC %i' % shocNr
        self.telescope = header.get('telescop')

        # date from header
        date, time = header['DATE'].split('T')
        self.date = Date(*map(int, date.split('-')))
        # oldSHOC: file creation date
        # starting date of the observing run: used for naming
        h = int(time.split(':')[0])
        namedate = self.date - datetime.timedelta(int(h < 12))
        self.nameDate = str(namedate).replace('-', '')

        # image binning
        self.binning = Binning(header['%sBIN' % _] for _ in 'VH')

        # image dimensions
        self.ndims = header['NAXIS']  # Number of image dimensions
        # Pixel dimensions
        self.ishape = header['NAXIS1'], header['NAXIS2']
        self.nframes = header.get('NAXIS3', 1)
        self.shape = tuple(header['NAXIS%i' % i]
                           for i in range(1, self.ndims + 1))

        # sub-framing
        self.subrect = np.array(header['SUBRECT'].split(','), int)
        # subrect stores the sub-region of the full CCD array captured for this
        #  observation
        # xsub, ysub = (xb, xe), (yb, ye) = \
        xsub, ysub = self.subrect.reshape(-1, 2) // self.binning
        # for some reason the ysub order is reversed
        self.sub = tuple(xsub), tuple(ysub[::-1])
        self.subSlices = list(map(slice, *np.transpose(self.sub)))

        # Readout stats
        # set the correct values here as attributes of the instance. These
        # though not values may be absent in the headers
        self.ron, self.sensitive, self.saturate = header.get_readnoise()

        # CCD mode
        self.acqMode = header['ACQMODE']
        self.preAmpGain = header['PREAMP']
        outAmpModeLong = header['OUTPTAMP']
        self.outAmpMode = 'CON' if outAmpModeLong.startswith('C') else 'EM'

        # readout speed
        readoutFrq = 1. / header['READTIME']
        self.readoutFrq = int(round(readoutFrq / 1.e6))  # MHz

        # gain   # should be printed as '--' when mode is CON
        self._emGain = header.get('gain', None)
        # Mode tuple
        self.mode = Mode(self.readoutFrq, self.preAmpGain,
                         self.outAmpMode, self._emGain)

        # orientation
        self.flip_state = yxTuple(header['FLIP%s' % _] for _ in 'YX')
        # NOTE: row, column order
        # WARNING: flip state wrong for old SHOC data : TODO confirm this

        # filters
        Filters = namedtuple('Filters', list('AB'))
        self.filters = Filters(*(header.get('WHEEL%s' % _) for _ in 'AB'))

        # Timing
        # self.kct = self.timing.kct
        # self.trigger_mode = self.timing.trigger.mode
        # self.duration

        # object name
        self.target = header.get('object')  # objName

        # coordinates
        self.coords = self.get_coords()
        # note: self.coords may be None

    @property
    def header(self):
        """retrieve PrimaryHDU header"""
        return self[0].header

    def _get_data(self):
        """retrieve PrimaryHDU data. """
        return self[0].data
        # note: SHOC data is always a 3D array.  Since we will always have only
        # one observation cube in the HDUList we can do this without
        # ambiguity.

    def _set_data(self, data):
        """set PrimaryHDU data"""
        self[0].data = data

    data = property(_get_data, _set_data)

    def get_filename(self, with_path=False, with_ext=True, suffix=(), sep='.'):

        path, filename = os.path.split(self.filename())

        if isinstance(with_path, str):
            filename = os.path.join(with_path, filename)
        elif with_path:
            filename = self.filename()

        *parts, ext = filename.split(sep)
        ext = [ext if with_ext else '']
        suffix = [suffix] if isinstance(suffix, str) else list(suffix)
        suffix = [s.strip(sep) for s in suffix]

        return sep.join(filter(None, parts + suffix + ext))

    def get_attr_repr(self):
        return AttrRepr(sep='.',
                        kind=self.kind,
                        obj=self.target,
                        # filter=header.get('FILTER', 'WL'),
                        basename=self.get_filename(0, 0),
                        date=self.nameDate,
                        binning=str(self.binning),
                        mode=self.mode.fn_repr(),
                        kct=self.timing.td_kct.hms)

    def get_coords(self, verbose=False):
        header = self.header

        ra, dec = header.get('objra'), header.get('objdec')
        coords = convert_skycooords(ra, dec)
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
        # note: These will lead to slightly less accurate timing solutions
        ra, dec = header.get('telra'), header.get('teldec')
        coords = convert_skycooords(ra, dec)

        # TODO: optionally query for named sources in this location
        if coords:
            warn('Using telescope pointing coordinates.')

        return coords

    @property
    def needs_timing(self):
        """
        check for date-obs keyword to determine if header information needs updating
        """
        return not ('date-obs' in self.header)
        # TODO: is this good enough???

    @property
    def has_coords(self):
        return self.coords is not None

    def get_instrumental_setup(self, attrs=None, headers=None):
        # TODO: units

        filename = self.get_filename() or '<Unsaved>'
        attrNames, attrDisplayNames = attrs, headers
        if attrs is None:
            attrNames = self.pprinter.attrs[:]

        if headers is None:
            attrDisplayNames = self.pprinter.headers[:]

        # check correct number of attrs / headers
        assert len(attrNames) == len(attrDisplayNames)
        attrVals = operator.attrgetter(*attrNames)(self)

        return filename, attrDisplayNames, attrNames, attrVals

    def get_field_of_view(self, telescope=None, unit='arcmin',
                          with_focal_reducer=False):
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

        """
        # Field of view in arcmin
        fov74 = (1.29, 1.29)
        fov74r = (2.79, 2.79)  # with focal reducer
        fov40 = (2.85, 2.85)
        # fov30 = (3.73, 3.73)

        # PS. welcome to the new millennium, we use the metric system now
        if telescope is None:
            telescope = self.header.get('telescop')

        telescope = str(telescope)
        telescope = telescope.rstrip('inm')  # strip "units" in name
        if with_focal_reducer:
            fov = {'74': fov74r,
                   '1.9': fov74r}.get(telescope)
        else:
            fov = {  # '30': fov30, '0.75': fov30,
                '40': fov40, '1.0': fov40, '1': fov40,
                '74': fov74, '1.9': fov74
            }.get(telescope)

        if fov is None:
            raise ValueError('Please specify telescope to get field of view.')

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

    get_FoV = get_field_of_view

    def get_pixel_scale(self, telescope=None, unit='arcmin',
                        with_focal_reducer=False):
        """pixel scale in `unit` per binned pixel"""
        return self.get_field_of_view(telescope, unit,
                                      with_focal_reducer) / self.ishape

    get_plate_scale = get_pixel_scale

    def cross_check(self, frame2, key, raise_error=0):
        """
        Check fits headers in this image against frame2 for consistency of
        key attribute

        Parameters
        ----------
        key:    str
            The attribute to be checked (binning / instrument mode / dimensions
            / flip state)
        frame2:
            shocObs Objects to check against

        Returns
        ------
        flag : Do the keys match?
        """
        flag = (getattr(self, key) == getattr(frame2, key))

        if not flag and raise_error:
            raise ValueError
        else:
            return flag

    def flip(self, state=None):

        state = self.flip_state if state is None else state
        header = self.header
        for axis, yx in enumerate('YX'):
            if state[axis]:
                self.logger.info('Flipping %r in %s.', self.get_filename(), yx)
                self.data = np.flip(self.data, axis + 1)
                header['FLIP%s' % yx] = int(not self.flip_state[axis])

        self.flip_state = tuple(header['FLIP%s' % s] for s in ['YX'])
        # FIXME: avoid this line by making flip_state a list

    @property
    def is_subframed(self):
        return np.any(self.sub[:, 1] != self.ishape)

    def subframe(self, subreg, write=True):

        cb, ce, rb, re = subreg
        self.logger.info('subframing %r to %s', self.filename(),
                         [rb, re, cb, ce])

        data = self.data[rb:re, cb:ce]
        header = self.header
        # header['SUBRECT']??

        print('!' * 8, self.sub)

        subext = 'sub{}x{}'.format(re - rb, ce - cb)
        outname = self.get_filename(1, 1, subext)
        fileobj = pyfits.file._File(outname, mode='ostream', overwrite=True)

        hdu = PrimaryHDU(data=data, header=header)
        stack = self.__class__(hdu, fileobj)
        # stack.instrumental_setup()
        # stack._is_subframed = 1
        # stack._needs_sub = []
        # stack.sub = subreg

        if write:
            stack.writeto(outname, output_verify='warn')

        return stack

    def combine(self, func, name=None):
        """
        Mean / Median combines the image stack

        Returns
        -------
        shocObs instance
        """

        # "Median combining can completely remove cosmic ray hits and
        # radioactive decay trails from the result, which cannot be done by
        # standard mean combining. However, median combining achieves an
        # ultimate signal to noise ratio about 80% that of mean combining he
        # same number of frames. The difference in signal to noise ratio can
        # be by median combining 57% more frames than if mean combining were
        # used. In addition, variants on mean combining, such as sigma
        # clipping, can remove deviant pixels while improving the S/N
        # somewhere between that of median combining and ordinary mean
        # combining. In a nutshell, if all images are "clean", use mean
        # combining. If the images have mild to severe contamination by
        # radiation events such as cosmic rays, use the median or sigma
        # clipping method." - Newberry

        # mean / median across images
        imnr = '001'  # FIXME:   #THIS WILL NEED TO CHANGE FOR MULTIPLE SINGLE IMAGES AS INPUT
        header = copy(self.header)
        data = apply_stack(func, self.data, axis=0)

        ncomb = header.pop('NUMKIN', 0)  # Delete the NUMKIN header keyword
        header['NCOMBINE'] = (ncomb, 'Number of images combined')
        header['ICMB' + imnr] = (
            self.filename(), 'Contributors to combined output image')
        # FIXME: THIS WILL NEED TO CHANGE FOR MULTIPLE SINGLE IMAGES AS INPUT

        # Load the stack as a shocObs
        if name is None:
            name = next(self.filename_gen())  # generate the filename

        hdu = shocHDU(data, header)
        fileobj = pyfits.file._File(name, mode='ostream', overwrite=True)
        stack = self.__class__(hdu,
                               fileobj)  # initialise the Cube with target file
        stack.instrumental_setup()

        return stack

    def unpack(self, count=1, padw=None, dryrun=0, w2f=1):
        # todo MULTIPROCESS
        """
        Unpack (split) a 3D cube of images along the 3rd axis.

        Parameters
        ----------
        outpath : The directory where the imags will be unpacked
        count : A running file count
        padw : The number of place holders for the number suffix in filename
        dryrun: Whether to actually unpack the stack

        Returns
        ------
        count
        """
        start_time = time.time()

        stack = self.get_filename()
        header = copy(self.header)
        naxis3 = self.nframes
        # number of digits in filename number string
        self.filename_gen.padwidth = padw if padw else len(str(naxis3))
        self.filename_gen.count = count

        if not dryrun:
            # edit header
            header.remove('NUMKIN')
            # Delete this keyword so it does not propagate into the headers of
            # the split files
            header.remove('NAXIS3')
            header['NAXIS'] = 2  # Number of axes becomes 2
            header.add_history('Split from %s' % stack)

            # open the txt list for writing
            if w2f:
                basename = self.get_filename(1, 0)
                self.unpacked = basename + '.split'
                fp = open(self.unpacked, 'w')

            self.logger.info('Unpacking the stack %s of %i images.', stack,
                             naxis3)

            # split the cube
            filenames = self.filename_gen(naxis3 + count - 1)
            bar.create(naxis3)
            for j, im, fn in zip(range(naxis3), self.data, filenames):
                bar.progress(count - 1)
                # count instead of j in case of sequential numbering for
                #  multiple cubes

                self.timing.stamp(
                        j)  # set the timing values in the header for frame j

                pyfits.writeto(fn, data=im, header=header, overwrite=True)

                if w2f:
                    fp.write(fn + '\n')  # OR outname???????????
                count += 1

            if w2f:
                fp.close()

            # how long did the unpacking take
            end_time = time.time()
            self.logger.debug('Time taken: %f', (end_time - start_time))

        self._is_unpacked = True

        return count

    # def writeto(self, fileobj, output_verify='exception', overwrite=False, checksum=False):
    #     self._in_write = True

    def plot(self, **kws):  # TODO: display...
        """Display the data"""
        if self.ndims == 2:
            from graphical.imagine import ImageDisplay
            im = ImageDisplay(self.data, **kws)

        elif self.ndims == 3:
            from graphical.imagine import VideoDisplay
            im = VideoDisplay(self.data, **kws)

        else:
            raise TypeError('Not an image!! WTF?')

        im.figure.canvas.set_window_title(self.get_filename())
        return im

    # def animate(self):
    #   TODO: OR incorporate in VideoDisplay


################################################################################
class shocBiasObs(shocObs):  # FIXME: DARK?
    kind = 'bias'

    def get_coords(self):
        return

    def compute_master(self, func=np.median, masterbias=None, name=None):
        return self.combine(func, name)


class shocFlatFieldObs(shocBiasObs):
    kind = 'flat'

    def compute_master(self, func=median_scaled_median, masterbias=None,
                       name=None):
        """ """
        master = self.combine(func, name)
        if masterbias:
            self.logger.info('Subtracting master bias %r from master flat %r.',
                             masterbias.get_filename(), master.basename)
            # match orientation
            masterbias.flip(master.flip_state)
            master.data -= masterbias.data

        self.logger.info('No master bias for %r', self.filename())

        # flat field normalization
        self.logger.info('Normalising flat field...')
        ffmean = np.mean(master.data)  # flat field mean
        master.data /= ffmean
        return master


class ClassProperty(property):
    def __get__(self, cls, owner):
        return self.fget.__get__(None, owner)()


################################################################################
class shocRun(LoggingMixin):
    # TODO: merge method?
    """
    Class containing methods to operate with sets of SHOC observations (shocObs
    instances.)
    Group, filter and compare across cubes based on keywords.
    Calibrate, calculate time stamps (including barycentrization)
    Merge, stack or combine the cubes, or unpack the individual frames.
    Write output fits and / or timing files.
    Pretty printed table representations of your SHOC runs
    """

    # Observation type
    obsClass = shocObs
    # Naming convention defaults
    nameFormat = '{basename}'
    # this is here so we can initialize new instances from existing instances
    _skip_init = False

    # options for pretty printing
    displayColour = 'g'
    _compactRepr = True
    # compactify the table representation form the `print_instrumental_setup`
    # method by removing columns (attributes) that are equal across all
    # constituent cubes and printing them as a top row in the table
    pprinter = PPrintHelper()

    @ClassProperty  # so we can access via shocRun.kind and shocRun().kind
    @classmethod
    def kind(cls):
        return cls.obsClass.kind

    @classmethod
    def load(cls, filenames, kind=None, mode='update', memmap=False,
             save_backup=False, **kwargs):
        """
        Load data from file(s).

        Parameters
        ----------
        filenames
        kind
        mode
        memmap
        save_backup
        kwargs

        Returns
        -------

        """
        label = kwargs.get('label', None)
        cls.logger.info('Loading data for %s run: %s', kind or '', label or '')

        # sanitize filenames:  input filenames may contain None - remove these
        filenames = list(filter(None, filenames)) if filenames else []
        hdus = []
        obsTypes = set()
        for i, fileobj in enumerate(filenames):
            # try:
            obs = shocObs.load(fileobj, kind=kind, mode=mode, memmap=memmap,
                               save_backup=save_backup, **kwargs)
            hdus.append(obs)
            obsTypes |= {obs.kind}

            # except Exception as err:
            #     import traceback
            #     warn('File: %s failed to load with exception:\n%s'
            #  % (fileobj, str(err)))

            # set the pretty printer as same object for all shocObs in the Run
            obs.pprinter = cls.pprinter

        #
        if len(obsTypes) > 1:
            cls.logger.warning(
                    'Observation types are not uniform: %s' % obsTypes)

        kls = cls  # defaults to parent object
        if kind is None and len(obsTypes) == 1:
            # we can safely guess the type of observation (bias / flat /science)
            kind = next(iter(obsTypes))
            kls = SHOCRunKinds[kind]

        return kls(hdus, label)

    @classmethod
    def load_dir(cls, path):
        """Load observing campaign fits files from a directory"""
        path = Path(path)
        if not path.is_dir():
            raise IOError('%r is not a directory' % str(path))

        filepaths = list(path.glob('*.fits'))

        if len(filepaths) == 0:
            # although we could load an empty run here, least surprise
            # dictates throwing an error
            raise IOError("Directory %s contains no valid '*.fits' files"
                          % str(path))

        return cls.load(filepaths)

    def __init__(self, hdus=None, label=None, groupId=None):

        if hdus is None:
            hdus = []

        for hdu in hdus:
            if not isinstance(hdu, HDUList):
                raise TypeError('Cannot initialize from %r. '
                                'Please use `shocRun.load(filename)`' % hdu)

        # put hdu objects in array to ease item getting. YAY!!
        self.cubes = np.empty(len(hdus), dtype='O')
        self.cubes[:] = hdus
        self.groupId = OrderedSet(groupId)
        self.label = label

    def __len__(self):
        return len(self.cubes)

    def __repr__(self):
        cls_name = self.__class__.__name__
        files = ' | '.join(self.get_filenames())
        return '<%s>' % ' : '.join((cls_name, files))

    def __getitem__(self, key):
        #
        items = self.cubes[key]

        if isinstance(key, (int, np.integer)):
            return items

        return self.__class__(items, self.label, self.groupId)

        # if np.size(items) > 1:
        #     return self.__class__(items, self.label, self.groupId)
        # return items

    def __add__(self, other):
        if self.kind != other.kind:
            self.logger.warning('Joining Runs of different kinds')
        if self.label != other.label:
            self.logger.info('Suppressing label %s', other.label)

        groupId = (self.groupId | other.groupId)
        cubes = np.r_[self.cubes, other.cubes]
        return self.__class__(cubes, self.label, groupId)

    def attrgetter(self, *attrs):
        """
        Fetch attributes from the inner class.
        see: builtin `operator.attrgetter` for details

        Parameters
        ----------
        attrs: tuple or str
            Attribute name(s) to retrieve

        Returns
        -------
        list of (tuples of) attribute values


        Examples
        --------
        >>> obs.attrgetter('emGain')
        >>> obs.attrgetter('date.year')
        """
        return list(map(operator.attrgetter(*attrs), self.cubes))

    def methodcaller(self, name, *args, **kws):
        """

        Parameters
        ----------
        name
        args
        kws

        Returns
        -------

        """
        return list(map(operator.methodcaller(name, *args, **kws),
                        self.cubes))

    @property
    def filenames(self):
        return [cube.filename() for cube in self.cubes]

    def pop(self, i):
        return np.delete(self.cubes, i)

    def join(self, *runs, unique=False):
        # TODO: option to filter duplicates
        # Filter empty runs (None)
        runs = filter(None, runs)
        if unique:
            runs = mit.unique_everseen(runs)
        return sum(runs, self)

        # for run in runs:
        # kinds = [run.kind for run in runs]
        # labels = [run.label for run in runs]
        # self.cubes, [r.cubes for r in runs]
        # hdus = sum([r.cubes for r in runs], self.cubes)

        # if np.all(self.label == np.array(labels)):
        #     label = self.label
        # else:
        #     self.logger.warning("Labels %s don't match %r!", str(labels), self.label)
        #     label = None

        # return self.__class__(hdus, label=label)

    def print_instrumental_setup(self, attrs=None, description='', **kws):
        """
        Print the instrumental setup for this run as a table.

        Parameters
        ----------
        attrs: array_like, optional
            Attributes of the instance that will be printed in the table.
            defaults to the list given in `self.pprinter.attrs`
        description: str, optional
            Description of the observation to be included in the table.
        **kws:
            Keyword arguments passed directly to the `ansi.table.Table`
            constructor.

        Returns
        -------
        `motley.table.Table` instance

        """
        filenames, attrDisplayNames, attrNames, attrVals = \
            zip(*(stack.get_instrumental_setup(attrs) for stack in self))
        attrDisplayNames = attrDisplayNames[0]  # all are the same
        name = self.label or ''  # ''<unlabeled>'      # todo: default?
        suptitle = 'Instrumental setup: SHOC %s frames' % self.kind.title()
        subtitle = ': '.join(filter(None, (str(name).title(), description)))
        title = os.linesep.join(filter(None, (suptitle, subtitle)))

        # if duration requested, als print total duration
        total = ()
        if len(self) > 1:
            for i, at in enumerate(attrNames[0]):
                if 'duration' in at:
                    total = (i,)
                    break

        kws.setdefault('width', range(140))
        table = sTable(attrVals,
                       title=title,
                       title_props=dict(text='bold', bg=self.displayColour),
                       col_headers=attrDisplayNames,
                       row_headers=['filename'] + list(filenames),
                       number_rows=True,
                       precision=5, minimalist=True, compact=True,
                       formatters={'duration': fmt_hms},
                       #  FIXME: uniform precision hms
                       total=total,
                       **kws)

        print(table)
        return table

    pprint = print_instrumental_setup

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Timing
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def set_times(self, coords=None):

        # initialize timing
        t0s = []
        for stack in self:
            t0s.append(stack.timing.t0mid)

        # check whether IERS tables are up to date given the cube starting times
        t_test = Time(t0s)
        status = t_test.check_iers_table()
        # Cached IERS table will be used if only some of the times are outside
        # the range of the current table. For the remaining values the predicted
        # leap second values will be used.  If all times are beyond the current
        # table a new table will be grabbed online.

        # update the IERS table and set the leap-second offset
        iers_a = get_updated_iers_table(cache=True, raise_=False)

        msg = 'Calculating timing arrays for data cube(s):'
        lm = len(msg)
        self.logger.info('\n\n' + msg)
        for i, stack in enumerate(self):
            self.logger.info(' ' * lm + stack.get_filename())
            t0 = t0s[i]  # t0 = stack.timing.t0mid
            if coords is None and stack.has_coords:
                coords = stack.coords
            stack.timing.set(t0, iers_a, coords)
            # update the header with the timing info
            stack.timing.stamp(0, t0, coords, verbose=True)
            stack.flush(output_verify='warn', verbose=True)

    def export_times(self, with_slices=False):
        for i, stack in enumerate(self):
            timefile = stack.get_filename(1, 0, 'time')
            stack.timing.export(timefile, with_slices, i)

    def gen_need_kct(self):
        """
        Generator that yields the cubes in the run that require KCT to be set
        """
        for stack in self:
            yield (stack.kct is None) and stack.timing.trigger.is_gps()

    def that_need_kct(self):
        """
        Return a shocRun object containing only those cubes that are missing KCT
        """
        return self[list(self.gen_need_kct())]

    def gen_need_triggers(self):
        """
        Generator that yields the cubes in the run that require GPS triggers to
        be set
        """
        for stack in self:
            # FIXME: value of date-obs?
            yield (stack.needs_timing and stack.timing.trigger.is_gps())

    def that_need_triggers(self):
        """
        Return a shocRun object containing only those cubes that are missing
        triggers
        """
        return self[list(self.gen_need_triggers())]

    def set_gps_triggers(self, times, triggers_in_local_time=True):
        # trigger provided by user at terminal through -g (args.gps)
        # if len(times)!=len(self):
        # check if single trigger OK (we can infer the remaining ones)
        if self.check_rollover_state():
            times = self.get_rolled_triggers(times)
            msg = ('A single GPS trigger was provided. Run contains '
                   'auto-split observation cubes (filesystem rollover due to '
                   '2Gb threshold on old windows server). Start time for  '
                   'rolled over ecubes will be inferred  from the length of '
                   'the preceding cube(s).\n')
            self.logger.info(msg)

        # at this point we expect one trigger time per cube
        if len(self) != len(times):
            raise ValueError('Only {} GPS trigger given. Please provide {} for '
                             '{}'.format(len(times), len(self), self))

        # warn('Assuming GPS triggers provided in local time (SAST)')
        for j, stack in enumerate(self):
            stack.timing.trigger.set(times[j], triggers_in_local_time)

    def check_rollover_state(self):
        """
        Check whether the filenames contain ',_X' an indicator for whether the
        datacube reached the 2GB windows file size limit on the shoc server, and
        was consequently split into a sequence of fits cubes.

        Notes:
        -----
        This applies for older SHOC data only
        """
        return np.any(['_X' in _ for _ in self.get_filenames()])

    def get_rolled_triggers(self, first_trigger_time):
        """
        If the cube rolled over while the triggering mode was 'External' or
        'External Start', determine the start times (inferred triggers) of the
        rolled over cube(s).
        """
        nframes = [cube.nframes for cube in self]  # stack lengths
        # sorts the file sequence in the correct order
        # re pattern to find the roll-over number (auto_split counter value
        # in  filename)
        matcher = re.compile('_X([0-9]+)')
        fns, nframes, idx = sorter(self.get_filenames(), nframes,
                                   range(len(self)),
                                   key=matcher.findall)

        print('get_rolled_triggers', 'WORK NEEDED HERE!')
        embed()
        # WARNING:
        # This assumes that the run only contains cubes from the run that
        # rolled-over. This should be ok for present purposes but might not
        # always be the case
        idx0 = idx[0]
        self[idx0].timing.trigger.start = first_trigger_time
        t0, td_kct = self[idx0].time_init(dryrun=1)
        # dryrun ==> don't update the headers just yet (otherwise it will be
        # done twice!)

        d = np.roll(np.cumsum(nframes), 1)
        d[0] = 0
        t0s = t0 + d * td_kct
        triggers = [t0.isot.split('T')[1] for t0 in t0s]

        # resort the triggers to the order of the original file sequence
        # _, triggers = sorter( idx, triggers )

        return triggers

    def export_headers(self):
        """save fits headers as a text file"""
        for stack in self:
            headfile = stack.get_filename(with_path=1, with_ext=0,
                                          suffix='.head')
            self.logger.info('Writing header to file: %r',
                             os.path.basename(headfile))
            stack.header.totextfile(headfile, overwrite=True)

    def get_filenames(self, with_path=False, with_ext=True, suffix=(), sep='.'):
        """filenames of run constituents"""
        return [stack.get_filename(with_path, with_ext, suffix, sep) for stack
                in self]

    def export_filenames(self, fn):

        if not fn.endswith('.txt'):  # default append '.txt' to filename
            fn += '.txt'

        self.logger.info('Writing names of %s to file %r', self.name, fn)
        with open(fn, 'w') as fp:
            for f in self.filenames:
                fp.write(f + '\n')

    def writeout(self, with_path=False, suffix='', dryrun=False,
                 header2txt=False):  # TODO:  INCORPORATE FILENAME GENERATOR
        fns = []
        for stack in self:
            fn_out = stack.get_filename(with_path, False, (
                suffix, 'fits'))  # FILENAME GENERATOR????
            fns.append(fn_out)

            if not dryrun:
                self.logger.info('Writing to file: %r',
                                 os.path.basename(fn_out))
                stack.writeto(fn_out, output_verify='warn', overwrite=True)

                if header2txt:
                    # save the header as a text file
                    headfile = stack.get_filename(1, 0, (suffix, 'head'))
                    self.logger.info('Writing header to file: %r',
                                     os.path.basename(headfile))
                    stack.header.totextfile(headfile, overwrite=True)

        return fns

    def zipper(self, keys, flatten=True):
        # TODO: eliminate this function

        # NOTE: this function essentially accomplishes what the one-liner below does
        # attrs = list(map(operator.attrgetter(*keys), self))

        if isinstance(keys, str):
            return keys, [getattr(s, keys) for s in self]  # s.__dict__[keys]??
        elif len(keys) == 1 and flatten:
            key = tuple(keys)[0]
            return key, [getattr(s, key) for s in self]
        else:
            return (tuple(keys),
                    list(zip(
                            *([getattr(s, key) for s in self] for key in
                              keys))))

    # def group_iter(self):
    #     'todo'

    def group_by(self, *keys, **kws):
        """
                Separate a run according to the attribute given in keys.
        keys can be a tuple of attributes (str), in which case it will
        separate into runs with a unique combination of these attributes.

        Parameters
        ----------
        keys
        kws

        optional keyword: return_index

        Returns
        -------
        atdict: dict
            (val, run) pairs where val is a tuple of attribute values mapped
            to by `keys` and run is the shocRun containing observations which
            all share the same attribute values
        flag:
            1 if attrs different for any cube in the run, 0 all have the same
            attrs

        """
        attrs = self.attrgetter(*keys)
        keys = OrderedSet(keys)
        return_index = kws.get('return_index', False)
        if self.groupId == keys:  # is already separated by this key
            gr = GroupedRun(zip([attrs[0]], [self]))
            gr.groupId = keys
            # gr.name = self.name
            if return_index:
                return gr, dict(attrs[0], list(range(len(self))))
            return gr

        atset = set(attrs)  # unique set of key attribute values
        atdict = OrderedDict()
        idict = OrderedDict()
        if len(atset) == 1:
            # all input files have the same attribute (key) value(s)
            self.groupId = keys
            atdict[attrs[0]] = self
            idict[attrs[0]] = np.arange(len(self))
        else:  # key attributes are not equal across all shocObs
            for ats in sorted(atset):
                # map unique attribute values to shocObs (indices) with those
                # attributes. list comp for-loop needed for tuple attrs
                l = np.array([attr == ats for attr in attrs])
                # shocRun object of images with equal key attribute
                eq_run = self.__class__(self.cubes[l], self.label, keys)
                atdict[ats] = eq_run  # put into dictionary
                idict[ats], = np.where(l)

        gr = GroupedRun(atdict)
        gr.groupId = keys
        # gr.name = self.name
        if return_index:
            return gr, idict
        return gr

    def varies_by(self, keys):
        """False if the run is homogeneous by keys and True otherwise"""
        attrs = self.attrgetter(keys)
        atset = set(attrs)
        return len(atset) != 1

    def select_by(self, **kws):
        out = self
        for key, val in kws.items():
            out = out.group_by(key)[val]
        return out

    def filter_by(self, **kws):
        attrs = self.attrgetter(*kws.keys())
        funcs = kws.values()

        predicate = lambda att: all(f(at) for f, at in zip(funcs, att))
        selection = list(map(predicate, attrs))
        return self[selection]

    def filter_duplicates(self):
        if len(set(map(id, self))) == len(self):
            return self
        return self.__class__([next(it) for _, it in itt.groupby(self, id)])

    def sort_by(self, *keys, **kws):
        """
        Sort the cubes by the value of attributes given in keys,
        kws can be (attribute, callable) pairs in which case sorting will be done according to value
        returned by callable on a given attribute.
        """

        # NOTE:
        # For python <3.5 order of kwargs is lost when passing so this
        # function may not work as expected when passing multiple sorting
        # functions as keyword argumonts.
        # see: https://docs.python.org/3/whatsnew/3.6.html

        def trivial(x):
            return x

        # compose sorting function
        triv = (trivial,) * len(keys)
        # will be used to sort by the actual values of attributes in *keys*
        kwkeys = tuple(kws.keys())
        kwattrs = self.attrgetter(kwkeys)
        kwfuncs = tuple(kws.values())  #
        funcs = triv + kwfuncs
        # tuple of functions, one for each requested attribute combine all
        # functions into single function that returns tuple that determines
        # sort position
        attrSortFunc = lambda *x: tuple(f(z) for (f, z) in zip(funcs, x))

        attrs = self.attrgetter(keys)
        keys += kwkeys
        attrs += kwattrs

        ix = range(len(self))
        *attrs, ix = sorter(attrs, ix, key=attrSortFunc)

        return self[ix]

    def combined(self, names, func):
        #     """return run"""
        assert len(names) == len(self)
        cmb = []
        for stack, name in zip(self, names):
            cmb.append(stack.combine(func, name))
        return self.__class__(cmb)

    def debias(self, bias):

        for cube in self:
            cube.data = cube.data - bias.data

    #
    # def combine_data(self, func):
    #     """
    #     Stack data and combine into single frame using func
    #     """
    #     # TODO:  MULTIPROCESS HERE
    #     data = self.stack_data()
    #     return func(data,  0)       # apply func (mean / median) across images

    def check_singles(self, verbose=True):
        # check for single frame inputs (user specified master frames)
        is_single = np.equal([b.ndims for b in self], 2)
        if is_single.any() and verbose:
            msg = (
                'You have input single image(s) named: {}. These will be used as the master {}'
                ' frames').format(self[is_single], self.name)
            self.logger.info(msg)
        return is_single

    def stack(self, name):

        header = copy(self[0].header)
        data = self.stack_data()

        # create a shocObs for the stacked data
        hdu = shocHDU(data, header)
        fileobj = pyfits.file._File(name, mode='ostream', overwrite=True)
        cube = self.__class__(hdu,
                              fileobj)  # initialise the Cube with target file
        cube.instrumental_setup()

    def stack_data(self):
        """return array of stacked data"""
        # if len(self) == 1:
        #     return

        dims = self.attrgetter('ndims')
        if not np.equal(dims, dims[0]).all():  # todo: use varies_by
            raise ValueError('Cannot stack cubes with differing image sizes: '
                             '%s' % str(dims))

        return np.vstack(self.attrgetter('data'))

    def merge_combine(self, name, func, *args, **kws):  # todo: merge_combine
        """
        Combines all of the stacks in the run

        Parameters
        ----------
        name: str
           filename of the output fits file
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
        # verbose = True
        # if verbose:
        if self.logger.getEffectiveLevel() > 10:
            self.logger.info('\n%s', self._get_combine_map(func.__name__, name))

        kws.setdefault('axis', 0)
        data = apply_stack(func, self.stack_data(), *args, **kws)

        # update header     # TODO: check which header entries are different
        header = copy(self[0].header)
        ncomb = sum(next(zip(*self.attrgetter('shape'))))
        header['NCOMBINE'] = (ncomb, 'Number of images combined')
        header['FCOMBINE'] = (
            func.__name__, 'Function used to combine the data')
        # fixme: better to use HISTORY than to invent keywords
        for i, fn in enumerate(self.get_filenames()):
            header['ICMB{:0{}d}'.format(i, 3)] = (
                fn, 'Contributors to combined output image')

        # create a shocObs
        hdu = shocHDU(data, header)
        fileobj = pyfits.file._File(name, mode='ostream', overwrite=True)
        # initialise the Cube with target file
        cube = self.obsClass(hdu, fileobj)
        cube.instrumental_setup()
        return cube

    def _get_combine_map(self, fname, out):
        #                         median_scaled_median
        # SHA_20170209.0002.fits ----------------------> f20170209.4x4.fits
        s = str(self)
        a = ' '.join([' ' * (len(s) + 1), fname, ' ' * (len(out) + 2)])
        b = ' '.join([s, '-' * (len(fname) + 2) + '>', out])
        return '\n'.join([a, b])

    def unpack(self, sequential=0, dryrun=0, w2f=1):
        # unpack datacube(s) and assign 'outname' to output images
        # if more than one stack is given 'outname' is appended with 'n_' where
        # `n` is the number of the stack (in sequence)

        if not dryrun:
            outpath = self[0].filename_gen.path
            self.logger.info(
                    'The following image stack(s) will be unpacked into %r:\n%s',
                    outpath, '\n'.join(self.get_filenames()))

        count = 1
        naxis3 = [stack.nframes for stack in self]
        tot_num = sum(naxis3)
        padw = len(str(tot_num)) if sequential else None
        # if padw is None, unpack_stack will determine the appropriate pad width
        #  for the cube

        if dryrun:
            # check whether the numer of images in the timing stack are equal
            # to the total number of frames in the cubes.
            if len(args.timing) != tot_num:  # WARNING - NO LONGER VALID
                raise ValueError('Number of images in timing list ({}) not '
                                 'equal to total number in given stacks ({}).'
                                 .format(len(args.timing), tot_num))

        for i, stack in enumerate(self):
            count = stack.unpack(count, padw, dryrun, w2f)

            if not sequential:
                count = 1

        if not dryrun:
            self.logger.info('A total of %i images where unpacked', tot_num)
            if w2f:
                catfiles([stack.unpacked for stack in self], 'all.split')
                # RENAME???????????????????????????????????

    def cross_check(self, run2, keys, raise_error=0):
        """
        check fits headers in this run agains run2 for consistency of key
        (binning / instrument mode / dimensions / flip state / etc)
        Parameters
        ----------
        keys :          The attribute(s) to be checked
        run2 :          shocRun Object to check against
        raise_error :   How to treat a mismatch.
                            -1 - silent, 0 - warning (default),  1 - raise

        Returns
        ------
        boolean array (same length as instance) that can be used to filter mismatched cubes.
        """

        # lists of attribute values (key) for given input lists
        keys, attr1 = self.zipper(keys)
        keys, attr2 = run2.zipper(keys)
        fn1 = np.array(self.get_filenames())
        fn2 = np.array(run2.get_filenames())

        # which of 1 occur in 2
        match1 = np.array([attr in attr2 for attr in attr1])

        if set(attr1).issuperset(set(attr2)):
            # All good, run2 contains all the cubes with matching attributes
            return match1
            # use this to select the minimum set of cubes needed (filter out
            # mismatched cubes)

        # some attribute mismatch occurs
        # which of 2 occur in 1
        #  (at this point we know that one of these values are False)
        match2 = np.array([attr in attr1 for attr in attr2])
        if any(~match2):
            # FIXME:   ERRONEOUS ERROR MESSAGES!
            fns = ',\n\t'.join(fn1[~match1])
            badfns = ',\n\t'.join(fn2[~match2])
            # set of string values for mismatched attributes
            mmset = set(np.fromiter(map(str, attr2), 'U64')[~match2])
            mmvals = ' or '.join(mmset)
            keycomb = ('{} combination' if isinstance(keys, tuple)
                       else '{}').format(keys)
            operation = (
                'de-biasing' if 'bias' in self.name else 'flat fielding')
            desc = ('Incompatible {} in'
                    '\n\t{}'
                    '\nNo {} frames with {} {} for {}'
                    '\n\t{}'
                    '\n\n').format(keycomb, fns, self.name, mmvals,
                                   keycomb, operation, badfns)
            # msg = '\n\n{}: {}\n'

            if raise_error == 1:
                raise ValueError(desc)
            elif raise_error == 0:
                warn(desc)

            return match1

    def close(self):
        [stack.close() for stack in self]

    # TODO: as a mixin?
    def match_and_group(self, cal_run, exact, closest=None, threshold_warn=7,
                        print_=1):
        """
        Match the attributes between sci_run and cal_run. Matches exactly to
        the attributes given in exact, and as closely as possible to the
        attributes in closest. Separates sci_run by the attributes in both
        exact and closest, and builds an index dictionary for the cal_run
        which can later be used to generate a StructuredRun instance.

        Parameters
        ----------


        Returns
        ------


        Parameters
        ----------
        cal_run: shocRun
            shocRun instance to be matched which will be trimmed by matching
        exact: tuple or str
            single or multiple keywords to match for equality between the two
            runs. note: No checks are performed to ensure cal_run forms a
            subset of sci_run w.r.t. these attributes
        closest: tuple or str
            single or multiple keywords to match as closely as possible between
            the two runs. The attributes which are pointed to by these should
            support item subtraction
        threshold_warn: int, optional
            If the difference in attribute values for attributes in `closest`
            are greater than `threshold_warn`, a warning is emitted
        print_: bool
            whether to print the resulting matches in a table

        Returns
        -------
        s_sr: GroupedRun
            a dict-like object keyed on the provided attributes and
            mapping to unique shocRun instances
        out_sr
        """

        def str2tup(keys):
            if isinstance(keys, str):
                keys = keys,  # a tuple
            return keys

        self.logger.info(
                'Matching %s frames to %s frames by:\tExact %s;\t Closest %r',
                cal_run.kind.upper(), self.kind.upper(), exact, closest)

        # create the GroupedRun for science frame and calibration frames
        exact, closest = str2tup(exact), str2tup(closest)
        groupId = OrderedSet(filter(None, mit.flatten([exact, closest])))
        s_sr = self.group_by(*groupId)
        c_sr = cal_run.group_by(*groupId)

        # Do the matching - map the science frame attributes to the calibration
        # GroupedRun element with closest match
        # NOTE AT THE MOMENT THIS ONLY USES THE FIRST KEYWORD IN closest TO
        # DETERMINE THE CLOSEST MATCH
        lme = len(exact)
        # groupId key attributes of the sci_run
        sciAttrs = self.attrgetter(*groupId)
        calAttrs = cal_run.attrgetter(*groupId)
        # get set of science frame attributes
        sciAttrSet = np.array(list(set(sciAttrs)), object)
        calAttrs = np.array(list(set(calAttrs)), object)
        # get state array to indicate where in data threshold is exceeded (used
        # to colourise the table)
        sss = sciAttrSet.shape
        states = np.zeros((2 * sss[0], sss[1] + 1))

        #
        runmap, attmap = {}, {}
        datatable = []
        for i, attrs in enumerate(sciAttrSet):
            # those calib cubes with same attrs (that need closest matching)
            lx = np.all(calAttrs[:, :lme] == attrs[:lme], axis=1)
            delta = abs(calAttrs[:, lme] - attrs[lme])

            if ~lx.any():  # NO exact matches
                threshold_warn = False  # Don't try issue warnings below
                cattrs = (None,) * len(groupId)
                crun = None
            else:
                lc = (delta == delta[lx].min())
                l = lx & lc
                cattrs = tuple(calAttrs[l][0])
                crun = c_sr[cattrs]

            tattrs = tuple(attrs)  # array to tuple
            attmap[tattrs] = cattrs
            runmap[tattrs] = crun

            # FIXME:  tables with 'mode' not printing nicely
            datatable.append((str(s_sr[tattrs]),) + tattrs)
            datatable.append((str(crun),) + cattrs)

            # Threshold warnings
            # FIXME:  MAKE WARNINGS MORE READABLE
            if threshold_warn:
                # type cast the attribute for comparison (datetime.timedelta for date attribute, etc..)
                deltatype = type(delta[0])
                threshold = deltatype(threshold_warn)
                if np.any(delta[l] > deltatype(0)):
                    states[2 * i:2 * (i + 1), lme + 1] += 1

                # compare to threshold value
                if np.any(delta[l] > threshold):
                    fns = ' and '.join(c_sr[cattrs].get_filenames())
                    sci_fns = ' and '.join(s_sr[tattrs].get_filenames())
                    msg = ('Closest match of {} {} in {}\n'
                           '\tto {} in {}\n'
                           '\texcedees given threshold of {}!!\n\n'
                           ).format(tattrs[lme], closest[0].upper(), fns,
                                    cattrs[lme], sci_fns, threshold_warn)
                    warn(msg)
                    states[2 * i:2 * (i + 1), lme + 1] += 1

        out_sr = GroupedRun(runmap)
        # out_sr.label = cal_run.label
        out_sr.groupId = s_sr.groupId

        if print_:
            # Generate data table of matches
            col_head = ('Filename(s)',) + tuple(map(str.upper, groupId))
            where_row_borders = range(0, len(datatable) + 1, 2)

            table = sTable(datatable,
                           title='Matches',
                           title_props=dict(text='bold', bg='blue'),
                           col_headers=col_head,
                           where_row_borders=where_row_borders,
                           precision=3, minimalist=True,
                           width=range(140))

            # colourise   #TODO: highlight rows instead of colourise??
            unmatched = [None in row for row in datatable]
            unmatched = np.tile(unmatched, (len(groupId) + 1, 1)).T

            states[unmatched] = 3
            colours = ('default', 'yellow', 202, 'red')
            table.colourise(states, colours)

            self.logger.info('The following matches have been made:')
            print(table)

        return s_sr, out_sr

    def identify(self):
        """Split science and calibration frames"""
        from recipes.iter import itersubclasses
        from recipes.dict import AttrDict

        idd = AttrDict()
        sr = self.group_by('kind')
        clss = list(itersubclasses(shocRun))
        for kind, run in sr.items():
            for cls in clss:
                if cls.obsClass.kind == kind:
                    break
            idd[kind] = cls(run)
        return idd

    def coalign(self, align_on=0, first=10, flip=True, return_index=False,
                **findkws):
        """
        Search heuristic that finds the positional and rotational offset between
        partially overlapping images.

        Parameters
        ----------
        align_on
        first
        flip
        return_index
        findkws

        Returns
        -------

        """

        # TODO: eliminate flip arg - this means figure out why the flip state
        # is being recorded erroneously. OR inferring the flip state
        # bork if no overlap ?
        from pySHOC.wcs import MatchImages

        npar = 3
        n = len(self)
        P = np.zeros((n, npar))
        FoV = np.empty((n, 2))
        scales = np.empty((n, 2))
        I = []

        self.logger.info('Extracting median images (first %d) frames', first)
        for i, cube in enumerate(self):
            image = np.median(cube.data[:first], 0)

            for axis in range(2):
                if flip and cube.flip_state[axis]:
                    self.logger.info('Flipping image from %r in %s.',
                                     cube.get_filename(), 'YX'[axis])
                    image = np.flip(image, axis)

            I.append(image)
            FoV[i] = fov = cube.get_FoV()
            scales[i] = fov / image.shape

        # align on highest res image if not specified
        a = align_on
        if align_on is None:
            a = scales.argmin(0)[0]
        others = set(range(n)) - {a}

        self.logger.info('Aligning run of %i images on %r', len(self),
                         self[a].get_filename())
        matcher = MatchImages(I[a], FoV[a], **findkws)
        for i in others:
            # print(i)
            p = matcher.match_image(I[i], FoV[i])
            P[i] = p

        if return_index:
            return I, FoV, P, a
        return I, FoV, P

    def coalignDSS(self, align_on=0, first=10, **findkws):
        from pySHOC.wcs import MatchDSS

        sr, idict = self.group_by('telescope', return_index=True)

        I = np.empty(len(self), 'O')
        P = np.empty((len(self), 3))
        FoV = np.empty((len(self), 2))
        aligned_on = np.empty(len(sr), int)
        # ensure that P, FoV maintains the same order as self
        for i, (tel, run) in enumerate(sr.items()):
            indices = idict[tel]
            images, fovs, ps, ali = run.coalign(first=first, return_index=True,
                                                **findkws)
            aligned_on[i] = indices[ali]
            FoV[indices], P[indices] = fovs, ps
            I[indices] = images

        # pick the DSS FoV to be slightly larger than the largest image
        fovDSS = np.ceil(FoV.max(0))
        dss = MatchDSS(self[align_on].coords, fovDSS, **findkws)

        for i, tel in enumerate(sr.keys()):
            a = aligned_on[i]
            p = dss.match_image(I[a], FoV[a])
            P[idict[tel]] += p

        return dss, I, FoV, P, idict


################################################################################
class shocSciRun(shocRun):
    # name = 'science'
    nameFormat = '{basename}'
    displayColour = 'g'


class shocBiasRun(shocRun):
    # name = 'bias'
    obsClass = shocBiasObs
    nameFormat = '{kind}{date}{sep}{binning}' \
                 '<{sep}{mode}><{sep}t{kct}><{sep}sub{sub}>'
    # TODO: add filter to nameFormat
    displayColour = 'm'
    _default_combine_func = staticmethod(np.median)

    # _default_grouping = (binning, preAmpGain, outAmpMode, emGain, subframe)

    # NOTE:
    # Bias frames here are technically dark frames taken at minimum possible
    # exposure time. SHOC's dark current is (theoretically) constant with time,
    # so these may be used as bias frames.
    # Also, there is therefore no need to normalised these to the science frame
    # exposure times.

    def combined(self, names, func=None):
        # overwriting just to set the default combine function
        if func is None:
            func = self._default_combine_func

        # TODO: auto gen names if None
        return shocRun.combined(self, names, func)

    def compute_masters(self, func=None, path=None,
                        group_by=('mode', 'binning'), date_merge_thresh=1):
        """
        Compute master bias image(s) and its dispersion image(s) for this run.
        The run will first be grouped by (binning, mode, subframe) before
        computing the master bias for each group using the func method. While
        the mean pixel values may not change across different (preAmpGain,
        mode) settings, the dispersion image will be different since the
        underlying distribution is different in each case.

        Parameters
        ----------
        func: Callable
            The function used to combine the stack of frames
        path: str or pathlib.Path
            Directory to place resulting combined fits files
        group_by: sequence
            Sequence of attribute names, the values of which will determine
            the groups of files that will be combined into a single master bias
        date_merge_thresh: int
            Observations that are separated in time by less days than this
            number will be merged together. Useful, for example, if you have
            multiple observations with the same instrumental setup that you
            wish to combine into a single master bias image.  A value of 0
            will merge all observations with the same group id, while a value of
            say 7, will merge combine all with that group idea taken within
            the same week.

        Returns
        -------
        GroupedRun
        """

        if func is None:
            func = self._default_combine_func

        unique = self.filter_duplicates()
        naming = NamingConvention(self.nameFormat)
        masters = GroupedRun()
        for gid, run in unique.group_by(group_by).items():
            # one filename per structured group
            name, = naming.unique(run[:1], path)

            if len(run) > 1 and date_merge_thresh:
                raise NotImplementedError
                # for each grouping, combine runs if they are below
                # `date_merge_thresh`
                dates = np.array(run.attrgetter('date'))
                deltaMatrix = abs(dates - dates[None].T)
                b = deltaMatrix < type(deltaMatrix[0, 0])(date_merge_thresh)
                for l in b:
                    # fixme: name, group_id ???
                    master = run[l].merge_combine(name, func)

            master = run.merge_combine(name, func)
            masters[gid] = master
            # write to disc
            master.flush(output_verify='warn', verbose=1)

        # print(masters)
        return masters


# return compute_master(how_combine, masterbias)

# def masters(self, name, how_combine, masterbias=None):
#     """individually"""
#     return .compute_master(how_combine, masterbias)


class shocFlatFieldRun(shocBiasRun):
    # name = 'flat'
    obsClass = shocFlatFieldObs
    nameFormat = 'f{date}{sep}{binning}<{sep}sub{sub}><{sep}f{filter}>'
    displayColour = 'c'
    _default_combine_func = staticmethod(median_scaled_median)

    def compute_masters(self, func=None, biases=None, path=None,
                        group_by=('mode', 'binning'), date_merge_thresh=1):
        # todo: inherit + edit docstring

        biases


SHOCRunKinds = {'science': shocSciRun,
                'flat': shocFlatFieldRun,
                'bias': shocBiasRun}


################################################################################
class GroupedRun(OrderedDict, LoggingMixin):
    """
    Emulates dict to hold multiple shocRun instances keyed by their shared
    common attributes. The attribute names given in groupId are the ones by
    which the run is separated into unique segments (which are also shocRun
    instances). This class attempts to eliminate the tedium of computing
    calibration frames for different  observational setups by enabling loops
    over various such groupings.
    """

    # @property
    # def runClass(self):
    #     if len(self):
    #         return type(list(self.values())[0])
    # fixme: this limits us to structured runs of the same kind

    # @property
    # def name(self):
    #     return getattr(self.runClass, 'name', None)

    def __repr__(self):
        # FIXME: 'REWORK REPR: look at table printed in shocRun.match_and_group
        # embed()
        # self.values()
        return '\n'.join(['%s: %s' % x for x in self.items()])

    def flatten(self):
        """Flatten to shocRun"""
        cleaned = filter(None, self.values())
        if self.varies_by('kind'):
            first = shocRun()  # generic kind
        else:
            first = next(cleaned)  # specific kind

        # construct shocRun
        run = first.join(*cleaned, unique=True)
        return run

    def writeout(self, with_path=True, suffix='', dryrun=False,
                 header2txt=False):
        return self.flatten().writeout(with_path, suffix, dryrun, header2txt)

    def group_by(self, *keys):
        if self.groupId == keys:
            return self
        return self.flatten().group_by(*keys)

    def varies_by(self, key):
        """
        Check whether the attribute value mapped to by `key` varies across
        the set of observing runs

        Parameters
        ----------
        key

        Returns
        -------
        bool
        """
        attrValSet = {getattr(o, key) for o in filter(None, self.values())}
        return len(attrValSet) > 1

    # @profiler.histogram()
    @timer
    def combined(self, func=None, path=None):  # , write=True
        """

        Parameters
        ----------
        func: Callable
        path: str or pathlib.Path
            Directory to place resulting combined fits files

        Returns
        -------
        GroupedRun
        """

        # TODO:
        # variance of sample median (need to know the distribution for
        # direct estimate)
        # OR bootstrap to estimate
        # OR median absolute deviation

        self.logger.info('Combining:')

        # detect duplicates
        unique = {}
        values, keys = self.values(), self.keys()
        for gid, (runs, keys) in itr.groupmore(id, list(values), list(keys)):
            unique[runs[0]] = keys

        # since all cubes in each attribute group (run) will be combined
        # into a single frame, we need only use the first cube from each
        # group to determine the naming signature. The naming convention may
        # also be different for different kinds of data (bias/flats/sci)
        combined = {}
        urun = shocRun(next(zip(*filter(None, unique))))
        for kind, run in urun.identify().items():
            # naming convention may be  different for different kinds of data
            # so first split by kind
            naming = NamingConvention(run.nameFormat)
            for key, sub in run.group_by(*self.groupId).items():
                # one filename per structured group
                name, = naming.unique(sub[0], path)
                # combine entire run (potentially multiple cubes) for this group
                cmb = sub.merge_combine(name, func, axis=0)
                # Make it a shocRun instance.
                combined[key] = run.__class__([cmb])  # FIXME: kind of sucky

        # GroupedRun instances returned by shocRun.match_and_group may
        # contain None values as placeholders for unmatched groups. Propagate
        # these groups by filling None
        output = GroupedRun()
        for run, keys in unique.items():
            for key in keys:
                output[key] = combined.get(key)

        # NOTE:
        # The GroupedRun here is keyed on the closest matching attributes
        # in *self*. The values of the key may therefore not equal exactly the
        # properties of the corresponding cube (see match_and_group)
        output.groupId = self.groupId
        return output

    def compute_masters(self, how_combine, mbias=None, load=False, w2f=True,
                        outdir=None):
        """
        Compute the master image(s) (bias / flat field)

        Parameters
        ----------
        how_combine:
            function to use when combining the stack
        mbias:
            A GroupedRun instance of master biases (optional)
        load: bool
            If True, the master frames will be loaded as shocObs.
            If False kept as filenames

        Returns
        -------
        A GroupedRun of master frames separated by the same keys as self
        """

        if mbias:
            if self.groupId != mbias.groupId:
                raise ValueError('Master biases GroupedRun needs to be '
                                 'grouped by identical attributes to flat run.')

        keys = self.groupId
        masters = {}  # master bias filenames
        dataTable = []
        for attrs, run in self.items():
            if run is None:  # Unmatched!
                masters[attrs] = None
                continue

            # master bias / flat frame for this group
            name = run.magic_filenames()[0]
            master = run.compute_master(how_combine, name)
            # write full frame master
            master.flush(output_verify='warn', verbose=1)

            # writes subframed master
            # submasters = [master.subframe(sub) for sub in stack._needs_sub]

            # master.close()

            masters[attrs] = master
            dataTable.append((master.get_filename(0, 1),) + attrs)

        # Table for master frames
        # bgcolours = {'flat': 'cyan', 'bias': 'magenta', 'sci': 'green'}
        title = 'Master {} frames:'.format(self.name)
        title_props = {'text': 'bold', 'bg': self.runClass.displayColour}
        col_head = ('Filename',) + tuple(map(str.upper, keys))
        table = sTable(dataTable, title, title_props, col_headers=col_head)
        print(table)
        # TODO: STATISTICS????????

        if load:
            # this creates a run of all the master frames which will be split
            # into individual shocObs instances upon the creation of the
            # GroupedRun at return
            label = 'master {}'.format(self.name)
            mrun = self.runClass(hdus=masters.values())  # label=label

        if w2f:
            fn = label.replace(' ', '.')
            outname = os.path.join(outdir, fn)
            mrun.export_filenames(outname)

        # NOTE: dict here is keyed on the closest matching attributes in self!
        gr = GroupedRun(masters)
        gr.groupId = self.groupId
        # gr.label = self.label
        return gr

    def compute_masters(self, master_biases=None, func=None, path=None):
        """
        Compute the master calibration image(s) (bias / flat field)

        Parameters
        ----------
        func:
            function to use when combining each stack
        master_biases:
            A GroupedRun instance of master biases (optional)
        path:
            output directory


        Returns
        -------
        A GroupedRun of master frames separated by the same keys as self
        """

        if master_biases is not None:
            # if isinstance(master_biases, GroupedRun):
            if self.groupId != master_biases.groupId:
                raise ValueError(
                        'Master biases GroupedRun needs to be grouped by'
                        ' identical attributes to flat run.')
            # else:
            #     raise NotImplementedError

        keys = self.groupId
        masters = {}  # master bias filenames
        dataTable = []
        for attrs, run in self.items():
            if run is None:  # Unmatched!
                masters[attrs] = None
                continue

            bias = master_biases[attrs]
            if bias:
                run = run.debias(bias)

            # master bias / flat frame for this group
            master = run.merge_combine(func, path)

            # write full frame master
            master.flush(output_verify='warn', verbose=1)

            # writes subframed master
            # submasters = [master.subframe(sub) for sub in stack._needs_sub]

            # master.close()

            masters[attrs] = master
            dataTable.append((master.get_filename(0, 1),) + attrs)

        # Table for master frames
        # bgcolours = {'flat': 'cyan', 'bias': 'magenta', 'sci': 'green'}
        title = 'Master {} frames:'.format(self.name)
        title_props = {'text': 'bold', 'bg': self.runClass.displayColour}
        col_head = ('Filename',) + tuple(map(str.upper, keys))
        table = sTable(dataTable, title, title_props, col_headers=col_head)
        print(table)
        # TODO: STATISTICS????????

        if load:
            # this creates a run of all the master frames which will be split
            # into individual shocObs instances upon the creation of the
            # GroupedRun at return
            label = 'master {}'.format(self.name)
            mrun = self.runClass(hdus=masters.values())  # label=label

        if w2f:
            fn = label.replace(' ', '.')
            outname = os.path.join(outdir, fn)
            mrun.export_filenames(outname)

        # NOTE: dict here is keyed on the closest matching attributes in self!
        gr = GroupedRun(masters)
        gr.groupId = self.groupId
        # gr.label = self.label
        return gr

    def subframe(self, c_sr):
        # Subframe
        print('sublime subframe')
        # i=0
        substacks = []
        # embed()
        for attrs, r in self.items():
            # _, csub = c_sr[attrs].sub   #what are the existing dimensions for this binning, mode combo
            stack = c_sr[attrs]
            # _, ssub = r.zipper('sub')
            _, srect = r.zipper('subrect')
            # if i==0:
            # i=1
            missing_sub = set(srect) - set([(stack.subrect)])
            print(stack.get_filename(0, 1), r)
            print(stack.subrect, srect)
            print(missing_sub)
            for sub in missing_sub:
                # for stack in c_sr[attrs]:
                substacks.append(stack.subframe(sub))

        # embed()

        print('substacks', [s.sub for s in substacks])

        b = c_sr.flatten()
        print('RARARARARRAAAR!!!', b.zipper('sub'))

        newcals = self.runClass(substacks) + b  # , label=c_sr.label
        return newcals

    # TODO: combine into calibration method
    @timer
    def debias(self, master_bias_groups):  # FIXME: RENAME Dark
        """
        Do the bias reductions on science / flat field data

        Parameters
        ----------
        mbias_dict : Dictionary with binning,filename key-value pairs for master biases
        sb_dict : Dictionary with (binning, run) key-value pairs for science data

        Returns
        ------
        Bias subtracted shocRun
        """
        for attrs, master in master_bias_groups.items():
            if master is None:
                continue

            if isinstance(master, shocRun):
                master = master[0]  # HACK!!

            stacks = self[attrs]  # the cubes (as shocRun) for this attrs value

            msg = 'Doing bias subtraction on the stack: '
            lm = len(msg)
            self.logger.info(msg)
            for stack in stacks:
                self.logger.info(' ' * lm, stack.get_filename())

                header = stack.header
                # Adds the keyword 'BIASCORR' to the image header to indicate
                # that bias correction has been done
                # header['BIASCORR'] = (True, 'Bias corrected')
                # Add the filename and time of bias subtraction to header
                # HISTORY
                hist = 'Bias frame %r subtracted at %s' % \
                       (master.get_filename(), datetime.datetime.now())
                header.add_history(hist, before='HEAD')

                # TODO: multiprocessing here...??
                stack.data = stack.data - master.data
                # avoid augmented assign -= here due to potential numpy casting
                # error for different types

        self.label = self.name + ' (bias subtracted)'
        return self

    @timer
    def flatfield(self, mflat_dict):
        """
        Do the flat field reductions
        Parameters
        ----------
        mflat_dict : Dictionary with (binning,run) key-value pairs for master flat images

        Returns
        ------
        Flat fielded shocRun
        """

        for attrs, masterflat in mflat_dict.items():

            if isinstance(masterflat, shocRun):
                masterflat = masterflat[0]  # HACK!!

            mf_data = masterflat.data
            # pyfits.getdata(masterflat, memmap=True)

            if round(np.mean(mf_data), 1) != 1:
                raise ValueError('Flat field not normalised!!!!')

            stacks = self[attrs]  # the cubes for this binning value

            msg = '\nDoing flat field division on the stack: '
            lm = len(msg)
            self.logger.info(msg, )
            for stack in stacks:
                self.logger.info(' ' * lm + stack.get_filename())

                # Adds the keyword 'FLATCORR' to the image header to indicate
                # that flat field correction has been done
                header = stack.header
                header['FLATCORR'] = (True, 'Flat field corrected')
                # Adds the filename used and time of flat field correction to
                # header HISTORY
                hist = 'Flat field {} subtracted at {}'.format(
                        masterflat.get_filename(), datetime.datetime.now())
                header.add_history(hist, before='HEAD')

                # flat field division
                stack.data = stack.data / mf_data
                # avoid augmented assignment due to numpy type errors

        self.label = 'science frames (flat fielded)'
