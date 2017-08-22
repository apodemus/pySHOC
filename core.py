# __version__ = '2.12'

import os
import re
import time
import datetime
import logging
import warnings
import operator
# import collections as col
from copy import copy
import itertools as itt

# WARNING: THESE IMPORT ARE MEGA SLOW!! ~10s  (localize to mitigate)
import numpy as np
# import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
from astropy.io.fits.hdu import HDUList, PrimaryHDU

import recipes.iter as itr
from recipes.io import warn
from recipes.list import sorter, flatten
from recipes.set import OrderedSet
from recipes.string import rreplace
from ansi.table import Table as sTable


# TODO: choose which to use for timing: spice or astropy
# from .io import InputCallbackLoop
from .utils import retrieve_coords, convert_skycooords
from .timing import timingFactory, Time, get_updated_iers_table, fmt_hms
from .header import shocHeader
from .convert_keywords import KEYWORDS as kw_old_to_new

# debugging
from IPython import embed
from decor.profiler.timers import timer#, profiler

# def warn(message, category=None, stacklevel=1):
# return warnings.warn('\n'+message, category=None, stacklevel=1)


# FIXME: Many of the functions here have verbosity argument. replace with logging
# FIXME: Autodetect corrupt files?  eg.: single exposure files (new) sometimes don't contain KCT

# TODO
# __all__ = ['']

#TODO: can you pickle these classes
#

def median_scaled_median(data, axis):
    """"""
    frame_med = np.median(data, (1,2))[:, None, None]
    scaled = data / frame_med
    return np.median(scaled, axis)


def mad(data, data_median=None):
    if data_median is None:
        data_median = np.median(data, 0)
    return np.median(np.abs(data - data_median), 0)


def apply_stack(func, *args, **kws):
    # TODO:  MULTIPROCESS HERE!
    return func(*args, **kws)


################################################################################
# class definitions
################################################################################
class Date(datetime.date):
    """
    We need this so the datetime.date instances print in date format instead
    of the class representation format, when print is called on, for eg. a tuple
    containing a date_time object.
    """
    def __repr__(self):
        return str(self)


# HDU Subclasses
class shocHDU(PrimaryHDU):
    def __init__(self, data=None, header=None, do_not_scale_image_data=False,
                 ignore_blank=False, uint=True, scale_back=None):

        # convert to shocHeader
        header = shocHeader(header)

        super(PrimaryHDU, self).__init__(
            data=data, header=header,
            do_not_scale_image_data=do_not_scale_image_data,
            uint=uint,
            ignore_blank=ignore_blank,
            scale_back=scale_back)


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


# # TODO: Consider shocBiasHDU, shocFlatFieldHDU + match headers by looking at obstype keywords
#
# register_hdu(shocNewHDU)
# register_hdu(shocOldHDU)


# class ModeHelper(object):
#     def __init__(self, header):
#         # readout speed
#         speed = 1. / header['READTIME']
#         speedMHz = int(round(speed / 1.e6))
#
#         # CCD mode
#         self.preAmpGain = header['PREAMP']       # FIXME: use seperately.....
#         self.outAmpMode = header['OUTPTAMP']


class shocObs(HDUList):
    """
    Extend the hdu.hdulist.HDUList class to perform simple tasks on the image stacks.
    """
    kind = 'science'
    location = 'sutherland'
    # default attributes for __repr__ / get_instrumental_setup
    _pprint_attrs = ['binning', 'shape', 'preAmpGain', 'outAmpMode', 'emGain',
                     ]  # 'ron'
    _nullGainStr = '--'

    @classmethod
    def set_pprint(cls, attrs):
        # TODO: safety checks - need abc for this? or is that overkill?
        cls._pprint_attrs = attrs

    @classmethod
    def load(cls, fileobj, mode='update', memmap=False, save_backup=False, **kwargs):
        """
        Load shocObs from file
        """
        return cls._readfrom(fileobj=str(fileobj), mode=mode, memmap=memmap,
                             save_backup=save_backup, ignore_missing_end=True,
                             # do_not_scale_image_data=True,
                             **kwargs)


    def __init__(self, hdus=None, file=None):

        if hdus is None:
            hdus = []
        # initialize HDUList
        super().__init__(hdus, file)

        #
        self.path, self.basename = os.path.split(self.filename())
        if self.basename:
            self.basename = self.basename.replace('.fits', '')

        # if len(self):   # FIXME: either remove these lines from init or detect new file write. or ??
        # Load the important header information as attributes
        self.instrumental_setup()   # NOTE: fails on self.writeto
        # Initialise timing class
        timingClass = timingFactory(self)
        self.timing = timingClass(self)
        # NOTE: this is a hack that allows us to set the timing associated methods dynamically
        self.kct = self.timing.kct
        # NOTE: Set this attribute here so the cross matching algorithm works. inherit from the timing classes directly to avoid the previous line
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

    def __repr__(self):
        filename, dattrs, values = self.get_instrumental_setup()
        attrRep = map('%s = %s'.__mod__, zip(dattrs, values))
        clsname = self.__class__.__name__
        sep = '; '
        return '%s (%s): %s' % (clsname, filename, sep.join(attrRep))

    def _get_data(self):
        """retrieve PrimaryHDU data"""
        return self[0].data
        # TODO: intercept here with FitsCube for optimized read access ?
        # NOTE: we will always have only one cube in the HDUList, so this in unambiguous

    def _set_data(self, data):
        """set PrimaryHDU data"""
        self[0].data = data

    data = property(_get_data, _set_data)

    @property
    def header(self):
        """retrieve PrimaryHDU header"""
        return self[0].header

    @property
    def emGain(self):
        """
        the gain is sometime erroneously recorded in the header as having a nonzero value even though the
        pre-amp mode is CON. We wrap the attribute as a property to retrieve the correct value
        """
        if self.is_EM:
            return self._emGain
        else:
            return self._nullGainStr # for display

    # @property
    # def is_old():

    # @property
    # def ron(self):


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

    def instrumental_setup(self):
        """
        Retrieve the relevant information about the observational setup from header
        and set them as attributes.
        """
        # note: some of these should be properties

        header = self.header

        # instrument
        serno = header['SERNO']
        self.instrument = 'SHOC ' + str([5982, 6448].index(serno) + 1)
        # else: self.instrument = 'unknown!'
        self.telescope = header.get('telescop')

        # date from header
        date, time = header['DATE'].split('T')
        self.date = Date(*map(int, date.split('-'))) # oldSHOC: file creation date
        # starting date of the observing run --> used for naming
        h = int(time.split(':')[0])
        namedate = self.date - datetime.timedelta(int(h < 12))
        self.nameDate = str(namedate).replace('-', '')

        # image binning
        self.binning = tuple(header[s + 'BIN'] for s in 'HV')

        # image dimensions
        self.ndims = header['NAXIS']  # Number of image dimensions
        self.shape = *self.ishape, self.nframes = \
            tuple(header['NAXIS' + str(i)] for i in np.r_[:self.ndims] + 1)
        # Pixel dimensions for 2D images in ishape

        # sub-framing
        self.subrect = np.array(header['SUBRECT'].split(','), int)
        # subrect stores the sub-region of the full CCD array captured for this obs
        # xsub, ysub = (xb, xe), (yb, ye) = \
        xsub, ysub = self.subrect.reshape(-1, 2) // self.binning
        self.sub = np.array([xsub, ysub[::-1]]) # for some reason the ysub order is reversed
        self.subSlices = list(map(slice, *self.sub.T))

        # readout speed
        speed = 1. / header['READTIME']
        self.preAmpSpeed = speedMHz = int(round(speed / 1.e6))

        # CCD mode
        self.preAmpGain = header['PREAMP']          # TODO: self.mode.preamp.gain ??
        self.outAmpModeLong = header['OUTPTAMP']    # TODO: self.mode.outamp ??
        self.acqMode = header['ACQMODE']

        # gain
        self._emGain = header.get('gain', None)       # should be printed as '--' when mode is CON
        self.outAmpMode = 'CON' if self.outAmpModeLong.startswith('C') else 'EM'
        self.is_EM = (self.outAmpMode == 'EM')      # self.outamp.mode.is_EM()
        self.mode = '{} MHz, PreAmp @ {}, {}'.format(speedMHz, self.preAmpGain, self.outAmpMode) #TODO: str(self.mode)
        self.mode_trim = '{}MHz{}{}'.format(speedMHz, self.preAmpGain, self.outAmpMode) #TODO: self.mode.suffix

        # TODO
        # if self.is_EM and (self._emGain is not None):
        #     warn('Erroneous gain values')

        # Readout stats
        # set the correct values here as properties of the instance though not yet in header.
        self.ron, self.sensitive, self.saturate = header.get_readnoise()

        # orientation
        self.flip_state = self.flipy, self.flipx = \
            tuple(header['FLIP' + s] for s in 'YX') # NOTE: row, column order
        # WARNING: flip state wrong for old shoc data : TODO confirm this

        # Timing
        # self.kct = self.timing.kct
        # self.trigger_mode = self.timing.trigger.mode
        # self.duration

        # object name
        self.target = header.get('object')

        # coordinates
        self.coords = self.get_coords()
        # NOTE: should we try load the coordinates immediately with self.get_coords since they
        # may not be there yet, and the pipeline will update them?

        # header keywords that need to be updated
        # if self.trigger_mode.startswith('External') and is old:
        # 'KCT', 'EXPOSURE', 'GPSSTART'

        # self.kct = header.get('kct')   #will only work for internal triggering.
        # self.exp = header.get('exposure')

        # telescope
        # self.telescope = header['telescope']

    # @property
    # def mode(self):
    #     return
    #
    # TODO: property?????????        NOTE: consequences for head_proc
    def get_coords(self, verbose=False):
        header = self.header

        ra, dec = header.get('objra'), header.get('objdec')
        coords = convert_skycooords(ra, dec)
        if coords:
            return coords

        if self.target:
            # No / bad coordinates in header, but object name available - try resolve
            coords = retrieve_coords(self.target)

        if coords:
            return coords

        # No header coordinates / Name resolve failed / No object name available

        # LAST resort use TELRA, TELDEC!!!! (Will only work for new SHOC data!!)
        # NOTE: These will lead to slightly less accurate timing solutions (Quantify?)
        ra, dec = header.get('telra'), header.get('teldec')
        coords = convert_skycooords(ra, dec)

        # TODO: optionally query for named sources in this location
        if coords:
            warn('Using telesope pointing coordinates.')

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

    def get_instrumental_setup(self, attrs=None):
        # TODO: YOU CAN MAKE THIS __REPR__????????
        # TODO: units

        filename = self.get_filename() or '<Unsaved>'
        attrNames = self._pprint_attrs[:] if (attrs is None) else attrs
        attrDisplayNames = attrNames[:] #.upper()

        # get timing attributes if available
        timingAttrNames = ['trigger', 'kct', 'duration.hms']
        if hasattr(self, 'timing'):
            attrNames.extend(
                map('timing.%s'.__mod__, timingAttrNames))
            attrDisplayNames.extend(
                (t.split('.')[0] for t in timingAttrNames))


        attrVals = operator.attrgetter(*attrNames)(self)

        return filename, attrDisplayNames, attrVals


    def get_field_of_view(self, telescope=None, unit='arcmin', with_focal_reducer=False):
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
        cube.get_field_of_view(1)               # 1m telescope
        cube.get_field_of_view(1.9)
        cube.get_field_of_view(74)
        cube.get_field_of_view('74in')

        """
        # Field of view in arcmin
        fov74 = (1.29, 1.29);   fov74r = (2.79, 2.79)  # with focal reducer
        fov40 = (2.85, 2.85)
        #fov30 = (3.73, 3.73)

        # PS. welcome to the new millennium, we use the metric system now
        if telescope is None:
            telescope = self.header.get('telescop')

        telescope = str(telescope)
        telescope = telescope.rstrip('inm') # strip "units" in name
        if with_focal_reducer:
            fov = {'74': fov74r, '1.9': fov74r}.get(telescope)
        else:
            fov = {#'30': fov30, '0.75': fov30,
                   '40': fov40, '1.0': fov40, '1': fov40,
                   '74': fov74, '1.9': fov74}.get(telescope)
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

    def get_pixel_scale(self, telescope=None, unit='arcmin', with_focal_reducer=False):
        """pixel scale in `unit` per binned pixel"""
        return self.get_field_of_view(telescope, unit, with_focal_reducer) / self.ishape

    get_plate_scale = get_pixel_scale


    def cross_check(self, frame2, key, raise_error=0):
        """
        Check fits headers in this image agains frame2 for consistency of key attribute

        Parameters
        ----------
        key : The attribute to be checked (binning / instrument mode / dimensions / flip state)
        frame2 : shocObs Objects to check against

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
                logging.info('Flipping %r in %s.', self.get_filename(), yx)
                self.data = np.flip(self.data, axis + 1)
                header['FLIP%s' % yx] = int(not self.flip_state[axis])

        self.flip_state = tuple(header['FLIP%s' % s] for s in ['YX'])
        #FIXME: avoid this line by making flip_state a list

    @property
    def is_subframed(self):
        return np.any(self.sub[:, 1] != self.ishape)

    def subframe(self, subreg, write=True):

        cb, ce, rb, re = subreg
        logging.info('subframing %r to %s', self.filename(), [rb, re, cb, ce])

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

        # "Median combining can completely remove cosmic ray hits and radioactive decay trails
        # from the result, which cannot be done by standard mean combining. However, median
        # combining achieves an ultimate signal to noise ratio about 80% that of mean combining
        # the same number of frames. The difference in signal to noise ratio can be compensated
        # by median combining 57% more frames than if mean combining were used. In addition,
        # variants on mean combining, such as sigma clipping, can remove deviant pixels while
        # improving the S/N somewhere between that of median combining and ordinary mean
        # combining. In a nutshell, if all images are "clean", use mean combining. If the
        # images have mild to severe contamination by radiation events such as cosmic rays,
        # use the median or sigma clipping method." - Newberry

        # mean / median across images
        imnr = '001'  # FIXME:   #THIS WILL NEED TO CHANGE FOR MULTIPLE SINGLE IMAGES AS INPUT
        header = copy(self.header)
        data = apply_stack(func, self.data, axis=0)

        ncomb = header.pop('NUMKIN', 0)  # Delete the NUMKIN header keyword
        # if 'naxis3' in header:          header.remove('NAXIS3')
        # header['NAXIS'] = 2     # NOTE: pyfits actually does this automatically...
        header['NCOMBINE'] = (ncomb, 'Number of images combined')
        header['ICMB' + imnr] = (self.filename(), 'Contributors to combined output image')
        # FIXME: THIS WILL NEED TO CHANGE FOR MULTIPLE SINGLE IMAGES AS INPUT

        # Load the stack as a shocObs
        if name is None:
            name = next(self.filename_gen())  # generate the filename

        hdu = shocHDU(data, header)
        fileobj = pyfits.file._File(name, mode='ostream', overwrite=True)
        stack = self.__class__(hdu, fileobj)  # initialise the Cube with target file
        stack.instrumental_setup()

        return stack

    def unpack(self, count=1, padw=None, dryrun=0, w2f=1):  # MULTIPROCESSING!!!!!!!!!!!!
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
        naxis3 = self.shape[-1]
        self.filename_gen.padwidth = padw if padw else len(str(naxis3))  # number of digits in filename number string
        self.filename_gen.count = count

        if not dryrun:
            # edit header
            header.remove('NUMKIN')
            header.remove('NAXIS3')  # Delete this keyword so it does not propagate into the headers of the split files
            header['NAXIS'] = 2  # Number of axes becomes 2
            header.add_history('Split from %s' % stack)

            # open the txt list for writing
            if w2f:
                basename = self.get_filename(1, 0)
                self.unpacked = basename + '.split'
                fp = open(self.unpacked, 'w')

            logging.info('Unpacking the stack %s of %i images.', stack, naxis3)

            # split the cube
            filenames = self.filename_gen(naxis3 + count - 1)
            bar.create(naxis3)
            for j, im, fn in zip(range(naxis3), self.data, filenames):
                bar.progress(count - 1)  # count instead of j in case of sequential numbering for multiple cubes

                self.timing.stamp(j)  # set the timing values in the header for frame j

                pyfits.writeto(fn, data=im, header=header, overwrite=True)  # MULTIPROCESSING!!!!!!!!!!!!

                if w2f:
                    fp.write(fn + '\n')  # OR outname???????????
                count += 1

            if w2f:
                fp.close()

            # how long did the unpacking take
            end_time = time.time()
            logging.debug('Time taken: %f', (end_time - start_time))

        self._is_unpacked = True

        return count


    def get_name_dict(self):

        # get nice representation of Kinetic Cycle Time
        kct = self.header.get('kct', 0)
        if int(kct / 10):
            kct = str(round(kct)) + 's'
        else:
            kct = str(round(kct * 1000)) + 'ms'

        return dict(sep='.',
                    obj=self.header.get('OBJECT', ''),
                    filter=self.header.get('FILTER', 'WL'),
                    basename=self.get_filename(0, 0),
                    date=self.nameDate,
                    binning='{}x{}'.format(*self.binning),
                    mode=self.mode_trim,
                    kct=kct)

    # def writeto(self, fileobj, output_verify='exception', overwrite=False, checksum=False):
    #     self._in_write = True


    def plot(self, **kws):
        """Display the data"""
        if self.ndims == 2:
            from grafico.imagine import ImageDisplay
            im = ImageDisplay(self.data, **kws)

        elif self.ndims == 3:
            from grafico.imagine import ImageCubeDisplay
            im = ImageCubeDisplay(self.data, **kws)

        else:
            raise TypeError('Not an image!! WTF?')

        im.figure.canvas.set_window_title(self.get_filename())
        return im


    # def animate(self):
    #   TODO: OR incorporate in ImageCubeDisplay

################################################################################
class shocBiasObs(shocObs): #FIXME: DARK?
    kind = 'bias'

    def get_coords(self):
        return

    def compute_master(self, func=np.median, masterbias=None, name=None, verbose=False):
        return self.combine(func, name)


class shocFlatFieldObs(shocBiasObs):
    kind = 'flat'

    def compute_master(self, func=median_scaled_median, masterbias=None, name=None, verbose=False):
        """ """
        master = self.combine(func, name)
        if masterbias:
            if verbose:
                logging.info(
                    'Subtracting master bias %r from master flat %r.',
                    masterbias.get_filename(), master.basename)
            # match orientation
            masterbias.flip(master.flip_state)
            master.data -= masterbias.data
        elif verbose:
            logging.info('No master bias for %r', self.filename())

        # flat field normalization
        ffmean = np.mean(master.data)  # flat field mean
        if verbose:
            logging.info('Normalising flat field...')
        master.data /= ffmean
        return master



class ClassProperty(property):
    def __get__(self, cls, owner):
        return self.fget.__get__(None, owner)()





################################################################################
class shocRun(object):
    # TODO: merge method?
    """
    Class containing methods to operate with sets of shocObs objects.
    perform comparitive tests between cubes to see if they are compatable.
    """
    obsClass = shocObs
    nameFormat = '{basename}'       # Naming convention defaults
    displayColour = 'g'
    _compactRepr = True
    # compactify the table representation form the `print_instrumental_setup` method by
    # removing columns (attributos) that are equal accross all constituent cubes and printing
    # them as a top row in the table
    _skip_init = False
    # this is here so we can initialize new instances from existing instances


    @ClassProperty      # so we can access via  shocRun.kind and shocRun().kind
    @classmethod
    def kind(cls):
        return cls.obsClass.kind


    @classmethod
    def load(cls, filenames, kind='science', mode='update', memmap=False, save_backup=False, **kwargs):
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
        logging.info('Loading data for %s run: %s', kind, label or '')

        # This method acts as a factory to create the underlying cube class
        kind = kind.lower()
        if kind.startswith('sci'):
            obsCls = shocObs
        elif kind.startswith('flat'):
            obsCls = shocFlatFieldObs
        elif kind.startswith('bias'):
            obsCls = shocBiasObs
        else:
            logging.warning('Invalid kind')
            obsCls = shocObs

        # sanitize filenames
        # filenames may contain None - these will be filtered
        filenames = list(filter(None, filenames)) if filenames else []
        # TODO: deal with duplicate filenames.....not desirable wrt efficiency.....
        hdus = []
        for i, fileobj in enumerate(filenames):
            # try:
            hdu = obsCls.load(fileobj, mode=mode, memmap=memmap,
                               save_backup=save_backup,
                               **kwargs)
            hdus.append(hdu)
            # except Exception as err:
            #     import traceback
            #     warn('File: %s failed to load with exception:\n%s' % (fileobj, str(err)))

        return cls(hdus, label)


    def __init__(self, hdus=None, label=None, groupId=None):
        # wrap objects in array to ease item getting

        if hdus is None:
            hdus = []

        for hdu in hdus:
            if not isinstance(hdu, HDUList):
                raise TypeError(
                    'Cannot initialize from %r. Please use `shocRun.load(filename)`' % wrongclass[0])

        self.cubes = np.empty(len(hdus), dtype='O')      #, dtype='O')
        self.cubes[:] = hdus
        self.groupId = OrderedSet(groupId)
        self.label = label


    def __len__(self):
        return len(self.cubes)

    def __repr__(self):
        clsname = self.__class__.__name__
        files = ' | '.join(self.get_filenames())
        return ' : '.join((clsname, files))

    def __getitem__(self, key):
        # NOTE: if you can wrap the cubes in an object array you will get
        # all the array indexing and slicing niceties for free, however,
        # since the cubes are already a container class HDUList (although
        # they really don't need to be such!!), the resultant array is 2D of
        # type shocHDU. :(
        # items = self.cubes[key]
        # if isinstance(key, (int, np.integer)):
        #     if key >= len(self):
        #         raise IndexError("The index (%d) is out of range." % key)
        #     return self.cubes[key]
        #
        # if isinstance(key, slice):
        #     items = self.cubes[key]

        # if isinstance(key, tuple):
        #     assert len(key) == 1
        #     key = key[0]  # this should be an array...

        # elif isinstance(key, (list, np.ndarray)):
        #     if isinstance(key[0], (bool, np.bool_)):
        #         assert len(key) == len(self)
        #         items = [self.cubes[i] for i in np.where(key)[0]]
        #
        #     elif isinstance(key[0], (int, np.integer)):  # NOTE: be careful bool isa int
        #         items = [self.cubes[i] for i in key]

        items = self.cubes[key]
        if np.size(items) > 1:
            return self.__class__(items, self.label, self.groupId)
        return items


    def __add__(self, other):
        if self.kind != other.kind:
            logging.warning('Jointing Runs of different kinds')
        if self.label != other.label:
            logging.info('Suppressing label %s', other.label)

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

    def pop(self, i):  # TODO: OR SUBCLASS LIST?
        return np.delete(self.cubes, i)

    def join(self, *runs):
        # Filter empty runs (None)
        runs = list(filter(None, runs))
        return sum(runs, self)

        # for run in runs:
            # kinds = [run.kind for run in runs]
            # labels = [run.label for run in runs]
            # self.cubes, [r.cubes for r in runs]
            # hdus = sum([r.cubes for r in runs], self.cubes)

        # if np.all(self.label == np.array(labels)):
        #     label = self.label
        # else:
        #     logging.warning("Labels %s don't match %r!", str(labels), self.label)
        #     label = None

        # return self.__class__(hdus, label=label)

    def print_instrumental_setup(self, attrs=None, description=''):
        """Print the instrumental setup for this run as a table.
        :param attrs:
        """
        filenames, attrDisplayNames, attrVals = zip(*(stack.get_instrumental_setup(attrs)
                                                      for stack in self))
        attrDisplayNames = attrDisplayNames[0] # all are the same
        name = self.label or ''# ''<unlabeled>'      # todo: default?
        title = 'Instrumental Setup: %s %s frames %s' \
                % (str(name).title(), self.kind.title(), description)

        table = sTable(attrVals,
                       title=title,
                       title_props=dict(text='bold', bg=self.displayColour),
                       col_headers=attrDisplayNames,
                       row_headers=['filename'] + list(filenames),
                       number_rows=True,
                       precision=5, minimalist=True, compact=True
                       )

        # try:
        print(table)
        # except:
        #     embed()
        #     raise
        return table

    pprint = print_instrumental_setup

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Timing
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def set_times(self, coords=None):

        # initialize timing
        t0s = []
        for stack in self:
            # stack.timing(self.location)
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

        msg = '\n\nCalculating timing arrays for datacube(s):'
        lm = len(msg)
        logging.info(msg )
        for i, stack in enumerate(self):
            logging.info(' ' * lm + stack.get_filename())
            t0 = t0s[i] # t0 = stack.timing.t0mid
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
        Generator that yields the cubes in the run that require GPS triggers to be set
        """
        for stack in self:
            # FIXME: value of date-obs?
            yield (stack.needs_timing and stack.timing.trigger.is_gps())

    def that_need_triggers(self):
        """Return a shocRun object containing only those cubes that are missing triggers"""
        return self[list(self.gen_need_triggers())]

    def set_gps_triggers(self, times, triggers_in_local_time=True):
        # trigger provided by user at terminal through -g  (args.gps)
        # if len(times)!=len(self):
        # check if single trigger OK (we can infer the remaining ones)
        if self.check_rollover_state():
            times = self.get_rolled_triggers(times)
            logging.info(
                '\nA single GPS trigger was provided. Run contains auto-split '
                'cubes (filesystem rollover due to 2Gb threshold on old windows server).'
                ' Start time for rolled over cubes will be inferred from the length'
                ' of the preceding cube(s).\n'
            )

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
        slints = [cube.shape[-1] for cube in self]  # stack lengths
        # sorts the file sequence in the correct order
        # re pattern to find the roll-over number (auto_split counter value in filename)
        matcher = re.compile('_X([0-9]+)')
        fns, slints, idx = sorter(self.get_filenames(), slints, range(len(self)),
                                  key=matcher.findall)

        print('get_rolled_triggers', 'WORK NEEDED HERE!')
        embed()
        # WARNING: This assumes that the run only contains cubes from the run that rolled-over.
        #         This should be ok for present purposes but might not always be the case
        idx0 = idx[0]
        self[idx0].timing.trigger.start = first_trigger_time
        t0, td_kct = self[idx0].time_init(dryrun=1)
        # dryrun ==> don't update the headers just yet (otherwise it will be done twice!)

        d = np.roll(np.cumsum(slints), 1)
        d[0] = 0
        t0s = t0 + d * td_kct
        triggers = [t0.isot.split('T')[1] for t0 in t0s]

        # resort the triggers to the order of the original file sequence
        # _, triggers = sorter( idx, triggers )

        return triggers

    def export_headers(self):
        """save fits headers as a text file"""
        for stack in self:
            headfile = stack.get_filename(with_path=1, with_ext=0, suffix='.head')
            logging.info('Writing header to file: %r', os.path.basename(headfile))
            stack.header.totextfile(headfile, overwrite=True)

    def get_filenames(self, with_path=False, with_ext=True, suffix=(), sep='.'):
        """filenames of run constituents"""
        return [stack.get_filename(with_path, with_ext, suffix, sep) for stack in self]

    def export_filenames(self, fn):

        if not fn.endswith('.txt'):  # default append '.txt' to filename
            fn += '.txt'

        logging.info('Writing names of %s to file %r', self.name, fn)
        with open(fn, 'w') as fp:
            for f in self.filenames:
                fp.write(f + '\n')

    def writeout(self, with_path=False, suffix='', dryrun=False, header2txt=False):  # TODO:  INCORPORATE FILENAME GENERATOR
        fns = []
        for stack in self:
            fn_out = stack.get_filename(with_path, False, (suffix, 'fits'))  # FILENAME GENERATOR????
            fns.append(fn_out)

            if not dryrun:
                logging.info('Writing to file: %r', os.path.basename(fn_out))
                stack.writeto(fn_out, output_verify='wbyarn', overwrite=True)

                if header2txt:
                    # save the header as a text file
                    headfile = stack.get_filename(1, 0, (suffix, 'head'))
                    logging.info('Writing header to file: %r', os.path.basename(headfile))
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
                    list(zip(*([getattr(s, key) for s in self] for key in keys))))

    def group_iter(self):
        'todo'

    def group_by(self, *keys, **kws):
        """
        Separate a run according to the attribute given in keys.
        keys can be a tuple of attributes (str), in which case it will seperate into runs with a unique combination
        of these attributes.

        optional keyword: return_index

        Returns
        -------
        atdict : dictionary containing attrs, run pairs where attrs are the attributes of run given by keys
        flag :  1 if attrs different for any cube in the run, 0 all have the same attrs
        """
        attrs = self.attrgetter(*keys)
        keys = OrderedSet(keys)
        return_index = kws.get('return_index', False)
        if self.groupId == keys:  # is already separated by this key
            SR = StructuredRun(zip([attrs[0]], [self]))
            SR.groupId = keys
            # SR.name = self.name
            if return_index:
                return SR, dict(attrs[0], list(range(len(self))))
            return SR

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
                # map unique attribute values to shocObs (indices) with those attributes
                l = np.array([attr == ats for attr in attrs])  # list for-loop needed for tuple attrs
                eq_run = self[l]  # shocRun object of images with equal key attribute
                eq_run.groupId = keys
                atdict[ats] = eq_run   # put into dictionary
                idict[ats], = np.where(l)

        SR = StructuredRun(atdict)
        SR.groupId = keys
        # SR.name = self.name
        if  return_index:
            return SR, idict
        return SR

    def varies_by(self, *keys):
        """False if the run is homogeneous by keys and True otherwise"""
        attrs = self.attrgetter(keys)
        atset = set(attrs)
        return (len(atset) != 1)

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


    def sort_by(self, *keys, **kws):
        """
        Sort the cubes by the value of attributes given in keys,
        kws can be (attribute, callable) pairs in which case sorting will be done according to value
        returned by callable on a given attribute.
        """
        # FIXME: order of kws lost when passing as dict. not desirable.
        # NOTE: this is not a problem in python >3.5!! yay! https://docs.python.org/3/whatsnew/3.6.html
        # for keys, func in kws.items():
        #     if

        def trivial(x):
            return x

        # compose sorting function
        triv = (trivial,) * len(keys)  # will be used to sort by the actual values of attributes in *keys*
        kwkeys = tuple(kws.keys())
        kwattrs = self.attrgetter(kwkeys)
        kwfuncs = tuple(kws.values())       #
        funcs = triv + kwfuncs              # tuple of functions, one for each requested attribute
        # combine all functions into single function that returns tuple that determines sort position
        attrSortFunc = lambda *x: tuple(f(z) for (f, z) in zip(funcs, x))

        attrs = self.attrgetter(keys)
        keys += kwkeys
        attrs += kwattrs

        ix = range(len(self))
        *attrs, ix = sorter(attrs, ix, key=attrSortFunc)

        return self[ix]

    def combined(self, names, how_combine):
        #     """return run"""
        assert len(names) == len(self)
        cmb = []
        for stack, name in zip(self, names):
            cmb.append(stack.combine(how_combine, name))
        return self.__class__(cmb)

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
            msg = ('You have input single image(s) named: {}. These will be used as the master {}'
                   ' frames').format(self[is_single], self.name)
            logging.info(msg)
        return is_single


    def stack(self, name):

        header = copy(self[0].header)
        data = self.stack_data()

        # create a shocObs for the stacked data
        hdu = shocHDU(data, header)
        fileobj = pyfits.file._File(name, mode='ostream', overwrite=True)
        cube = self.__class__(hdu, fileobj)  # initialise the Cube with target file
        cube.instrumental_setup()


    def stack_data(self):
        """return array of stacked data"""
        # if len(self) == 1:
        #     return

        dims = self.attrgetter('ndims')
        if not np.equal(dims, dims[0]).all():
            raise ValueError('Cannot stack cubes with differing frame sizes: %s' %str(dims))

        return np.vstack(self.attrgetter('data'))

    def combine_all(self, name, func, *args, **kws):
        """
        Mean / Median combines all of the stacks in the run

        Parameters
        ----------
        name:   filename of the output fits file
        func:   function used to combine
        args:   extra arguments to *func*
        kws:    extra keyword arguments to func

        Returns
        ------
        shocObs instance
        """
        # verbose = True
        # if verbose:
        #     self._print_combine_map(func.__name__, name)

        data = apply_stack(func, self.stack_data(), *args, **kws)

        # update header     # TODO: check which header entries are different
        header = copy(self[0].header)
        ncomb = sum(next(zip(*self.attrgetter('shape'))))
        header['NCOMBINE'] = (ncomb, 'Number of images combined')
        header['FCOMBINE'] = (func.__name__, 'Function used to combine the data')
        for i, fn in enumerate(self.get_filenames()):
            header['ICMB{:0{}d}'.format(i, 3)] = (fn, 'Contributors to combined output image')

        # create a shocObs
        hdu = shocHDU(data, header)
        fileobj = pyfits.file._File(name, mode='ostream', overwrite=True)
        cube = self.obsClass(hdu, fileobj)  # initialise the Cube with target file
        cube.instrumental_setup()

        return cube

    def _print_combine_map(self, fname, out):
        #                         median_scaled_median
        # SHA_20170209.0002.fits ----------------------> f20170209.4x4.fits
        s = str(self)
        a = ' '.join([' ' * (len(s) + 1), fname, ' ' * (len(out) + 2)])
        b = ' '.join([s, '-' * (len(fname) + 2) + '>', out])
        print('\n'.join([a, b]))

    def unpack(self, sequential=0, dryrun=0, w2f=1):
        # unpack datacube(s) and assign 'outname' to output images
        # if more than one stack is given 'outname' is appended with 'n_' where n is the number of the stack (in sequence)

        if not dryrun:
            outpath = self[0].filename_gen.path
            logging.info('The following image stack(s) will be unpacked into %r:\n%s',
                         outpath, '\n'.join(self.get_filenames()))

        count = 1
        naxis3 = [stack.shape[-1] for stack in self]
        tot_num = sum(naxis3)
        padw = len(str(tot_num)) if sequential else None
        # if padw is None, unpack_stack will determine the appropriate pad width for the cube

        if dryrun:
            # check whether the numer of images in the timing stack are equal to the total number of frames in the cubes.
            if len(args.timing) != tot_num:  # WARNING - NO LONGER VALID
                raise ValueError(
                    'Number of images in timing list ({}) not equal to total number in given '
                    'stacks ({}).'.format(len(args.timing), tot_num))

        for i, stack in enumerate(self):
            count = stack.unpack(count, padw, dryrun, w2f)

            if not sequential:
                count = 1

        if not dryrun:
            logging.info('A total of %i images where unpacked', tot_num)
            if w2f:
                catfiles([stack.unpacked for stack in self],
                         'all.split')  # RENAME??????????????????????????????????????????????????????????????????????????????????????????????

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
            # use this to select the minimum set of cubes needed (filter out mismatched cubes)

        # some attribute mismatch occurs
        # which of 2 occur in 1     # (at this point we know that one of these values are False)
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
            operation = ('de-biasing' if 'bias' in self.name else 'flat fielding')
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

    #TODO: as a mixin?
    def match_and_group(self, cal_run, exact, closest=None, threshold_warn=7, _pr=1):
        """
        Match the attributes between sci_run and cal_run.
        Matches exactly to the attributes given in exact, and as closely as possible to the
        attributes in closest. Separates sci_run by the attributes in both exact and
        closest, and builds an index dictionary for the cal_run which can later be used to
        generate a StructuredRun instance.

        Parameters
        ----------
        sci_run     :   The shocRun to which comparison will be done
        cal_run   :   shocRun which will be trimmed by matching
        exact       :   tuple or str. keywords to match exactly
                        NOTE: No checks run to ensure cal_run forms a subset of
                        sci_run w.r.t. these attributes
        closest     :   tuple or str. keywords to match as closely as possible

        Returns
        ------
        s_sr        :   StructuredRun of science frames separated
        out_sr
        """

        def str2tup(keys):
            if isinstance(keys, str):
                keys = keys,  # a tuple
            return keys

        logging.info('Matching %s frames to %s frames by:\tExact %s;\t Closest %r',
                     cal_run.name.upper(), self.name.upper(), exact, closest)

        # create the StructuredRun for science frame and calibration frames
        exact, closest = str2tup(exact), str2tup(closest)
        groupId = OrderedSet(filter(None, flatten([exact, closest])))
        s_sr = self.group_by(*groupId)
        c_sr = cal_run.group_by(*groupId)

        # Do the matching - map the science frame attributes to the calibration
        # StructuredRun element with closest match
        # NOTE AT THE MOMENT THIS ONLY USES THE FIRST KEYWORD IN closest TO DETERMINE
        # THE CLOSEST MATCH
        lme = len(exact)
        sciAttrs = self.attrgetter(groupId)  # groupId key attributes of the sci_run
        calAttrs = cal_run.attrgetter(groupId)
        sciAttrSet = np.array(list(set(sciAttrs)), object)  # a set of the science frame attributes
        calAttrs = np.array(list(set(calAttrs)), object)
        sss = sciAttrSet.shape
        where_thresh = np.zeros((2 * sss[0], sss[1] + 1))
        # state array to indicate where in data threshold is exceeded (used to colourise the table)

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

            datatable.append((str(s_sr[tattrs]),) + tattrs)
            datatable.append((str(crun),) + cattrs)

            # Threshold warnings
            # FIXME:  MAKE WARNINGS MORE READABLE
            if threshold_warn:
                # type cast the attribute for comparison (datetime.timedelta for date attribute, etc..)
                deltatype = type(delta[0])
                threshold = deltatype(threshold_warn)
                if np.any(delta[l] > deltatype(0)):
                    where_thresh[2 * i:2 * (i + 1), lme + 1] += 1

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
                    where_thresh[2 * i:2 * (i + 1), lme + 1] += 1

        out_sr = StructuredRun(runmap)
        # out_sr.label = cal_run.label
        out_sr.groupId = s_sr.groupId

        if _pr:
            try:
                # Generate data table of matches
                col_head = ('Filename(s)',) + tuple(map(str.upper, groupId))
                where_row_borders = range(0, len(datatable) + 1, 2)

                table = sTable(datatable,
                               title='Matches',
                               title_props=dict(text='bold', bg='light blue'),
                               col_headers=col_head,
                               where_row_borders=where_row_borders,
                               precision=3, minimalist=True)

                # colourise           #TODO: highlight rows instead of colourise??
                unmatched = [None in row for row in datatable]
                unmatched = np.tile(unmatched, (len(groupId) + 1, 1)).T
                states = where_thresh
                states[unmatched] = 3
                table.colourise(states, 'default', 'yellow', 202, {'bg': 'red'})

                logging.info('The following matches have been made:')
                print(table)
            except:
                print('TABLE FAIL! '*5)
                embed()

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


    def coalign(self, align_on=0, first=10, flip=True, return_index=False, **findkws):
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

        logging.info('Extracting median images (first %d) frames', first)
        for i, cube in enumerate(self):
            image = np.median(cube.data[:first], 0)

            for axis in range(2):
                if flip and cube.flip_state[axis]:
                    logging.info('Flipping image from %r in %s.',
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

        logging.info('Aligning run of %i images on %r', len(self), self[a].get_filename())
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
            images, fovs, ps, ali = run.coalign(first=first, return_index=True, **findkws)
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
    #name = 'science'
    nameFormat = '{basename}'
    displayColour = 'g'



class shocBiasRun(shocRun):
    # name = 'bias'
    obsClass = shocBiasObs
    nameFormat = 'b{date}{sep}{binning}[{sep}m{mode}][{sep}t{kct}][sub{sub}]'
    displayColour = 'm'
    # NOTE: Bias frames here are technically dark frames taken at minimum possible
    # exposure time.  SHOC's dark current is (theoretically) constant with time,
    # so these may be used as bias frames.
    # Also, there is therefore no need to normalised these to the science frame
    # exposure times.

    def combined(self, names, how_combine=np.median):
        # overwriting just to set the default function
        return shocRun.combined(self, names, how_combine)


    #     return compute_master(how_combine, masterbias)

    # def masters(self, name, how_combine, masterbias=None):
    #     """individually"""
    #     return .compute_master(how_combine, masterbias)


class shocFlatFieldRun(shocRun):
    # name = 'flat'
    obsClass = shocFlatFieldObs
    nameFormat = 'f{date}{sep}{binning}[{sep}sub{sub}][{sep}fltr{filter}]'
    displayColour = 'c'

    def combined(self, names, how_combine=median_scaled_median):
        # overwriting just to set the default function
        return shocRun.combined(self, names, how_combine)

    # def combine_data(self, names, how_combine=median_scaled_median): # overwriting just to set the default function
    #     return shocRun.combine_data(self, names, how_combine)

    # def masters(self, names, how_combine=np.median, masterbiases=None):
    #     """return run"""
    #     assert len(names) == len(self)
    #     masters = []
    #     for stack, name in zip(self, names, masterbiases):
    #         masters.append(stack.compute_master(how_combine, name))
    #     return self.__class__(masters)

    # def compute_master(self, name, how_combine, masterbias=None):
    #     self.stack(name).compute_master(how_combine, masterbias)
    #     return cmb.compute_master()

from collections import OrderedDict
################################################################################
class StructuredRun(OrderedDict): # TODO: Maybe rename as GroupedRun??
    """
    Emulates dict to hold multiple shocRun instances keyed by their shared common attributes.
    The attribute names given in groupId are the ones by which the run is separated
    into unique segments (which are also shocRun instances).
    This class attempts to eliminate the tedium of computing calibration frames for different
    observational setups by enabling loops over various such groupings.
    """
    @property
    def runClass(self):
        if len(self):
            return type(list(self.values())[0])

    @property
    def name(self):
        return getattr(self.runClass, 'name', None)

    def __repr__(self):
        # FIXME: 'REWORK REPR: look at table printed in shocRun.match_and_group
        # embed()
        # self.values()
        return '\n'.join([' : '.join(map(str, x)) for x in self.items()])

    def flatten(self):
        if isinstance(list(self.values())[0], shocObs):
            run = self.runClass(list(self.values()), groupId=self.groupId)
        else:
            run = self.runClass().join(*self.values())

        # eliminate duplicates
        _, idx = np.unique(run.get_filenames(), return_index=1)
        dup = np.setdiff1d(range(len(run)), idx)
        for i in reversed(dup):
            run.pop(i)

        return run

    def writeout(self, with_path=True, suffix='', dryrun=False, header2txt=False):
        return self.flatten().writeout(with_path, suffix, dryrun, header2txt)

    def group_by(self, *keys):
        if self.groupId == keys:
            return self
        return self.flatten().group_by(*keys)

    def magic_filenames(self, path='', sep='.', extension='.fits'):
        return self.flatten().magic_filenames(path, sep, extension)

    # TODO: maybe make specific to calibrationRun
    # @profiler.histogram()
    @timer
    def combined(self, func): #, write=True
        verbose = True

        #TODO: split duplicate_aware_compute

        #TODO: variance of sample median (need to know the distribution for direct estimate)
        # OR bootstrap to estimate
        # OR median absolute deviation

        if verbose:
            logging.info('\nCombining:')

        # detect duplicates
        dup = {}
        for gid, (runs, keys) in itr.groupmore(id, self.values(), self.keys()):
            dup[runs[0]] = keys

        combined = {}
        for run, keys in dup.items():
            if run is None:  # Unmatched!
                for attr in keys:
                    combined[attr] = None
                continue

            name = run.magic_filenames()[0]
            if verbose:
                run._print_combine_map(func.__name__, name)

            # FIXME: will have to implement generating filenames like: 'f2017020[89].4x4' or something
            # combine entire run (potentially multiplo cubes) for this group
            cmb = run.combine_all(name, func, axis=0)
            #cmb.flush(output_verify='warn', verbose=1)  # writes to file
            cmbrun = run.__class__([cmb])  # Make it a run.  FIXME: kind of sucky
            for attr in keys:
                combined[attr] = cmbrun

        # NOTE: The dict here is keyed on the closest matching attributes in *self*
        # NOTE  The values of the key will thus not reflect the exact properties of the corresponding cube
        # (see match_and_group)
        sr = StructuredRun(combined)
        sr.groupId = self.groupId
        # SR.label = self.label
        return sr


    def compute_master(self, how_combine, mbias=None, load=False, w2f=True, outdir=None):
        """
        Compute the master image(s) (bias / flat field)

        Parameters
        ----------
        how_combine: function to use when combining the stack
        mbias :     A StructuredRun instance of master biases (optional)
        load  :     If set, the master frames will be loaded as shocObss.
                    If unset kept as filenames

        Returns
        -------
        A StructuredRun of master frames separated by the same keys as self
        """

        if mbias:
            # print( 'self.groupId, mbias.groupId', self.groupId, mbias.groupId )
            assert self.groupId == mbias.groupId

        keys = self.groupId
        masters = {}  # master bias filenames
        dataTable = []
        for attrs, run in self.items():

            if run is None:  # Unmatched!
                masters[attrs] = None
                continue

            # FIXME: will have to implement generating filenames like: 'f2017020[89].4x4' or something
            # master bias / flat frame for this group
            name = run.magic_filenames()[0]
            master = run.compute_master(how_combine, name)
            master.flush(output_verify='warn', verbose=1)  # writes full frame master

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
            # this creates a run of all the master frames which will be split into individual
            # shocObs instances upon the creation of the StructuredRun at return
            label = 'master {}'.format(self.name)
            mrun = self.runClass(hdus=masters.values())     #label=label

        if w2f:
            fn = label.replace(' ', '.')
            outname = os.path.join(outdir, fn)
            mrun.export_filenames(outname)

        # NOTE:  The dict here is keyed on the closest matching attributes in self!
        SR = StructuredRun(masters)
        SR.groupId = self.groupId
        # SR.label = self.label
        return SR

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

        newcals = self.runClass(substacks) + b  #, label=c_sr.label
        return newcals

    # TODO: combine into calibration method
    @timer
    def debias(self, m_bias_dict, ):       #FIXME: RENAME Dark
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
        for attrs, master in m_bias_dict.items():
            if master is None:
                continue

            if isinstance(master, shocRun):
                master = master[0]  # HACK!!

            stacks = self[attrs]  # the cubes (as shocRun) for this attrs value

            msg = '\nDoing bias subtraction on the stack: '
            lm = len(msg)
            logging.info(msg)
            for stack in stacks:
                logging.info(' ' * lm, stack.get_filename())

                header = stack.header
                # Adds the keyword 'BIASCORR' to the image header to indicate that bias correction has been done
                # header['BIASCORR'] = (True, 'Bias corrected')
                # Add the filename and time of bias subtraction to header HISTORY
                hist = 'Bias frame %r subtracted at %s' %\
                       (master.get_filename(), datetime.datetime.now())
                header.add_history(hist, before='HEAD')

                # TODO: multiprocessing here...??
                stack.data = stack.data - master.data
                # NOTE avoid augmented assign -= here due to potential numpy casting error for different types

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

            mf_data = masterflat.data  # pyfits.getdata(masterflat, memmap=True)

            if round(np.mean(mf_data), 1) != 1:
                raise ValueError('Flat field not normalised!!!!')

            stacks = self[attrs]  # the cubes for this binning value

            msg = '\nDoing flat field division on the stack: '
            lm = len(msg)
            logging.info(msg, )
            for stack in stacks:
                logging.info(' ' * lm + stack.get_filename())

                # Adds the keyword 'FLATCORR' to the image header to indicate that
                # flat field correction has been done
                header = stack.header
                header['FLATCORR'] = (True, 'Flat field corrected')
                # Adds the filename used and time of flat field correction to header HISTORY
                hist = 'Flat field {} subtracted at {}'.format(masterflat.get_filename(), datetime.datetime.now())
                header.add_history(hist, before='HEAD')

                try:
                    # flat field division
                    stack.data = stack.data / mf_data
                    # NOTE: avoiding augmented assignment due to numpy type errors
                except Exception as err:
                    print('FAIL ' * 10)
                    print(err)
                    embed()

        self.label = 'science frames (flat fielded)'

