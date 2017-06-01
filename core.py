# __version__ = '2.12'

import os
import re
import time
import datetime
import warnings
import operator
# import collections as col
from copy import copy
import itertools as itt

# WARNING: THESE IMPORT ARE MEGA SLOW!! ~10s  (localize to mitigate)
import numpy as np
# import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
from astropy.io.fits.hdu.image import PrimaryHDU

import recipes.iter as itr
from recipes.io import warn
from recipes.list import sorter, flatten
from recipes.string import rreplace
from ansi.table import Table as sTable


# TODO: choose which to use for timing: spice or astropy
# from .io import InputCallbackLoop
from .utils import retrieve_coords, convert_skycooords
from .timing import timingFactory, Time, get_updated_iers_table
from .header import shocHeader
from .convert_keywords import KEYWORDS as kw_old_to_new

# debugging
from IPython import embed
from decor.profile.timers import timer#, profiler

# def warn(message, category=None, stacklevel=1):
# return warnings.warn('\n'+message, category=None, stacklevel=1)


# FIXME: Many of the functions here have verbosity argument. can you wrap these somehow in a
# centralized logging interface?
# FIXME: Autodetect corrupt files?  eg.: single exposure files (new) sometimes don't contain KCT

# TODO
# __all__ = ['']


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
    # ===========================================================================
    def __repr__(self):
        return str(self)


################################################################################
class FilenameGenerator(object):
    # ===========================================================================
    def __init__(self, basename, reduction_path='', padwidth=None, sep='.',
                 extension='.fits'):
        self.count = 1
        self.basename = basename
        self.path = reduction_path
        self.padwidth = padwidth
        self.sep = sep
        self.extension = extension

    # ===========================================================================
    def __call__(self, maxcount=None, **kw):
        """Generator of filenames of unpacked cube."""
        path = kw.get('path', self.path)
        sep = kw.get('sep', self.sep)
        extension = kw.get('extension', self.extension)

        base = os.path.join(path, self.basename)

        if maxcount:
            while self.count <= maxcount:
                imnr = '{1:0>{0}}'.format(self.padwidth, self.count)  # image number string. eg: '0013'
                outname = '{}{}{}{}'.format(base, sep, imnr, extension)  # name string eg. 'darkstar.0013.fits'
                self.count += 1
                yield outname
        else:
            yield '{}{}'.format(base, self.extension)



################################################################################
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



################################################################################
class shocCube(pyfits.hdu.hdulist.HDUList):     #TODO: maybe rename shocObs
    # TODO: location as class attribute??
    """
    Extend the hdu.hdulist.HDUList class to perform simple tasks on the image stacks.
    """
    name = 'science'        # TODO: or should this be a subclass ??
    # default attributes for __repr__ / get_instrumental_setup
    _pprint_attrs = ['binning', 'shape', 'preAmpGain', 'outAmpMode', 'emGain']  # ['binning', 'shape', 'mode', 'emGain', 'ron']
    _nullGainStr = '--'

    @classmethod
    def set_pprint(cls, attrs):
        # TODO: safety checks - need abc for this? or is that overkill?
        cls._pprint_attrs = attrs

    # ===========================================================================
    @classmethod
    def load(cls, fileobj, mode='readonly', memmap=False, save_backup=False, **kwargs):
        """Load shocCube from file"""
        hdulist = cls._readfrom(fileobj=fileobj, mode=mode, memmap=memmap,
                                save_backup=save_backup, ignore_missing_end=True,
                                # do_not_scale_image_data=True,
                                **kwargs)

        return hdulist

    # ===========================================================================
    def __init__(self, hdus=None, file=None):

        hdus = [] if hdus is None else hdus
        super().__init__(hdus, file)

        # FIXME: these should be properties...
        self._needs_flip = False
        self._needs_sub = []
        self._is_master = False
        self._is_unpacked = False
        self._is_subframed = False

        self.path, self.basename = os.path.split(self.filename())
        if self.basename:
            self.basename = self.basename.replace('.fits', '')

        # if len(self):   # FIXME: either remove these lines from init or detect new file write. or ??
        # Load the important header information as attributes
        self.instrumental_setup()   # NOTE: fails on self.writeto
        # Initialise timing class
        self.timing = timingFactory(self)
        # NOTE: this is a hack that allows us to set the timing associated methods dynamically
        self.kct = self.timing.kct
        # NOTE: Set this attribute here so the cross matching algorithm works. inherit from the timing classes directly to avoid the previous line

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

    # ===========================================================================
    def __repr__(self):
        name, dattrs, values = self.get_instrumental_setup()
        ref = tuple(itr.interleave(dattrs, values))
        r = name + ':\t' + '%s = %s;\t' * len(values) % ref
        return '{} ==> {}'.format(self.__class__.__name__, r)

    # ===========================================================================
    def _get_data(self):
        """retrieve PrimaryHDU data"""
        return self[0].data
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
    # def ron(self):


    # ===========================================================================
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

    # ===========================================================================
    def instrumental_setup(self):
        """
        Retrieve the relevant information about the observational setup. Used for comparitive
        tests.
        """
        header = self.header

        # instrument
        serno = header['SERNO']
        self.instrument = 'SHOC ' + str([5982, 6448].index(serno) + 1)
        # else: self.instrument = 'unknown!'

        # date
        date, time = header['DATE'].split('T')
        self.date = Date(*map(int, date.split('-')))  # file creation date
        # starting date of the observing run --> used for naming
        h = int(time.split(':')[0])
        self.namedate = self.date - datetime.timedelta(int(h < 12))

        # image binning
        self.binning = tuple(header[s + 'BIN'] for s in 'HV')

        # image dimensions
        self.ndims = header['NAXIS']  # Number of image dimensions
        self.shape = *self.ishape, self.nframes = \
            tuple(header['NAXIS' + str(i)] for i in np.r_[:self.ndims] + 1)
        # self.ishape = self.shape[:2]  # Pixel dimensions for 2D images

        # sub-framing
        self.subrect = np.array(header['SUBRECT'].split(','), int)
        xsub, ysub = (xb, xe), (yb, ye) = self.subrect.reshape(-1, 2) // self.binning
        self.sub = np.r_[xsub, ysub]
        self._is_subframed = (xe, ye) != self.ishape

        # readout speed
        speed = 1. / header['READTIME']
        self.preAmpSpeed = speedMHz = int(round(speed / 1.e6))

        # CCD mode
        self.preAmpGain = header['PREAMP']          # TODO: self.mode.pre.gain ??
        self.outAmpModeLong = header['OUTPTAMP']    # TODO: self.mode.outamp ??
        self.acqMode = header['ACQMODE']

        #TODO: as properties???
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
        self.flip_state = tuple(header['FLIP' + s] for s in 'YX')
        # NOTE: the order of the axes here is row, column

        # Timing
        # self.kct = self.timing.kct
        # self.trigger_mode = self.timing.trigger.mode
        # self.duration

        # object name
        self.target = header.get('object')

        # coordinates
        self.coords = self.get_coords()     #verbose=False
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
    # ===========================================================================
    # TODO: property?????????        NOTE: consequences for head_proc
    def get_coords(self, verbose=False):
        header = self.header

        ra, dec = header.get('objra'), header.get('objdec')
        coords = convert_skycooords(ra, dec)
        if coords:
            return coords

        if self.target:
            # No / bad coordinates in header, but object name available - try resolve
            coords = retrieve_coords(self.target, verbose=verbose)

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

    # ===========================================================================
    @property
    def needs_timing_fix(self):
        """check for date-obs keyword to determine if header information needs updating"""
        return not ('date-obs' in self.header)  # FIXME: is this good enough???

    # ===========================================================================
    @property
    def has_coords(self):
        return self.coords is not None

    # ===========================================================================
    def get_instrumental_setup(self, attrs=None):
        # TODO: YOU CAN MAKE THIS __REPR__????????
        # TODO: units

        filename = self.get_filename() or 'Unsaved'
        attrNames = self._pprint_attrs[:] if (attrs is None) else attrs
        attrDisplayNames = attrNames[:] #.upper()

        # get timing attributes if initialized
        timingAttrNames = ['trigger', 'kct', 'duration']
        if hasattr(self, 'timing'):
            attrNames.extend(map('timing.%s'.__mod__, timingAttrNames))
            attrDisplayNames.extend(timingAttrNames)

        attrVals = operator.attrgetter(*attrNames)(self)  # COOL, but no default option

        # for at in attrNames:
        #     attrDisplayNames.append(at) #.upper()
        #     attrVals.append(getattr(self, at, None))

        # get timing attributes if initialized
        #timing = getattr(self, 'timing', None)
        #if timing:
            #for at in timingAttrNames:
                # attrDisplayNames.append(at.replace('_', ' ')) #.upper()
                #attrVals.append(getattr(timing, at, None))

        return filename, attrDisplayNames, attrVals

    # ===========================================================================
    def get_pixel_scale(self, telescope):
        """get pixel scale in arcsec """
        pixscale = {'1.9': 0.076,
                    '1.9+': 0.163,  # with focal reducer
                    '1.0': 0.167,
                    '0.75': 0.218}

        tel = rreplace(telescope, ('focal reducer', 'with focal reducer'), '+')
        tel = tel.replace('m', '').strip()

        return np.array(self.binning) * pixscale[tel]

    # ===========================================================================
    def get_field_of_view(self, telescope):
        """get FoV in arcmin"""
        fov = {'1.9': (1.29, 1.29),
               '1.9+': (2.79, 2.79),  # with focal reducer
               '1.0': (2.85, 2.85),
               '0.75': (3.73, 3.73)}

        tel = rreplace(telescope, ('focal reducer', 'with focal reducer'), '+')
        tel = tel.replace('m', '').strip()

        return fov[tel]

    get_FoV = get_field_of_view

    # ===========================================================================
    def cross_check(self, frame2, key, raise_error=0):
        """
        Check fits headers in this image agains frame2 for consistency of key attribute

        Parameters
        ----------
        key : The attribute to be checked (binning / instrument mode / dimensions / flip state)
        frame2 : shocCube Objects to check against

        Returns
        ------
        flag : Do the keys match?
        """
        flag = (getattr(self, key) == getattr(frame2, key))

        if not flag and raise_error:
            raise ValueError
        else:
            return flag

    # ===========================================================================
    def flip(self, state=None):

        state = self.flip_state if state is None else state
        header = self.header
        for axis, yx in enumerate('YX'):
            if state[axis]:
                print('Flipping {} in {}.'.format(self.get_filename(), yx))
                self.data = np.flip(self.data, axis)
                header['FLIP%s' % yx] = int(not self.flip_state[axis])

        self.flip_state = tuple(header['FLIP%s' % s] for s in ['YX'])
        #FIXME: avoid this line by making flip_state a list

        # flipx, flipy = self.flip_state
        # data = self.data
        # if flipx:
        #     print('Flipping {} in X.'.format(self.get_filename()))
        #     self.data = np.fliplr(self.data)
        #     header['FLIPX'] = int(not flipx)
        #     self.flip_state = tuple(header['FLIP' + s] for s in ['X', 'Y'])
        #
        # if flipy:
        #     print('Flipping {} in Y.'.format(self.get_filename()))
        #     self.data = np.flipud(data)
        #     header['FLIPY'] = int(not flipy)
        #     self.flip_state = tuple(header['FLIP' + s] for s in ['X', 'Y'])

    # ===========================================================================
    def subframe(self, subreg, write=1):
        if self._is_subframed:
            raise TypeError('{} is already sub-framed!'.format(self.filename()))

        embed()

        cb, ce, rb, re = subreg
        print('subframing {} to {}'.format(self.filename(), [rb, re, cb, ce]))

        data = self.data[rb:re, cb:ce]
        header = self.header
        # header['SUBRECT']??

        print('!' * 8, self.sub)

        subext = 'sub{}x{}'.format(re - rb, ce - cb)
        outname = self.get_filename(1, 1, subext)
        fileobj = pyfits.file._File(outname, mode='ostream', overwrite=True)

        hdu = pyfits.hdu.PrimaryHDU(data=data, header=header)
        # embed()

        stack = self.__class__(hdu, fileobj)
        stack.instrumental_setup()

        # stack._is_subframed = 1
        # stack._needs_sub = []
        # stack.sub = subreg

        if write:
            stack.writeto(outname, output_verify='warn')

        return stack

    # ===========================================================================
    def combine(self, func, name=None):
        """
        Mean / Median combines the image stack

        Returns
        -------
        shocCube instance
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

        imnr = '001'  # FIXME:   #THIS WILL NEED TO CHANGE FOR MULTIPLE SINGLE IMAGES AS INPUT
        header = copy(self.header)
        data = apply_stack(func, self.data, axis=0)   # mean / median across images

        ncomb = header.pop('NUMKIN', 0)  # Delete the NUMKIN header keyword
        # if 'naxis3' in header:          header.remove('NAXIS3')
        # header['NAXIS'] = 2     # NOTE: pyfits actually does this automatically...
        header['NCOMBINE'] = (ncomb, 'Number of images combined')
        header['ICMB' + imnr] = (self.filename(), 'Contributors to combined output image')
        # FIXME: THIS WILL NEED TO CHANGE FOR MULTIPLE SINGLE IMAGES AS INPUT

        # Load the stack as a shocCube
        if name is None:
            name = next(self.filename_gen())  # generate the filename

        hdu = shocHDU(data, header)
        fileobj = pyfits.file._File(name, mode='ostream', overwrite=True)
        stack = self.__class__(hdu, fileobj)  # initialise the Cube with target file
        stack.instrumental_setup()

        return stack

    # ===========================================================================
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

            print('\n\nUnpacking the stack {} of {} images...\n\n'.format(stack, naxis3))

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
            print('Time taken: %f' % (end_time - start_time))

        self._is_unpacked = True

        return count

    # ===========================================================================
    def set_name_dict(self):
        header = self.header
        obj = header.get('OBJECT', '')
        filter = header.get('FILTER', 'WL')

        kct = header.get('kct', 0)
        if int(kct / 10):
            kct = str(round(kct)) + 's'
        else:
            kct = str(round(kct * 1000)) + 'ms'

        self.name_dict = dict(sep='.',
                              obj=obj,
                              basename=self.get_filename(0, 0),
                              date=str(self.namedate).replace('-', ''),
                              filter=filter,
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
class shocBiasCube(shocCube): #FIXME: DARK?
    name = 'bias'

    def get_coords(self):
        return

    def compute_master(self, func=np.median, masterbias=None, name=None, verbose=False):
        return self.combine(func, name)


class shocFlatFieldCube(shocBiasCube):
    name = 'flat'

    def compute_master(self, func=median_scaled_median, masterbias=None, name=None, verbose=False):
        """ """
        master = self.combine(func, name)
        if masterbias:
            if verbose:
                print('Subtracting master bias {} from master flat {}.'.format(
                    masterbias.get_filename(), master.basename))
            # match orientation
            masterbias.flip(master.flip_state)
            master.data -= masterbias.data
        elif verbose:
            print('No master bias for {}'.format(self.filename()))

        # flat field normalization
        ffmean = np.mean(master.data)  # flat field mean
        if verbose:
            print('Normalising flat field...')
        master.data /= ffmean
        return master



class ClassProperty(property):
    def __get__(self, cls, owner):
        return self.fget.__get__(None, owner)()

################################################################################
class shocRun(object):
    # TODO: merge method?
    """
    Class containing methods to operate with sets of shocCube objects.
    perform comparitive tests between cubes to see if they are compatable.
    """
    # ===========================================================================
    cubeClass = shocCube
    nameFormat = '{basename}'       # Naming convention defaults
    _compactRepr = True
    # compactify table repr by removing columns (properties) that are all equal

    @ClassProperty      # so we can access via  shocRun.name and shocRun().name
    @classmethod
    def name(cls):
        return cls.cubeClass.name

    # ===========================================================================
    def __init__(self, hdus=None, filenames=None, label=None, grouped_by=None,
                 location='sutherland'):

        # WARNING:  filenames may contain None as well as duplicate entries.....??????
        # not sure if duplicates is desirable wrt efficiency.....

        self.cubes = list(filter(None, hdus)) if hdus else []

        self.grouped_by = grouped_by
        self.label = label

        if not filenames is None:
            self.filenames = list(filter(None, filenames))  # filter None
            self.load(self.filenames)
        elif not hdus is None:
            self.filenames = [hdulist.filename() for hdulist in self]

    # ===========================================================================
    def __len__(self):
        return len(self.cubes)

    # ===========================================================================
    def __repr__(self):
        return '{} : {}'.format(self.__class__.__name__, ' | '.join(self.get_filenames()))

    # ===========================================================================
    def __getitem__(self, key):

        if isinstance(key, int):
            if key >= len(self):
                raise IndexError("The index (%d) is out of range." % key)
            return self.cubes[key]

        if isinstance(key, slice):
            rl = self.cubes[key]

        if isinstance(key, tuple):
            assert len(key) == 1
            key = key[0]  # this should be an array...

        if isinstance(key, (list, np.ndarray)):

            if isinstance(key[0], (bool, np.bool_)):
                assert len(key) == len(self)
                rl = [self.cubes[i] for i in np.where(key)[0]]

            elif isinstance(key[0], (int, np.int0)):  # NOTE: be careful bool isa int
                rl = [self.cubes[i] for i in key]

        return self.__class__(rl, label=self.label, grouped_by=self.grouped_by)

    # ===========================================================================
    def __add__(self, other):
        return self.join(other)

    # ===========================================================================
    # def __eq__(self, other):
    # return vars(self) == vars(other)

    # ===========================================================================
    # def pullattr(self, attr, return_as=list):
    # return return_as([getattr(item, attr) for item in self])

    # ===========================================================================
    def load(self, filenames, mode='update', memmap=False, save_backup=False, **kwargs):
        """
        Load data from file. populate data for instrumental setup from fits header.
        """
        self.filenames = filenames

        label = kwargs.pop('label', self.label)
        print('\nLoading data for {} run...'.format(label))  # TODO: use logging

        # cubes = []
        for i, fileobj in enumerate(filenames):
            try:
                hdu = self.cubeClass.load(fileobj, mode=mode, memmap=memmap, save_backup=save_backup,
                                          **kwargs)
                # NOTE: YOU CAN BYPASS THIS INTERMEDIATE STORAGE IF YOU MAKE THE PRINT OPTION
                # A KEYWORD ARGUMENT FOR THE shocCube __init__
                self.cubes.append(hdu)
            except Exception as err:
                import traceback
                warn('File: %s failed to load with exception:\n%s' % (fileobj, str(err)))


    # ===========================================================================
    def pop(self, i):  # TODO: OR SUBCLASS LIST?
        return self.cubes.pop(i)

    # ===========================================================================
    def join(self, *runs):

        runs = list(filter(None, runs))  # Filter empty runs (None)
        labels = [r.label for r in runs]
        hdus = sum([r.cubes for r in runs], self.cubes)

        if np.all(self.label == np.array(labels)):
            label = self.label
        else:
            warn("Labels {} don't match {}!".format(labels, self.label))
            label = None

        return self.__class__(hdus, label=label)

    # ===========================================================================
    def print_instrumental_setup(self, attrs=None, description=''):
        """Print the instrumental setup for this run as a table.
        :param attrs:
        """
        filenames, attrDisplayNames, attrVals = zip(*(stack.get_instrumental_setup(attrs)
                                                      for stack in self))
        attrDisplayNames = attrDisplayNames[0] # all are the same

        # TODO: compactify method of Table ?
        # if 'emGain' in attrDisplayNames:
        #     ixEM = attrDisplayNames.index('emGain')
        #     gain = itr.nthzip(ixEM, *attrVals)
        #     if np.all(np.array(gain) == self.cubeClass._nullGainStr):
        #         # unnecessary to display emGain values when all cubes are CON mode - remove from display table
        #         n = len(attrDisplayNames)
        #         #ix = attrDisplayNames.index('emGain')
        #         _, *fltrs = itr.filtermore(lambda i: i != ixEM, range(n), attrDisplayNames, *attrVals)
        #         attrDisplayNames, *attrVals = map(tuple, fltrs)

        name = self.label or '<unlabeled>'      # fixme: property
        title = 'Instrumental Setup: {} frames {}'.format(str(name).title(), description)
        table = sTable(attrVals,
                       title=title,
                       title_props=dict(text='bold', bg=self.displayColour),
                       col_headers=attrDisplayNames,
                       row_headers=['filename'] + list(filenames),
                       number_rows=True,
                       precision=5, minimalist=True, compact=True
                       )

        # print(table)
        return table

    pprint = print_instrumental_setup
    # ===========================================================================
    def reload(self, filenames=None, mode='update', memmap=True,
               save_backup=False, **kwargs):
        if len(self):
            self.cubes = []
        self.load(filenames, mode, memmap, save_backup, **kwargs)

    ############################################################################
    # Timing
    ############################################################################
    # ===========================================================================
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
        print(msg, )
        for i, stack in enumerate(self):
            print(' ' * lm + stack.get_filename())
            t0 = t0s[i] # t0 = stack.timing.t0mid
            if coords is None and stack.has_coords:
                coords = stack.coords
            stack.timing.set(t0, iers_a, coords)
            # update the header with the timing info
            stack.timing.stamp(0, t0, coords, verbose=True)
            stack.flush(output_verify='warn', verbose=True)

    # ===========================================================================
    def export_times(self, with_slices=False):
        for i, stack in enumerate(self):
            timefile = stack.get_filename(1, 0, 'time')
            stack.timing.export(timefile, with_slices, i)

    # ===========================================================================
    def gen_need_kct(self):
        """
        Generator that yields the cubes in the run that require KCT to be set
        """
        for stack in self:
            yield (stack.needs_timing_fix() and stack.trigger.is_gps())

    # ===========================================================================
    def that_need_kct(self):
        """Return a shocRun object containing only those cubes that are missing KCT"""
        return self[list(self.gen_need_kct())]

    # ===========================================================================
    def gen_need_triggers(self):
        """
        Generator that yields the cubes in the run that require GPS triggers to be set
        """
        for stack in self:
            yield (stack.needs_timing_fix() and stack.trigger.is_gps())

    # ===========================================================================
    def that_need_triggers(self):
        """Return a shocRun object containing only those cubes that are missing triggers"""
        return self[list(self.gen_need_triggers())]

    # ===========================================================================
    def set_gps_triggers(self, times, triggers_in_local_time=True):
        # trigger provided by user at terminal through -g  (args.gps)
        # if len(times)!=len(self):
        # check if single trigger OK (we can infer the remaining ones)
        if self.check_rollover_state():
            times = self.get_rolled_triggers(times)
            print(
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
            stack.timing.trigger.set(triggers_in_local_time)

    # ===========================================================================
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

    # ===========================================================================
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

    # ===========================================================================
    def export_headers(self):
        """save fits headers as a text file"""
        for stack in self:
            headfile = stack.get_filename(with_path=1, with_ext=0, suffix='.head')
            print('\nWriting header to file: {}'.format(os.path.basename(headfile)))
            # TODO: remove existing!!!!!!!!!!
            stack.header.totextfile(headfile, overwrite=True)

    # ===========================================================================
    def make_slices(self, suffix):
        for i, cube in enumerate(self):
            cube.make_slices(suffix, i)

    # ===========================================================================
    def make_obsparams_file(self, suffix):
        for i, cube in enumerate(self):
            cube.make_obsparams_file(suffix, i)

    # ===========================================================================
    # TODO: as Mixin ???
    def magic_filenames(self, path='', sep='.', extension='.fits'):
        """Generates a unique sequence of filenames based on the name_dict."""

        self.set_name_dict()

        # re pattern matchers
        # matches the optional keys sections (including square brackets) in the
        # format specifier string from the args.names namespace
        opt_pattern = '\[[^\]]+\]'
        opt_matcher = re.compile(opt_pattern)
        # matches the key (including curly brackets) and key (excluding curly
        # brackets) for each section of the format string
        key_pattern = '(\{(\w+)\})'
        key_matcher = re.compile(key_pattern)

        # get format specification string from label
        # for label in ('sci', 'bias', 'flat'):
        #     if label in self.label:
        #         # get the naming format string from the argparse namespace
        #         fmt_str = getattr(self.NAMES, label)
        #         break

        # check which keys help in generating unique set of filenames - these won't be used
        # print('Check that this function is actually producing minimal unique filenames!!')
        non_unique_keys = [key for key in self[0].name_dict.keys()
                           if all([self[0].name_dict[key] == stack.name_dict[key]
                                   for stack in self])]
        non_unique_keys.pop(non_unique_keys.index('sep'))

        filenames = [self.nameFormat] * len(self)
        nfns = []
        for cube, fn in zip(self, filenames):
            nd = copy(cube.name_dict)

            badoptkeys = [key for _, key in key_matcher.findall(self.nameFormat)
                          if not (key in nd and nd[key])]
            # This checks whether the given key in the name format specifier should be used
            # (i.e. if it has a corresponding entry in the shocCube instance's name_dict.
            # If one of the keys are unset in the name_dict, this optional key will be eliminated
            # when generating the filename below.

            for opt_sec in opt_matcher.findall(self.nameFormat):
                if any(key in opt_sec for key in badoptkeys + non_unique_keys):
                # or any(key in opt_sec for key in non_unique_keys)):
                    fn = fn.replace(opt_sec, '')
                    # replace the optional sections which contain keywords that
                    # are not in the corresponding name_dict and replace the
                    # optional sections which contain keywords that don't
                    # contribute to the uniqueness of the filename set
            nfns.append(fn.format(**nd))

        # eliminate square brackets
        filenames = [fn.replace('[', '').replace(']', '') for fn in nfns]
        # last resort append numbers to the non-unique filenames
        if len(set(filenames)) < len(set(self.filenames)):
            unique_fns, idx = np.unique(filenames, return_inverse=True)
            nfns = []
            for basename in unique_fns:
                count = filenames.count(basename)  # number of files with this name
                if count > 1:
                    padwidth = len(str(count))
                    g = FilenameGenerator(basename, padwidth=padwidth, sep='_', extension='')
                    fns = list(g(count))
                else:
                    fns = [basename]
                nfns += fns

            # sorts by index. i.e. restores the order of the original filename sequence
            _, filenames = sorter(idx, nfns)

        # create a FilenameGenerator for each stack
        for stack, fn in zip(self, filenames):
            padwidth = len(str(stack.shape[-1]))
            stack.filename_gen = FilenameGenerator(fn, path, padwidth, sep, extension)

        filenames = list(map(''.join, zip(filenames, itt.repeat(extension))))
        return filenames

    # ===========================================================================
    def genie(self, i=None):
        """returns list of generated filename tuples for cubes up to file number i"""
        return list(itt.zip_longest(*[cube.filename_gen(i) for cube in self]))

    # ===========================================================================
    def get_filenames(self, with_path=False, with_ext=True, suffix=(), sep='.'):
        """filenames of run constituents"""
        return [stack.get_filename(with_path, with_ext, suffix, sep) for stack in self]

    # ===========================================================================
    def export_filenames(self, fn):

        if not fn.endswith('.txt'):  # default append '.txt' to filename
            fn += '.txt'

        print('\nWriting names of {} to file {}...\n'.format(self.name, fn))
        with open(fn, 'w') as fp:
            for f in self.filenames:
                fp.write(f + '\n')

    # ===========================================================================
    def writeout(self, with_path=False, suffix='', dryrun=False, header2txt=False):  # TODO:  INCORPORATE FILENAME GENERATOR
        fns = []
        for stack in self:
            fn_out = stack.get_filename(with_path, False, (suffix, 'fits'))  # FILENAME GENERATOR????
            fns.append(fn_out)

            if not dryrun:
                print('\nWriting to file: {}'.format(os.path.basename(fn_out)))
                stack.writeto(fn_out, output_verify='wbyarn', overwrite=True)

                if header2txt:
                    # save the header as a text file
                    headfile = stack.get_filename(1, 0, (suffix, 'head'))
                    print('\nWriting header to file: {}'.format(os.path.basename(headfile)))
                    stack.header.totextfile(headfile, overwrite=True)

        return fns

    # ===========================================================================
    def list_attr(self, keys):
        return self.zipper(keys)[1]

    def zipper(self, keys, flatten=True):

        # attrs = list(map(operator.attrgetter(keys), self))

        if isinstance(keys, str):
            return keys, [getattr(s, keys) for s in self]  # s.__dict__[keys]??
        elif len(keys) == 1 and flatten:
            key = tuple(keys)[0]
            return key, [getattr(s, key) for s in self]
        else:
            return (tuple(keys),
                    list(zip(*([getattr(s, key) for s in self] for key in keys))))

    # ===========================================================================
    def group_by(self, *keys):
        """
        Separate a run according to the attribute given in keys.
        keys can be a tuple of attributes (str), in which case it will seperate into runs with a unique combination
        of these attributes.

        Returns
        -------
        atdict : dictionary containing attrs, run pairs where attrs are the attributes of run given by keys
        flag :  1 if attrs different for any cube in the run, 0 all have the same attrs
        """
        keys, attrs = self.zipper(keys)

        if self.grouped_by == keys:  # is already separated by this key
            SR = StructuredRun(zip([attrs[0]], [self]))
            SR.grouped_by = keys
            # SR.name = self.name
            return SR#, 0

        atset = set(attrs)  # unique set of key attribute values
        atdict = OrderedDict()
        if len(atset) == 1:  # all input files have the same attribute (key) value(s)
            # flag = 0
            atdict[attrs[0]] = self
            self.grouped_by = keys
        else:  # binning is not the same for all the input cubes
            # flag = 1
            for ats in sorted(atset):  # map unique attribute values to slices of run with those attributes
                l = np.array([attr == ats for attr in attrs])  # list for-loop needed for tuple attrs
                eq_run = self[l]  # shocRun object of images with equal key attribute
                eq_run.grouped_by = keys
                atdict[ats] = eq_run   # put into dictionary

        SR = StructuredRun(atdict)
        SR.grouped_by = keys
        # SR.name = self.name
        return SR#, flag

    def varies_by(self, *keys):
        """False if the run is homogeneous by keys and True otherwise"""
        keys, attrs = self.zipper(keys, flatten=False)
        atset = set(attrs)
        return (len(atset) != 1)

    def select_by(self, **kws):
        out = self
        for key, val in kws.items():
            out = out.group_by(key)[val]
        return out

    def filter_by(self, **kws):
        keys, attrs = self.zipper(kws.keys(), flatten=False)
        funcs = kws.values()

        predicate = lambda att: all(f(at) for f, at in zip(funcs, att))
        selection = list(map(predicate, attrs))
        return self[selection]


    def sort_by(self, *keys, **kws):
        """
        Sort the cubes by the value of attributes given in keys,
        kws can be attribute, callables pairs in which case sorting will be done according to value
        returned by callable on given attribute.
        """
        # FIXME: order of kws lost when passing as dict. not desirable.
        # NOTE: this is not a problem in python >3.5!! yay! https://docs.python.org/3/whatsnew/3.6.html
        # for keys, func in kws.items():
        #     if

        def trivial(x):
            return x

        # compose sorting function
        triv = (trivial,) * len(keys)  # will be used to sort by the actual values of attributes in *keys*
        kwkeys, kwattrs = self.zipper(kws.keys(), flatten=False)
        kwfuncs = tuple(kws.values())       #
        funcs = triv + kwfuncs              # tuple of functions, one for each requested attribute
        # combine all functions into single function that returns tuple that determines sort position
        attrSortFunc = lambda *x: tuple(f(z) for (f, z) in zip(funcs, x))

        keys, attrs = self.zipper(keys, flatten=False)
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
            print(msg)
        return is_single


    def stack(self, name):

        header = copy(self[0].header)
        data = self.stack_data()

        # create a shocCube
        hdu = shocHDU(data, header)
        fileobj = pyfits.file._File(name, mode='ostream', overwrite=True)
        cube = self.__class__(hdu, fileobj)  # initialise the Cube with target file
        cube.instrumental_setup()


    def stack_data(self):
        """return array of stacked data"""
        # if len(self) == 1:
        #     return

        dims = self.list_attr('dimension')
        if not np.equal(dims, dims[0]).all():
            raise ValueError('Cannot stack cubes with differing frame sizes: %s' %str(dims))

        return np.vstack(self.list_attr('data'))

    # ===========================================================================
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
        shocCube instance
        """
        # verbose = True
        # if verbose:
        #     self._print_combine_map(func.__name__, name)

        data = apply_stack(func, self.stack_data(), *args, **kws)

        # update header     # TODO: check which header entries are different
        header = copy(self[0].header)
        ncomb = sum(next(zip(*self.list_attr('shape'))))
        header['NCOMBINE'] = (ncomb, 'Number of images combined')
        header['FCOMBINE'] = (func.__name__, 'Function used to combine the data')
        for i, fn in enumerate(self.get_filenames()):
            header['ICMB{:0{}d}'.format(i, 3)] = (fn, 'Contributors to combined output image')

        # create a shocCube
        hdu = shocHDU(data, header)
        fileobj = pyfits.file._File(name, mode='ostream', overwrite=True)
        cube = self.cubeClass(hdu, fileobj)  # initialise the Cube with target file
        cube.instrumental_setup()

        return cube

    def _print_combine_map(self, fname, out):
        #                         median_scaled_median
        # SHA_20170209.0002.fits ----------------------> f20170209.4x4.fits
        s = str(self)
        a = ' '.join([' ' * (len(s) + 1), fname, ' ' * (len(out) + 2)])
        b = ' '.join([s, '-' * (len(fname) + 2) + '>', out])
        print('\n'.join([a, b]))

     # ===========================================================================
    def unpack(self, sequential=0, dryrun=0, w2f=1):
        # unpack datacube(s) and assign 'outname' to output images
        # if more than one stack is given 'outname' is appended with 'n_' where n is the number of the stack (in sequence)

        if not dryrun:
            outpath = self[0].filename_gen.path
            print('The following image stack(s) will be unpacked into {}:\n{}'
                  ''.format(outpath, '\n'.join(self.get_filenames())))

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
            print('\n' * 3 + 'A total of %i images where unpacked' % tot_num + '\n' * 3)
            if w2f:
                catfiles([stack.unpacked for stack in self],
                         'all.split')  # RENAME??????????????????????????????????????????????????????????????????????????????????????????????

    # ===========================================================================
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

     # ===========================================================================
    def flag_sub(self, science_run, raise_error=0):
        return
        # print( 'flag' )
        # embed()
        # dim_mismatch = self.check(science_run, 'dimension', 0)
        # if dim_mismatch:
        # keys = 'binning', 'dimension'
        # sprops = set( science_run.zipper(keys)[1] )
        # cprops = set( self.zipper(keys)[1] )
        # missing_props = sprops - cprops
        # for binning, dim in zip(*missing_props):


        dim_match = self.cross_check(science_run, 'dimension', raise_error=-1)
        if dim_match.any():
            is_subframed = np.array([s._is_subframed for s in science_run])
            if np.any(is_subframed):
                # flag the cubes that need to be subframed
                sf_bins = [s.binning for s in science_run if s._is_subframed]
                # those binnings that have subframed cubes
                for cube in self:
                    if cube.binning in sf_bins:
                        cube._needs_sub = 1

    # ===========================================================================
    # def set_airmass(self, coords=None):
    # for stack in self:
    # stack.set_airmass(coords)

    # ===========================================================================
    def set_name_dict(self):
        for stack in self:
            stack.set_name_dict()

    # ===========================================================================
    def close(self):
        [stack.close() for stack in self]

    # ===========================================================================
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

        # ===========================================================================
        def str2tup(keys):
            if isinstance(keys, str):
                keys = keys,  # a tuple
            return keys

        # ===========================================================================
        msg = ('\nMatching {} frames to {} frames by:\tExact {};\t Closest {}'
               '').format(cal_run.name.upper(), self.name.upper(), exact, repr(closest))
        print(msg)

        # create the StructuredRun for science frame and calibration frames
        exact, closest = str2tup(exact), str2tup(closest)
        grouped_by = tuple(filter(None, flatten([exact, closest])))
        s_sr = self.group_by(*grouped_by)
        c_sr = cal_run.group_by(*grouped_by)

        # Do the matching - map the science frame attributes to the calibration
        # StructuredRun element with closest match
        # NOTE AT THE MOMENT THIS ONLY USES THE FIRST KEYWORD IN closest TO DETERMINE
        # THE CLOSEST MATCH
        lme = len(exact)
        _, sciatts = self.zipper(grouped_by)  # grouped_by key attributes of the sci_run
        _, calibatts = cal_run.zipper(grouped_by)
        ssciatts = np.array(list(set(sciatts)), object)  # a set of the science frame attributes
        calibatts = np.array(list(set(calibatts)), object)
        sss = ssciatts.shape
        where_thresh = np.zeros((2 * sss[0], sss[1] + 1))
        # state array to indicate where in data threshold is exceeded (used to colourise the table)

        runmap, attmap = {}, {}
        datatable = []

        for i, attrs in enumerate(ssciatts):
            # those calib cubes with same attrs (that need closest matching)
            lx = np.all(calibatts[:, :lme] == attrs[:lme], axis=1)
            delta = abs(calibatts[:, lme] - attrs[lme])

            if ~lx.any():  # NO exact matches
                threshold_warn = False  # Don't try issue warnings below
                cattrs = (None,) * len(grouped_by)
                crun = None
            else:
                lc = (delta == delta[lx].min())
                l = lx & lc
                cattrs = tuple(calibatts[l][0])
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
        out_sr.grouped_by = s_sr.grouped_by

        if _pr:
            try:
                # Generate data table of matches
                col_head = ('Filename(s)',) + tuple(map(str.upper, grouped_by))
                where_row_borders = range(0, len(datatable) + 1, 2)

                table = sTable(datatable,
                               title='Matches',
                               title_props=dict(text='bold', bg='light blue'),
                               col_headers=col_head,
                               where_row_borders=where_row_borders,
                               precision=3, minimalist=True)

                # colourise           #TODO: highlight rows instead of colourise??
                unmatched = [None in row for row in datatable]
                unmatched = np.tile(unmatched, (len(grouped_by) + 1, 1)).T
                states = where_thresh
                states[unmatched] = 3
                table.colourise(states, 'default', 'yellow', 202, {'bg': 'red'})

                print('The following matches have been made:')
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
        sr = self.group_by('name')
        clss = list(itersubclasses(shocRun))
        for name, run in sr.items():
            for cls in clss:
                if cls.cubeClass.name == name:
                    break
            idd[name] = cls(run)
        return idd


################################################################################
class shocSciRun(shocRun):
    #name = 'science'
    nameFormat = '{basename}'
    displayColour = 'g'



class shocBiasRun(shocRun):
    # name = 'bias'
    cubeClass = shocBiasCube
    nameFormat = 'b{date}{sep}{binning}[{sep}m{mode}][{sep}t{kct}]'
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
    cubeClass = shocFlatFieldCube
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
    The attribute names given in grouped_by are the ones by which the run is separated
    into unique segments (which are also shocRun instances).
    """
    # ===========================================================================
    @property
    def runClass(self):
        if len(self):
            return type(list(self.values())[0])

    # ===========================================================================
    @property
    def name(self):
        return getattr(self.runClass, 'name', None)

    # ===========================================================================
    def __repr__(self):
        # FIXME: 'REWORK REPR: look at table printed in shocRun.match_and_group
        # embed()
        # self.values()
        return '\n'.join([' : '.join(map(str, x)) for x in self.items()])

    # ===========================================================================
    def flatten(self):
        if isinstance(list(self.values())[0], shocCube):
            run = self.runClass(list(self.values()), grouped_by=self.grouped_by)
        else:
            run = self.runClass().join(*self.values())

        # eliminate duplicates
        _, idx = np.unique(run.get_filenames(), return_index=1)
        dup = np.setdiff1d(range(len(run)), idx)
        for i in reversed(dup):
            run.pop(i)

        return run

    # ===========================================================================
    def writeout(self, with_path=True, suffix='', dryrun=False, header2txt=False):
        return self.flatten().writeout(with_path, suffix, dryrun, header2txt)

    # ===========================================================================
    def group_by(self, *keys):
        if self.grouped_by == keys:
            return self

        return self.flatten().group_by(*keys)

    # ===========================================================================
    def magic_filenames(self, path='', sep='.', extension='.fits'):
        return self.flatten().magic_filenames(path, sep, extension)

    # ===========================================================================
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
            print('\nCombining:')

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
        sr.grouped_by = self.grouped_by
        # SR.label = self.label
        return sr


    def compute_master(self, how_combine, mbias=None, load=False, w2f=True, outdir=None):
        """
        Compute the master image(s) (bias / flat field)

        Parameters
        ----------
        how_combine: function to use when combining the stack
        mbias :     A StructuredRun instance of master biases (optional)
        load  :     If set, the master frames will be loaded as shocCubes.
                    If unset kept as filenames

        Returns
        -------
        A StructuredRun of master frames separated by the same keys as self
        """

        if mbias:
            # print( 'self.grouped_by, mbias.grouped_by', self.grouped_by, mbias.grouped_by )
            assert self.grouped_by == mbias.grouped_by

        keys = self.grouped_by
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

        print()

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
            # shocCube instances upon the creation of the StructuredRun at return
            label = 'master {}'.format(self.name)
            mrun = self.runClass(hdus=masters.values())     #label=label

        if w2f:
            fn = label.replace(' ', '.')
            outname = os.path.join(outdir, fn)
            mrun.export_filenames(outname)

        # NOTE:  The dict here is keyed on the closest matching attributes in self!
        SR = StructuredRun(masters)
        SR.grouped_by = self.grouped_by
        # SR.label = self.label
        return SR

    # ===========================================================================
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

    # ===========================================================================
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
            print(msg)
            for stack in stacks:
                print(' ' * lm, stack.get_filename())

                # Adds the keyword 'BIASCORR' to the image header to indicate that bias correction has been done
                header = stack.header
                header['BIASCORR'] = (True, 'Bias corrected')
                hist = 'Bias frame {} subtracted at {}'.format(master.get_filename(), datetime.datetime.now())
                # Add the filename and time of bias subtraction to header HISTORY
                header.add_history(hist, before='HEAD')

                # TODO: multiprocessing here...??
                # NOTE avoid inplace -= here due to potential numpy casting error for different types
                stack.data = stack.data - master.data

        self.label = self.name + ' (bias subtracted)'
        return self

    # ===========================================================================
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
            print(msg, )
            for stack in stacks:
                print(' ' * lm + stack.get_filename())

                # Adds the keyword 'FLATCORR' to the image header to indicate that
                # flat field correction has been done
                header = stack.header
                header['FLATCORR'] = (True, 'Flat field corrected')
                hist = 'Flat field {} subtracted at {}'.format(masterflat.get_filename(), datetime.datetime.now())
                header.add_history(hist, before='HEAD')
                # Adds the filename used and time of flat field correction to header HISTORY

                try:
                    # flat field division
                    stack.data = stack.data / mf_data        # NOTE: avoiding augmented assignment due to type errors
                except Exception as err:
                    print('FAIL ' * 10)
                    print(err)
                    embed()

        self.label = 'science frames (flat fielded)'
