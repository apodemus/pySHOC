# -*- coding: utf-8 -*-

import re
import os
import copy
import logging
import warnings
import operator
import itertools as itt
from pathlib import Path
from datetime import date as Date

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii

from tsa import fold
from tsa.smoothing import smoother
from tsa.spectral import Spectral
from grafico.ts import TSplotter
from grafico.multitab import MplMultiTab
from fastfits import quickheader
from recipes.list import flatten
from recipes.iter import (interleave, groupmore, first_true_idx,
                          roundrobin)
# from grafico.interactive import PointSelector
# from tsa.tfr import TimeFrequencyRepresentation as TFR
# from tsa.outliers import WindowOutlierDetection, generalizedESD

from pySHOC.io import Conversion as convert

from IPython import embed
# import decor
# profiler = decor.profile.profile()
# from ansi.core import banner

tsplt = TSplotter()             #TODO: attach to PhotResult class


def quadadd(a, axis):
    return np.sqrt(np.square(a).sum(axis))

def as_flux(mag, mag_std=None, mag0=0):
    """Convert magnitudes to Fluxes"""
    flux = 10 ** ((mag0 - mag) / 2.5)
    if mag_std is None:
        return flux

    fluxerr = mag_std * (np.log(10) / 2.5) * flux
    return flux, fluxerr

def as_magnitude(flux, flux_std=None, mag0=0):
    """Convert fluxes to magnitudes"""
    mag = -2.5 * np.log10(flux) + mag0
    mag_std = None
    return mag, mag_std

def extract_2Dfloat(table, colid, n):
    colix = range(1, n + 1)
    colnames = map(('%s{}' % colid).format, colix)
    table = table[tuple(colnames)]
    return table.as_array().view(float).reshape(len(table), -1)


def _make_ma(data):
    if data is None:
        data = []

    if np.ndim(np.squeeze(data)) > 1:
        raise ValueError('Signal should be at most 1 dimensional, not %iD'
                         % np.ndim(data))

    mask = False
    if np.ma.isMA(data):
        mask = data.mask
    mask |= (np.isnan(data) | np.isinf(data))

    data = np.ma.asarray(data)
    data.mask = mask
    return data


def read(filename):
    from recipes.io import read_data_from_file
    header = read_data_from_file(filename, N=3)  # read first 3 lines
    (metaline, starline, headline) = header

    fill_value = -99
    data = np.genfromtxt(str(filename),
                         skip_header=3,
                         usemask=True)
    data.mask |= (data == fill_value)

    return header, data


def load_times(timefile):
    """read timestamps from file."""
    from recipes.io import read_file_line

    hline = read_file_line(timefile, 0)  # read header line
    colnames = hline.strip('\n #').lower().split()  # extract column names (lower case)

    t = np.genfromtxt(str(timefile),
                         dtype=None,
                         names=colnames,
                         skip_header=1,
                         usecols=range(1, len(colnames)))
    # if len(t)
    # TODO: check that it is consistent with data
    return t

def get_name_date(fitsfile=None):
    if fitsfile and os.path.exists(fitsfile):
        logging.info('Getting info from %r', os.path.split(fitsfile)[1])
        header = quickheader(fitsfile)
        name = header['OBJECT']
        size = header['NAXIS3']
        try:
            date = header['DATE-OBS'].split('T')[0]
        except KeyError:
            ds = header['DATE']  # fits generation date and time
            date = ds.split('T')[0].strip('\n')  # date string e.g. 2013-06-12

        return name, date, size
    return None, None, None



class SelfAwareContainer(object):
    _skip_init = False

    def __new__(cls, *args):
        # this is here to handle initializing the object from an already existing
        # istance of the class
        if len(args) and isinstance(args[0], cls):
            instance = args[0]
            instance._skip_init = True
            return instance
        else:
            return super().__new__(cls)


# ****************************************************************************************************
class PhotRun(SelfAwareContainer):
    """ Class for containing / merging / plotting PhotResult objects."""
    # TODO: inherit from OrderedDict / list ??
    # TODO: merge by: target; date; etc...
    # TODO: multiprocess generic methods

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self, filenames, databases=None, coordinates=None,
                 targets=None, target_name='', outpath=None):

        if self._skip_init:
            # self._skip_init = False
            logging.debug('PhotRun Skipping init')
            return

        self.target_name = target_name

        # FIXME: check if filenames are non-str sequence
        if len(filenames):
            itr = itt.zip_longest(filenames,
                                  databases or [],  # magnitudes database files
                                  coordinates or [],
                                  targets or [])
            self.results = [PhotResult(fn, db, cf, ix)
                            for (fn, db, cf, ix) in itr]
        else:
            raise ValueError('No filenames')


    @property
    def fitsfiles(self):
        return self.attrgetter('fitsfile')

    @property
    def basenames(self):
        return self.attrgetter('basename')

    @property
    def datafiles(self):
        return self.attrgetter('datafile')

    def attrgetter(self, *attrs):
        """fetch attributes from the inner class"""
        return list(map(operator.attrgetter(*attrs), self.results))

    # TODO: attrsetter

    def set_targets(self, targets):
        # assert len(targets) == len(self)
        tmp = np.zeros(len(self))
        tmp[:] = targets

        for obs, trg in zip(self.results, tmp):
            obs.target = int(trg)

    def get_targets(self):
        return self.attrgetter('target')

    targets = property(get_targets, set_targets)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __getitem__(self, key):
        """Can be indexed numerically, or by corresponding filename / basename."""

        if isinstance(key, str):
            if key.endswith('.fits'):
                key = key.replace('.fits', '')
            key = self.basenames.index(key)
            return self.results[key]

        elif isinstance(key, slice):
            return self.__class__(self.results[key])
        else:
            return self.results[key]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __len__(self):
        return len(self.results)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __str__(self):
        return '%s of %i observation%s: %s' \
               % (self.__class__.__name__,
                  len(self),
                  's' * bool(len(self) - 1),
                  '| '.join(self.basenames))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # def __repr__(self):
    #     return str(self)

    def remove_empty(self):
        self.results = [r for r in self.results if r.datafile]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def get_outname(self, with_name=True, extension='txt', sep='.'):  # TODO: MERGE WITH gen_filename??
        name = ''
        if with_name:
            name = self.target_name.replace(' ', '') + sep

        date_str = self.date.replace('-', '')
        outname = '{}{}.{}'.format(name, date_str, extension)
        return os.path.join(self.data_path, outname)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def add_cube(self, filename, Nstars=None):
        """add a cube to the run."""
        Nstars = Nstars if Nstars else self.nstars
        basename = filename.replace('.fits', '')
        filename = os.path.join(self.data_path, filename)

        self.filenames.append(filename)
        self.basenames.append(basename)
        self.results.append(PhotResult(filename, Nstars))

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # def get_date(self, fn):
        # ds = pyfits.getval( self.template, 'date' )                #fits generation date and time
        # self.date = ds.split('T')[0].strip('\n')                         #date string e.g. 2013-06-12

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def get_time(self, tkw):
        # if not tkw in self.Tkw:
        # raise ValueError( '{} is not an valid time format.  Choose from {}'.format(tkw, self.Tkw) )
        # else:
        t = np.concatenate([getattr(cube, tkw) for cube in self])
        return t

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # @decor.expose.args()
    def conjoin(self):
        """Join cubes together to form contiguous data block."""
        from numpy.lib.recfunctions import stack_arrays

        # concatenate data
        npoints = self.attrgetter('npoints')
        nstars = max(self.attrgetter('nstars'))
        naps = self.attrgetter('naps')
        assert len(set(naps)) == 1
        naps = naps[0]

        data = np.ma.empty((sum(npoints), nstars, naps))
        data.mask = True
        std = np.ma.empty((sum(npoints), nstars, naps))
        std.mask = True
        cum = np.cumsum(npoints)

        try:
            for i, obs in enumerate(self.results):
                end = cum[i]
                start = end - obs.npoints
                # FIXME:  match stars!
                order = obs.reorder
                data[start:end, :obs.nstars] = obs.data[:, order]
                std[start:end, :obs.nstars] = obs.std[:, order]
        except Exception as err:
            print(str(err))
            embed()
            raise

        new = PhotResult()
        new.data = data
        new.std = std
        new.timedata = stack_arrays(self.attrgetter('timedata'))
        new.target = 0  # since we re-ordered everything
        new.target_name = self.target_name
        new.date = self[0].date
        new.fix_ut()

        return new

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def compute_ls(self, tkw='lmst', **kw):
        t = self.get_time(tkw)

        for star in self.stars:
            star.compute_ls(t, **kw)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def plot_lcs(self, **kw):

        figs = []
        for obs in self:
            fig, lines = obs.plot_lc(**kw)
            figs.append(fig)

        ui = MplMultiTab(figures=figs, labels=self.basenames)
        ui.show()

        return ui

    # def plot_phased(self, offsets):



# ****************************************************************************************************


# from .lc import LightCurve
# TODO: redesign so that methods like diff return a LightCurve object

# ****************************************************************************************************
class Star(object):

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def running_stats(self, nwindow, center=True, which='clipped'):
        import pandas as pd

        # first reflect the edges of the data array
        # if center is True, the window will be centered on the data point -
        # i.e. data point preceding and following the current data point will
        # be used to calculate the statistics (mean & var) else right window
        # edge is aligned with data points - i.e. only the preceding values are
        # used. The code below ensures that the resultant array will have the
        # same dimension as the input array

        x = self.data

        if center:
            div, mod = divmod(nwindow, 2)
            if mod:  # i.e. odd window length
                pl, ph = div, div + 1
            else:  # even window len
                pl = ph = div

            s = np.ma.concatenate(
                [x[pl:0:-1], x, x[-1:-ph:-1]])  # pad data array with reflection of edges on both sides
            iu = -ph + 1

        else:
            pl = nwindow - 1
            s = np.ma.concatenate([x[pl:0:-1], x])  # pad data array with reflection of the starting edge
            iu = len(s)

        s[s.mask] = np.nan
        max_nan_frac = 0.5  # maximum fraction of invalid values (nans) of window that will still yield a result
        mp = int(nwindow * (1 - max_nan_frac))
        self.median = pd.rolling_median(s, nwindow, center=center, min_periods=mp)[pl:iu]
        self.var = pd.rolling_var(s, nwindow, center=center, min_periods=mp)[pl:iu]

        # return med, var

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def plot_clippings(self, *args):
        # TODO:  plot clippings from outliers.py
        ax, t, nwindow = args
        self.ax_clp = ax
        med = self.median
        std = np.sqrt(self.var)
        lc = self.data
        threshold = self.sig_thresh

        ax.plot(t, self.clipped, 'g.', ms=2.5, label='data')
        ax.plot(t[self.clipped.mask], lc[self.clipped.mask], 'x', mfc='None', mec='r', mew=1, label='clipped')

        # print( 'top', len(top), 'bottom', len(bottom), 't', len(t[st:end]) )
        sigma_label = r'{}$\sigma$ ($N_w={}$)'.format(threshold, nwindow)
        median_label = r'median ($N_w={}$)'.format(nwindow)
        ax.plot(t, med + threshold * std, '0.6')
        ax.plot(t, med - threshold * std, '0.6', label=sigma_label)
        ax.plot(t, med, 'm-', label=median_label)

        # clp = sigma_clip( lcr, 3 )
        # ax.plot( t[clp.mask], lcr[clp.mask], 'ko', mfc='None', mec='k', mew=1.75, ms=10 )
        # m, ptp = np.mean(med), np.ptp(med)
        # ax.set_ylim( m-3*ptp, m+3*ptp )

        white_frac = 0.025
        xl, xu, xd = t.min(), t.max(), t.ptp()
        ax.set_xlim(xl - white_frac * xd, xu + white_frac * xd)

        ax.set_title(self.name)
        ax.invert_yaxis()
        ax.legend(loc='best')

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def smooth(self, nwindow, window='hanning', fill='mean'):
        # data = self.clipped[~self.clipped.mask]
        self.smoothed = smoother(self.clipped, nwindow, window, fill)
        return self.smoothed

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def argmin(self):
        return np.argmin(self.counts())

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def compute_ls(self, t, signal, **kw):
        print('Doing Lomb Scargle for star {}'.format(self.name))
        """Do a LS spectrum on light curve of star.
        If 'split' is a number, split the sequence into that number of roughly equal portions.
        If split is a list, split the array according to the indeces in that list."""
        return Spectral(t, signal, **kw)



class NullPath(type(Path())):
    def exists(self):
        return False
    def with_suffix(self, _):
        return


class AutoFileToPath(object):
    # NOTE: this is probably a huge anti-pattern
    def __getattr__(self, item):
        if item.endswith('path'):
            key = item.replace('path', 'file')
            try:
                filename = object.__getattribute__(self, key)
            except AttributeError as err:
                msg = str(err).replace('file', 'path')
                raise AttributeError(msg) from err

            if filename:
                return Path(filename)
            else:
                return NullPath()

        return object.__getattribute__(self, item)


def _file_checker(filename, what=None, borks=True):
    if not exists(filename):
        s = ' for %s' % what if what else ''
        msg = 'Nothing to load at %r. Please give valid filename%s.'
        if borks:
            raise ValueError(msg % (filename, s))
        else:
            logging.warning(msg, filename, s)
            return
    return str(filename)


def exists(filename):
    return filename and os.path.exists(str(filename))

def first_exists(*files):
    return next(filter(exists, files), None)

# ****************************************************************************************************
class PhotResult(AutoFileToPath):
    """
    Data container for photometry results.  Datna stored internally as
    array of fluxes (npoints, nstars, naps)
    """
    # TODO: some elements here are FileManagement, some are Plotting
    # TODO: split classes to avoid god-objects
    # TODO: integrate with shocObs

    # TODO:  STORE/LOAD data AS FITS.
    # TODO:  PICKLE / json?
    #TODO: maybe autoload class attribute that enable automatic loading with
    # self.datafile = 'some/file/that/exists'

    #TODO: if coordinates are given, we can order the stars logically so that
    # stars correspond across cubes.
    # sorting methods? checkout pandas / astropy.table

    _skip_init = False

    _flx_sort = True # will order data according to brightness
    _default_cmap = 'nipy_spectral'
    _label_fmt = 'C%i'
    _title_fmt = 'Light curve: {s.fitspath.name}'

    # Available time formats    # TODO: TIME UNITS!!
    Tkw = ['jd', 'bjd', 'utc', 'utsec', 'lmst']
    Tunits = ['', '', '', 's', ]

    @property
    def basename(self):
        if self.fitsfile:
            return self.fitspath.stem

    # @property
    # def basepath(self):
    #     if self.fitsfile:
    #         return self.fitspath.parent / self.basename

    # TODO:
    # @property
    # def lightcurves

    def globsearch(self, ext):
        if self.fitsfile:
            fp = self.fitspath
            pattern = '%s*.%s' % (fp.stem, ext)
            yield from fp.parent.glob(pattern)
        return

    def discover_files(self, *exts):
        blob = []
        for ext in exts:
            blob.append(reversed(list(self.globsearch(ext))))
        yield from roundrobin(*blob)


    def __new__(cls, *args, **kws):
        # this is here to handle initializing the object from an already existing
        # istance of the class
        if len(args) > 0 and isinstance(args[0], cls):
            instance = args[0]
            instance._skip_init = True
            return instance
        else:
            return super().__new__(cls)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self, fitsfile=None, datafile=None, timefile=None,
                 coordfile=None, target=None, target_name=None, ephem=None):
        """

        Parameters
        ----------
        fitsfile
        datafile
        coordfile
        target
            if given, this star will be plotted over all others
        """

        if self._skip_init:
            logging.debug('PhotResult Skipping init')
            return
            # this happens if the first argument to the initializer is already an
            # instance of the class

        self.fitsfile = str(fitsfile) if fitsfile else None
        self.datafile = datafile
        self.coordfile = coordfile

        name_guess = self.fitspath.with_suffix('.time')
        self.timefile = first_exists(timefile, name_guess)

        #optional ephemeris
        # self.ephem = ephem

        # load info from fitsfile if avaliable
        self.target_name, self.date, self.size = get_name_date(self.fitsfile)
        # FIXME: self.date should be self.startdate ???

        # store data internally as masked array (Nframes, Nstars, Naps)
        # init null data containers
        self.data = np.ma.empty((0, 0, 0))
        self.std = np.empty((0,0,0))
        self.timedata = np.empty(0)

        # load timing data if available
        if self.timefile:
            self.timedata = load_times(self.timefile)

        # load data if available
        if exists(self.datafile):
            self.load_data()
            # self.load_times()


        # NOTE: this is restrictive in that it fixes the Nstars, target, ref across
        # all cubes.  Think of a way of dynamically creating the Stars class
        self._target = target

        # for name in ('fits', 'data', 'time', 'coord'):
        #     if not getattr(self, '%spath' % name).exists():
        #         logging.warning('Non-existant %s file: %r', name,
        #                         getattr(self, '%sfile' %name))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __getitem__(self, key):
        """Can be indexed numerically, or by using 'target' or 'ref' strings."""
        raise NotImplementedError
        if isinstance(key, (int, np.integer, slice)):
            #TODO: return lightcurve object
            pass



        if isinstance(key, str):
            key = key.lower()
            if key.startswith('t'):
                return self[self.target]
            # if key.startswith('r') or key == 'c0':
            #     return self.stars[self._ref]
            # if key.startswith('c'):
            #     return self.stars[self._others[int(key.lstrip('c')) - 1]]

        else:
            raise KeyError

    # def __str__(self):
    #     'TODO'

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __len__(self):
        # Number of data points *not* stars
        return self.data.shape[0]

    @property
    def npoints(self):
        return self.data.shape[0]

    @property
    def nstars(self):
        return self.data.shape[1]

    @property
    def naps(self):
        return self.data.shape[-1]


    def get_target(self):
        return self._target

    def set_target(self, target):
        self._target = target
        #self.stars.set_names(target, self.target_name)
        #self.stars.set_colours(target)

    target = property(get_target, set_target)

    @property
    def others(self):
        return np.setdiff1d(np.arange(self.nstars), self.target)

    @property
    def reorder(self):
        if self._flx_sort and self.data.size:
            order = list(self.data.mean(0)[:, -1].argsort()[::-1])
        else:
            order = list(range(self.nstars))

        if self.target is not None:
            order.insert(0, order.pop(order.index(self.target)))

        return order

    @property
    def labels(self):
        return list(self.gen_labels(self.target, self.target_name))


    def gen_labels(self, target=None, target_name=None, fmt=None):
        fmt = fmt or self._label_fmt
        count = 0
        for i in range(self.nstars):
            if i == target:
                yield target_name
            else:
                yield fmt % count
                count += 1

    # def gen_colours(self, unique=True, **kws):
    #     """Assign unique colours to stars for plotting"""
    #     from matplotlib import rcParams
    #     cyc = rcParams['axes.prop_cycle']
    #
    #     if unique:
    #         if len(cyc) < len(self):
    #             # Use cmap to make unique colours
    #             cmap = kws.get('cmap', self._default_cmap)
    #             cm = plt.get_cmap(cmap)
    #             colours = cm(np.linspace(0, 1, len(self)))
    #             return colours
    #
    #     return [p['color'] for p in list(cyc[:len(self)])]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def load_data(self, filename=None, coordfile=None, **kws):
        # if (filename is None) and self.datapath.exists():
        #     filename = self.datafile

        filename = _file_checker(filename or self.datafile, 'data')
        logging.info('Loading data for %r', os.path.basename(filename))

        # TODO: better way of identifying file structure
        if filename.endswith('lc'):
            self.load_from_lc(filename, **kws)
        elif filename.endswith('npz'):
            self.load_from_npz(filename, **kws)
        elif filename.endswith('tbl'):
            self.load_from_tbl(filename, **kws)
        elif 'apcor' in filename:
            coordfile = _file_checker(coordfile or self.coordfile)
            self.load_from_apcor(filename, coordfile, **kws)
        else:
            self.load_from_mag(filename)

        self.datafile = filename

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def load_from_npz(self, filename):

        try:
            # lz = np.load(str(filename))  # psf_flux, psf_fwhm, problematic
            # self.data = lz['flux_ap']  # shape (Nframes, Nstars, Naps)
            # coo = lz['coords']      #shape (Nframes, Nstars, 2)

            lz = np.load(str(filename))
            self.timedata = lz['t']
            self.data = np.ma.array(lz['data'], mask=lz['data_mask'])
            self.std = np.ma.array(lz['std'], mask=lz['std_mask'])

            #todo: target, name etc..
        except Exception as orr:
            embed()
            raise

        self.load_times()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def load_from_lc(self, filename, mag2flux=False, mag0=0):
        """load from ascii lc"""
        (metaline, starline, headline), data = read(filename)

        data_start_col = first_true_idx(headline.split('\t'),
                                        lambda s: s.startswith('MAG'))
        newdata = fold.fold(data.T[data_start_col:], 2)
        signals, std = np.swapaxes(newdata, 1, 0)

        if mag2flux:
            signals, std = as_flux(signals, std, mag0=mag0)

        # upcast for consistency
        self.data = signals[..., None]
        self.std = std[..., None]

        # extract headers
        # starheads = list(filter(None, map(str.strip, starline.strip('#').split('  '))))
        # target = first_false_idx(starheads, lambda s: re.match('C\d|ref', s))  # [2::2]
        # self.target = target
        # colheads = kill_brackets(headline).strip('#').lower().split('\t')
        # matcher = re.compile('C(\d|ref)')
        # nrs = sorted(flatten(map(matcher.findall, starheads)))
        # start = nrs[0]
        # ref = starheads.index('C%s' % start)


        # timefields = list(map( lambda s: rreplace(s, '() ', '').lower(),
        # filter( lambda s: not s.strip() in ('', 'flux','err'),
        # headline.strip('#').lower().split('  ') ) ))
        # dtype = list(itt.zip_longest( timefields, [float], fillvalue=float ))
        # embed()
        timefields = 'utsec', 'utdate'
        formats = float, 'U30'
        # data[:,:len(timefields)]
        self.timedata = np.recarray(len(data),
                                    dtype=list(zip(timefields, formats)))
        self.timedata['utsec'] = data[:, 0]
        self.timedata['utdate'] = self.date

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # @profiler.histogram
    def _load_daophot(self, filename):
        """load data from IRAF daophot multi-aperture ascii database."""
        reader = ascii.Daophot()
        table = reader.read(filename)

        mag0 = float(table.meta['keywords']['ZMAG']['value'])
        # Naps = len(reader.header.aperture_values)
        return table, mag0


    def load_from_tbl(self, filename, mag0=0):
        tbl = ascii.read(filename)
        self._load_from_table(tbl, mag0)


    def load_from_mag(self, filename, save='npz'):
        # TODO: default format as class attr

        filename = str(filename)
        filepath = Path(filename)

        table, mag0 = self._load_daophot(filename)
        self._load_from_table(table, mag0)

        # TODO: move to Saver ?
        outpath = filepath.with_suffix('.%s' % save)
        if save == 'npz':
            logging.info('Saving as: %s', outpath.name)
            np.savez(str(outpath),
                     t=self.timedata, data=self.data, std=self.std)
            #FIXME: NotImplementedError: MaskedArray.tofile() not implemented yet.

        if save == 'tbl':
            # finally write the table in a more accesible ascii form
            logging.info('Saving as: %s', outpath.name)
            table.write(str(outpath),
                        format='ascii.fast_commented_header',
                        overwrite=True)


    def _load_from_table(self, table, mag0=0):

        # extract magnitudes and their uncertainty # shape (n, naps)
        naps = int(re.search('(\d+)', table.colnames[-1]).group())
        mags = extract_2Dfloat(table, 'MAG', naps)
        magstd = extract_2Dfloat(table, 'MERR', naps)

        #TODO: optionally show some stats ala table.info

        # HACK Daophot databases are incredible annoying things. Since some
        # of the stars are sometimes dropped and moreover the filenames
        # are cut by the way-too-small fixed column width of (23 characters)
        ids, counts = np.unique(table['LID'], return_counts=True)
        nstars = len(ids)
        any_dropped = len(np.unique(counts)) > 1
        if any_dropped:
            assert any_dropped, 'Database error'

        # convert to flux
        flx, std = as_flux(mags, magstd, mag0)
        self.data = flx.reshape(-1, nstars, naps)
        self.std = std.reshape(-1, nstars, naps)
        self.load_times()

    def recommend_ap(self):
        # for multiaperture dbs. determine aperture with highest mean SNR
        snr = self.data / self.std
        ixap = int(np.ceil(snr.argmax(-1).mean()))
        return ixap

    def select_best_ap(self):
        ixap = self.recommend_ap()
        data = self.data[..., ixap:ixap+1]
        std = self.std[..., ixap:ixap+1]
        return data, std

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def load_from_apcor(self, filename, coofile, splitfile=None):
        """load data from aperture correction ascii database."""

        # self.size = 74000

        convert_indef = lambda s: np.nan if s == b'INDEF' else float(s)
        converters = {7: convert_indef,
                      8: convert_indef}
        bigdata = np.genfromtxt(filename,
                                dtype=None,
                                skip_header=2,
                                names=True,
                                converters=converters)

        kcoords = np.atleast_2d(np.loadtxt(coofile, usecols=(0, 1)))
        coords = np.array((bigdata['Xcenter'], bigdata['Ycenter'])).T

        try:
            get_fileno = lambda fn: int(fn.strip(']').rsplit(',', 1)[-1])
            fileno = np.fromiter(map(get_fileno, bigdata['Image'].astype(str)), int)
        except ValueError:
            try:  # Older data with different naming convention
                get_fileno = lambda fn: int(fn.rstrip('.fits').rsplit('.', 1)[-1])
                fileno = np.fromiter(map(get_fileno, bigdata['Image'].astype(str)), int)
            except ValueError:
                pass

            # WARNING:  This will only work if there are no dropped frames! (IRAF SUX!!)
            nstars, ndata = len(kcoords), len(bigdata)
            nframes, remainder = divmod(ndata, nstars)
            if remainder:
                raise ValueError(('Dropped Frames in {}! '
                                  'Number of data lines ({}) does not equally divide '
                                  'by number of stars {}').format(filename, ndata, nstars))

            fileno = np.mgrid[:nstars, :nframes][1].ravel()


        # group the stars by the closest matching coordinates in the coordinate file list.
        starmatcher = lambda coo: np.sqrt(np.square(kcoords - coo).sum(1)).argmin()
        for starid, (coo, ix) in groupmore(starmatcher, coords, range(len(bigdata))):
            ix = np.array(ix)
            stardata = bigdata[ix]

            data, err = np.ma.zeros((2, self.size))
            mask = np.ones(self.size, bool)

            try:
                z = fileno[ix] - 1
                data[z], err[z] = as_flux(bigdata[ix]['Mag'],
                                          bigdata[ix]['Merr'], mag0=25)
            except:
                print('ERROR CAUGHT!')
                embed()
            # bigdata[ix]['Exptime']
            mask[z] = False
            data.mask = mask


        self.load_times()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def load_times(self, filename=None):
        filename = _file_checker(filename or self.timefile, 'time')
        logging.info('Loading timing data from %r', os.path.basename(filename))
        self.timedata = load_times(filename)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def fix_ut(self, set_=True):
        """fix UTSEC to be relative to midnight on the starting date of the first frame"""
        t = self.timedata['utsec'].copy()
        dates = self.timedata['utdate']
        dateset = set(dates)
        delim = b'-' if isinstance(next(iter(dateset)), (bytes, np.bytes_)) else '-'
        to_datetime = lambda d: Date(*map(int, d.split(delim)))  # .decode()
        if len(dateset) > 1:
            warnings.warn('Data spans multiple UTC dates! Adjusting time origin.')

            d0 = to_datetime(dates[0])
            for i, d in enumerate(dateset - {dates[0]}):
                tshift = (to_datetime(d) - d0).total_seconds()
                t[dates == d] += tshift
                if set_:
                    self.timedata['utsec'] = t

        return t

    def add_phase_info(self, emphem):
        import numpy.lib.recfunctions as rfn
        ph = emphem.phase(self.timedata['bjd'])
        self.timedata = rfn.append_fields(self.timedata, ('phase', 'phaseMod1'), (ph, ph%1))


    def date_split(self):
        ''#TODO

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # def optimal_smoothing(self):
    #     """Determine the optimal amount of smoothing for differential photometry
    #     by minimising the variance of the differentail light curve for the given
    #     length of the smoothing window.  Note: This operation may be slow..."""
    #
    #     data = self.datablock('clipped')

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def compute_dl(self, mode='flux', **kws):

        if self.target is None:
            raise ValueError('Please set target star first')
            # TODO: # de - correlate somehow

        logging.info('Computing differential light curve..')

        # summed flux of all non-target stars
        flx = self.data#[:, self.target, :]
        flxRef = self.data[:, self.others, :].sum(1)[:, None]
        # compute std
        std = self.std#[:, self.target, :]
        stdRef = quadadd(self.std[:, self.others, :], 1)[:, None]

        # scale ref flux
        mu = flxRef.mean(0)
        flxScl = flxRef / mu
        stdScl = stdRef / mu

        flxRatio = flx / flxScl
        stdRatio = flxRatio * ((std / flx) - (stdScl / flxScl))

        # create new instance
        # FIXME: OO LightCurves needed!!
        new = self.__class__()
        new.data = flxRatio
        new.std = stdRatio
        new.timedata = self.timedata
        new.target = self.target
        new.target_name = self.target_name
        new.date = self.date
        return new


    def std_cuts(self, lower=0, upper=100):
        l = (lower > self.std) | (self.std > upper)
        self.data[l] = np.ma.masked


    # def mask_outliers(self):



    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # MultiAperture plot gui

    def plot_lc(self, tkw='utsec', ixap=None, **kw):

        if self.naps == 1:
            ixap = 0
        elif ixap is None:
            ixap = self.recommend_ap()
            logging.info('Choosing aperture %i (highest SNR)', ixap)

        # Reshape for plotting
        y = self.data[:, self.reorder, ixap].T
        e = self.std[:, self.reorder, ixap].T
        labels = np.array(self.labels)[self.reorder]
        # If target specified, re-order the array so we get a consistent
        # colour sequence

        logging.info('Plotting light curves for %s', self)
        mode = kw.pop('mode', 'flux')

        t = self.timedata[tkw]
        title = self._title_fmt.format(s=self)
        xlabel = tkw.upper()
        ylabel = mode.title() #'Instr. ' +
        axlabels = xlabel, ylabel

        # if kw.get('twinx'):
        timescales = {'utsec': 's',
                      'utc': 'h',
                      }  #TODO etc...

        fig, plots, *rest = tsplt(t, y, e,
                                  title=title,
                                  labels=labels,
                                  # colours=colours,
                                  axlabels=axlabels,
                                  timescale=timescales.get(tkw),
                                  start=self.date,
                                  **kw)


        # This will plot the target *over* the other light curves
        if self.target:
            plots.lines[self.target].set_zorder(10)

        # TODO: interactively switch to mag scale with key
        ax = fig.axes[0]
        if mode.lower() == 'mag':
            ax.invert_yaxis()

        if self.date:
            ax.text(0, ax.xaxis.labelpad,
                    'DATE: ' + self.date,
                    transform=ax.xaxis.label.get_transform())

        return fig, plots

    def save_npz(self, filename):
        filename = str(filename)
        logging.info('saving as: %s', filename)
        np.savez(filename,
                 t=self.timedata,
                 data=self.data.data, data_mask=self.data.mask,
                 std=self.std.data, std_mask=self.std.mask)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def save_lc(self, filename=None, which='raw', tkw=('utsec',), clobber=None):
        # TODO: MAYBE YOU CAN FIX THE ASTROPY TABLE TO ALIGN COLUMN NAMES WITH VALUES

        if filename is None:
            ext = '{}lc'.format(which[0])
            # self.target_name.replace(' ', '_'),
            filename = '.'.join([self.date.replace('-', ''),
                                 ext])
            filename = os.path.join(self.write_path, filename)

        if saver.clobber_check(filename, clobber):
            nstars = len(self.stars)

            # get time columns
            times = self.timedata[list(tkw)].view(float).reshape(-1, len(tkw))
            # get data columns
            datablock = np.ma.array(self.datablock(which, 1))
            output = datablock.T.reshape(datablock.shape[-1], -1)
            output = np.c_[times, output.filled(-99)]

            fmt = ('%-12.6f',) + ('%-9.3f',) * 2 * nstars  # , '%-18.9f'
            col_head = [''] + interleave(self.stars.names, [''] * nstars)
            col_head2 = ['UTC (sec)'] + ['Flux', 'Err'] * nstars  # , 'BJD'

            delimiter = ' '
            try:
                dbs = '; '.join(map(os.path.basename, self.databases))
            except AttributeError:
                dbs = self.database

            header0 = 'Fluxes for {} stars extracted from {}.'.format(nstars, dbs)
            header1 = saver.make_header_line(col_head, fmt, delimiter)
            header2 = saver.make_header_line(col_head2, fmt, delimiter)
            header = '\n'.join([header0, header1, header2])  # header0,

            np.savetxt(filename, output, header=header, fmt=fmt, delimiter=delimiter)
            print('Halleluja!', filename)
        else:
            print('Nothing written')

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def heteroskedacity(self, whichstars, nwindow, window='flat', tkw='lmst'):

        fig, ax = plt.subplots()

        for whichstar in whichstars:
            star = self[whichstar]
            t = getattr(self, tkw)
            ax.plot(t, star.var, star.colour, label=star.name)

        white_frac = 0.025
        xl, xu, xd = t.min(), t.max(), t.ptp()
        ax.set_xlim(xl - white_frac * xd, xu + white_frac * xd)

        ax.set_title('Heteroskedacity')
        ax.set_ylabel(r'Variance $\sigma^2 (N={})$'.format(nwindow))
        ax.set_xlabel(tkw.upper())
        ax.legend()


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def plot_spread(self, tkw='utsec'):
        """plot all aperture data as time series"""
        fig, axes = plt.subplots(2, 1, figsize=(18, 12),
                                 sharex=True,
                                 gridspec_kw=dict(hspace=0))

        t = self.timedata[tkw]

        print(t)

        for ax, star in zip(axes, self):
            sdata = get_spread(star, 1e4)
            naps = sdata.shape[0]
            labels = ['ap %d' % i for i in range(naps)]
            tsplt(t, sdata, ax=ax, labels=labels)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_spread(star, sfactor):
    naps = star._data.shape[1]
    return (star._data + np.arange(naps) * sfactor).T





class Saver():
    @staticmethod
    def make_header_line(info, fmt, delimiter):
        import re
        matcher = re.compile('%-?(\d{1,2})')
        padwidths = [int(matcher.match(f).groups()[0]) for f in fmt]
        padwidths[0] -= 2
        colheads = [s.ljust(p) for s, p in zip(info, padwidths)]
        return delimiter.join(colheads)

    @staticmethod
    def clobber_check(filename, clobber):
        if os.path.exists(filename):
            if clobber is None:
                msg = 'A file named {} already exists! Overwrite ([y]/n)?? '.format(filename)
                return InputLoop.str(msg, 'y', convert=convert.yn2TF)
            return clobber
        else:
            return True

    def __call__(self, func, filename, data, clobber=None, **kw):
        clobber = self.clobber_check(filename)
        if clobber:
            np.savetxt(filename, data, **kw)
        else:
            print('Nothing written')


saver = Saver()






# ****************************************************************************************************###################################################################
# import cProfile as cpr
if __name__ == '__main__':
    import os, sys, argparse
    from recipes.io import iocheck, parse
    from recipes.misc import is_interactive, Unbuffered

    if is_interactive():
        # convert stdout to unbuffered
        _stdout = sys.stdout  # backup
        sys.stdout = Unbuffered(sys.stdout)
        # FIXME: causes AttributeError: 'IOStream' object has no attribute 'flush'
        # when using embed()
        # print('IN IPYTHON!!')

    # bar = ProgressBar()

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-d', '--dir', default=os.getcwd(), dest='dir',
                        help='The data directory. Defaults to current working directory.')
    parser.add_argument('-x', '--coords', dest='coo',
                        help='File containing star coordinates.')
    parser.add_argument('-m', '--mags', default='all.mags', dest='mags',
                        help='Database file containing star magnitudes.')
    parser.add_argument('-c', '--cubes', nargs='+', type=str,
                        help='Science data cubes to be processed.  Requires at least one argument.  Argument can be explicit list of files, a glob expression, or a txt list.')
    parser.add_argument('-w', '--write-to-file', action='store_true', default=False, dest='w2f',
                        help='Controls whether the script writes the light curves to file.')

    parser.add_argument('-t', '--target', default=0, type=int,
                        help='The position of the target star in the coordinate file.')
    parser.add_argument('-r', '--ref', type=int,
                        help='The position of the reference star in the coordinate file.')

    # parser.add_argument('-i', '--instrument', default='SHOC', nargs=None, help='Instrument. Switches behaviour for loading the data.')
    parser.add_argument('-l', '--image-list', default='all.split.txt', dest='ims',
                        help='File containing list of image fits files.')

    # Arguments for light curve / spectral analysis
    parser.add_argument('-dl', '--diff-phot', default=True, action='store_true', dest='dl',
                        help="Perform differential photometry?  The star with lowest variance will be used reference star ( Unless explicitly given via the 'r' argument.)")
    parser.add_argument('-ls', '--lomb-scargle', default=False, action='store_true', dest='ls',
                        help='Perform Lomb-Scargle periodogram on light curves?')
    args = parser.parse_args()

    path = iocheck(args.dir, os.path.exists, 1)
    path = os.path.abspath(path) + os.sep
    args.cubes = parse.to_list(args.cubes, os.path.exists,
                             path=path, raise_error=1)
    args.mags = parse.to_list(args.mags, os.path.exists,
                            include=('mag', 'lc'),
                            path=path, raise_error=1)
    args.coo = parse.to_list(args.coo, os.path.exists,
                           include='coo',
                           path=path, raise_error=0)

    # coords = np.loadtxt(args.coo, unpack=0, usecols=(0,1))
    # target = args.target	 #index of target    #WARN IF LARGER THAN LEN COORDS
    target_name, _, _ = get_name_date(args.cubes[0])
    run = PhotRun(args.cubes, args.mags, args.coo,
                  args.target, args.ref, target_name)
    run.data_path = path  # FIXME!
    # run.target = target

    for cube in run:
        # cube.load_data( args.instrument, args.mags )
        cube.read_path = cube.write_path = path  # FIXME!
        cube.load_data()

        # embed()

        if len(cube.stars) > 1:
            cube.ref = ref = cube.check_ref(args.ref)
            # FIXME:  TARGET AND REFERENCE CANNOT BE THE SAME!  CONJOIN WILL MERGE 2 STARS INTO THE SAME!!


            # cube.check_ref( args.ref )

            # differential light curves!                     #NOTE:  Find a method which can handle systematic offsets between cubes (TVM????)
            # cube.compute_dl( poly=1 )

    conjc = run.conjoin()

    # raise SystemExit

    if args.ls:

        for cube in run:
            fn = '{}.ls'.format(cube.basename)
            if os.path.exists(fn):
                cube.load_ls(fn)
            else:
                cube.compute_ls(set_ls=1)  # compute ls and save data_path

            if args.w2f:
                cube.save_ls()

        # for combined time series of entire run

        conjc.compute_ls(set_ls=1)

        if args.w2f:
            outname = run.get_outname(with_name=False, extension='lc')
            conjc.save_lc(outname)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plots setup
    import matplotlib as mpl

    # plt.close('all')

    # mpl.rc( 'figure', figsize=(18,8) )                                #set the figure size for this session
    mpl.rc('savefig', directory=path)
    # mpl.rc( 'figure.subplot', left=0.065, right=0.95, bottom=0.075, top=0.95 )    # the left, right, bottom, top of the subplots of the figure
    # mpl.rc( 'figure', dpi=600 )

    #######################################################################################################################
    # RAW LIGHT CURVE
    tkw = 'utsec'
    starplots = conjc.plot_lc(tkw=tkw,
                              mode='flux',
                              relative_time=True,
                              twinx='sexa')

    starplots.connect()

    # ps = PointSelector(starplots.draggables)
    # ps.connect()

    raise SystemExit

    if args.w2f:
        # outname = run.get_outname( with_name=False, extension='lc')
        conjc.save_lc()

    #######################################################################################################################
    # TODO: PLEASE STORE THIS AS A 3D RECARRAY / FITS??!
    testfile = Path(conjc.databases[0]).with_suffix('.clc')
    if testfile.exists():
        _, data = read(testfile)
        for i, d in enumerate(fold.fold(data.T[1:], 2)):
            conjc[i].clipped = d[0]

    else:
        # OUTLIER CLIPPING
        nwindow = 100
        overlap = '50%'
        for star in conjc:
            print('outliers', star)
            # NOTE: OUTLIER DETECTION DOES NOT TAKE ACCOUNT OF UNCERTAINTIES
            ix = WindowOutlierDetection(star.data,
                                        nwindow, overlap,
                                        generalizedESD,
                                        maxOLs=25, alpha=0.05)
            mask = np.zeros_like(star.data)
            mask[ix] = True
            star.clipped = np.ma.array(star.data, mask=mask)

    starplots = conjc.plot_lc(which='clipped',
                              tkw=tkw,
                              mode='flux',
                              show_masked='x',
                              relative_time=True,
                              twinx='sexa')
    starplots.connect()
    # TODO: Line selector

    if args.w2f:
        conjc.save_lc(which='clipped')

    raise SystemExit

    #######################################################################################################################
    # HETEROSKEDACITY
    # whichstars = range(Nstars)
    # conjc.heteroskedacity( whichstars, 100, tkw=tkw )


    #######################################################################################################################
    # DIFFERENTIAL LIGHT CURVE
    testfile = Path(conjc.databases[0]).with_suffix('.dlc')
    if testfile.exists():
        _, data = read(testfile)
        for i, d in enumerate(fold.fold(data.T[1:], 2)):
            conjc[i].dl = d[0]
    else:
        conjc.compute_dl(which='clipped', poly=1)  # smooth=50, poly=3

    # whichstars = [target, ref]
    starplots = conjc.plot_lc(which='diff',
                              tkw=tkw,
                              mode='flux',
                              twinx='sexa')
    starplots.connect()

    if args.w2f:
        # outname = run.get_outname( with_name=False, extension='dlc', which='diff')
        conjc.save_lc(which='diff')

        #######################################################################################################################
        # FILL DATA GAPS
        # for i in [target, ref]:
        # star = conjc[i]
        # ax = star.ax_clp
        # fill_mode = 'median'
        # opt = 20
        # Tfiller, Mfiller = Spectral.fill_gaps(t, star.dl, mode=fill_mode, option=opt, fill=False)

        # *star.filled, idx = Spectral.fill_gaps(t, star.dl, mode=fill_mode, option=opt, ret_idx=1)

        ##plot filler values
        # lbl = 'filled ({} {})'.format(fill_mode, opt)
        # M = np.polyval( polycoof, Tfiller )         #filled values projected onto the trend
        # Mde = M - np.ma.mean(M)                      #de-project filled points to original data
        # ax.plot(Tfiller, Mfiller+Mde, 'o', mew=1, mec='b', mfc='None', label=lbl)

        ##plot differential trend
        # if i==ref:
        # tfilled = star.filled[0]
        # trend = np.polyval( polycoof, tfilled )
        # ax.plot( tfilled, trend, 'r-', label='polyfit (n=%i)'%(len(polycoof)-1) )

        # ax.legend( loc='best' )
        # ax.grid()

        #######################################################################################################################
        # NEED TO EXPAND TO AM / AR / SSA MODELLING!!!
        # def plot_spec(self, whichstars=None, fig=None, **kw):
        # whichstars = whichstars if whichstars else [self.target, self.ref]

        # fig = plt.figure()
        # ax = DualAxes(fig, 1, 1, 1)

        # fig.add_subplot(ax)

        # for i in whichstars:
        # spec = self[i].spec
        # ax.plot( spec.frq, spec.power.mean(0), self[i].colour, label=self[i].name)

        ##ax.set_xscale('log')
        # ax.setup_ticks()
        # ax.set_xlabel('Frequency (Hz)')
        # ax.set_ylabel('RMS Power')
        # ax.legend( framealpha=0.25 )

        # ax.parasite.set_xlabel( 'Period (s)' )

        # ax.grid( b=True, which='both')
        ##ax2.grid( b=True, which='major', linestyle='--')

        # ax.text( 0, ax.xaxis.labelpad,
        # 'DATE: '+self.date,
        # transform=ax.xaxis.label.get_transform() )

        # return fig


    def save_spec(self, ext, clobber=None):
        nstars = 2
        filename = '.'.join([self.date.replace('-', ''),
                             ext])  # self.target_name.replace(' ', '_')
        filename = os.path.join(path, filename)

        if saver.clobber_check(filename, clobber):

            data = np.c_[self['t'].spec.frq,
                         self['t'].spec.power.mean(0),
                         self['r'].spec.power.mean(0)]  # np.nanmean()

            fmt = ('%-12.9f',) + ('%-12.9f',) * nstars
            col_head = [''] + [self['t'].name, self['r'].name]
            col_head2 = ('Frq',) + ('Power',) * nstars

            delimiter = ' '
            dbs = '; '.join(map(os.path.basename, self.databases))
            header0 = '{} for {} stars extracted from {}.'.format('RMS Power', nstars, dbs)
            header1 = saver.make_header_line(col_head, fmt, delimiter)
            header2 = saver.make_header_line(col_head2, fmt, delimiter)
            header = '\n'.join([header0, header1, header2])

            np.savetxt(filename, data, header=header, fmt=fmt, delimiter=delimiter)
            print('Halleluja!', filename)
        else:
            print('Nothing written')


    raise SystemExit

    print('Computing periodograms...')

    # Welch's periodogram
    conjc.compute_ls(which='clipped',
                     nwindow=1024,
                     noverlap='50%',
                     use='fft',
                     normalise='rms',
                     detrend=3,
                     timescale='s',
                     gaps=('mean', 25))

    save_spec(conjc, 'clc.welch.fft')
    fig = plot_spec(conjc)

    # Welch's periodogram
    conjc.compute_ls(which='clipped',
                     nwindow=1024,
                     noverlap='50%',
                     use='ls',
                     normalise=None,
                     detrend=3,
                     timescale='s')

    save_spec(conjc, 'clc.welch.ls')
    fig = plot_spec(conjc)
    # fig.savefig( )



    ######################################################################################################################
    t = conjc.timedata['utsec']
    flux = conjc['t'].clipped
    # flux = np.ma.masked_where( np.isnan(flux), flux )
    tfr = TFR(t, flux,
              nwindow=2 ** 10,
              overlap=-1,
              apodise=2 ** 11,
              use='fft',
              normalise='rms',
              detrend=3,
              timescale='s',
              gaps=('mean', 25))
    tfr.connect()

    #######################################################################################################################



    ######################################################################################################################

    ########################################################################################################################

    plt.show()





    # gix = detect_gaps( conjc.timedata['utsec'] ) +1
    # sst = np.split( conjc['target'].clipped, gix )
    # ssr = np.split( conjc['ref'].clipped, gix )
    # ts = np.split( conjc.timedata['utsec'], gix )
    # Sp = []
    # for t, ft, fr in zip(ts, sst, ssr):
    # specT = Spectral( t, ft, nwindow=2**10, overlap=0, detrend=0, normalise='rms' )
    # specR = Spectral( t, fr, nwindow=2**10, overlap=0, detrend=0, normalise='rms' )
    ##spec = Spectral( t, signal, nwindow=1024, overlap='50%', use='fft', normalise=0, detrend=3, timescale='s' )
    ##Sp.append( spec )
    # fig, (ax1, ax2) = plt.subplots(2,1)
    # ax1.plot( t, ft, label=conjc['target'].name, color=conjc['target'].colour )
    # ax1.plot( t, fr, label=conjc['ref'].name, color=conjc['ref'].colour )
    # ax1.set_xlabel( 't (s)' )
    # ax1.set_ylabel( 'Instr. Flux' )
    # ax1.grid()

    # ax2.plot( specT.frq, specT.power.mean(0), color=conjc['target'].colour )
    # ax2.plot( specT.frq, specR.power.mean(0), color=conjc['ref'].colour )
    # ax2.set_xlabel( 'Frequency (Hz)' )
    # ax2.set_ylabel( 'RMS Power' )
    # ax2.grid()
    ##ax2.plot( specR.frq,  )
    # ax1.legend( framealpha=0.25 )
