# -*- coding: utf-8 -*-
# std
from recipes.io import read_lines
import re
import numbers
import logging
import warnings
import operator
import itertools as itt
import os
import sys
import argparse
from pathlib import Path
from datetime import date as Date

# third-party
import numpy as np
import more_itertools as mit
import matplotlib.pyplot as plt
from tsa import ts
from astropy.io import ascii

# local
from mpl_multitab import MplMultiTab
from obstools.fastfits import quickheader
from recipes.introspect.utils import get_module_name
from recipes.iter import first_true_idx, group_more, interleave
from tsa import fold
from tsa.spectral import Spectral
from tsa.smoothing import smoother


# tsplt = TSplotter()  # TODO: attach to PhotResult class


# TODO: from tsa.ts import TimeSeries

# class LightCurve(TimeSeries):
#     pass


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


def read(filename):

    # (metaline, sourceline, headline) = header
    header = read_lines(filename, 3)  # read first 3 lines

    fill_value = -99
    data = np.genfromtxt(str(filename),
                         skip_header=3,
                         usemask=True)
    data.mask |= (data == fill_value)

    return header, data


def load_times(filename):
    """read timestamps from file."""

    # read header line
    hline = read_line(filename, 0)
    # extract column names (lower case)
    col_names = hline.strip('\n #').lower().split()

    t = np.genfromtxt(str(filename),
                      dtype=None,
                      names=col_names,
                      skip_header=1,
                      usecols=range(len(col_names)),
                      encoding=None
                      # this prevents np.VisibleDeprecationWarning
                      )
    # if len(t)
    # TODO: check that it is consistent with data
    return t.view(np.recarray)


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


# class LightCurve


class PhotRun(SelfAwareContainer):  # make dataclass ???
    """ Class for collecting / merging / plotting PhotResult (objects."""

    # TODO: inherit from OrderedDict / list ??
    # TODO: merge by: target; date; etc...
    # TODO: multiprocess generic methods
    # TODO: make picklable !!!!!!!!!
    # TODO: methods to plot on same axes... !!!

    # @classmethod
    # def from_list(cls, stack):
    #
    #     self.results = stack

    def __init__(self, filenames, databases=(), coordinates=(),
                 targets=(), target_name=None):
        # FIXME: dont need all this crap filenames / databases?

        if self._skip_init:
            # self._skip_init = False
            logging.debug('PhotRun Skipping init')
            return

        self.target_name = target_name

        # TODO: implement optional ephemeris that allows:
        #  - plot folded light curve etc...

        # FIXME: check if filenames are non-str sequence
        if len(filenames):
            itr = itt.zip_longest(filenames,
                                  databases,  # magnitudes database files
                                  coordinates,
                                  targets)

            self.results = [PhotResult(fn, db, None, cf, ix, target_name)
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

    # def inner(self):

    def set_targets(self, targets):
        # assert len(targets) == len(self)
        tmp = np.zeros(len(self))
        tmp[:] = targets

        for obs, trg in zip(self.results, tmp):
            obs.target = int(trg)

    def get_targets(self):
        return self.attrgetter('target')

    targets = property(get_targets, set_targets)

    def __getitem__(self, key):
        """
        Can be indexed numerically, or by corresponding filename / basename.
        """

        if isinstance(key, str):
            if key.endswith('.fits'):
                key = key.replace('.fits', '')
            key = self.basenames.index(key)
            return self.results[key]

        elif isinstance(key, slice):
            return self.__class__(self.results[key])
        else:
            return self.results[key]

    def __len__(self):
        return len(self.results)

    def __str__(self):
        return '%s of %i observation%s: %s' \
               % (self.__class__.__name__,
                  len(self),
                  's' * bool(len(self) - 1),
                  '| '.join(self.basenames))

    # def __repr__(self):
    #     return str(self)

    def remove_empty(self):
        self.results = [r for r in self.results if r.datafile]

    def get_outname(self, with_name=True, extension='txt',
                    sep='.'):
        # TODO: filenaming.FilenameGenerator?
        name = ''
        if with_name:
            name = self.target_name.replace(' ', '') + sep

        date_str = self.date.replace('-', '')
        outname = '{}{}.{}'.format(name, date_str, extension)
        return os.path.join(self.data_path, outname)

    def add_cube(self, filename, Nstars=None):
        """add a cube to the run."""
        Nstars = Nstars if Nstars else self.nstars
        basename = filename.replace('.fits', '')
        filename = os.path.join(self.data_path, filename)

        self.filenames.append(filename)
        self.basenames.append(basename)
        self.results.append(PhotResult(filename, Nstars))

        # def get_date(self, fn):
        # ds = pyfits.getval( self.template, 'date' )
        # #fits generation date and time
        # self.date = ds.split('T')[0].strip('\n')
        #       #date string e.g. 2013-06-12

    def get_time(self, tkw):
        # if not tkw in self.Tkw:
        # raise ValueError( '{} is not an valid time format.  Choose from {}'.format(tkw, self.Tkw) )
        # else:
        t = np.concatenate([getattr(cube, tkw) for cube in self])
        return t

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

        for i, obs in enumerate(self.results):
            end = cum[i]
            start = end - obs.npoints
            # FIXME:  match stars!  cross_calibrate?
            order = obs.reorder
            data[start:end, :obs.nstars] = obs.data[:, order]
            std[start:end, :obs.nstars] = obs.std[:, order]

        new = PhotResult()
        new.data = data
        new.std = std
        # convert timedata to recarray for quicker attr access
        new.t = stack_arrays(self.attrgetter('t')).view(np.recarray)
        new.target = 0  # since we re-ordered everything
        new.target_name = self.target_name
        new.date = self[0].date
        new.fix_ut()

        return new

    # def split_by_date(self):
    #     ''  # TODO

    # TODO:
    def phase_select(self, eph, min_phase=0, max_phase=1, jd=True, bjd=False):
        # Split light curves by cycle since we have the ephemeris

        # prs = np.size(phase_range)
        # if prs == 0:
        #     min_phase, max_phase = (0, 1)
        # elif prs == 1:
        #     # interpret as maximal phase
        #     max_phase = float(phase_range[0])
        #     min_phase = 0.
        # elif prs == 2:
        #     min_phase, max_phase = sorted(phase_range)
        # else:
        #     raise ValueError('phase range not understood')

        # checks
        for arg in (min_phase, max_phase):
            assert isinstance(arg, numbers.Real)
            assert arg > 0

        t, y, e = [], [], []

        # embed()
        # print(min_phase, max_phase)
        if bjd:
            jd = False
        tkw = 'jd' if jd else 'bjd'

        j = 0
        n_total = n_select = 0
        for lc in self:
            # TODO: for all stars
            jd, data, std = lc.t[tkw], lc.data[lc.target], lc.std[lc.target]
            ph = eph.phase(jd)

            # adjust zero point for light curve with negative phase
            if ph[0] < 0:
                ph -= np.floor(ph[0])

            # iterate through the segments
            ph_range = np.floor(ph[[0, -1]] + [0, 1])
            # print('ph', ph[[0, -1]])
            # print(np.arange(*ph_range))
            n_select = 0
            n_total += len(ph)
            for zero_point in np.arange(*ph_range):
                ph0 = ph - zero_point
                # todo could be faster if array already  split?
                selected = (ph0 >= min_phase) & (ph0 <= max_phase)
                if selected.any():
                    # TODO: better: itemized print for each LightCurve being
                    #  split / created
                    n_points = selected.sum()
                    n_select += n_points

                    print('E{: <2d}: Splitting: {:d} / {:d}'.format(
                        int(zero_point), n_points, len(selected)))
                    #
                    y.append(data[selected])
                    t.append(jd[selected])
                    e.append(std[selected])
                    j += 1
                else:
                    print('E%i: nothing selected' % zero_point)

        print(f'Selected {j} segments: {n_select} / {n_total}')
        # TODO: return list of LightCurve / MultiVariateTimeSeries objects
        return t, y, e

    # def phase_fold(self, min_phase=0, max_phase=1):

    # merge and sort

    # def compute_ls(self, tkw='lmst', **kw):
    #     t = self.get_time(tkw)
    #
    #     for star in self.stars:
    #         star.compute_ls(t, **kw)

    def plot_lcs(self, **kw):

        figs = []
        for obs in self:
            tsp = obs.plot_lc(**kw)
            figs.append(tsp.fig)

        ui = MplMultiTab(figures=figs, labels=self.basenames)
        ui.show()

        return ui

    # def plot_phased(self, offsets):


# ****************************************************************************************************


# from .lc import LightCurve
# TODO: redesign so that methods like diff return a LightCurve object

# ****************************************************************************************************
class Star(object):

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
                [x[pl:0:-1], x, x[
                    -1:-ph:-1]])  # pad data array with reflection of edges on both sides
            iu = -ph + 1

        else:
            pl = nwindow - 1
            s = np.ma.concatenate([x[pl:0:-1],
                                   x])  # pad data array with reflection of the starting edge
            iu = len(s)

        s[s.mask] = np.nan
        # maximum fraction of invalid values (nans) of window that will still yield a result
        max_nan_frac = 0.5
        mp = int(nwindow * (1 - max_nan_frac))
        self.median = pd.rolling_median(s, nwindow, center=center,
                                        min_periods=mp)[pl:iu]
        self.var = pd.rolling_var(s, nwindow, center=center, min_periods=mp)[
            pl:iu]

        # return med, var

    def plot_clippings(self, *args):
        # TODO:  plot clippings from outliers.py
        ax, t, nwindow = args
        self.ax_clp = ax
        med = self.median
        std = np.sqrt(self.var)
        lc = self.data
        threshold = self.sig_thresh

        ax.plot(t, self.clipped, 'g.', ms=2.5, label='data')
        ax.plot(t[self.clipped.mask], lc[self.clipped.mask], 'x', mfc='None',
                mec='r', mew=1, label='clipped')

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

    def smooth(self, nwindow, window='hanning', fill='mean'):
        # data = self.clipped[~self.clipped.mask]
        self.smoothed = smoother(self.clipped, nwindow, window, fill)
        return self.smoothed

    def argmin(self):
        return np.argmin(self.counts())

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
    # NOTE: this is probably a huge anti-pattern:
    # Explicit is better than implicit
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


class PhotFileManager:
    #                                             ↓ kill
    _default_search_order = ('npz', 'tbl', 'mag', 'dlc')

    def __init__(self, filename):
        self.filename = filename

    @property
    def basename(self):
        if self.fitsfile:
            return self.fitspath.stem

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
        yield from mit.roundrobin(*blob)


# ****************************************************************************************************
class PhotResult(AutoFileToPath):  # FixMe eliminate AutoFileToPath
    """
    Data container for photometry results.  Data stored internally as
    array of fluxes (npoints, nstars, naps)
    """
    # TODO: some elements here are FileManagement, some are Plotting
    # TODO: split classes to avoid god-objects
    # TODO: integrate with shocObs / LightCurve

    # TODO:  STORE/LOAD data AS FITS.
    # TODO:  PICKLE / json?
    # TODO: maybe autoload class attribute that enable automatic loading with
    # self.datafile = 'some/file/that/exists'

    # TODO: if coordinates are given, we can order the stars logically so that
    # stars correspond across cubes.
    # sorting methods? checkout pandas / astropy.table

    # TODO: - implement pdm, lomb-scargle

    # TODO: `data` attr must be `TimeSeries` object

    _skip_init = False  # initializer flag
    _flx_sort = True  # will order data according to brightness

    # plotting config
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
        yield from mit.roundrobin(*blob)

    def __new__(cls, *args, **kws):
        # this is here to handle initializing the object from an already
        # existing instance of the class
        # embed()
        if len(args) > 0 and (args[0].__class__.__name__ == cls.__name__):
            # autoreload hack!
            # isinstance(args[0], cls):
            instance = args[0]
            instance._skip_init = True
            return instance
        else:
            return super().__new__(cls)

    def __init__(self, fitsfile=None, datafile=None, timefile=None,
                 coordfile=None, target=None, target_name=None):
        """

        Parameters
        ----------
        fitsfile
        datafile
        coordfile
        target
            if given, this star will be plotted over all others
        """

        # if self._skip_init:
        #     logging.debug('PhotResult Skipping init')
        #     # print('hi   ')
        #     return
        # this happens if the first argument to __init__ is already an
        # instance of the class

        # TODO: maybe put the stuff below in a load classmethod ?

        self.fitsfile = str(fitsfile) if fitsfile else None
        self.datafile = datafile
        self.coordfile = coordfile

        name_guess = self.fitspath.with_suffix('.time')
        self.timefile = first_exists(timefile, name_guess)

        # optional ephemeris
        # self.ephem = ephem

        # load info from fitsfile if avaliable
        self.target_name, self.date, self.size = get_name_date(self.fitsfile)
        # FIXME: move to subclass ??
        # FIXME: self.date should be self.startdate ???
        if target_name is not None:
            self.target_name = target_name

        # store data internally as masked array (Nframes, Nstars, Naps)
        # init null data containers
        self.t = np.empty(0)  # TimeIndex ????
        self.data = np.ma.empty((0, 0))
        self.std = np.empty((0, 0))

        # load timing data if available
        if self.timefile:
            self.t = load_times(self.timefile)

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

    def __getitem__(self, key):
        """Can be indexed numerically, or by using 'target' or 'ref' strings."""
        raise NotImplementedError
        # if isinstance(key, (int, np.integer, slice)):
        #     # TODO: return lightcurve object
        #     pass
        #
        # if isinstance(key, str):
        #     key = key.lower()
        #     if key.startswith('t'):
        #         return self[self.target]
        #     # if key.startswith('r') or key == 'c0':
        #     #     return self.stars[self._ref]
        #     # if key.startswith('c'):
        #     #     return self.stars[self._others[int(key.lstrip('c')) - 1]]
        #
        # else:
        #     raise KeyError

    # def __str__(self):
    #     'TODO'

    def __len__(self):
        # Number of data points *not* stars
        return self.data.shape[1]

    @property
    def npoints(self):
        return self.data.shape[1]

    @property
    def nstars(self):
        return self.data.shape[0]

    # @property  # TODO: derive MultiAperturePhotResult?
    # def naps(self):
    #     return self.data.shape[-1]

    def get_target(self):
        return self._target

    def set_target(self, target):
        self._target = target
        # self.stars.set_names(target, self.target_name)
        # self.stars.set_colours(target)

    target = property(get_target, set_target)  # fixme: won't be inherited

    @property
    def others(self):
        return np.setdiff1d(np.arange(self.nstars), self.target)

    @property
    def reorder(self):
        if self._flx_sort and self.data.size:
            order = list(self._ordered_by_flux())
        else:
            order = list(range(self.nstars))

        if self.target is not None:
            order.insert(0, order.pop(order.index(self.target)))

        return order

    def _ordered_by_flux(self):
        return self.data.mean(1).argsort()[::-1]

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

    def load_from_npz(self, filename):

        # try:
        # lz = np.load(str(filename))  # psf_flux, psf_fwhm, problematic
        # self.data = lz['flux_ap']  # shape (Nframes, Nstars, Naps)
        # coo = lz['coords']      #shape (Nframes, Nstars, 2)

        lz = np.load(str(filename))
        self.t = lz['t'].view(np.recarray)
        self.data = np.ma.array(lz['data'], mask=lz['data_mask'])
        self.std = np.ma.array(lz['std'], mask=lz['std_mask'])

        # todo: target, name etc..
        # except Exception as orr:
        #     embed()
        #     raise

        self.load_times()

    def load_from_lc(self, filename, mag2flux=False, mag0=0):
        """load from ascii lc"""
        (metaline, starline, headline), data = read(filename)

        data_start_col = first_true_idx(
            filter(None, headline.split('  ')),  # '\t'
            lambda s: s.startswith(('MAG', 'Flux')))
        # print('DATA START COL', data_start_col)

        newdata = fold.fold(data.T[data_start_col:], 2)
        signals, std = np.swapaxes(newdata, 1, 0)

        if mag2flux:
            signals, std = as_flux(signals, std, mag0=mag0)

        # upcast for consistency
        self.data = signals[...]
        self.std = std[...]

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
        self.t = np.recarray(len(data),
                             dtype=list(zip(timefields, formats)))
        self.t['utsec'] = data[:, 0]
        self.t['utdate'] = self.date

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
                     t=self.t, data=self.data, std=self.std)
            # FIXME: NotImplementedError: MaskedArray.tofile() not implemented yet.

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

        # TODO: optionally show some stats ala table.info

        # HACK
        # Daophot databases are incredible inefficient and annoying data
        # structures. Since sometimes results for a particular star may be
        # silently omitted, and moreover, since the filename entries are
        # truncated by the fixed column width of (23 characters), and the
        # naming convention for SHOC data eg.: SHA_20181212.001.003.fits
        # (is longer than 23 characters!) it may not be possible to
        # match the table row entry with the star it belongs to a priori.
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
        # print(self.data.shape, '!!!!!!!!!!!!!!!!!!!!!!!!!')

    def recommend_ap(self):
        # TODO: Move! better to have MultiApertureResult
        # for multi-aperture dbs. determine aperture with highest mean SNR
        snr = self.data / self.std
        ixap = int(np.ceil(snr.argmax(-1).mean()))
        return ixap

    def select_best_ap(self):
        # TODO: Move! better to have MultiApertureResult
        ixap = self.recommend_ap()
        data = self.data[..., ixap:ixap + 1]
        std = self.std[..., ixap:ixap + 1]
        return data, std

    def load_from_apcor(self, filename, coofile, splitfile=None):
        """load data from aperture correction ascii database."""

        # self.size = 74000

        def convert_indef(s): return np.nan if s == b'INDEF' else float(s)
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
            def get_fileno(fn): return int(fn.strip(']').rsplit(',', 1)[-1])
            fileno = np.fromiter(map(get_fileno, bigdata['Image'].astype(str)),
                                 int)
        except ValueError:
            try:  # Older data with different naming convention
                def get_fileno(fn): return int(
                    fn.rstrip('.fits').rsplit('.', 1)[-1])
                fileno = np.fromiter(
                    map(get_fileno, bigdata['Image'].astype(str)), int)
            except ValueError:
                pass

            # WARNING:  This will only work if there are no dropped frames!
            # (IRAF SUX!!)
            nstars, ndata = len(kcoords), len(bigdata)
            nframes, remainder = divmod(ndata, nstars)
            if remainder:
                raise ValueError(
                    ('Dropped Frames in {}!  Number of data lines ({}) '
                     'does not equally divide  by number of stars {}'
                     ).format(filename,
                              ndata,
                              nstars))

            fileno = np.mgrid[:nstars, :nframes][1].ravel()

        # group the stars by the closest matching coordinates in the
        # coordinate  file list.
        def starmatcher(coo): return np.sqrt(
            np.square(kcoords - coo).sum(1)).argmin()
        for starid, (coo, ix) in group_more(starmatcher, coords,
                                            range(len(bigdata))):
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

    def load_times(self, filename=None):
        filename = _file_checker(filename or self.timefile, 'time')
        logging.info('Loading timing data from %r', os.path.basename(filename))
        self.t = load_times(filename)

    def fix_ut(self, set_=True):  # TODO: more descriptive name please
        """
        fix UTSEC to be relative to midnight on the starting date of the
        first frame"""
        t = self.t['utsec'].copy()
        dates = self.t['utdate']
        dateset = set(dates)
        delim = b'-' if isinstance(next(iter(dateset)),
                                   (bytes, np.bytes_)) else '-'

        def to_datetime(d): return Date(*map(int, d.split(delim)))  # .decode()
        if len(dateset) > 1:
            warnings.warn(
                'Data spans multiple UTC dates! Adjusting time origin.')

            d0 = to_datetime(dates[0])
            for i, d in enumerate(dateset - {dates[0]}):
                tshift = (to_datetime(d) - d0).total_seconds()
                t[dates == d] += tshift
                if set_:
                    self.t['utsec'] = t

        return t

    def add_phase_info(self, ephem):
        import numpy.lib.recfunctions as rfn

        # Get the phases
        ph = ephem.phase(self.t['bjd'])

        # add to time array
        if 'phase' in self.t.dtype.names:
            # overwrite
            self.t['phase'] = ph
            self.t['phaseMod1'] = ph % 1
        else:
            # append
            t = rfn.append_fields(
                self.t, ('phase', 'phaseMod1'), (ph, ph % 1))
            # make it a recarray
            self.t = t.view(np.recarray)

    def compute_dl(self, mode='flux', **kws):

        if self.target is None:
            raise ValueError('Please set target star first')
            # TODO: # de - correlate somehow

        logging.info('Computing differential light curve..')

        # summed flux of all non-target stars
        flx = self.data  # [:, self.target, :]
        flxRef = self.data[:, self.others, :].sum(1)[:, None]
        # compute std
        std = self.std  # [:, self.target, :]
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
        new.t = self.t
        new.target = self.target
        new.target_name = self.target_name
        new.date = self.date
        return new

    def std_cuts(self, lower=0, upper=100):
        l = (lower > self.std) | (self.std > upper)
        self.data[l] = np.ma.masked

    # def mask_outliers(self):

    # MultiAperture plot gui

    def plot_lc(self, tkw='utsec', **kws):

        # Reshape for plotting
        y = self.data[self.reorder]
        e = self.std[self.reorder]
        kws.setdefault('labels', np.array(self.labels)[self.reorder])

        # If target specified, re-order the array so we get a consistent
        # colour sequence

        # self.logger.info('Plotting light curves for %s', self)
        mode = kws.pop('mode', 'flux')

        t = self.t[tkw]
        kws.setdefault('title', self._title_fmt.format(s=self))

        # xlabel = tkw.upper()
        # ylabel = mode.title()  # 'Instr. ' +
        axes_labels = (tkw.upper(), mode.title())

        # if kw.get('twinx'):
        timescales = {'utsec': 's',
                      'utc': 'h',
                      }  # TODO etc...

        tsp = ts.plot(t, y, e,
                      # colours=colours,
                      axes_labels=axes_labels,
                      timescale=timescales.get(tkw),
                      start=self.date,
                      **kws)

        # This will plot the target *over* the other light curves
        if self.target:
            tsp.art.lines[self.target].set_zorder(10)

        # TODO: interactively switch to mag scale with key
        ax = tsp.ax  # fig.axes[0]
        if mode.lower() == 'mag':
            ax.invert_yaxis()

        if self.date:
            ax.text(0, ax.xaxis.labelpad,
                    'DATE: ' + self.date,
                    transform=ax.xaxis.label.get_transform())

        return tsp

    # def plot_lc(self, tkw='utsec', ixap=None, **kw):
    #
    #     if self.naps == 1:
    #         ixap = 0
    #     elif ixap is None:
    #         # fixme: elliminate the need for this shitty hack!!
    #         ixap = self.recommend_ap()
    #         logger.info('Choosing aperture %i (highest SNR)', ixap)
    #
    #     # Reshape for plotting
    #     y = self.data[:, self.reorder, ixap].T
    #     e = self.std[:, self.reorder, ixap].T
    #     labels = np.array(self.labels)[self.reorder]
    #     # If target specified, re-order the array so we get a consistent
    #     # colour sequence
    #
    #     # self.logger.info('Plotting light curves for %s', self)
    #     mode = kw.pop('mode', 'flux')
    #
    #     t = self.t[tkw]
    #     title = self._title_fmt.format(s=self)
    #     xlabel = tkw.upper()
    #     ylabel = mode.title()  # 'Instr. ' +
    #     axes_labels = xlabel, ylabel
    #
    #     # if kw.get('twinx'):
    #     timescales = {'utsec': 's',
    #                   'utc': 'h',
    #                   }  # TODO etc...
    #
    #     tsp = ts.plot(t, y, e,
    #                   title=title,
    #                   labels=labels,
    #                   # colours=colours,
    #                   axes_labels=axes_labels,
    #                   timescale=timescales.get(tkw),
    #                   start=self.date,
    #                   **kw)
    #
    #     # This will plot the target *over* the other light curves
    #     if self.target:
    #         tsp.art.lines[self.target].set_zorder(10)
    #
    #     # TODO: interactively switch to mag scale with key
    #     ax = fig.axes[0]
    #     if mode.lower() == 'mag':
    #         ax.invert_yaxis()
    #
    #     if self.date:
    #         ax.text(0, ax.xaxis.labelpad,
    #                 'DATE: ' + self.date,
    #                 transform=ax.xaxis.label.get_transform())
    #
    #     return tsp

    def save_npz(self, filename):
        filename = str(filename)
        self.logger.info('saving as: %s', filename)
        np.savez(filename,
                 t=self.t,
                 data=self.data.data, data_mask=self.data.mask,
                 std=self.std.data, std_mask=self.std.mask)

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
            times = self.t[list(tkw)].view(float).reshape(-1, len(tkw))
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

            header0 = 'Fluxes for {} stars extracted from {}.'.format(nstars,
                                                                      dbs)
            header1 = saver.make_header_line(col_head, fmt, delimiter)
            header2 = saver.make_header_line(col_head2, fmt, delimiter)
            header = '\n'.join([header0, header1, header2])  # header0,

            np.savetxt(filename, output, header=header, fmt=fmt,
                       delimiter=delimiter)
            print('Halleluja!', filename)
        else:
            print('Nothing written')

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

    def plot_spread(self, tkw='utsec'):
        """plot all aperture data as time series"""
        fig, axes = plt.subplots(2, 1, figsize=(18, 12),
                                 sharex=True,
                                 gridspec_kw=dict(hspace=0))

        t = self.t[tkw]

        print(t)

        for ax, star in zip(axes, self):
            sdata = get_spread(star, 1e4)
            naps = sdata.shape[0]
            labels = ['ap %d' % i for i in range(naps)]
            ts.plot(t, sdata, ax=ax, labels=labels)


def get_spread(star, sfactor):
    naps = star._data.shape[1]
    return (star._data + np.arange(naps) * sfactor).T
