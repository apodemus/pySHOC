# TODO:  trim down all the cruft!!!!!
# TODO: localise imports

# -*- coding: utf-8 -*-
import re
import os
import copy
import warnings
import itertools as itt
# from collections import Iterable
# from pathlib import Path
from datetime import date as Date

import numpy as np
import matplotlib.pyplot as plt

# import numpy.linalg as la

# from matplotlib import cm, ticker
# import matplotlib.gridspec as gridspec
# import matplotlib.image as mimage               #LOCALISE IMPORTS??
# from mpl_toolkits.axes_grid1 import make_axes_locatable

# import pyfits

# from pySHOC.io import InputCallbackLoop
from pySHOC.io import Conversion as convert

from recipes.iter import interleave, groupmore, first_false_idx
from recipes.list import find_missing_numbers, sortmore
# from superstring import rreplace#kill_brackets   #ProgressBar

from fastfits import quickheader
from grafico.ts import TSplotter
# from grafico.interactive import PointSelector
# from superplot.spectra import DualAxes
# from superplot.spectra import
# from draggables import DraggableErrorbars



# from lombscargle import fasper, getSignificance
from tsa import fold
from tsa.smoothing import smoother
from tsa.spectral import Spectral
# from tsa.tfr import TimeFrequencyRepresentation as TFR
# from tsa.outliers import WindowOutlierDetection, generalizedESD

# from IPython import embed
import decor
# profiler = decor.profile.profile()

from ansi.str import banner

tsplt = TSplotter()


# TODO:  EXPLORE PANDAS DATAFRAMES

# ****************************************************************************************************
class Run(object):
    # TODO make modular
    # TODO: merge by: target; date; etc...
    # TODO: initialize from cubes!!
    ''' Class for containing / merging / plotting Cube objects.'''

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self, filenames, databases=None, coordinates=None,
                 target=0, ref=1, target_name='', outpath=None):

        self.coordinates = coordinates
        self.databases = databases or []  # magnitudes database files
        # self.data_path = path

        self.ncubes = len(filenames)

        self.target_name = target_name
        # self.date = date
        if len(filenames):  # TODO: This is ugly... filenames should be a property
            if isinstance(filenames[0], Cube):  # TODO: This logic should be inside Cube!!
                self.filenames = [cube.filename for cube in filenames]
                self.basenames = [cube.basename for cube in filenames]
                self.cubes = filenames
            # elif isinstance(filenames[0], Path):        #TODO: This logic should be inside Cube!!


            else:
                # full_filenames = [ os.path.join(self.data_path,fn) for fn in filenames ]
                # TODO: send to Cube...

                self.filenames = list(map(str, filenames))  # NOTE: THIS SUCKS

                self.basenames = [os.path.split(fn)[1].replace('.fits', '')
                                  for fn in self.filenames]

                self.cubes = [Cube(fn, db, cf, target, ref) for (fn, db, cf) in
                              itt.zip_longest(self.filenames, self.databases,
                                              self.coordinates or [])]  # NOT DYNAMIC!

        else:
            raise ValueError('No filenames')

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __getitem__(self, key):
        '''Can be indexed numerically, or by corresponding filename / basename.'''

        if isinstance(key, str):
            if key.endswith('.fits'):
                key = key.replace('.fits', '')
            key = self.basenames.index(
                key)  # self.cube_filenames.index(True)     ##BE CAREFUL WITH THIS......................................
            return self.cubes[key]

        elif isinstance(key, slice):
            newrun = copy(self)
            newrun.filenames = self.filenames[key]
            newrun.basenames = self.basenames[key]
            newrun.databases = self.databases[key]
            newrun.coordinates = self.coordinates[key]
            newrun.cubes = self.cubes[key]
            newrun.ncubes = len(newrun.cubes)

            return newrun
        else:
            return self.cubes[key]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __len__(self):
        return self.ncubes

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __str__(self):
        return 'Run of {} cube{}: {}'.format(self.ncubes,
                                             's' * bool(self.ncubes - 1),
                                             '| '.join(self.basenames))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __repr__(self):
        return str(self)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def get_outname(self, with_name=True, extension='txt', sep='.'):  # TODO: MERGE WITH gen_filename??
        if with_name:
            name = self.target_name.replace(' ', '') + sep
        else:
            name = ''
        date_str = self.date.replace('-', '')
        outname = '{}{}.{}'.format(name, date_str, extension)
        return os.path.join(self.data_path, outname)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # def get_template(self, n):

        # self.template = read_data_from_file( args.ims, n )

        # return self.template

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def add_cube(self, filename, Nstars=None):
        '''add a cube to the run.'''
        Nstars = Nstars if Nstars else self.nstars
        basename = filename.replace('.fits', '')
        filename = os.path.join(self.data_path, filename)

        self.filenames.append(filename)
        self.basenames.append(basename)
        self.cubes.append(Cube(filename, Nstars))

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
    def join(self, whichstar):  # FIXME: INEFFICIENT!!!!
        '''join stars'''
        # Grab the properties from the same star in the first cube
        # coo = self[0][whichstar].coo
        # name = self[0][whichstar].name
        # colour = self[0][whichstar].colour

        #FIXME: match target and ref

        # WARNING! stars may be missing in some cubes??
        # for which in ('raw', 'clipped', 'diff'):
        data = np.ma.concatenate([cube[whichstar].data for cube in self])
        err = np.ma.concatenate([cube[whichstar].err for cube in self])
        try:
            coo = np.ma.concatenate([cube[whichstar].coo for cube in self])
        except ValueError:
            coo = None

        star = Star(data, err, coo)

        star.dl = np.ma.concatenate([cube[whichstar].dl for cube in self])
        star.clipped = np.ma.concatenate([cube[whichstar].clipped for cube in self])

        # star.ls_data = np.concatenate( [cube[whichstar].ls_data[:] for cube in self], axis=1 )

        return star

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # @decor.expose.args()
    def conjoin(self):
        '''Join cubes together to form contiguous data block.'''

        # if len(self)==1:
        # return self[0]

        cube = Cube()
        cube.databases = self.databases

        # TODO:  match coordinates to stars...

        # stack star data
        N = max(len(cube.stars) for cube in self)
        for name in ['target'] + ['C' + str(i) for i in range(N - 1)]:
            cube.stars.append(self.join(name))

        # stack timing data
        cube.timedata = np.r_[tuple(c.timedata for c in self)]

        # set total length
        cube._len = sum(len(cube) for cube in self)

        cube.target_name = self.target_name
        cube.set_target_ref(self[0].target, self[0].ref)
        cube.date = self[0].date

        cube.fix_ut()

        return cube

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def compute_ls(self, tkw='lmst', **kw):
        t = self.get_time(tkw)

        for star in self.stars:
            star.compute_ls(t, **kw)

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # def plot_raw(self, ax, whichstar, tkw='lmst', **kw):
            # starname = self[0][whichstar].name
            # print( 'Plotting raw light curves for star {}'.format(starname) )

            # offset = kw['offset'] if 'offset' in kw else 0
            ##plots = []
            # whole = self.conjoin()
            # if len(whole[whichstar]):
            # label = cube[whichstar].name
            ##print('label', label)
            # t = getattr(whole,tkw)[ whole[whichstar].l ]
            # pl = whole[whichstar].plot_raw( ax, t, label=label )
            ##plots.append(pl)
            ##else:
            ##print( 'WARNING: No data found for star {} in cube {}.'.format(starname, cube.filename) )
            # return pl

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # def plot_ls(self, which_stars=[0], fig=None, **kw):
            # option = kw.pop('option') if 'option' in kw else 'combined'
            # if option=='combined':
            # for star in which_stars:
            # f, A = self.combine_ls(star)
            # cstar = Star()
            # cstar.name = self[0][star].name
            # cstar.colour = self[0][star].colour
            # cstar.ls_data[:] = [f], [A], []
            # cstar.plot_ls(fig, **kw)

            # elif option=='stack':
            # scube = Cube('', len(which_stars))
            # for star in which_stars:
            # scube[star].ls_data.f = [ cube[star].ls_data.f for cube in self if len(cube[star]) ]
            # scube[star].ls_data.A = [ cube[star].ls_data.A for cube in self if len(cube[star]) ]
            # scube.plot_ls(which_stars, fig, **kw)


# ****************************************************************************************************
class Cube(object):
    # TODO: integrate with SHOC_Cube ---> Observation / ShocObs ?

    # TODO:  MULTIDIMENSIONAL REC ARRAYS ---- WITH UNCERTAINTIES!
    # TODO:  STORE/LOAD AS FITS.
    # TODO:  PICKLE / json?
    # TODO:  KILL REF!!!!!!!!!!!!!!!!!!!
    ''' ... '''
    Tkw = ['jd', 'bjd', 'utc', 'utsec', 'lmst']  # 'utsec'              #Available time formats

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # @decor.expose.args( pre='!', post='@' )
    # @decor.path.to_string.auto
    def __init__(self, filename='', database=None, coordinates=None,
                 target=None, ref=None):
        # self.coo = []  #star coordinates on image
        self.filename = str(filename)
        self.basename = self.filename.replace('.fits', '')
        self.database = database
        self.coordinates = coordinates

        if filename:
            self.target_name, self.date, self.size = get_name_date(self.filename)

        self.stars = Stars()
        # NOTE: this is restrictive in that it fixes the Nstars, target, ref across
        # all cubes.  Think of a way of dynamically creating the Stars class

        self._target = target
        self._ref = ref

        # if filename:
        # self.set_times()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __getitem__(self, key):
        '''Can be indexed numerically, or by using 'target' or 'ref 'strings.'''
        if isinstance(key, (int, np.int_, slice)):
            return self.stars[key]

        if isinstance(key, str):
            key = key.lower()
            if key.startswith('t'):
                return self.stars[self._target]
            if key.startswith('r') or key == 'c0':
                return self.stars[self._ref]
            if key.startswith('c'):
                return self.stars[self._others[int(key.lstrip('c')) - 1]]

        else:
            raise KeyError

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __len__(self):
        return len(self.stars[0])

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def get_target(self):
        return self._target

    def set_target(self, target):
        print('set target', target)
        self._target = target
        self._others = find_missing_numbers([target, self.ref, -1, len(self.stars)])

        self.stars.set_names(target, self.ref, self.target_name)
        self.stars.set_colours(target, self.ref)

    target = property(get_target, set_target)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # @property
    def get_ref(self):
        return self._ref

    # @ref.setter
    def set_ref(self, ref):
        self._ref = ref
        self._others = find_missing_numbers([self.target, ref, -1, len(self.stars)])

        self.stars.set_names(self.target, ref, self.target_name)
        self.stars.set_colours(self.target, ref)

    ref = property(get_ref, set_ref)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def set_target_ref(self, target, ref):
        self._target = target
        self._ref = ref
        self._others = np.setdiff1d(range(len(self.stars)),
                                    [self.target, self.ref or self.target])  # ref None if no ref star
        # find_missing_numbers( [target, ref, -1, len(self.stars)] )

        self.stars.set_names(target, ref, self.target_name)
        self.stars.set_colours(target, ref)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # def get_data(self):
        # TODO:  as property

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def set_times(self):
        timefile = self.basename + '.time'
        print('Loading timing data from {}'.format(repr(timefile)))

        self.timedata = load_times(timefile)


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def fix_ut(self, set_=True):
        '''fix UTSEC to be relative to midnight on the starting date of the first frame'''
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

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def load_data(self, filename=None, coofile=None):
        # mode='shoc',

        # for which in ['lc']:#['rlc', 'dlc']:

        # print( 'Loading light curve data from file {}...'.format(filename) )
        # masterdata = np.loadtxt(filename, unpack=1)

        # if 'SHOC' in mode.upper():
        # filename = '{}.{}'.format( self.basename, which )
        # datastartcol, Nreadcols, Nskipcol = 2, 2, 0

        # if mode.upper() in ['SCAM' ,'SALT']:
        ##filename =
        # datastartcol, Nreadcols, Nskipcol = 6, 2, 2
        # self.utsec = masterdata[1]

        # for i in itt.count():
        # if datastartcol + Nskipcol*i + Nreadcols*(i+1) > len(masterdata):
        # break
        # idx0 = datastartcol + (Nreadcols+Nskipcol)*i

        # self.stars.append( *masterdata[idx0:idx0+Nreadcols] )

        print('loading data. Patience is a virtue')
        filename = filename or self.database
        coordinates = coofile or self.coordinates
        # print( coordinates )
        filename = str(filename)
        print(filename)

        # embed()

        if filename.endswith('lc'):
            self.load_from_lc(filename)
        elif filename.endswith('npz'):
            self.load_from_npz(filename)
        elif 'apcor' in filename:
            self.load_from_apcor(filename, coordinates)
        else:
            self.load_from_mag(filename, 8)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def load_from_npz(self, filename, mode='flux', nap=0):

        lz = np.load(filename)  # psf_flux, psf_fwhm, problematic
        data = lz['flux_ap']  # shape (Nframes, Nstars, Naps)
        # coo = lz['coords']      #shape (Nframes, Nstars, 2)
        Nframes, Nstars, Naps = data.shape
        for i in range(Nstars):
            star = Star(data[:, i, nap],
                        err=None,
                        coo=None)  # coo[:,i,::-1])
            star._data = data[:, i]
            self.stars.append(star)

        self.set_times()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def load_from_lc(self, filename, mode='flux', data_start_col=1):
        # TODO: instead of having mode argument, store data internally as flux ang have a
        # property that handles the flux conversion
        (metaline, starline, headline), data = read(filename)

        for i, d in enumerate(fold.fold(data.T[data_start_col:], 2)):
            if mode == 'mag':
                mag0 = 25
                d = self.asflux(*d, mag0=mag0)
            self.stars.append(Star(*d))

        # ref = starheads[2::2].index('ref')
        # starheads = starline.strip('#').split('\t')
        starheads = list(filter(None, map(str.strip, starline.strip('#').split('  '))))
        # colheads = kill_brackets(headline).strip('#').lower().split('\t')

        # embed()

        ref = starheads.index('C0')
        target = first_false_idx(starheads, lambda s: re.match('C\d|ref', s))  # [2::2]
        self.set_target_ref(target, ref)

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
    def load_from_mag(self, filename, nap=None, write=True):
        '''load data from IRAF daophot multi-aperture ascii database.'''
        from astropy.io import ascii

        def as2Dfloat(colid):
            table = bigdata[tuple(colid + str(i) for i in range(1, Naps + 1))]
            return table.as_array().view(float).reshape(len(table), -1)

        reader = ascii.Daophot()
        bigdata = reader.read(filename or self.database)

        if nap is None:
            # determine aperture with highest SNR for each observation
            Naps = len(reader.header.aperture_values.split())
            MAG = as2Dfloat('MAG')
            ERR = as2Dfloat('MERR')
            iap = np.argmax(MAG / ERR, 1)
            mag = MAG[np.arange(len(iap)), iap]
            err = ERR[np.arange(len(iap)), iap]
        else:
            datafield = 'MAG%i' % nap
            errorfield = 'MERR%i' % nap
            mag, err = bigdata[datafield], bigdata[errorfield]

        mag0 = float(bigdata.meta['keywords']['ZMAG']['value'])

        # Separate the stars by their initial coordinates
        cooinit = bigdata['XINIT', 'YINIT']
        for coo in np.unique(cooinit):
            # select the data for this star

            banner('extracting star at coo', coo, bg='magenta', width=100)

            # embed()

            l = cooinit == coo
            stardata = bigdata[l]
            flux, fluxerr = self.asflux(mag[l], err[l], mag0)
            coo = np.array([stardata['XCENTER'], stardata['YCENTER']]).T

            # Initialize the Star object and add it to the Cube
            star = Star(flux, fluxerr, coo)
            star._data = stardata
            self.stars.append(star)

        self.set_times()

        if write:  # finally write the table in a more accesible form
            # embed()
            bigdata[:10].write('{}.tbl'.format(filename),
                               format='ascii.fast_commented_header')

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def load_from_apcor(self, filename, coofile, splitfile=None):
        '''load data from aperture correction ascii database.'''

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

        # embed()

        # group the stars by the closest matching coordinates in the coordinate file list.
        starmatcher = lambda coo: np.sqrt(np.square(kcoords - coo).sum(1)).argmin()
        for starid, (coo, ix) in groupmore(starmatcher, coords, range(len(bigdata))):
            ix = np.array(ix)
            stardata = bigdata[ix]

            data, err = np.ma.zeros((2, self.size))
            mask = np.ones(self.size, bool)

            try:
                z = fileno[ix] - 1
                data[z], err[z] = self.asflux(bigdata[ix]['Mag'], bigdata[ix]['Merr'], mag0=25)
            except:
                print('ERROR CAUGHT!')
                embed()
            # bigdata[ix]['Exptime']
            mask[z] = False
            data.mask = mask

            star = Star(data, err, coo)
            star._data = stardata
            self.stars.append(star)

        self.set_times()

    def date_split(self):
        ''#TODO

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @staticmethod
    def asflux(mag, magerr=None, mag0=0):
        '''Convert magnitudes to Fluxes'''
        flux = 10 ** ((mag0 - mag) / 2.5)
        if magerr is None:
            return flux

        fluxerr = magerr * (np.log(10) / 2.5) * flux
        return flux, fluxerr

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # TODO:  as property??
    def datablock(self, which='raw', with_error=False):
        if with_error:
            return (np.ma.vstack([star.get_data(which) for star in self]),
                    np.ma.vstack([star.err for star in self]))  #FIXME: ERRORS FOR DL!!!
        else:
            return np.ma.vstack([star.get_data(which) for star in self])

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def optimal_smoothing(self):
        '''Determine the optimal amount of smoothing for differential photometry
        by minimising the variance of the differentail light curve for the given
        length of the smoothing window.  Note: This operation may be slow...'''

        data = self.datablock('clipped')

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def compute_dl(self, which='raw', mode='flux', **kw):
        # TODO: instead of having mode argument, store data internally as flux ang have a
        # proporty that handles the flux conversion
        print('Computing differential light curve..')

        straight = kw.get('straight')
        smoothing = kw.get('smooth')
        poly = kw.get('poly')
        mean = kw.get('mean')

        #
        if mode == 'flux':
            op = np.ma.divide
        else:
            op = np.ma.subtract

        if straight:
            for j, star in enumerate(self):
                print('Star %i' % j)

                star.dl = op(star.data, self['ref'].data)  # NEED ERROR CALCULATION!!!!!!!
                # star.dl_diff = diff

        # mag_ref = np.ma.array( mag_ref, mask=self[ref].clipped.mask )

        # NEEDS WORK --> COMBINED POLY + SMOOTH DETRENDING...

        if mean:
            compix = tuple(set(range(len(self.stars))) - {self._target})  # self._others???
            comps = self.datablock(which)[compix, :]  # data for comparison stars
            Fr = comps / comps.mean(1)[None].T
            ref = Fr.mean(0)
            # TODO:  SMOOTH BY CONSIDERING STDEV OF Fr????
            # TODO:  UNCERTAINTY propagation!!!!
            # diff = np.mean(comps - np.atleast_2d(np.ma.median(comps, 1)).T

        else:
            ref = self['ref'].get_data(which)  # 'clipped'

        if smoothing:  # smoothing gives window size
            ref = smoother(ref, smoothing, fill=None)  # , return_type='masked')

        if poly:
            t = self.timedata['utsec']
            coof = np.ma.polyfit(t, ref, poly)
            ref = np.polyval(coof, t)

        # embed()

        # scale = np.ma.median( ref )

        # Error computation
        # mean_err = (1./len(err_ref))*la.norm(err_ref)                                               #Add errors in quadrature
        # diff_err = np.array( [la.norm([err, mean_err]) for err in err_ref] )                        #Add errors in quadrature

        for j, star in enumerate(self):
            print('Star %i!!!!!!!!!!' % j)

            # mag = star.data                             #OR CLIPPED???
            star.dl = op(star.get_data(which), ref)  # * scale

            # print np.array( [np.linalg.norm([e1, e2]) for e1,e2 in zip(err,diff_err)] )
            # err_dl = np.array( [np.linalg.norm([e1, e2]) for e1,e2 in zip(err,diff_err)] )              #Add errors in quadrature
            # star.dl = np.array([mag_dl, err_dl])
            # print('*'*3, star.dl.shape)

            # return coof

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def plot_lc(self, which='raw', tkw='utsec', **kw):

        which = which.lower()
        mode = kw.pop('mode', 'flux')
        # plots = []

        t = self.timedata[tkw]
        data = self.datablock(which)

        print('Plotting light curves for {}'.format(self))
        starnames = [getattr(star, 'name', 'unknown') for star in self]
        colours = [star.colour for star in self]  # FIXME!  This sux!

        #kw.get('title')

        if which.startswith('r'):
            desc, extra = 'Raw', ''
        elif which.startswith('d'):
            desc, extra = 'Differential', ''
        elif which.startswith('c'):
            desc, extra = 'clipped', ''  # '  ({}$\sigma$ clipped)'.format(star.sig_thresh)

        # title = '{} light curve:\t{}'.format(desc, self['target'].name).expandtabs() #on \n{}.{}, self.date, extra)
        title = desc + ' light curve'
        xlabel = {'utsec': 'UTC (s)'}[tkw]
        if which == 'diff':
            if mode == 'flux':
                ylabel = 'Flux ratio'
            else:
                ylabel = 'Diff. mag.'
        else:
            ylabel = 'Instr. ' + mode.title()
        axlabels = xlabel, ylabel

        # if kw.get('twinx'):
        timescales = {'utsec': 's',
                      'utc': 'h'}  # etc...

        fig, plots, *rest = tsplt(t, data,
                                  title=title,
                                  labels=starnames,
                                  colours=colours,
                                  axlabels=axlabels,
                                  timescale=timescales[tkw],
                                  start=self.date,
                                  **kw)

        # This will plot the target *over* the other light curves
        plots.lines[self._target].set_zorder(10)  # FIXME!!!!!!!!!!!!

        ax = fig.axes[0]
        if mode.lower() == 'mag':
            ax.invert_yaxis()

        ax.text(0, ax.xaxis.labelpad,
                'DATE: ' + self.date,
                transform=ax.xaxis.label.get_transform())
        # plt.subplots_adjust(top=0.95)
        # ax.grid()

        return plots

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def load_ls(self, fn):
        Nstars = len(self.stars)
        ls_data = np.loadtxt(fn, unpack=1)
        frq = ls_data[0]
        # assert ls_data.shape[0] == Nstars//2
        for i in range(Nstars):
            ls_data[:, i + 1]
            self.stars[i].ls_data = frq, ls_data[:, i + 1], None

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def save_ls(self, fn):
        Amps = [star.ls_data.A[0] for star in self]
        data = [self[0].ls_data.f[
                    0]] + Amps  # frequencies should be the same for all the stars.  If not, we have problems
        outarray = np.array(data)
        np.savetxt(fn, outarray)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def compute_ls(self, tkw='utsec', **kw):
        print('Computing Ldbomb Scargle spectrum for cube {}'.format(self.filename))
        if not tkw in self.timedata.dtype.names:
            raise ValueError('{} is not an valid time format'.format(tkw))
        else:
            'warn if time crosses 00.00h for utc or lmst ---> aliases'
            t = self.timedata[tkw]

        # 'which': [],'set_ls': []
        whichstars = kw.pop('whichstars') if 'whichstars' in kw   else range(len(self.stars))
        which = kw.pop('which') if 'which' in kw        else 'raw'
        set_ls = kw.pop('set_ls') if 'set_ls' in kw       else 1

        for idx in whichstars:
            star = self[idx]
            if len(star.data):
                signal = star.get_data(which)

                # print( 'kw', kw, len(signal), len(t) )

                star.spec = Spectral(t, signal, **kw)
                star.welch = np.nanmean(star.spec.power, 0)


                # if set_ls:
                # star.ls_data[:] = Frq, Amp, Sig
                # else:
                # 'create data array and return it'
                # return T, Frq, Amp, Sig

                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # def plot_ls(self, whichstars=None, fig=None, **kw ):

                # whichstars = whichstars if whichstars else [self._target, self._ref]
                # n_sp = len(whichstars)
                # height_ratios = [3] + [1]*n_sp
                # gs = gridspec.GridSpec(n_sp, 1, height_ratios=height_ratios)

                # fig = fig if fig else plt.figure()
                # self.ls_plots = []
                # for i,g in zip(whichstars, gs):
                # if i==whichstars[0]:
                # ax = fig.add_subplot(g)
                # else:
                # ax = fig.add_subplot(g, sharex=ax)

                # print('Plotting LS for star %i.' %i)
                # setup_ticks(ax)
                # plts = self[i].plot_ls( ax, 0., **kw )
                # self.ls_plots.append( plts )

                # plt.suptitle( 'LS periodogram of {} in {}'.format(cube['target'].name, cube.filename) )
                ##ax.set_xlim(f[0],f[-1])
                # ax.set_xlabel('f (Hz)')
                # ax.legend(loc='best')

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
            col_head = [''] + interleave(self.stars.get_names(), [''] * nstars)
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
    def minvar(self):
        '''return index of star with minimal variance'''
        return np.ma.var(self.datablock(), axis=1).argmin()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def check_ref(self, ref):
        # FIXME: THIS FUNCTION WILL PICK THE WRONG REFERENCE IF THE TARGET STAR
        # IS THE ONE WITH THE HIGHEST SNR!!
        # check which star has maximal SNR

        d, e = self.datablock(with_error=True)
        snr = d / np.sqrt(d + e)

        ssnr = sortmore(snr.mean(1), range(len(self.stars)), order='reverse')
        others = find_missing_numbers([-1, self.target, len(self.stars)])
        msnr, ix = np.array(ssnr).T[others].T
        maxsnr = int(ix[msnr.argmin()])

        # minvar = self.minvar()

        if not ref is None:
            if ref != maxsnr:  # minvar:
                warnings.warn('The selected reference star {} is not the one '
                              'with the best SNR!  Use {} instead.'.format(args.ref, maxsnr))
            return ref

        else:
            print('Using star {} as reference'.format(maxsnr))  # minvar
            return maxsnr  # minvar

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def plot_spread(self, tkw='utsec'):
        '''plot all aperture data as time series'''
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


# ****************************************************************************************************
class Stars(object):
    # TODO: Merge into Cube
    COLOURS = ['b', 'g', 'm', 'c', 'k', 'r', 'y', 'orange']  # colormap???

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self):
        # self.target = target
        # self.ref = ref

        self.stars = []
        # self.set_names()                                                #SHOULD FORM PART OF STAR __INIT__
        # self.set_colours()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __getitem__(self, n):
        return self.stars[n]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __setitem__(self, key, val):
        self.stars[key] = val

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __len__(self):
        return len(self.stars)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def append(self, data=None, err=None, coo=None):

        if isinstance(data, Star):
            self.stars.append(data)
        else:
            self.stars.append(Star(data, err, coo))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def set_names(self, target, ref, target_name, basename=None):  # THS SHOULD BE A METHOD OF CUBE
        count = 0
        for i, star in enumerate(self):
            if i == target:
                star.name = target_name
                # elif i==ref:
                # star.name = 'ref'                       #FIXME:  RELEVANCE??
            else:
                star.name = 'C' + str(count)
                count += 1

                # name = star.name.replace(' ','_')
                # star.fn_LS = '{}.{}.ls'.format(basename, name)                         #Lomb-Scargle txt file

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def get_names(self):
        return [star.name for star in self]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def set_colours(self, target, ref):  # THS SHOULD BE A METHOD OF CUBE
        '''
        Set the plot colours for the stars.
        Ensures target and ref always have the same colours (0,1) indices Stars.COULOURS
        '''

        colours = Stars.COLOURS[:]  # create a copy
        self[target].colour = colours.pop(0)
        if not ref is None:
            self[ref].colour = colours.pop(1)

        i, j = 0, 0
        while i < len(self):
            if not (i == target or i == ref):
                self[i].colour = colours[j]
                j += 1
            i += 1


            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # def allmags(self):
            ##alldata = np.empty( (len(self), len(self[shortest].data) ,2) )
            ##[q.T for q in s.allmags()]
            # return np.ma.hstack( [s.data for s in self] )

            ##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # def clipper(self):                                                                                          #THIS IS MEMORY INEFFICIENT!!!!!!!
            # for star in self:
            # for cube in star:
            # cube.clipper()


# ****************************************************************************************************
class Star(object):
    ''' ... '''

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self, data=None, err=None, coo=None, name=None, colour=None):
        # datalen = datalen if datalen else 0
        if not data is None:
            if np.ma.isMA(data):
                self.data = data  # raw data points (mag)
                self.err = err  # errors
            else:
                self.data = np.ma.array(data, mask=np.isnan(data))
                if err is None:
                    self.err = np.ma.array(np.empty_like(data),
                                           mask=True)
                else:
                    self.err = np.ma.array(err, mask=np.isnan(err))

        self.coo = None if coo is None          else np.array(coo)

        self.dl = np.empty(0)  # differential light curve (mag,err)
        self.clipped = np.empty(0)

        # self.ls_data = SpectralDataWrapper()                    #Lomb Scargle data

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __len__(self):
        if ~self.data.mask.any():
            return len(self.data)
        else:
            return sum(~self.data.mask)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def get_data(self, which):
        # TODO: STORE THIS DATA IN REC ARRAY
        if which == 'raw':
            data = self.data
        if which.startswith('clip'):
            data = self.clipped
        if which == 'diff':
            data = self.dl
        return data

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def running_stats(self, nwindow, center=True, which='clipped'):
        import pandas as pd

        # first reflect the edges of the data array
        # if center is True, the window will be centered on the data point - i.e. data point preceding and following the current data point will be used to calculate the statistics (mean & var)
        # else right window edge is aligned with data points - i.e. only the preceding values are used.
        # The code below ensures that the resultant array will have the same dimension as the input array

        x = self.get_data(which)

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
        '''Do a LS spectrum on light curve of star.
        If 'split' is a number, split the sequence into that number of roughly equal portions.
        If split is a list, split the array according to the indeces in that list.'''
        return Spectral(t, signal, **kw)


        # def set_ls(self, f, A, signif):
        # self.ls_data.f, self.ls_data.A, self.ls_data.signif = f, A, signif

        # def get_ls(self):
        # return self.ls_data.f, self.ls_data.A, self.ls_data.signif


        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # def plot_ls(self, ax=None, offsets=None, **kw):
        ##TODO: OFFSETS are redundant with draggables
        # '''Plot the LS spectra for the star.'''
        # print( 'Plotting Lomb Scargle spectrum for star {}'.format(self.name) )
        ##fignum = eval(str(2)+'.'+str(which))
        ##plt.figure(fignum*10, figsize=figsize)

        # if not ax:
        # fig, ax = plt.subplots()
        # else:
        # fig = ax.figure

        # which = kw['which'] if 'which' in kw else ''
        # yscale = kw['yscale'] if 'yscale' in kw else 'log'

        ##if x is None:
        ##x = self.ls_data.f
        ##if y is None:
        ##y = self.ls_data.A

        # n = len(self.ls_data.f)				#number of splits for data
        # offsetmap = 10**np.arange(n) if yscale=='log' else 10*np.arange(n)
        # offsets = offsets if offsets!=None else offsetmap
        # offsets = offsets if isinstance(offsets, Iterable) else [offsets]*n
        ##colour = 'b' if which==self.target else 'r'

        # plots = []
        # for j in range(n):
        # f = self.ls_data.f[j]
        # A = self.ls_data.A[j]

        # label = '{}{}'.format(self.name, (which if j==0 else '') )
        ##print 'label ', label
        # setup_ticks( ax )
        # if yscale=='log':             plt.yscale('log')
        # pl = ax.plot(f, A + offsets[j], self.colour, label=label)
        # plots.append( pl )
        ##ax.plot(wk1,signif,label = 'False detection probability')
        # ax.legend()
        # ax.set_xlabel('f (Hz)')
        # ax.set_ylabel('LS Power')
        ##fig.text(0.,0., 'Detrend: n=%i'%dtrend, transform=fig.transFigure)

        # self.ls_plots = plots
        # return plots

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def plot_lc(self, ax, t, which='raw', **kw):

        label = getattr(self, 'label', kw.get('label', ''))
        fmt = 'o'

        data = self.get_data(which)
        err = self.err

        fig, plots, *rest = tsplt(t, self.get_data(which), self.err)

        # pl = ax.plot(t, data, fmt, ms=2.5, color=self.colour, label=self.name)

        # erbpl = ax.errorbar(t, mag, yerr=err, fmt=fmt, ms=1, label=label)


        # print('Done!')
        return pl  # erbpl


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


# TODO:
# class JoinedCube( Cube ):
# def __init__(self, cubes):


# THIS SHOULD BE FITS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# @decor.path.to_string.auto
def read(filename):
    from myio import read_data_from_file
    header = read_data_from_file(filename, N=3)  # read first 3 lines
    (metaline, starline, headline) = header

    fill_value = -99
    data = np.genfromtxt(str(filename),
                         skip_header=3,
                         usemask=True)
    data.mask |= (data == fill_value)

    return header, data


# THIS SHOULD BE FITS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



# ****************************************************************************************************
# @decor.path.to_string.auto
def load_times(timefile):
    '''read timestamps from file.'''
    from myio import read_file_line

    hline = read_file_line(timefile, 0)  # read header line
    colnames = hline.strip('\n #').lower().split()  # extract column names (lower case)

    return np.genfromtxt(str(timefile),
                         dtype=None,
                         names=colnames,
                         skip_header=1,
                         usecols=range(1, len(colnames)))  # ???


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_name_date(fn=None):
    if fn:
        print("Getting info from {}... ".format(os.path.split(fn)[1]))
        header = quickheader(fn)
        name = header['OBJECT']
        size = header['NAXIS3']
        try:
            date = header['DATE-OBS'].split('T')[0]
        except KeyError:
            ds = header['DATE']  # fits generation date and time
            date = ds.split('T')[0].strip('\n')  # date string e.g. 2013-06-12

        # print( 'Done' )
        return name, date, size


# ****************************************************************************************************###################################################################
# import cProfile as cpr
if __name__ == '__main__':
    import os, sys, argparse
    from myio import iocheck, parsetolist
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
    args.cubes = parsetolist(args.cubes, os.path.exists,
                             path=path, raise_error=1)
    args.mags = parsetolist(args.mags, os.path.exists,
                            include=('mag', 'lc'),
                            path=path, raise_error=1)
    args.coo = parsetolist(args.coo, os.path.exists,
                           include='coo',
                           path=path, raise_error=0)

    # coords = np.loadtxt(args.coo, unpack=0, usecols=(0,1))
    # target = args.target	 #index of target    #WARN IF LARGER THAN LEN COORDS
    target_name, _, _ = get_name_date(args.cubes[0])
    run = Run(args.cubes, args.mags, args.coo,
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
