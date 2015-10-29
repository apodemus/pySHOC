# -*- coding: utf-8 -*-
import matplotlib as mpl
import pylab as plt
import numpy as np
#import numpy.linalg as la

import re
import os
import itertools as itt
from collections import Iterable

from matplotlib import cm, ticker
import matplotlib.gridspec as gridspec
import matplotlib.image as mimage               #LOCALISE IMPORTS??

import pandas as pd
from astropy.io import ascii
#import pyfits


from misc import interleave, sortmore, unique_rows, lzip, groupmore, make_ipshell
from myio import iocheck, parsetolist, warn
from superstring import ProgressBar, SuperString, as_superstrings
from superfits import quickheader
from draggables import *

#from lombscargle import fasper, getSignificance
from tsa.tsa import smooth
from tsa.spectral import *

#from misc import make_ipshell
ipshell = make_ipshell()
from myio import warn

##############################################################################################################################################
class Run(object):
    ''' Class for containing / merging / plotting Cube objects.'''
    #====================================================================================================
    def __init__(self, filenames, databases=None, coordinates=None, 
                    target=0, ref=1, target_name='', outpath=None):
        
        #from ApphotDatabase import ApphotDatabase
        self.coordinates = coordinates
        self.databases =  databases                                           #magnitudes database files 
        self.data_path = path
        
        #if outpath is None:
            #self.data_path = 
            
        #if isinstance(database, ApphotDatabase):
            #self.data_path = database.path
        #elif database:
            #self.data_path, _ = os.path.split( database )
        
        #if not self.data_path:
            #self.data_path, _ = os.path.split( filenames[0] )
            #if not self.data_path:
                #cwd = os.path.getcwd()
                #warn( 'Using CWD {} as data path...'.format( repr(cwd) ) )
                #self.data_path = cwd
        
        
        #self.nstars = Nstars
        self.ncubes = len(filenames)
        self._targets = self.make_iterable( target, self.ncubes )
        self._refs = self.make_iterable( ref, self.ncubes )
        
        self.target_name = target_name
        #self.date = date
        
        #full_filenames = [ os.path.join(self.data_path,fn) for fn in filenames ]
        self.filenames = filenames
        self.basenames = [ os.path.split(fn)[1].replace('.fits','') for fn in self.filenames ]
        
        self.cubes = [ Cube(fn, db, cf, t, r) for (fn, db, cf , t, r) in
                         itt.zip_longest(self.filenames, self.databases, self.coordinates, 
                                self._targets, self._refs) ]           #NOT DYNAMIC!
        
        self.current_cube = ''
    
    #====================================================================================================
    def __getitem__(self, key):
        '''Can be indexed numerically, or by corresponding filename / basename.'''
        if isinstance(key, str):
            if key.endswith('.fits'):
                key = key.replace('.fits','')
            key = self.basenames.index(key)            #self.cube_filenames.index(True)     ##BE CAREFUL WITH THIS......................................
                    
        return self.cubes[key]
        
    #====================================================================================================
    def __len__(self):
        return self.ncubes
    
    #====================================================================================================
    @staticmethod
    def make_iterable(obj, n):          #TODO: NEEDS WORK - see misc.as_iter
        if isinstance(obj, Iterable):
            assert len(obj)==n
            return obj
        else:
            return [obj]*n
    
    #====================================================================================================
    def get_outname(self, with_name=True, extension='txt', sep='.'):           #TODO: MERGE WITH gen_filename??
        if with_name:
            name = self.target_name.replace(' ','') + sep
        else:
            name = ''
        date_str = self.date.replace('-','')
        outname = '{}{}.{}'.format( name, date_str, extension )
        return os.path.join( self.data_path, outname )
    
    #====================================================================================================
    #def get_template(self, n):
        
        #self.template = read_data_from_file( args.ims, n )
        
        #return self.template
    
     
    #====================================================================================================
    def add_cube(self, filename, Nstars=None):
        '''add a cube to the run.'''
        Nstars = Nstars if Nstars else self.nstars
        basename = filename.replace('.fits','')
        filename = os.path.join( self.data_path, filename )
        
        self.filenames.append( filename )
        self.basenames.append( basename )
        self.cubes.append( Cube(filename, Nstars) )
        
    #====================================================================================================
    #def get_date(self, fn):
        #ds = pyfits.getval( self.template, 'date' )                #fits generation date and time
        #self.date = ds.split('T')[0].strip('\n')                         #date string e.g. 2013-06-12    
    
    #====================================================================================================
    def get_time(self, tkw):
        if not tkw in Cube.Tkw:
            raise ValueError( '{} is not an valid time format.  Choose from {}'.format(tkw, Cube.Tkw) )
        else:
            t = np.concatenate([getattr(cube, tkw) for cube in self])
        return t
        
    #====================================================================================================
    def print_summary_table(self):
        #TODO: Use Table from superstring.py
        '''print summary table for values read from database.'''
        
        def create_row(columns):
            '''apply properties each item in the list of columns create a single string'''
            columns = list( map(SuperString, columns) )
            col_widths =  [padwidth + col.ansi_len() for col in columns]
            columns = [col_format.format(col, cw) for col,cw in zip(columns,col_widths)]
            row = row_format.format(*columns)    
            return row
        
        def colourise_array(data, greenvals):
            if isinstance(greenvals, (int,float)):
                greenvals = [greenvals]*data.shape[0]
            if len(data.shape)==1:
                data = data[None]
            r,c = data.shape
            for ii in range(r):
                for jj in range(c):
                    n = data[ii,jj]
                    if      n == 0:                colour = 'red'
                    elif    n < greenvals[jj]:     colour = 'yellow'
                    elif    n == greenvals[jj]:    colour = 'green'
                    data[ii,jj] = as_superstrings(n, colour)
            return data.squeeze()
        
        table = []
        Ncols = len(self[0].stars)+2
        padwidth = max(len(fn) for fn in args.cubes) + 1
        col_format = '{0:<{1}}'
        row_format = '{}|' * (Ncols)
        
        top_line = col_format.format('',padwidth+1)*Ncols
        top_line = as_superstrings(top_line, 'underline')
        table.append(top_line)
        
        col_head = ['', 'Cube length'] + self[0].stars.get_names()
        col_head = [as_superstrings(col, 'green') if col==self[0][target].name else col for col in col_head]
        col_head = as_superstrings(col_head, 'bold')
        head_row = create_row(col_head)
        head_row = as_superstrings( head_row, 'underline' )
        table.append(head_row)
        
        cubelengths = [len(cube) for cube in self]
        Ndata = np.array( [[len(star) for star in cube.stars] for cube in self], dtype=object ).T
        Ntotals = Ndata.sum(axis=1)
        
        #print( ' greenvals', cubelengths )
        #from IPython import embed
        #embed()
        
        Ndata = colourise_array(Ndata, cubelengths)
        Ntotals = colourise_array(Ntotals, sum(cubelengths))
        
        columns = zip(args.cubes, cubelengths, *Ndata)
        for i, col_items in enumerate(columns):
            row = create_row( col_items )
            if i==len(args.cubes)-1:
                row = as_superstrings(row, 'underline' )
            table.append( row )
        
        columns = ['Total', sum(cubelengths)] + list(Ntotals)
        end_row = create_row(columns)
        end_row = as_superstrings( end_row, 'underline' )
        table.append( end_row )
        
        table = '\n'.join(table)
        self.summary_table = table
        print( table )
        print( '\n'*2 )
        
    
    #====================================================================================================
    def join(self, whichstar):          #MEMORY INEFFICIENT
        
        data = np.ma.concatenate( [cube[whichstar].data for cube in self] )
        err = np.ma.concatenate( [cube[whichstar].err for cube in self] )
        
        #star.ls_data = np.concatenate( [cube[whichstar].ls_data[:] for cube in self], axis=1 )
        
        #Grab the name, colour from the same star in the first cube
        coo = self[0][whichstar].coo
        name = self[0][whichstar].name
        colour = self[0][whichstar].colour
        
        return Star( data, err, coo, name, colour )
    
    
    #====================================================================================================
    def conjoin(self):
        '''Join cubes together to form contiguous data block.'''
        if len(self)==1:
            return self[0]
            
        cube = Cube( )
        
        #TODO:  match coordinates to stars...
        
        #stack star data
        N = min(len(cube.stars) for cube in run)
        for i in range(N):                         #WARNING! stars may be missing in some cubes??
            cube.stars.append( self.join(i) )
        
        #stack timing data
        for tkw in Cube.Tkw:
            t = np.hstack( [getattr(cube, tkw) for cube in run.cubes] )
            setattr(cube, tkw, t)
        
        #set total length
        cube._len = sum( len(cube) for cube in self )
        for which in ['targets', 'refs']:
            torr = getattr( self, '_'+which )
            if not np.equal( torr, torr[0] ).all():
                warn( 'Conjoining cubes with different {}!!!'.format(which) )
            
        cube._target = self._targets[0]
        cube._ref = self._refs[0]
        cube.stars.set_names( cube._target, cube._ref, self.target_name )
        cube.stars.set_colours( cube._target, cube._ref )
        cube.date = self[0].date
        
        return cube
        
    
    #====================================================================================================
    def compute_ls(self, tkw='lmst', **kw):
        t = self.get_time(tkw)
        
        for star in self.stars:
            star.compute_ls(t, **kw)
        
    #====================================================================================================
    #def plot_raw(self, ax, whichstar, tkw='lmst', **kw):
        #starname = self[0][whichstar].name
        #print( 'Plotting raw light curves for star {}'.format(starname) )
        
        #offset = kw['offset'] if 'offset' in kw else 0
        ##plots = []
        #whole = self.conjoin()
        #if len(whole[whichstar]):
            #label = cube[whichstar].name
            ##print('label', label)
            #t = getattr(whole,tkw)[ whole[whichstar].l ]
            #pl = whole[whichstar].plot_raw( ax, t, label=label )
            ##plots.append(pl)
        ##else:    
            ##print( 'WARNING: No data found for star {} in cube {}.'.format(starname, cube.filename) )
        #return pl
    
    #====================================================================================================
    def combine_ls(self, which_star):
        '''
        Interpolated rms for spectra with unequal frequency steps.
        '''
        from scipy import interpolate as inter
        
        if any( [len(cube[which_star].ls_data.f)>1 for cube in self] ):
            raise ValueError( 'Cannot (yet) combine LS data for multiple splits per cube' )
        
        F = [cube[which_star].ls_data.f[0] for cube in self if len(cube[which_star])]
        A = [cube[which_star].ls_data.A[0] for cube in self if len(cube[which_star])]
        leneq = [len(f)==len(F[0]) for f in F]
        if all(leneq):
            eq = [np.allclose(F[0],f) for f in F]
            if all(eq):
                return F[0], np.sqrt( np.mean( np.square(A), axis=0) )
        
        #truncate the spectra such that only overlapping data is used
        warn( 'The spectra have unequal frequency steps and/or range. Interpolation will be done to compute the combined rms spectrum' )
        f0 = [np.min(f) for f in F]
        fm = [np.max(f) for f in F]
        if np.ptp(f0):                                  #TOLERANCE?
            imaxfs = np.argmax( f0 )
            F = [f[f>=F[imaxfs][0]] for f in F]
            A = [a[f>=F[imaxfs][0]] for f,a in zip(F,A)]
        if np.ptp(fm):                                  #TOLERANCE?
            iminfs = np.argmin( fm )
            F = [f[f<=F[iminfs][-1]] for f in F]
            A = [a[f<=F[iminfs][-1]] for f,a in zip(F,A)]
        
        #Interpolate
        shortest = np.argmin( [f.shape for f in F] )
        #ensures that the longer arrays are interpolated to the frequency values of the shortes array
        A_interp = np.empty( (len(A),len(A[shortest])) )
        i = 0
        for f, a in zip(F, A):
            if i==shortest:
                A_interp[i] = a
            else:
                spline = inter.InterpolatedUnivariateSpline(f, a)
                A_interp[i] = spline( F[shortest] )                   #interpolated amplitude spectrum
            i += 1
        
        return F[shortest], np.sqrt( np.mean( np.square(A_interp), axis=0 ) )
        
    #====================================================================================================
    def plot_ls(self, which_stars=[0], fig=None, **kw):
        option = kw.pop('option') if 'option' in kw else 'combined'
        if option=='combined':
            for star in which_stars:
                f, A = self.combine_ls(star)
                cstar = Star()
                cstar.name = self[0][star].name
                cstar.colour = self[0][star].colour
                cstar.ls_data[:] = [f], [A], []
                cstar.plot_ls(fig, **kw)
            
        elif option=='stack':
            scube = Cube('', len(which_stars))
            for star in which_stars:
                scube[star].ls_data.f = [ cube[star].ls_data.f for cube in self if len(cube[star]) ]
                scube[star].ls_data.A = [ cube[star].ls_data.A for cube in self if len(cube[star]) ]
            scube.plot_ls(which_stars, fig, **kw)
    
    
##############################################################################################################################################
class Cube(object):
    ''' ... '''
    Tkw = ['jd', 'bjd', 'utc', 'utsec', 'lmst']     #'utsec'              #Available time formats
    
    #====================================================================================================
    def __init__(self, filename='', database=None, coordinates=None, target=None, ref=None):
        #self.coo = []  #star coordinates on image
        self.filename = filename
        self.basename = filename.replace('.fits','')
        self.database = database
        self.coordinates = coordinates
        
        if filename:
            self.target_name, self.date = get_name_date( filename )
        
        self.stars = Stars()                                    #NOTE: this is restrictive in that it fixes the Nstars, target, ref across all cubes.  Think of a way of dynamically creating the Stars class
        
        self._target = target
        self._ref = ref
        
        if filename: 
            self.set_times()
    
    #====================================================================================================
    def __getitem__(self, key):
        '''Can be indexed numerically, or by using 'target' or 'ref 'strings.'''
        if isinstance( key, (int, np.int_, slice) ):
            return self.stars[key]
        
        if isinstance( key, str ):
            if key.startswith( 't' ):
                return self.stars[self._target]
            if key.startswith( 'r' ):
                return self.stars[self._ref]
        else:
            raise KeyError
            
    #====================================================================================================
    def __len__(self):
        return len( self.stars[0] ) 
    
    #====================================================================================================
    def set_times(self):
        '''read timestamps from file.'''
        #path, fn = os.path.split( self.filename )
        #filesindir = next(os.walk(path))[-1]
        
        timefile = self.basename + '.time'
        #TODO: DO THIS READ INTELLIGENTLY
        self.utc, self.utsec, self.lmst, self.jd, self.bjd = np.loadtxt( timefile, usecols=(1,2,3,5,7), unpack=1 )
        #self.bjd -= 2457078
        
        #for tkw in Cube.Tkw:
            #tfn =  fn.replace('fits',tkw)
            #if tfn in filesindir:
                #ffn = os.join( path, tfn )
                #t = np.loadtxt( ffn )  #os.join( path,
            #else:
                #warn( 'Setting {} to None'.format(tkw) )
                #t = None
            
            #setattr(self, tkw, t)
    
    #====================================================================================================
    #def set_date(self, date):
        
        
    #====================================================================================================
    def load_data(self, filename=None, coofile=None):
        #mode='shoc',
        
        #for which in ['lc']:#['rlc', 'dlc']:
            
            #print( 'Loading light curve data from file {}...'.format(filename) )
            #masterdata = np.loadtxt(filename, unpack=1)
            
            #if 'SHOC' in mode.upper():
                #filename = '{}.{}'.format( self.basename, which )
                #datastartcol, Nreadcols, Nskipcol = 2, 2, 0
            
            #if mode.upper() in ['SCAM' ,'SALT']:
                ##filename = 
                #datastartcol, Nreadcols, Nskipcol = 6, 2, 2
                #self.utsec = masterdata[1]
                
            #for i in itt.count():
                #if datastartcol + Nskipcol*i + Nreadcols*(i+1) > len(masterdata):
                    #break
                #idx0 = datastartcol + (Nreadcols+Nskipcol)*i
                
                #self.stars.append( *masterdata[idx0:idx0+Nreadcols] )

        print( 'loading data. Patience is a virtue' )
        filename        = filename or self.database
        coordinates     = coofile or self.coordinates
        #print( coordinates )
        #self.load_from_apcor( filename, coordinates )
        self.load_from_mag( filename, 3 )
    
    #====================================================================================================
    def load_from_mag(self, filename, nap=3):
        '''load data from daophot multi-aperture ascii database.'''
        datafield = 'FLUX%i'%nap
        errorfield = 'MERR%i'%nap
        data = ascii.read( filename or self.database )

        cooinit = lzip(data['XINIT'], data['YINIT'])
        coords  = lzip(data['XCENTER'], data['YCENTER'])
        
        for _, (_, dat, err, coo) in groupmore( None, cooinit, data[datafield], data[errorfield], coords ):
            self.stars.append(dat, err, coo )
        
    #====================================================================================================
    def load_from_apcor(self, filename, coofile, splitfile, ):
        '''load data from aperture correction ascii database.'''
        data = np.genfromtxt(filename, dtype=None, skip_header=2, names=True)
        
        kcoords = np.loadtxt( coofile )
        coords = np.array((data['Xcenter'], data['Ycenter'])).T
        
        #group the stars by the closest matching coordinates in the coordinate file list.
        #starmatcher = lambda coo: np.sqrt(np.square(kcoords-coo).sum(1)).argmin()    
        
        #for starid, (coo, dat, err) in groupmore( starmatcher, coords, data['Mag'], data['Merr'] ):
            #self.stars.append(dat, err, coo )
            
        

            
    #====================================================================================================
    def datablock(self, which='raw', with_error=False):
        if with_error:
            return np.ma.vstack( [(star.get_data(which)[0], star.err) for star in self] )
        else:
            return np.ma.vstack( [star.get_data(which)[0] for star in self] )
    
    #====================================================================================================
    def optimal_smoothing(self):
        '''Determine the optimal amount of smoothing for differential photometry
        by minimising the variance of the differentail light curve for the given 
        length of the smoothing window.  Note: This operation may be slow...'''
        
        data = self.datablock('clipped')
        
    
    #====================================================================================================
    def compute_dl(self, tkw='lmst', straight=True, **kw):
        print('Computing differential light curve..')
        
        smoothing       = kw.get( 'smooth' )
        poly            = kw.get( 'poly' )
        mean            = kw.get( 'mean' )
        
        if straight:
            for j, star in enumerate(self):
                print('Star %i' %j)
            
                star.dl = star.data / self['r'].data                        #NEED ERROR CALCULATION!!!!!!!
                #star.dl_diff = diff
        
        #mag_ref = np.ma.array( mag_ref, mask=self[ref].clipped.mask )
        
        #NEEDS WORK --> COMBINED POLY + SMOOTH DETRENDING... (BILINEAR FILTERING???)
        if mean:
            compix = tuple(set(range(len(self.stars))) - {self._target})
            comps = conjc.datablock()[compix,:]         #data for comparison stars
            ref = comps.mean(0)
            #diff = np.mean(comps - np.atleast_2d(np.ma.median(comps, 1)).T
        
        else:
            ref, l = self['ref'].get_data('raw')     #'clipped'
        
        if smoothing:
            ref = smooth(ref, smoothing, fill=None      , return_type='unmasked')
            l = np.ones( len(ref), dtype=bool )
            
        if poly:
            t = getattr( self, tkw )
            coof = np.polyfit(t[l], mag_ref[l], poly)
            ref = np.polyval(coof, t)
        
        diff = ref - np.ma.median(ref)
        
        #Error computation
        #mean_err = (1./len(err_ref))*la.norm(err_ref)                                               #Add errors in quadrature
        #diff_err = np.array( [la.norm([err, mean_err]) for err in err_ref] )                        #Add errors in quadrature
        
        for j, star in enumerate(self):
            print('Star %i' %j)
            
            mag = star.data                             #OR CLIPPED???
            star.dl = mag - diff                           #NEED ERROR CALCULATION!!!!!!!
            star.dl_diff = diff
            
        #print np.array( [np.linalg.norm([e1, e2]) for e1,e2 in zip(err,diff_err)] )
        #err_dl = np.array( [np.linalg.norm([e1, e2]) for e1,e2 in zip(err,diff_err)] )              #Add errors in quadrature
        #star.dl = np.array([mag_dl, err_dl])
        #print('*'*3, star.dl.shape)
        
        #return coof

    #====================================================================================================
    def plot_lc(self, ax, whichstars=None, which='raw', tkw='lmst', **kw):
        
        which = which.lower()
        #TODO: Make the whichstars argument redundant by enabling legend picking.....
        whichstars = whichstars         if not whichstars is None       else range(len(self.stars)) 
        #offsets = kw['offsets']         if 'offset' in kw               else np.zeros(len(whichstars))
        mode = kw.get( 'mode', 'flux' )
        plots = []
        
        for whichstar in whichstars:

            star = self[whichstar]
            starname = star.name

            print( 'Plotting {} light curve for star {}'.format(which, starname) )
            
            if len(star):
                t = getattr(self, tkw)
                pl = star.plot_lc( ax, t, which, label=starname )
                plots.append(pl)
            #else:    
                #print( 'WARNING: No data found for star {} in cube {}.'.format(starname, cube.filename) )
        
        if which.startswith('r'):
            desc, extra = 'Raw', ''
        elif which.startswith('d'):
            desc, extra = 'Differential', ''
        elif which.startswith('c'):
            desc, extra = '', '  ({}$\sigma$ clipped)'.format(star.sig_thresh)
        

        title = '{} light curve of {} on {}.{}'.format(desc, self['target'].name, self.date, extra)
        ax.set_title( title )
        ax.set_xlabel( tkw.upper() )
        if mode=='flux':
            ax.set_ylabel( 'Instr. Flux' )
        else:
            ax.set_ylabel( 'Instr. mag' )
            ax.invert_yaxis()
        ax.grid()
        
        white_frac = 0.025
        xl, xu, xd = t.min(), t.max(), t.ptp()
        ax.set_xlim( xl-white_frac*xd, xu+white_frac*xd )
        
        return plots
    
    #====================================================================================================
    def load_ls(self, fn):
        Nstars = len(self.stars)
        ls_data = np.loadtxt( fn, unpack=1 )
        frq = ls_data[0]
        #assert ls_data.shape[0] == Nstars//2
        for i in range(Nstars):
            ls_data[:,i+1]
            self.stars[i].ls_data = frq, ls_data[:,i+1], None
    
    #====================================================================================================        
    def save_ls(self, fn):
        Amps = [ star.ls_data.A[0] for star in self ]
        data = [ self[0].ls_data.f[0] ] + Amps          #frequencies should be the same for all the stars.  If not, we have problems
        outarray = np.array( data )
        np.savetxt(fn, outarray)
    
    #====================================================================================================
    def compute_ls(self, tkw='lmst', **kw):
        print( 'Computing Lomb Scargle spectrum for cube {}'.format(self.filename) )
        if not tkw in Cube.Tkw:
            raise ValueError( '{} is not an valid time format'.format(tkw) )
        else:
            'warn if time crosses 00.00h for utc or lmst ---> aliases'
            t = getattr(self, tkw)
        
        #'which': [],'set_ls': []
        whichstars = kw.pop('whichstars')       if 'whichstars' in kw   else range(len(self.stars))
        which = kw.pop('which')                 if 'which' in kw        else 'raw'
        set_ls = kw.pop('set_ls')               if 'set_ls' in kw       else 1
        
        for idx in whichstars:
            star = self[idx]
            if len(star.data):
                signal, l = star.get_data( which )
                    
                print( 'kw', kw, len(signal), len(t) )
                #ipshell()
                
                T, Frq, Amp, Sig = star.compute_ls(t, signal, **kw)
                
                if set_ls:
                    star.ls_data[:] = Frq, Amp, Sig
                else:
                    'create data array and return it'
                return T, Frq, Amp, Sig
    
    #====================================================================================================
    def plot_ls(self, whichstars=None, fig=None, **kw ):
        
        whichstars = whichstars if whichstars else [self._target, self._ref]
        n_sp = len(whichstars)
        height_ratios = [3] + [1]*n_sp
        gs = gridspec.GridSpec(n_sp, 1, height_ratios=height_ratios)
        
        fig = fig if fig else plt.figure()
        self.ls_plots = []
        for i,g in zip(whichstars, gs):
            if i==whichstars[0]:
                ax = fig.add_subplot(g)
            else:
                ax = fig.add_subplot(g, sharex=ax)
            
            print('Plotting LS for star %i.' %i)
            setup_ticks(ax)
            plts = self[i].plot_ls( ax, 0., **kw )
            self.ls_plots.append( plts )
        
        plt.suptitle( 'LS periodogram of {} in {}'.format(cube['target'].name, cube.filename) )
        #ax.set_xlim(f[0],f[-1])
        ax.set_xlabel('f (Hz)')
        ax.legend(loc='best')
        
    #====================================================================================================
    def save_lc(self, fn=None, which='raw'):

        fn = fn or '{}.lc'.format( self.basename )
        if os.path.exists( fn ):
            yn = input( 'A file named {} already exists! Overwrite ([y]/n)?? '.format(fn) )
        else:
            yn = 'y'

        if yn in ['y','Y','']:

            print( 'Halleluja!', fn )

            nstars = len(self.stars)

            tsec = (self.lmst - self.lmst[0])*3600.
            T = np.c_[ tsec, self.bjd ].T

            data = np.r_[ T, self.datablock(which, 1) ].T
            
            fmt = ('%15.9f', '%18.9f') + ('%6.3f',)*2*nstars
            col_head = ['', ''] + interleave( self.stars.get_names(), ['']*nstars)
            col_head2 = ['T (sid. sec)', 'BJD'] + ['MAG','MERR']*nstars
            
            make_head = lambda info: '\t'.join( [ s.ljust( int(re.match( '%(\d{1,2})', f).groups()[0])) 
                                                    for s,f in zip(info,fmt) ] )   
            
            header0 = 'Magnitudes for {} stars extracted from {}.'.format(nstars, self.database)
            header1 = make_head( col_head )
            header2 = make_head( col_head2 )
            header = '\n'.join( [header0, header1, header2] )

            np.savetxt( fn, data, header=header, fmt=fmt, delimiter='\t' )
    
    #====================================================================================================
    def heteroskedacity( self, whichstars, nwindow, window='flat', tkw='lmst' ):
        
        fig, ax = plt.subplots()
        
        for whichstar in whichstars:
            star = self[whichstar]
            t = getattr( self, tkw )
            ax.plot(t , star.var, star.colour, label=star.name )
        
        white_frac = 0.025
        xl, xu, xd = t.min(), t.max(), t.ptp()
        ax.set_xlim( xl-white_frac*xd, xu+white_frac*xd )
        
        ax.set_title( 'Heteroskedacity' )
        ax.set_ylabel( r'Variance $\sigma^2 (N={})$'.format(nwindow) )
        ax.set_xlabel( tkw.upper() )
        ax.legend()
    
    #====================================================================================================
    def minvar(self):
        '''return index of star with minimal variance'''
        return np.ma.var( self.datablock(), axis=1 ).argmin()
        
    #====================================================================================================
    def check_ref(self, ref):
        #check which star has mininal variance
        minvar = self.minvar()
        
        if not ref is None:
            if ref != minvar:
                warn( 'The selected reference star {} is not the one with the lowest variance!  Use {} instead.'.format(args.ref, minvar) )
            return ref
            
        else:
            print( 'Using star {} as reference'.format(minvar) )
            return minvar
        

    #====================================================================================================
    def psd(self, t, whichstar=0):
        '''Power Spectral density map'''
        
        
        #maxlen = max(maplen)
        #if np.any(maplen!=maxlen):
            #for i in range(len(amp)):
                #pad = maxlen - len(amp[i])
                #if pad:
                    #amp[i] = np.pad( amp[i], (0,pad), mode='constant', constant_values=0 )
        
        frq = self[whichstar].ls_data.f
        amp = self[whichstar].ls_data.A
        
        maplen = np.fromiter((map(len, amp)), dtype=int)
        idxs = np.where( maplen != len(amp[0]))[0]
        for idx in idxs:
            frq.pop(idx)
            amp.pop(idx)
            t.pop(idx)
        
        t, frq, amp = np.array(t), np.array(frq[0]), np.array(amp)
        
        # definitions for the axes
        left, bottom, width, height = 0.1, 0.1, 0.65, 0.65
        spacing = 0.01
        bottom_h = left_h = left + width + spacing
        
        rect_map = [left, bottom, width, height]
        rect_lc = [left, bottom_h, width, 0.2]
        rect_spec = [left_h, bottom, 0.2, height]
        
        fig = plt.figure()
        ax_map = fig.add_axes(rect_map)
        ax_lc = fig.add_axes(rect_lc, sharex=ax_map)
        ax_spec = fig.add_axes(rect_spec, sharey=ax_map)
        
        l = ~self[whichstar].data.mask
        ax_lc.plot(self.lmst[l], self[whichstar].data[l], 'go', ms=2.5)
        
        ax_spec.plot(self[whichstar].ls_data.A[0], self[whichstar].ls_data.f[0], 'g')                   #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #ax_spec.set_xscale('log')
        
        #Plot NonUniformImage of data set
        im = mimage.NonUniformImage( ax_map, origin='lower', extent=(t[0],t[-1],frq[0],frq[-1]) )
        im.set_clim( 0,100 )
        data = im.cmap( amp.T )
        
        #im.set_figure( fig_map )
        
        #detect_gaps
        #narrow_gap_inds = Spectral.detect_gaps(t, self.KCT*128/3600, 2)
        #narrow_gap_inds = narrow_gap_inds
        ##print( narrow_gap_inds )
     
        ##fade gaps
        #alpha = data[...,-1]
        #alpha[:,narrow_gap_inds] = 0.25
        #alpha[:,narrow_gap_inds+1] = 0.25
        
        #print( t.shape, frq.shape, data.shape )
        
        im.set_data(t, frq, data )
        ax_map.images.append(im)
        ax_map.set_xlim(t[0],t[-1])
        ax_map.set_ylim(frq[0],frq[-1])
        ax_map.figure.colorbar(im, ax=(ax_map,ax_spec), orientation='horizontal', fraction=0.01, pad=0.1, anchor=(1,0), aspect=100*(width+spacing), extend='max')
        
        #Ticks and labels
        setup_ticks(ax_map)
        #ax_map.tick_params(which='both', labeltop=0)
        plt.setp( ax_lc.get_xmajorticklabels(), visible=False )     #set major xticks invisible on light curve plot
        plt.setp( ax_lc.get_xminorticklabels(), visible=False )     #set minor xticks invisible on light curve plot
        plt.setp( ax_spec.get_ymajorticklabels(), visible=False )   #set yticks invisible on frequency spectum plot
        plt.setp( ax_spec.get_yminorticklabels(), visible=False )
        ax_lc.invert_yaxis()
        ax_lc.set_ylabel('mag')
        ax_spec.set_xlabel('Power')
        ax_map.set_xlabel('LMST')
        ax_map.set_ylabel('f (Hz)')
        return im
        
    
############################################################################################################################################## 
class Stars(object):
    COLOURS = ['b', 'g', 'm', 'c', 'k', 'r', 'y','orange']   #colormap???
    
    #====================================================================================================
    def __init__(self):
        #self.target = target
        #self.ref = ref
        
        self.stars = [ ]
        #self.set_names()                                                #SHOULD FORM PART OF STAR __INIT__
        #self.set_colours()

    #====================================================================================================
    def __getitem__(self,n):
        return self.stars[n]
        
    #====================================================================================================
    def __setitem__(self, key, val):
        self.stars[key] = val
        
    #====================================================================================================
    def __len__(self):
        return len(self.stars)
    
    #====================================================================================================
    def append(self, data=None, err=None, coo=None):
        
        if isinstance( data, Star ):
            self.stars.append( data )
        else:
            self.stars.append( Star(data, err, coo) )
        
    #====================================================================================================
    def set_names(self, target, ref, target_name, basename=None):       #THS SHOULD BE A METHOD OF CUBE
        count = 1
        for i, star in enumerate(self):
            if i==target:
                star.name = target_name
            #elif i==ref:
                #star.name = 'ref'                       #FIXME:  RELEVANCE??
            else:
                star.name = 'C'+str(count)
                count += 1
        
            #name = star.name.replace(' ','_')
            #star.fn_LS = '{}.{}.ls'.format(basename, name)                         #Lomb-Scargle txt file
    
    
    #====================================================================================================
    def get_names(self):
        return [star.name for star in self]
    
    #====================================================================================================
    def set_colours(self, target, ref):              #THS SHOULD BE A METHOD OF CUBE   
        '''
        Set the plot colours for the stars.  
        Ensures target and ref always have the same colours (0,1) indices Stars.COULOURS
        '''
        
        colours = Stars.COLOURS[:]      #create a copy
        self[target].colour = colours.pop(0)
        if not ref is None:
            self[ref].colour = colours.pop(1)
        
        i,j = 0,0
        while i<len(self):
            if not (i==target or i==ref):
                self[i].colour = colours[j]
                j += 1
            i += 1
            
        
    #====================================================================================================
    #def allmags(self):
        ##alldata = np.empty( (len(self), len(self[shortest].data) ,2) )
        ##[q.T for q in s.allmags()]
        #return np.ma.hstack( [s.data for s in self] )
    
    ##====================================================================================================
    #def clipper(self):                                                                                          #THIS IS MEMORY INEFFICIENT!!!!!!!
        #for star in self:
            #for cube in star:
                #cube.clipper()
        


    
##############################################################################################################################################
class Star(object):
    ''' ... '''
    #====================================================================================================
    def __init__(self, data=None, err=None, coo=None, name=None, colour=None):
        #datalen = datalen if datalen else 0
        if not data is None:
            if np.ma.isMA(data):
                self.data = data                         #raw data points (mag)
                self.err = err                          #errors
            else:
                #try:
                self.data = np.ma.array( data, mask=np.isnan(data) )
                self.err = np.ma.array( err, mask=np.isnan(err) )
                #except:
                    #ipshell()
                
        #self.dl = np.empty( (datalen,2) )                      #differential light curve (mag,err)
        #self.l = np.empty( (datalen,2), dtype=bool )           #logical array user for removing bad values from data
        self.coo = np.array(coo)
        
        self.ls_data = SpectralDataWrapper()                    #Lomb Scargle data
        
    #====================================================================================================
    def __len__(self):
        if ~self.data.mask.any():
            return len(self.data)
        else:
            return sum(~self.data.mask)
    
    #====================================================================================================
    def get_data(self, which):
        if which=='raw':
            mag = self.data
            l = ~self.data.mask
        if which=='clipped':
            mag = self.clipped
            l = ~self.clipped.mask
        if which=='diff':
            mag = self.dl
            l = ~self.dl.mask
        return mag, l
    
    #====================================================================================================
    def running_stats(self, nwindow, center=True, which='clipped'):
        
        #first reflect the edges of the data array
        #if center is True, the window will be centered on the data point - i.e. data point preceding and following the current data point will be used to calculate the statistics (mean & var)
        #else right window edge is aligned with data points - i.e. only the preceding values are used.
        #The code below ensures that the resultant array will have the same dimension as the input array
       
        x, _ = self.get_data( which )
        
        if center:
            div, mod = divmod( nwindow, 2 )
            if mod: #i.e. odd window length
                pl, ph = div, div+1
            else:  #even window len
                pl = ph = div
            
            s = np.ma.concatenate([ x[pl:0:-1], x, x[-1:-ph:-1] ])  #pad data array with reflection of edges on both sides
            iu = -ph+1
        
        else:
            pl = nwindow-1
            s = np.ma.concatenate([ x[pl:0:-1], x ])                  #pad data array with reflection of the starting edge
            iu = len(s)
        
        s[s.mask] = np.nan
        max_nan_frac = 0.5            #maximum fraction of invalid values (nans) of window that will still yield a result
        mp = int( nwindow*( 1 - max_nan_frac ) )
        self.median = pd.rolling_median( s, nwindow, center=center, min_periods=mp )[pl:iu]
        self.var = pd.rolling_var( s, nwindow, center=center, min_periods=mp )[pl:iu]
        
        #return med, var
    
    #====================================================================================================
    def plot_clippings(self, *args):
        #TODO:  plot clippings from outliers.py
        ax, t, nwindow = args
        self.ax_clp = ax
        med = self.median
        std = np.sqrt( self.var )
        lc = self.data
        threshold = self.sig_thresh
        
        ax.plot( t, self.clipped, 'g.', ms=2.5, label='data' )
        ax.plot( t[self.clipped.mask], lc[self.clipped.mask], 'x', mfc='None', mec='r', mew=1, label='clipped' )

        #print( 'top', len(top), 'bottom', len(bottom), 't', len(t[st:end]) )
        sigma_label = r'{}$\sigma$ ($N_w={}$)'.format(threshold, nwindow)
        median_label = r'median ($N_w={}$)'.format(nwindow)
        ax.plot( t, med+threshold*std, '0.6' )
        ax.plot( t, med-threshold*std, '0.6', label=sigma_label )
        ax.plot( t, med, 'm-', label=median_label )

        #clp = sigma_clip( lcr, 3 )
        #ax.plot( t[clp.mask], lcr[clp.mask], 'ko', mfc='None', mec='k', mew=1.75, ms=10 )
        #m, ptp = np.mean(med), np.ptp(med)
        #ax.set_ylim( m-3*ptp, m+3*ptp )
        
        white_frac = 0.025
        xl, xu, xd = t.min(), t.max(), t.ptp()
        ax.set_xlim( xl-white_frac*xd, xu+white_frac*xd )
        
        ax.set_title( self.name )
        ax.invert_yaxis()
        ax.legend( loc='best' )
        
    
    #====================================================================================================
    def smooth(self, nwindow, window='hanning', fill='mean'):
        #data = self.clipped[~self.clipped.mask]
        self.smoothed = smooth( self.clipped, nwindow, window, fill)
        return self.smoothed
    
    #====================================================================================================
    def argmin(self):
        return np.argmin(self.counts())
    
        
    #====================================================================================================
    def compute_ls(self, t, signal, **kw):
        print( 'Doing Lomb Scargle for star {}'.format(self.name) )
        '''Do a LS spectrum on light curve of star.
        If 'split' is a number, split the sequence into that number of roughly equal portions.
        If split is a list, split the array according to the indeces in that list.'''
        return Spectral.compute(t, signal, **kw)
  
  
    #def set_ls(self, f, A, signif):
        #self.ls_data.f, self.ls_data.A, self.ls_data.signif = f, A, signif
  
    #def get_ls(self):
        #return self.ls_data.f, self.ls_data.A, self.ls_data.signif
    
    
    #====================================================================================================
    def plot_ls(self, ax=None, offsets=None, **kw):
        #TODO: OFFSETS are redundant with draggables
        '''Plot the LS spectra for the star.'''
        print( 'Plotting Lomb Scargle spectrum for star {}'.format(self.name) )
        #fignum = eval(str(2)+'.'+str(which))
        #plt.figure(fignum*10, figsize=figsize)
        
        if not ax:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        
        which = kw['which'] if 'which' in kw else ''
        yscale = kw['yscale'] if 'yscale' in kw else 'log'
        
        #if x is None:
            #x = self.ls_data.f
        #if y is None:
            #y = self.ls_data.A
            
        n = len(self.ls_data.f)				#number of splits for data
        offsetmap = 10**np.arange(n) if yscale=='log' else 10*np.arange(n)
        offsets = offsets if offsets!=None else offsetmap
        offsets = offsets if isinstance(offsets, Iterable) else [offsets]*n
        #colour = 'b' if which==self.target else 'r'
        
        plots = []
        for j in range(n):
            f = self.ls_data.f[j]
            A = self.ls_data.A[j]
            
            label = '{}{}'.format(self.name, (which if j==0 else '') )
            #print 'label ', label
            setup_ticks( ax )
            if yscale=='log':             plt.yscale('log')
            pl = ax.plot(f, A + offsets[j], self.colour, label=label)
            plots.append( pl )
            #ax.plot(wk1,signif,label = 'False detection probability')
            ax.legend()
            ax.set_xlabel('f (Hz)')
            ax.set_ylabel('LS Power')
            #fig.text(0.,0., 'Detrend: n=%i'%dtrend, transform=fig.transFigure)
        
        self.ls_plots = plots
        return plots
  
    #====================================================================================================
    def plot_lc(self, ax, t, which='raw', **kw):
        
        label = self.name if 'label' in kw and kw['label'] else ''
        fmt = 'o'
        
        mag, l = self.get_data( which )
        err = self.err
        
        if isinstance(l, (bool, np.bool_)):
            pl = ax.plot(t, mag, fmt, ms=2.5, color=self.colour,  label=self.name)
        else:
            pl = ax.plot(t[l], mag[l], fmt, ms=2.5, color=self.colour, label=self.name)
        
        #erbpl = ax.errorbar(t, mag, yerr=err, fmt=fmt, ms=1, label=label)
        
       
        #print('Done!')
        return pl#erbpl
  

##############################################################################################################################################  

#====================================================================================================
def get_name_date( fn=None ):
    
    if fn:
        print( "Getting 'OBJECT' and 'DATE' info from {}... ".format( os.path.split(fn)[1] ) )
        header = quickheader( fn )
        name = header['OBJECT']
        try:
            date = header['DATE-OBS']
        except KeyError:
            ds = header['DATE']                 #fits generation date and time
            date = ds.split('T')[0].strip('\n') #date string e.g. 2013-06-12
        
        #print( 'Done' )
        return name, date
   

    
#################################################################################################################################################################################################################
#import cProfile as cpr
if __name__=='__main__':
    
    import argparse, os
    bar = ProgressBar()

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-d', '--dir', default=os.getcwd(), dest='dir', help = 'The data directory. Defaults to current working directory.')
    parser.add_argument('-x', '--coords', default='all.coo', dest='coo', help = 'File containing star coordinates.')
    parser.add_argument('-m', '--mags', default='all.mags', dest='mags', help = 'Database file containing star magnitudes.')
    parser.add_argument('-c', '--cubes', nargs='+', type=str, help = 'Science data cubes to be processed.  Requires at least one argument.  Argument can be explicit list of files, a glob expression, or a txt list.')
    parser.add_argument('-w', '--write-to-file', action='store_true', default=False, dest='w2f', help = 'Controls whether the script writes the light curves to file.')
    #parser.add_argument('-e', '--extract', action='store_true', default=False, help = 'Extract magnitudes from database?')
    parser.add_argument('-t', '--target', default=0, type=int, help = 'The position of the target star in the coordinate file.')
    parser.add_argument('-r', '--ref', type=int, help = 'The position of the reference star in the coordinate file.')
    parser.add_argument('-i', '--instrument', default='SHOC', nargs=None, help='Instrument. Switches behaviour for loading the data.')
    parser.add_argument('-l', '--image-list', default='all.split.txt', dest='ims', help = 'File containing list of image fits files.')

    #Arguments for light curve / spectral analysis
    parser.add_argument('-dl', '--diff-phot', default=True, action='store_true', dest='dl', help="Perform differential photometry?  The star with lowest variance will be used reference star ( Unless explicitly given via the 'r' argument.)")
    parser.add_argument('-ls', '--lomb-scargle', default=False, action='store_true', dest='ls', help='Perform Lomb-Scargle periodogram on light curves?')
    args = parser.parse_args()

    path = iocheck(args.dir, os.path.exists, 1)
    path = os.path.abspath(path) + os.sep
    args.cubes = parsetolist(args.cubes, os.path.exists, path=path)
    args.mags = parsetolist(args.mags, os.path.exists, read_ext='.mag', path=path)
    args.coo = parsetolist(args.coo, os.path.exists, read_ext='.coo', path=path)

    #coords = np.loadtxt(args.coo, unpack=0, usecols=(0,1))
    target = args.target	        			  #index of target                              #WARN IF LARGER THAN LEN COORDS
    target_name, _ = get_name_date( args.cubes[0] )
    run = Run( args.cubes, args.mags, args.coo,
                args.target, args.ref, target_name )
    #run.target = target

    for cube in run:
        #cube.load_data( args.instrument, args.mags )
        cube.load_data( )
        cube.set_times( )
        
        cube._ref = ref = cube.check_ref( args.ref )
        
        cube.stars.set_names( target, ref, cube.target_name )
        cube.stars.set_colours( target, ref )
        

    conjc = run.conjoin()

    if args.ls:
        
        for cube in run:
            fn = '{}.ls'.format( cube.basename )
            if os.path.exists( fn ):
                cube.load_ls( fn )
            else:
                cube.compute_ls( set_ls=1 )                   #compute ls and save data_path
            
            if args.w2f:
                cube.save_ls()
        
        #for combined time series of entire run
        
        conjc.compute_ls( set_ls=1 )
        
        if args.w2f:
            outname = run.get_outname( with_name=False, extension='lc' )
            conjc.save_lc( outname )
        

    #==================================================================================================== 

    #Plots setup
    plt.close('all')

    mpl.rc( 'figure', figsize=(18,8) )                                #set the figure size for this session
    mpl.rc( 'savefig', directory=path )
    mpl.rc( 'figure.subplot', left=0.065, right=0.95, bottom=0.075, top=0.95 )    # the left, right, bottom, top of the subplots of the figure
    #mpl.rc( 'figure', dpi=600 )

    def setup_ticks(ax):  
        majorLocator = ticker.AutoLocator()                 #NEED PLOT INIT METHOD.....................................
        #majorFormatter = ticker.FormatStrFormatter('%.2f')
        minorLocator   = ticker.AutoMinorLocator()
        minorFormatter = ticker.ScalarFormatter()
        ax.xaxis.set_major_locator(majorLocator)
        #ax.xaxis.set_major_formatter(majorFormatter)
        ax.xaxis.set_minor_locator(minorLocator)
        ax.xaxis.set_minor_formatter(minorFormatter)
        ax.tick_params(axis='both',which='both',direction='out')
        ax.tick_params(axis='x',which='minor',pad=4, labelsize=10)
        #ax.fmt_xdata = lambda x: "{0:f}".format(x)
            
        #ax2.minorticks_on()

        #ax2.axis((0,5, 0,1.1*max(wk2)))

    #######################################################################################################################
    #RAW LIGHT CURVE

    fig0, ax0 = plt.subplots( )

    tkw = 'utc'
    starplots = conjc.plot_lc( ax0, tkw=tkw, mode='flux' )

    drag = DraggableLine( starplots, markerscale=2 )
    drag.connect()


    #handler_map = {ErrorbarContainer: ReorderedErrorbarHandler(numpoints=2)}
    #leg = ax0.legend(handler_map=handler_map, loc='best')
    #lined = map_legend_dict(leg, erbpls)

    #fig0.canvas.mpl_connect('pick_event', onpick)



    #######################################################################################################################
    #SIGMA CLIPPING & SMOOTHING

    #from outliers import *

    #threshold = 5
    #nwindow = 50
    #Nstars = len(conjc.stars)
    #t = getattr( conjc, tkw )
    ##Clip at thereshold and plot clipped points
    ##print( 'check outlier detection ' )
    ##ipshell()
    #for j in range(Nstars):
        ##star = conjc[j]
        ##fig, ax = plt.subplots( )
        
        ##star.clipped = running_sigma_clip( star.data, threshold, nwindow )
        ##star.sig_thresh = threshold
        #star.running_stats( nwindow )
        
        #star.plot_clippings(ax, t, nwindow )
        
        ##plot smoothed light curves for reference star
        ##if j==ref:
            ##refstar = conjc[j]
            ##clpd = refstar.clipped
            ##colours = ['c', 'm', 'y', 'orange', 'r']
            ##for N, c in zip( range(100,501,100), colours ):
                ##s = refstar.smooth(N)
                ##ax.plot( t, s, c, label='smooth '+str(N) ) #[~clpd.mask]

            ##ax.legend( loc='best' )
            ##figname = run.get_outname('clp.ref.png')
            ##fig.savefig( figname )
            
        ##if j == target:
            ##figname = run.get_outname('clp.trg.png')
            ##fig.savefig( figname )

    ##plot sigma clipped light curves
    #fig1, ax1 = plt.subplots( )
    #whichstars = [target, ref]
    #starplots = conjc.plot_lc(ax1, whichstars, 'clipped', tkw)

    #plt.title( r'light curve of {} on {} '.format(run[0][target].name, run.date, threshold) )


    #######################################################################################################################
    #HETEROSKEDACITY
    #whichstars = range(Nstars)
    #conjc.heteroskedacity( whichstars, 100, tkw=tkw )


    #######################################################################################################################
    #DIFFERENTIAL LIGHT CURVE
    conjc.compute_dl( tkw, straight=True )		#smooth=50, poly=3				

    fig2, ax2 = plt.subplots( )

    #whichstars = [target, ref]
    starplots = conjc.plot_lc(ax2, which='diff', tkw=tkw, mode='flux')

    drag = DraggableLine( starplots, markerscale=2 )
    drag.connect()

    if args.w2f:
        outname = run.get_outname( with_name=False, extension='dlc', which='diff')
        conjc.save_lc( outname )
        


    raise SystemExit
    #######################################################################################################################
    #FILL DATA GAPS
    for i in [target, ref]:
        star = conjc[i]
        ax = star.ax_clp
        fill_mode = 'median'
        opt = 20
        Tfiller, Mfiller = Spectral.fill_gaps(t, star.dl, mode=fill_mode, option=opt, fill=False)

        *star.filled, idx = Spectral.fill_gaps(t, star.dl, mode=fill_mode, option=opt, ret_idx=1)
        
        #plot filler values
        lbl = 'filled ({} {})'.format(fill_mode, opt)
        M = np.polyval( polycoof, Tfiller )         #filled values projected onto the trend
        Mde = M - np.ma.mean(M)                      #de-project filled points to original data
        ax.plot(Tfiller, Mfiller+Mde, 'o', mew=1, mec='b', mfc='None', label=lbl)
        
        #plot differential trend
        if i==ref:
            tfilled = star.filled[0]
            trend = np.polyval( polycoof, tfilled )
            ax.plot( tfilled, trend, 'r-', label='polyfit (n=%i)'%(len(polycoof)-1) )
        
        ax.legend( loc='best' )
        ax.grid()

    #######################################################################################################################
    #NEED TO EXPAND TO AM / AR / SAS MODELLING!!!


    ######################################################################################################################



    #######################################################################################################################



    ######################################################################################################################

    ########################################################################################################################

    plt.show()

