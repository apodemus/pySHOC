#TODO:
#Parallel processing


print( 'Importing modules' )
import numpy as np
import numpy.linalg as la
import pyfits
import os

#from stsci.tools import capable

from scipy.spatial.distance import cdist

from matplotlib import pyplot as plt
from matplotlib import gridspec, rc
#from matplotlib.widgets import Button
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cbook
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
#from matplotlib.colors import colorConverter

from ApertureCollections import *
ApertureCollection.WARN = False
SkyApertures.WARN = False

#import multiprocessing as mpc
from zscale import Zscale
from SHOC_readnoise import ReadNoiseTable
#from completer import Completer
from SHOC_user_input import Input
from SHOC_user_input import ValidityTests as validity
from pySHOC import SHOC_Run

from misc import Table, flatten, interleave, sorter, warn, as_iter
from time import time
from copy import copy, deepcopy

#import multiprocessing as mpc

import itertools
from string import Template

from scipy.optimize import leastsq

#print( 'Done!')

#def background_imports( queue ):
#print( 'Doing background imports..' )
#print( 'Importing IRAF...' )
#from pyraf import iraf
#from stsci.tools import capable
#capable.OF_GRAPHICS = False                     #disable pyraf graphics                                 

#queue.put( (daofind, phot) )
print( 'Done!')

from misc import make_ipshell
ipshell = make_ipshell()

######################################################################################################    
# Decorators
######################################################################################################    
#needed to make ipython / terminal input prompts work with pyQt
from PyQt4.QtCore import pyqtRemoveInputHook, pyqtRestoreInputHook
def unhookPyQt( func ):
    '''Decorator that removes the PyQt input hook during the execution of the decorated function.
    Used for functions that need ipython / terminal input prompts to work with pyQt.'''
    def unhooked_func(*args, **kwargs):
        pyqtRemoveInputHook()
        out = func(*args, **kwargs)
        pyqtRestoreInputHook()
        return out
    
    return unhooked_func

######################################################################################################
# Misc Functions
######################################################################################################    

def Gaussian2D(p, x, y):
    '''Elliptical Gaussian function for fitting star profiles'''
    A, a, b, c, x0, y0 = p
    return A*np.exp(-(a*(x-x0)**2 + 2*b*(x-x0)*(y-y0) + c*(y-y0)**2))

def residuals( p, data, X, Y ):
    '''Difference between data and model'''
    return data - Gaussian2D(p, X, Y)

def err( p, data, X, Y ):
    return abs(residuals(p, data, X, Y).flatten() )
    

def starfucker( args ):
    
    p0, grid, star_im, sky_mean, sky_sigma, win_idx, = args
    
    Y, X = grid
    args = star_im, X, Y
    
    plsq, _, info, msg, success = leastsq( err, p0, args=args, full_output=1 )
    
    if success!=1:
        print( msg )
        print( info )
        return
    else:
        #print( '\nSuccessfully fit {func} function to stellar profile.'.format(func=func) )
        pdict = StellarFit.get_param_dict( plsq )
        
        skydict = {     'sky_mean'      :       sky_mean,
                        'sky_sigma'     :       sky_sigma,
                        'star_im'       :       star_im,
                        '_window_idx'   :       win_idx          }
        pdict.update( skydict )
        return Star( **pdict )

    


class StellarFit(object):
    #Currently only implemented for Gaussian2D!
    #===============================================================================================
    FUNCS = ['Gaussian', 'Gaussian2D', 'Moffat']
    #===============================================================================================
    def __init__(self, func='Gaussian2D', alg=None):
        ''' '''
        if alg is None:
            self.algo = leastsq
        else:
            raise NotImplementedError
        
        if func in self.FUNCS:
            self.F = getattr(self, func)
            if func=='Moffat':
                raise NotImplementedError( 'Currently only implemented for Gaussian2D' )
                self.to_cache = slice(1,3)
            elif func=='Gaussian2D':
                self.to_cache = slice(1,4)
        else:
            raise ValueError( 'func must be chosen from amongst {}'.format(self.FUNCS) )
        
        self.cache = []         #parameter cache
    
    #===============================================================================================    
    def __call__(self, xy0, grid, data):
        
        p0 = self.param_hint( xy0, data )
        Y, X = grid
        args = data, X, Y
        
        plsq, _, info, msg, success = self.algo( self.err, p0, args=args, full_output=1 )
        
        if success!=1:
            print( msg )
            print( info )
        else:
            print( '\nSuccessfully fit {func} function to stellar profile.'.format(func=self.F.__name__) )
            self.cache.append( plsq[self.to_cache] )
        
        return plsq
    
    #===============================================================================================    
    #@staticmethod
    def param_hint(self, xy0, data):
        '''initial parameter guess'''
        z0 = np.max(data)
        x0, y0 = xy0
        func = self.F.__name__
        
        if len(self.cache):
            cached_params = self.get_params_from_cache()
            
        elif func=='Moffat':
            cached_params = 1., 2.
        
        elif func=='Gaussian2D':    
            cached_params = 0.2, 0, 0.2
            
        return (z0,) +cached_params+ (x0, y0)
    
    #===============================================================================================    
    def get_params_from_cache(self):
        '''return the mean value of the cached parameters.  Useful as initial guess.'''
        return tuple(np.mean( self.cache, axis=0 ))
    
    #===============================================================================================    
    @staticmethod
    def get_param_dict( plsq ):
        A, a, b, c, x, y = plsq
        
        sigx, sigy = 1/(2*a), 1/(2*c)           #standard deviation along the semimajor and semiminor axes
        fwhm_a = 2*np.sqrt(np.log(2)/a)         #FWHM along semi-major axis
        fwhm_c = 2*np.sqrt(np.log(2)/c)         #FWHM along semi-minor axis
        fwhm = np.sqrt( fwhm_a*fwhm_c )         #geometric mean of the FWHMa along each axis
        
        ratio = min(a,c)/max(a,c)              #Ratio of minor to major axis of Gaussian kernel
        theta = 0.5*np.arctan2( -b, a-c )       #rotation angle of the major axis in sky plane
        ellipticity = np.sqrt(1-ratio**2)
        
        coo = x, y
        
        pdict = {'coo'          :       coo,
                'peak'          :       A,
                'fwhm'          :       fwhm,
                'sigma_xy'      :       (sigx, sigy),
                'theta'         :       np.degrees(theta),
                'ratio'         :       ratio,
                'ellipticity'   :       ellipticity          }
        
        return pdict
        
    #===============================================================================================    
    @staticmethod
    def print_params( pdict ):
        
        print( ('Stellar fit parameters: \nCOO = {coo[0]:3.2f}, {coo[1]:3.2f}'
                                        '\nPEAK = {peak:3.1f}'
                                        '\nFWHM = {fwhm:3.2f}'
                                        '\nTHETA = {theta:3.2f}'
                                        '\nRATIO = {ratio:3.2f}\n'
                                        
                'Sky parameters:         \nMEAN = {sky_mean:3.2f}'
                                        '\nSIGMA = {sky_sigma:3.2f}'
                                        
                ).format( **pdict ) )
    
    #===============================================================================================    
    @staticmethod
    def Gaussian(p, x):
        '''Gaussian function for fitting radial star profiles'''
        A, b, mx = p
        return A*np.exp(-b*(x-mx)**2 )
    
    @staticmethod
    def Gaussian2D(p, x, y):
        '''Elliptical Gaussian function for fitting star profiles'''
        A, a, b, c, x0, y0 = p
        return A*np.exp(-(a*(x-x0)**2 + 2*b*(x-x0)*(y-y0) + c*(y-y0)**2 ))
    
    @staticmethod
    def Moffat(p, x, y):
        A, a, b, mx, my = p
        return A*(1 + ((x-mx)**2 + (y-my)**2) / (a*a))**-b
    
    
    def residuals(self, p, data, X, Y ):
        '''Difference between data and model'''
        return data - self.F(p, X, Y)

    def err(self, p, data, X, Y ):
        return abs( self.residuals(p, data, X, Y).flatten() )
    

    
def get_func_name(func):
    return func.__name__
    
######################################################################################################
# Class definitions
######################################################################################################

class ImageApertures( SkyApertures ):
    #NOTE: THIS CLASS IS UNECESSARY AND CONFUSING AND CAN BE INCORPORATE INTO Stars
    #===============================================================================================
    
    #===============================================================================================
    
    def __init__(self, **kw):
        '''
        '''
        fwhm = kw.pop('fwhm')           if 'fwhm' in kw         else None
        psfradii = [0.5*fwhm, 1.5*fwhm] if not fwhm is None     else []
        skyradii = [SkyApertures.RADII] if not fwhm is None     else []
        
        print( 'IMAGEAPS', kw )
        
        self.psfaps = ApertureCollection( ls=':', gc=['k','w'], picker=False, radii=psfradii, **kw)
        self.photaps = ApertureCollection( gc='c', **kw)
        self.skyaps = SkyApertures( radii=skyradii, **kw )
        
        #ipshell()
        #self.annotations = []
    
    #===============================================================================================
    #def __getitem__(self, key) :
       #return self.get_apertures()[key]
    
    ##===============================================================================================
    #def get_apertures(self):
        #return [self.psfaps, self.photaps, self.skyaps]
    
    #===============================================================================================
    def get_unique_coords(self):
        '''get unique coordinates'''
        nq_coords = self.psfaps.coords          #non-unique coordinates (possibly contains duplicates)   # RESHAPING IN THE GETTER????
        _, idx = np.unique( nq_coords[:,0], return_index=1 )
        _, coords = sorter( list(idx), nq_coords[idx] )
        return np.array(coords)                   #filtered for duplicates
        
    coords = property(get_unique_coords)
    #===============================================================================================
    #def append(self, aps):
        #self.psfaps.append( aps[0] )
        #self.skyaps.append( aps[1] )
        
        
        #self.auto_colour()
    
    #===============================================================================================
    #def auto_colour(self):
        #'''
        #Checks whether any of the sky apertures cross the 3 sigma psf fwhm aperture.  Set colours accordingly
        #'''
        #if len(self):
            #print( '\n'*2, 'NEEDS WORKS>>>>>>>>>>>>>>>>>>>>>>>' )
            #lc = self.skyaps.cross( self.psfaps )
            #ls = collection.state
                
            #print( lc, ls )
            
            #colours = np.where( ls|lc, self.badcolour, self.goodcolour )
            #collection.set( 'colours', colours )
    
    #def axadd( self, ax ):
        #for aps in self.get_apertures():
            #aps.axadd( ax )
    
#################################################################################################################################################################################################################          
    
class Star( object ):   #ApertureCollection
    '''
    '''
    ATTRS = ['coo', 'peak', 'fwhm', 'sigma_xy', 'ratio', 'ellipticity', 
                'sky_mean', 'sky_sigma', 'star_im', 'id', '_window_idx']
    #===============================================================================================
    def __init__(self, **params):
        
        #check = params.pop('check') if 'check' in params else 1
        #apertures = params.pop('apertures') if 'apertures' in params else None
        
        for key in self.ATTRS:
            setattr( self, key, params[key] if key in params else None )           #set the star attributes given in the params dictionary
        
        #apertures = apertures if apertures!=None else [1.5*self.fwhm if self.fwhm else None]
        #super( Star, self ).__init__( coords=self.coo, radii=apertures, check=check )
        
        drp = params.pop('rad_prof') if 'rad_prof' in params else 1     #default is to compute the radial profile
        drp = 0 if self.star_im is None else drp                        #if no stellar image is supplied don't try compute the radial profile
        
        self.rad_prof = self.radial_profile( ) if drp else [[], [], []]               #containers for radial profiles of fit, data, cumulative data
        
    #===============================================================================================
    def get_params(self):
        raise NotImplementedError
    
    #===============================================================================================
    def set_params(self, dic):
        raise NotImplementedError
    
    #===============================================================================================
    def radial_profile( self ):
        
        #im_shape = np.array( self.star_im.shape, ndmin=3 ).T
        il,iu, jl,ju = self._window_idx                           #position of pixel window in main image
        pix_cen = np.mgrid[il:iu, jl:ju]  + 0.5                   #the index grid for pixel centroids
        staridx = tuple(reversed(self.coo))                     #star position in pixel coordinates
        rfc = np.sqrt(np.sum(np.square(pix_cen.T-staridx), -1))    #radial distance of pixel centroid from star centoid
        rmax = SkyApertures.R_OUT_UPLIM + 1                       #maximal radial distance of pixel from image centroid
        
        return np.array([ np.mean( self.star_im[(rfc>=r-1)&(rfc<r)] ) for r in range(1, rmax) ])

        
#######################################################################################################    
class MeanStar( Star ):   #ApertureCollection
    #===============================================================================================     
    def __init__( self, window ):
        self.window = window
        fakecoo = window, window
        
        Star.__init__( self, coo=fakecoo, check=0 )      #won't calculate rad_prof
        self.apertures = ApertureCollection( radii=np.zeros(4), coords=fakecoo, 
                                                colours=['k','w','g','g'], 
                                                ls=['dotted','dotted','solid','solid'])
        
        self.rad_prof = [[], [], []]               #containers for radial profiles of fit, data, cumulative data
        self.has_plot = False
        
    #=============================================================================================== 
    def init_plots( self, fig ):
        
        ##### Plots for mean profile #####
        self.mainfig = fig
        gs2 = gridspec.GridSpec(2, 2, width_ratios=(1,2), height_ratios=(3,1) )
        self.ax_zoom = fig.add_subplot( gs2[2], aspect='equal' )
        self.ax_zoom.set_title( 'PSF Model' )
        self.pl_zoom = self.ax_zoom.imshow([[0]], origin='lower', cmap = 'gist_heat')   #vmin=self.zlims[0], vmax=self.zlims[1]
        self.colour_bar = fig.colorbar( self.pl_zoom, ax=self.ax_zoom )
        
        self.ax_prof = fig.add_subplot( gs2[3] )
        self.ax_prof.set_title( 'Mean Radial Profile' )
        self.ax_prof.set_ylim( 0, 1.1 )
        
        ##### stellar profile + psf model #####
        labels = ['', 'Cumulative', 'Model']
        self.pl_prof = self.ax_prof.plot( 0,0,'g-', 0,0,'r.', 0,0, 'bs' )
        [pl.set_label(label) for pl,label in zip( self.pl_prof, labels )]
        
        ##### apertures #####
        self.apertures.convert_to_lines( self.ax_prof )
        
        ##### sky fill #####
        from matplotlib.patches import Rectangle
        
        trans = self.apertures.line_transform
        self.sky_fill = Rectangle( (0,0), width=0, height=1, transform=trans, color='b', alpha=0.3)
        self.ax_prof.add_artist( self.sky_fill )
        #self.pl_aps_postxt = []
        #print( 'just initializing, will set values at plot time!!!!!!!!!!!!!!!!!!!!' )
        self.ax_prof.add_collection( self.apertures.aplines )           #NOTE:
        self.apertures.axadd( self.ax_zoom )
        #print( self.apertures.aplines )
        #print( vars(self.apertures.aplines) )
        #print( '\n'*10 )
        self.has_plot = True
        fig.canvas.draw() 
        
    #===============================================================================================
    def update( self, cached_params, data_profs ):
        
        window = self.window
        Y, X = np.mgrid[:2*window, :2*window] + 0.5
        p = (1,) +cached_params+ (window, window)               #update with mean of cached_params
        pdict = StellarFit.get_param_dict( p )          #TODO: set_params
        
        self.fwhm = fwhm = pdict['fwhm']
        b = 4*np.log(2) / fwhm / fwhm                           #derive the kernel width parameter for the 1D Gaussian radial profile from the geometric mean of the FWHMa of the 2D Gaussian
        
        self.star_im = StellarFit.Gaussian2D( p, X, Y )
        self.rad_prof[0] = StellarFit.Gaussian( (1., b, 0.), np.linspace(0, window) )
        self.rad_prof[1:] = data_profs
        
        self.apertures.radii = [0.5*fwhm, 1.5*fwhm, SkyApertures.R_IN, SkyApertures.R_OUT]
        
        #print( '^'*88 )
        #print( self.rad_prof )
        #print( list(map(len, self.rad_prof)) )
        #print( self.apertures )
        #print( self.apertures.coords )
        
    #===============================================================================================
    def update_plots(self):
        
        Z = self.star_im
        self.pl_zoom.set_data( Z )
        self.pl_zoom.set_clim( 0, 1 )
        self.pl_zoom.set_extent( [0, 2.*self.window, 0, 2.*self.window] )
        
        self.ax_prof.set_xlim( 0, SkyApertures.R_OUT_UPLIM + 0.5 )
        
        rpx = np.arange( 0, self.window )
        rpxd = np.linspace( 0, self.window )
        
        for pl, x, y in zip( self.pl_prof, [rpxd, rpx, rpx], self.rad_prof ):
            pl.set_data( x, y )
        
    #===============================================================================================
    def update_aplines(self):
        ##### set aperture line position + properties #####
        self.apertures.update_aplines()
        
        #print( '*'*88 )
        #print( self.apertures )
        #print( self.apertures.aplines )
        
        ##### Shade sky region #####
        sky_width = np.ptp( SkyApertures.RADII )
        self.sky_fill.set_x( SkyApertures.R_IN )
        self.sky_fill.set_width( sky_width )
        
        ##### Update figure texts #####
        #plot_texts = self.plot_texts
        #bb = plot_texts[2].get_window_extent( renderer=self.mainfig.canvas.get_renderer() )
        #bb = bb.transformed( self.ax_prof.transAxes.inverted() )
        #sky_txt_offset = 0.5*sky_width - bb.width
        #offsets = 0.1, 0.1, sky_txt_offset, 0
        #xposit = np.add( radii, offsets )
        #y = 1.0
        #for txt, x in zip( plot_texts, xposit ):
            #txt.set_position( (x, y) )
    
#######################################################################################################    
                
class Stars( ImageApertures ):
    '''
    Class to contain measured values for selected stars in image.
    '''
    #===============================================================================================
    def __init__(self, **kw):
        
        
        fwhm = kw.pop('fwhm')           if 'fwhm' in kw         else []
        coords = kw.pop('coords')       if 'coords' in kw       else []
        self.window = kw.pop('window')  if 'window' in kw       else SkyApertures.R_OUT_UPLIM
        
        self.stars = [Star(coo=coo, fwhm=fwhm) 
                        for (coo,fwhm) in itertools.zip_longest(coords, as_iter(fwhm)) ]
        self.star_count = len(self.stars)
        
        self.meanstar = MeanStar( self.window )
        self.plots = []
        self.annotations = []
        
        if 'psfradii' in kw:
            psfradii = kw.pop('psfradii')
        else:
            psfradii = np.tile( fwhm, (2,1)).T * [0.5, 1.5]
        
        if 'skyradii' in kw:
            skyradii = kw.pop('skyradii')
        else:
            skyradii = np.tile( SkyApertures.RADII, (len(fwhm),1) )
        
        apcoo = np.array( interleave( coords, coords ) )                #each stars has 2 skyaps and 2 psfaps!!!!
        self.psfaps = ApertureCollection( coords=apcoo, radii=psfradii, ls=':', gc=['k','w'], picker=False, **kw)
        self.photaps = ApertureCollection( gc='c' )
        self.skyaps = SkyApertures( radii=skyradii, coords=apcoo, **kw )
        
        self.has_plot = 0
        
    #===============================================================================================
    def __str__(self):
        attrs = copy(Star.ATTRS)
        attrs.pop( attrs.index('star_im') )
        attrs.pop( attrs.index('_window_idx') ) #Don't print these attributes
        data = [self.pullattr(attr) for attr in attrs]
        table = Table(data, title='Stars', row_headers=attrs)
        
        #apdesc = 'PSF', 'PHOT', 'SKY'
        #rep = '\n'.join( '\t{} Apertures: {}'.format( which, ap.radii ) 
                    #for which, radii in zip(apdesc, self.get_apertures()) )
        return str(table)
    
    #===============================================================================================
    def __repr__(self):
        return str(self)
    
    #===============================================================================================    
    def __len__(self):
        return len(self.stars)
    
    #===============================================================================================    
    def __getitem__(self, key):
        return self.stars[key]
    
    #===============================================================================================
    def pullattr(self, attr):
        return [getattr(star, attr) for star in self]
    
    #===============================================================================================
    def get_apertures(self):
        return [self.psfaps, self.photaps, self.skyaps]
    
    #===============================================================================================
    def append(self, star=None, **star_params):
        
        if star_params:
            star = Star( **star_params )             #star parameters dictionary passed not star object
        
        self.stars.append( star )
        #print()
        #print('UPDATING PSFAPS')
        coo, fwhm = star.coo, star.fwhm
        
        if not np.size(self.psfaps.coords):
            coo = [coo]
        
        self.psfaps.append( coords = coo, radii = [0.5*fwhm, 1.5*fwhm], picker=False, ls=':', gc=['k','w'] )
        #print()
        #print('UPDATING SKYAPS')
        self.skyaps.append( coords = coo, radii = SkyApertures.RADII )
        #self.photaps.append( something )
        
        ax = self.psfaps.axes
        txt_offset = 1.5*fwhm
        coo = np.squeeze(coo)
        anno = ax.annotate( str(self.star_count), coo, coo+txt_offset, color='w',size='small') #transform=self.ax.transData
        self.annotations.append( anno )
        
        self.star_count += 1
        
        return star
    
    #===============================================================================================
    #def duplicate(self, scale, offset):
        #stars = copy( self )
        #coords = self.psfaps.coords
        #psfradii = self.psfaps.radii
        #skyradii = self.skyaps.radii
        
        #coords = (coords + offset) * scale
        #psfradii *= scale[0]
        #skyradii *= scale[0]
        
        #stars.psfaps.coords = coords
        #stars.psfaps.radii = psfradii
        #stars.skyaps.coords = coords
        #stars.skyaps.radii = skyradii
        
        #return stars
    
    #===============================================================================================
    def axadd( self, ax ):
        self.skyaps.axadd( ax )
        for aps in self.get_apertures():
            aps.axadd( ax )
    
    #===============================================================================================
    def remove(self, idx):
        
        self.psfaps.remove( idx )
        self.skyaps.remove( idx )
        #self.photaps.remove( idx )
        
        self.stars.pop( idx[0][0] )
        self.annotations.pop(idx[0][0]).set_visible( False )
        self.star_count -= 1
    
    #===============================================================================================
    def remove_all( self ):
        print( 'Attempting remove all on', self )
        while len(self):
            print( self )
            self.remove([[0]])
    
    #===============================================================================================
    def init_plots(self, mainfig):
        '''
        '''
        ##### Plots for current fit #####
        fig = self.fitfig = plt.figure( figsize=(12,9) )
        gs = gridspec.GridSpec(2, 3)
        fig.suptitle('PSF Fitting')
        titles = ['Data', 'Fit', 'Residual']
        for i,g in enumerate(gs):
            if i<3:
                ax = fig.add_subplot(g, projection='3d')
                pl = ax.plot_wireframe( [],[],[] )
                plt.title( titles[i] )
                _,ty = ax.title.get_position( )
                ax.title.set_y(1.1*ty)
            else:
                ax = fig.add_subplot( g )
                pl = ax.imshow([[0]], origin='lower')
            self.plots.append( pl )
            plt.show( block=False )
        
        self.has_plot = 1
        
        self.meanstar.init_plots( mainfig )
    
    #===============================================================================================
    @staticmethod
    def update_segments(X, Y, Z):
        lines = []
        for v in range( len(X) ):
            lines.append( list(zip(X[v,:],Y[v,:],Z[v,:])) )
        for w in range( len(X) ):
            lines.append( list(zip(X[:,w],Y[:,w],Z[:,w])) )
        return lines
    
    #===============================================================================================
    def update_plots(self, X, Y, Z, data):
        res = data - Z
        self.plots[0].set_segments( self.update_segments(X,Y,data) )
        self.plots[1].set_segments( self.update_segments(X,Y,Z) )
        self.plots[2].set_segments( self.update_segments(X,Y,res) )
        self.plots[3].set_data( data )
        self.plots[4].set_data( Z )
        self.plots[5].set_data( res )
        
        zlims = [np.min(Z), np.max(Z)]
        for q, pl in enumerate( self.plots ):
            ax = pl.axes
            ax.set_xlim( [X[0,0],X[0,-1]] ) 
            ax.set_ylim( [Y[0,0],Y[-1,0]] )
            if q<3:
                ax.set_zlim( zlims )
            else:
                pl.set_clim( zlims )
                pl.set_extent( [X[0,0], X[0,-1], Y[0,0], Y[-1,0]] )
                #plt.colorbar()
        self.fitfig.canvas.draw()
        #SAVE FIGURES..................

    #===============================================================================================
    def get_mean_val(self, key):
        if not len(self):
            raise ValueError( 'The {} instance is empty!'.format(type(self)) )
        if not hasattr( self[0], key ):
            raise KeyError( 'Invalid key: {}'.format(key) )
        vals = [getattr(star,key) for star in self]
        if not None in vals:
            return np.mean( vals, axis=0 )
    
    #===============================================================================================
    #def get_sky_mean(self):
        #return np.mean( [star.sky_mean for star in self] )
    
    ##===============================================================================================
    #def get_sky_sigma(self):
        #return np.mean( [star.sky_sigma for star in self] )
    
    #=============================================================================================== 
    @unhookPyQt
    def mean_radial_profile( self ):
        
        rp = [star.rad_prof for star in self]                    #radial profile of mean stellar image\
        rpml = max(map(len, rp))
        rpa = np.array( [np.pad(r, (0,rpml-len(r)), 'constant', constant_values=-1) for r in rp] )
        rpma = np.ma.masked_array( rpa, rpa==-1 )
        
        rpm = rpma.mean( axis=0 );      rpm /= np.max(rpm)       #normalized mean radial data profile
        cpm = np.cumsum( rpm )   ;      cpm /= np.max(cpm)       #normalized cumulative
        
        return rpm, cpm
        
    
####################################################################################################    
class ApertureInteraction(object):
    
    msg = ( 'Please select a few stars (by clicking on them) for measuring psf and sky brightness.\n ' 
        'Right click to resize apertures.  Middle mouse to restart selection.  Close the figure when you are done.\n\n' )
    
    #===============================================================================================
    def __init__(self, basename=None, **kwargs):
        
        
        #self.mode = mode                  if 'mode' in kwargs           else 'fit'
        
        if 'window' not in kwargs:          window = SkyApertures.R_OUT_UPLIM
        self.window = window
        self.cbox = cbox                 if 'cbox' in kwargs          else 12
        
        self.stars = Stars( window=window )
        self.zoom_pix = np.mgrid[:2*window, :2*window]                         #the pixel index grid
            
        self.selection = None                   # allows artist resizing only when an artist has been picked
        self.status = 0                         #ready for photometry??
        
        self._write_par = 1                     #should parameter files be written

        self.cid = []                           #connection ids
        
        self.fit = StellarFit()
        
    #===============================================================================================
    def load_image(self, filename=None, data=None, WCS=True):
        '''
        Load the image for display, applying IRAF's zscale algorithm for colour limits.
        '''
        
        print( 'Loading image {}'.format(filename) )
        
        ##### Load the image #####
        if filename and data is None:
            self.filename = filename
            self.image_data = data = pyfits.getdata( filename )
            
            #get readout noise and saturation if available
            self.ron = float( pyfits.getval(self.filename,'ron') )
            try:        #Only works for SHOC data
                self.saturation = RNT.get_saturation(filename)                                             #NOTE: THIS MEANS RNT NEEDS TO BE INSTANTIATED BEFORE StarSelector!!!!!!!!
            except BaseException as err:
                warn( 'Error in retrieving saturation value:\n'+err )
                self.saturation = 'INDEF'
            
        elif not data is None:
            self.image_data = data
        else:
            raise ValueError( 'Please provide either a FITS filename, or the actual data' )
        
        if WCS:            data = np.fliplr( data )
        self.image_data_cleaned = copy(data)
        r,c =  self.image_shape = data.shape
        self.pixels = np.mgrid[:r, :c]                      #the pixel grid
        
        
        ##### Set up axes geometry #####
        fig = self.figure = Figure( figsize=(8,18), tight_layout=1 )                                        #fig.set_size_inches( 12, 12, forward=1 )
        
        gs1 = gridspec.GridSpec(2, 1, height_ratios=(3,1))
        ax = self.ax = fig.add_subplot( gs1[0] )
        ax.set_aspect( 'equal' )

        
        ##### Plot star field #####
        from zscale import Zscale
        zscale = Zscale( sigma_clip=5. )
        zlims = zscale.range(data, contrast=1./99)
        
        
        self.image = ax.imshow( data, origin='lower', cmap = 'gist_heat', vmin=zlims[0], vmax=zlims[1])                  #gist_earth, hot
        #ax.set_title( 'PSF Measure' )                                                                                    #PRINT TARGET NAME AND COORDINATES ON IMAGE!! DATE, TIME, ETC.....
        self.colour_bar = fig.colorbar( self.image, ax=ax )                                                                                                              #PLOT N E ON IMAGE
        self.stars.zlims = zlims
        #box = ax.get_position()
        #ax.set_position([box.x0, box.y0 + box.height * 0.1,
                #box.width, box.height * 0.9])
        
        ##### plot NE arrows #####
        if WCS:
            apos, alen = 0.02, 0.15                     #arrow position and length as fraction of image dimension
            x, y = (1-apos)*r, apos*c                   #arrow coordinates
            aclr = 'g'
            
            ax.arrow( x, y, -alen*r, 0, color=aclr, width=0.1 )
            ax.text( x-alen*r, 2*apos*c, 'E', color=aclr, weight='semibold' )
            ax.arrow( x, y, 0, alen*c, color=aclr, width=0.1 )
            ax.text( (1-3*apos)*r, y+alen*c, 'N', color=aclr, weight='semibold' )
        
        
        ##### Add ApertureCollection to axes #####
        self.stars.axadd( ax )
        
        return fig
    
    #===============================================================================================    
    def connect(self, msg='', **kwargs):
        '''
        Connect to the figure canvas for mouse event handeling.
        '''
        print(msg)
        #self.mode = kwargs['mode']          if 'mode' in kwargs         else 'fit'
        self.cid.append( self.figure.canvas.mpl_connect('button_press_event', self.select_stars) )
        #self.cid.append( self.figure.canvas.mpl_connect('close_event', self._on_close) )
        
        self.cid.append( self.figure.canvas.mpl_connect('pick_event', self._on_pick) )
        self.cid.append( self.figure.canvas.mpl_connect('button_release_event', self._on_release) )
        self.cid.append( self.figure.canvas.mpl_connect('motion_notify_event', self._on_motion) )
        
        self.cid.append( self.figure.canvas.mpl_connect('key_press_event', self._on_key) )             #IF MODE==SELECT --> A=SELECT ALL
        #self.apertures.connect( self.figure )    
    
    #===============================================================================================    
    def disconnect(self):
        '''
        Disconnect from figure canvas.
        '''
        for cid in self.cid:
            self.figure.canvas.mpl_disconnect( cid )
        print('Disconnected from figure {}'.format(self.figure) )
        
    #====================================================================================================
    def map_coo( self, coords, WCS=1 ):
        ''' Apply coordinate mapping to render in coordinates with N up and E left. 
            This is also the inverse mapping'''
        coords = np.array( coords, ndmin=2 )
        if WCS:
            #print( 'WM'*100 )
            #print( coords )
            coords[:,0] = self.image_data.shape[1] - coords[:,0]
            #print( 'WM'*100 )
            #print( coords )
        return np.squeeze( coords )
        
    #===============================================================================================
    def resize_selection( self, mouseposition ):
        apertures, idx = self.selection.artist, self.selection.idx
        rm = apertures.edge_proximity( mouseposition, idx )       #distance moved by mouse
        
        if SkyApertures.SAMESIZE:
            if len(idx)>1:
                idx = ..., idx[1]      #indexes all apertures of this size 
        
        apertures.resize( rm, idx )
        
    #===============================================================================================
    def _on_release(self, event):
        '''
        Resize and deselct artist on mouse button release.
        '''
        
        if self.selection:
            #print( '**\n'*3, 'Releasing', self.selection.artist.radii )
            self.resize_selection( (event.xdata, event.ydata) )
            
            self.figure.canvas.draw()
        
        self.selection = None # the artist is deselected upon button released
        
    #===============================================================================================    
    def _on_pick(self, event):
        '''
        Select an artist (aperture)
        '''
        self.selection = event
    
    #===============================================================================================    
    def _on_motion(self, event):
        '''
        Resize aperture on motion.  Check for validity and colourise accordingly.
        '''
        if event.inaxes!=self.image.axes:
            return
        
        if self.selection:
            self.resize_selection( (event.xdata, event.ydata) )
            
            apertures, idx = self.selection.artist, self.selection.idx
            psfaps, photaps, skyaps = self.stars.get_apertures()
            skyaps.auto_colour( check='all', edge=self.image_data.shape, cross=psfaps )
            #skyaps.auto_colour()
            
            if SkyApertures.SAMESIZE:
                idx = ... if len(self.stars)==1 else 0
                SkyApertures.RADII = skyaps.radii[idx].ravel()

            #### mean star apertures #####
            meanstar = self.stars.meanstar
            meanstar.apertures.radii[-2:] = SkyApertures.RADII
            #self.stars.meanstar.radii = radii
            meanstar.apertures.auto_colour( check='sky' )
            #meanstar.apertures.auto_colour()
            meanstar.update_aplines( )
            
            #self.update_legend()
            self.figure.canvas.draw()
    
    #=============================================================================================== 
    def _on_key(self, event):
        print( 'ONKEY!!' )
        print( repr(event.key) )
        
        if event.key.lower()=='d':
            
            print( 'psfaps.radii', self.stars.psfaps.radii )
            prox = self.stars.psfaps.center_proximity( (event.xdata, event.ydata) )
            
            print( 'prox', prox )
            idx = np.where( prox==np.min(prox) )
            
            print( 'where', np.where( prox==np.min(prox) ) )
            
            star = self.stars[idx[0][0]]
            
            try:
                il,iu, jl,ju = star._window_idx
                im = star.star_im
            except TypeError:
                #if the stars where found by daofind, the _window_idx would not have been determined!
                #Consider doing the fits in a MULTIPROCESSING queue and joining here...
                print( 'star.coo', star.coo )
                x, y = np.squeeze(star.coo)
                j,i = int(round(x)), int(round(y))
                il,iu ,jl,ju, im = self.zoom(i,j)
                
            self.image_data_cleaned[il:iu, jl:ju] = im                     #replace the cleaned image data section with the original star image
            
            self.stars.remove( idx )
            if len(self.stars):
                self.stars.skyaps.auto_colour( check='all', edge=self.image_data.shape, cross=self.stars.psfaps )
            
            self.figure.canvas.draw()
    
    #===============================================================================================
    def update_legend(self):
        
        markers = []; texts = []
        #print( 'WTF~??!! '*8 )
        #print( self.stars[0].apertures )
        for ap in self.stars[0].apertures:           #CURRENT SELECTION?????????????
            mec = ap.get_ec()
            #print( 'mec' )
            #print( mec )
            mew = ap.get_linewidth()
            marker = Line2D(range(1), range(1), ls='', marker='o', mec=mec, mfc='none', mew=mew, ms=15)
            text = str( ap )
            markers.append( marker )
            texts.append( text )
            
        leg = self.figure.legend( markers, texts, loc='lower center', 
                fancybox=True, shadow=True, ncol=3, numpoints=1  )
    
    #bbox_to_anchor=(0.5,-0.25),
           
            
    #===============================================================================================    
    def snap2peak(self, x, y, offset_tolerance=10):
        '''
        Snap to the peak value within the window.
        Parameters
        ----------
        x, y :                  input coordinates (mouse position)
        offset_tolerance :      the size of the image region to consider. i.e offset tolerance
        
        Returns
        -------
        xp, yp, zp :            pixel coordinates and value of peak
        '''
        j,i = int(round(x)), int(round(y))
        il,_ ,jl,_, im = self.zoom(i,j, offset_tolerance)
        
        zp = np.max(im)
        ip, jp = np.where(im==zp)                                    #this is a rudamnetary snap function NOTE: MAY NOT BE SINGULAR.
        yp = ip[0]+il; xp = jp[0]+jl                                      #indeces of peak within sub-image
        
        return xp, yp, zp
    
    #===============================================================================================
    def snap(self, x, y, threshold=5., edge_cutoff=5., offset_tolerance=10, pr=True):
            
        r,c = self.image_shape
        
        xp, yp, zp = self.snap2peak(x, y, offset_tolerance)
        j, i = xp, yp
        sky_mean, sky_sigma = self.measure_sky(i,j, pr=0)
        
        #check threshold
        if zp - sky_mean < threshold*sky_sigma:
            print('Probably not a star!')
            return None, None
        
        #check edge proximity
        if any(np.abs([xp, xp-c, yp, yp-r]) < edge_cutoff):
            print('Too close to image edge!')
            return None, None
        
        #check if known
        xs, ys = self.snap2known(xp, yp, offset_tolerance=5)
        if xs and ys:
            print('Duplicate selection!')
            return None, None
            
        return xp, yp
            
    #===============================================================================================
    def snap2known(self, x, y, known_coo=None, offset_tolerance=10):
        
        if not known_coo:                   known_coo = self.stars.coords
        if not len(known_coo):
            return None, None

        if not offset_tolerance:            offset_tolerance = self.window
        
        known_coo = np.array(known_coo)
        rs = list( map(la.norm, known_coo - (x,y)) )                                              #distance from user input and known stars
        l_r = np.array(rs) < offset_tolerance

        if any(l_r):
            ind = np.argmin( rs )                                                                     #fin star that has minimal radial distance from selected point
            return known_coo[ind]
        else:
            print('No known star found within %i pixel radius' %offset_tolerance)
            return None, None
    
    #===============================================================================================
    def get_sky(self, i, j):                            #COMBINE THIS ALGORITHM WITH THE RAD PROF ALG FOR EFFICIENCY
        Yp, Xp = self.pixels + 0.5                      #pixel centroids
        Rp = np.sqrt( (Xp-j)**2 + (Yp-i)**2 )

        sky_inner, sky_outer = SkyApertures.RADII
        l_sky = np.all( [Rp > sky_inner, Rp < sky_outer], 0 )
        return l_sky
    
    #===============================================================================================
    def measure_sky(self, i, j, pr=1):                           #WARN IF STARS IN SKY ANNULUS!!!!!
        '''Measure sky mean and deviation.'''
        if pr:
            print( 'Doing sky measurement...' )
        
        l_sky = self.get_sky(i, j)
        sky_im = self.image_data_cleaned[l_sky]
       
        sky_mean = np.mean(sky_im)
        sky_sigma = np.std(sky_im)

        return sky_mean, sky_sigma
    
    #===============================================================================================
    def zoom(self, i, j, window=None, use_clean=1):
        '''limits indeces to fall within image and returns zoomed section'''
        r,c = self.image_shape
        window = window if window else self.window
        il,iu = i-window, i+window
        jl,ju = j-window, j+window
        if il<0: il=0; iu = 2*window
        if iu>r: iu=r; il = r-2*window
        if jl<0: jl=0; ju = 2*window
        if ju>c: ju=c; jl = c-2*window
        if use_clean:
            star_im = self.image_data_cleaned[il:iu, jl:ju]
        else:
            star_im = self.image_data[il:iu, jl:ju]
        return il,iu, jl,ju, star_im
    
    #===============================================================================================
    #def radial_profile( self, coo, rmax=self.window ):
        #'''
        #rmax : maximal radial distance of pixel from stellar centroid
        #'''
        
        #coo = np.array( coo, ndmin=3 ).T
        #pix_cen = self.pixels + 0.5                                  #pixel centroids
       
        #rfc = np.linalg.norm(pix_cen - coo , axis=0 )                #radial distance of pixels from stellar centroid
        
        #return np.array([ np.sum( self.image_data[(rfc>=r-1)&(rfc<r)] ) for r in range(1, rmax) ])
    
    #===============================================================================================
    def pre_fit(self, xy):
        xs, ys = self.snap( *xy, pr=0 )
        if not (xs and ys):
            return None
        
        j, i = xs, ys
        sky_mean, sky_sigma = self.measure_sky(i,j, pr=0)     #NOTE:  YOU CAN ELIMINATE THE NEED FOR THIS IF YOU ADD A BACKGROUND PARAM TO THE FITS
        il,iu, jl,ju, star_im = self.zoom(i,j)
        star_im = star_im - sky_mean
        Y, X = grid = np.mgrid[il:iu,jl:ju]
        
        p0 = self.fit.param_hint( (xs,ys), star_im )
        
        return p0, grid, star_im, sky_mean, sky_sigma, (il,iu, jl,ju)
            
    #===============================================================================================
    def fit_psf(self, xs, ys, plot=0):          #TODO: DISPLAY MEASUREMENTS ON FIGURE?????
        '''
        Fit 2D Gaussian distribution to stellar image
        Parameters
        ----------
        xs, ys : Initial coordinates for fit (maximal value in window)
        
        Returns
        ------
        coo : Star coordinates from fit
        fwhm : Full width half maximum from fit
        ellipticity : from fit
        sky_mean : measured mean sky counts
        sky_sigma : measured standard deviation of sky
        star_im : sky subtracted image of star
        '''
        j, i = xs, ys
        sky_mean, sky_sigma = self.measure_sky(i,j)     #NOTE:  YOU CAN ELIMINATE THE NEED FOR THIS IF YOU ADD A BACKGROUND PARAM TO THE FIT
        il,iu, jl,ju, star_im = self.zoom(i,j)
        star_im = star_im - sky_mean
        Y, X = grid = np.mgrid[il:iu,jl:ju]
        
        skydict = {     'sky_mean'      :       sky_mean,
                        'sky_sigma'     :       sky_sigma,
                        'star_im'       :       star_im,
                        '_window_idx'   :       (il,iu, jl,ju)          }
        
        ##### Fit function {self.fit.F.__name__} to determine fwhm psf ###
        plsq = self.fit( (xs,ys), grid, star_im )
        pdict = self.fit.get_param_dict( plsq )
        pdict.update( skydict )                                 #extend the fit parameters dictionary with the sky paramaters
        
        Z = self.fit.F(plsq, X, Y)
        self.image_data_cleaned[il:iu, jl:ju] -= Z              #remove star from image by subtracting Moffat fit
        
        if plot:
            if not self.stars.has_plot:
                self.stars.init_plots( self.figure )                #initialise the figure for displaying the fits

            #star_im = self.image_data[il:iu, jl:ju] - sky_mean
            self.stars.update_plots( X, Y, Z, star_im )
        
        return pdict
    
    #===============================================================================================
    def select_stars(self, event):
        
        if event.inaxes!=self.image.axes:
            return
        
        if event.button==3:
            #print( '!'*888 )
            #print( 'BLLUUUAAAAAAAGGGGGHHHHHHHHX!!!!!!!!!!' )
            return
        
        if event.button==2:
            self.restart()
            return
        
        x,y = event.xdata, event.ydata
        if None in (x,y):
            return
        
        else:
            xs, ys = self.snap(x,y)                                           #coordinates of maximum intensity within window / known star
            if not (xs and ys):
                return
            else:
                stars = self.stars
                
                params = self.fit_psf( xs,ys, plot=1 )
                #xf, yf = params['coo']
                #print('*'*8)
                #print('Star No. %i' %(stars.star_count))
                #print('User in: x, y = %6.3f, %6.3f;\nPixel value (%i, %i) = %6.3f\n' %(x, y, xs, ys, self.image_data[ys,xs]))
      
                info = Table(params, title='Star No. %i' %(stars.star_count), 
                                col_headers='Fit params', ignore_keys=('star_im', '_window_idx'))
                print( info )
                
                star = stars.append( **params )
                
                stars.skyaps.auto_colour( check='all', cross=stars.psfaps, edge=self.image_data.shape )
                
                stars.meanstar.update( self.fit.get_params_from_cache(), stars.mean_radial_profile() )
                stars.meanstar.update_plots()
                
                self.figure.canvas.draw()               #BLITTING????
                
                
####################################################################################################    

#====================================================================================================
def samesize( imdata, interp='cubic' ):
    '''resize a bunch of images to the same resolution for display purposes.'''
    from scipy.misc import imresize
    
    shapes = np.array([d.shape for d in imdata])
    resize_to = shapes.max(0)
    scales = shapes / resize_to
    
    new = np.array( [imresize(im, resize_to, interp, mode='F') for im in imdata] )
    return new, scales

#====================================================================================================
def get_pixel_offsets( imdata ):
    '''Determine the shifts between images from the indices of the maximal pixel in each image.
    This is not a robust algorithm!
    '''
    maxi = np.squeeze( [np.where(d==d.max()) for d in imdata] )
    return maxi - maxi[0]
    
#====================================================================================================
def blend_prep(data, offsets):
    '''shift and zero pad the data in preperation for blending.'''
    omax, omin = offsets.max(0), offsets.min(0)
    newshape = data.shape[:1] + tuple(data.shape[1:] + offsets.ptp(0))
    nd = np.empty(newshape)
    for i, (dat, padu, padl) in enumerate(zip(data, omax-offsets, offsets-omin)):
        padw = tuple(zip(padu,padl))
        nd[i] = np.pad( dat, padw, mode='constant', constant_values=0 )
    return nd      
                
                
####################################################################################################    
from mplMultiTab import *
class StarSelector(MplMultiTab):
    '''
    class which contains methods for selecting stars from image and doing basic psf fitting and sky measurements
    '''
    
    WCS = True                                  #Flag for whether to display the image as it would appear on sky
    #====================================================================================================
    def __init__(self, im_fns, coord_fns=None, offsets=None, show_blend=1):
        
        #Filename wrapper
        self.filenames = type('DataWrapper', (), 
                                {       'images'      :       im_fns, 
                                        'coo'         :       coord_fns        } )
        
        self.apis = []
        self.figures = []
        labels = []
        for i, fn in enumerate(im_fns):
            api = ApertureInteraction( )
            fig = api.load_image( fn, WCS=self.WCS )
            
            self.figures.append( fig )
            self.apis.append( api )
            labels.append( 'Tab %i' %i )
        
        self.imshapes = np.array([api.image_data.shape for api in self.apis])
        self.has_blend = bool(show_blend)
        
        if offsets is None:
            data = np.array( [api.image_data for api in self.apis] )    #remember that this is flipped left to right if WCS
            data, self.scales = samesize(data)
            
            m = np.array([np.median(_) for _ in data], ndmin=3).T           #median of each image
            data = data / m                                             #normalised
            
            #ipshell()
            
            print( 'Determining image shifts...' )
            self.pixoff = get_pixel_offsets( data )                     #offsets in pixel i,j coordinates
            
            if show_blend:
                print( 'Doing data blend...' )
                data = blend_prep(data, self.pixoff)
                weights = np.ones(data.shape[0])
                self.blend = np.average(data, 0, weights)
                
                api_blend = ApertureInteraction( )
                fig = api_blend.load_image( data=self.blend, WCS=self.WCS )
                self.apis.append( api_blend )
                self.figures.append( fig )
                labels.append( 'Blend' )
                self.pixoff = np.vstack( (self.pixoff, (0,0)) )
                self.scales = np.vstack( (self.scales, (1,1)) )
        
        self.offsets = np.fliplr( self.pixoff )                         #offsets in x,y coordinates
        if self.WCS:
            self.offsets[:,0] = -self.offsets[:,0]
        #ipshell()
        
        MplMultiTab.__init__(self, figures=self.figures, labels=labels )
        
        self.tabWidget.currentChanged.connect( self.tabChange )
        
        #Connect to the canvases for aperture interactions
        for api in self.apis:           api.connect()
        
        #ipshell()
    
    #====================================================================================================
    def __len__(self):
        return len(self.apis)
    
    #====================================================================================================
    def create_main_frame(self, figures, labels):
        
        MplMultiTab.create_main_frame( self, figures, labels )
        
        buttonBox = self.create_buttons()
        
        hbox = qt.QHBoxLayout()
        hbox.addWidget(self.tabWidget )
        #hbox.addWidget(buttonBox)
        hbox.addLayout(buttonBox)
        self.vbox.addLayout(hbox)
        
        
        self.main_frame.setLayout(self.vbox)
        self.setCentralWidget(self.main_frame)
    
    #====================================================================================================
    def create_buttons(self):
        ''' Add axes Buttons '''
        
        buttonBox = qt.QVBoxLayout()
         
        self.buttons = {}
        labels = ['load coords', 'daofind', 'phot', 'Ap. Corr.', 'Propagate']
        func_names = ['load_coo', '_on_find_button', '_on_phot_button',  '_on_apcor_button', 'propagate']
        colours = ['g','g', 'orange', 'orange', 'orange']
        for i, label in enumerate(labels):
            F = getattr( self,  func_names[i] )
            #F = self._on_click#lambda s : print(s)
            button = self.add_button(F, labels[i], colours[i])
            self.buttons[labels[i]] = button
            buttonBox.addWidget( button )
        
        buttonBox.addItem( qt.QSpacerItem(20,200) )
        
        return buttonBox
    
    #====================================================================================================
    def add_button(self, func, label, colour):
            
        def colourtuple( colour, alpha=1, ncols=255 ):
            rgba01 = np.array(colorConverter.to_rgba( colour, alpha=alpha ))    #rgba array range [0,1]
            return tuple( (ncols*rgba01).astype(int) )
        
        bg_colour = colourtuple( colour )
        hover_colour = colourtuple( colour, alpha=.7 )
        press_colour = colourtuple( "blue", alpha=.7 )
        
        button = qt.QPushButton(label, self)
        style = Template("""
            QPushButton
            { 
                background-color: rgba$bg_colour;
                border-style: outset;
                border-width: 1px;
                border-radius: 3px;
                border-color: black;
                font: bold 14px;
                min-width: 10em;
                padding: 6px;
            }
            QPushButton:hover { background-color: rgba$hover_colour }
            QPushButton:pressed { background-color: rgba$press_colour }
            """).substitute( bg_colour=bg_colour, hover_colour=hover_colour,  press_colour=press_colour )
        
        palette = qt.QPalette( button.palette() ) # make a copy of the palette
        palette.setColor( qt.QPalette.ButtonText, qt.QColor('black') )
        button.setPalette( palette )
        
        button.setStyleSheet( style ) 
        button.clicked.connect( func )
        
        return button
    
    #====================================================================================================
    def _on_click(self):
        sender = self.sender()
        self.statusBar().showMessage(sender.text() + ' was pressed')  
    
    #===============================================================================================
    def tabChange(self, i):
        '''When a new tab is clicked, 
            assume apertures already added to axis
            initialise the meanstar plot (if needed)
            redraw the canvas.
        '''
        api = self.apis[i]
        fig = api.figure
        if not api.stars.meanstar.has_plot:
            api.stars.meanstar.init_plots( fig )
            fig.canvas.draw()
        #ipshell()
        
    #===============================================================================================
    @unhookPyQt
    def propagate(self, *args):
        import multiprocessing as mpc
        
        N = len(self)
        cidx = self.tabWidget.currentIndex()    #index of currently active tab
        stars = self.apis[cidx].stars
        Nstars = len(stars)
        
        #reshape, shift + rescale the star coordinates to match the images
        offsets = self.offsets[:,None]
        coords = np.tile( stars.coords, (N,1,1) )
        scales = self.scales[:,None]
        #fwhma = np.tile(stars.pullattr('fwhm'), (N,1))          #NOTE: this should be a suggestion for the fit
        coords = (coords + offsets) * scales
        
        idx = np.mgrid[1:N,:Nstars].T.reshape((N-1)*Nstars,2)
        idx = list(map(tuple, idx))
        
        args = [self.apis[ij[0]].pre_fit( coords[ij] ) for ij in idx]
        
        print( 'Fitting for {} stars.'.format(Nstars) )
        t0 = time()
        pool = mpc.Pool()       #pool of fitting tasks
        cluster = pool.map( starfucker, args )
        print( 'took {} sec'.format(time()-t0) )
        
        print( 'Appending for {} stars.'.format(Nstars) )
        t0 = time
        for i, (j, k) in enumerate(idx):
            self.apis[j].stars.append( cluster[i] )
        print( 'took {} sec'.format(time()-t0) )
        
    #===============================================================================================
    def get_coo( self, fn ):
        '''load star cordinates from file and return as array.'''
   
        #ffn, phfn = self.filenames.found_coo, self.filenames.phot_coo
        #for fn in (ffn, phfn):
        if os.path.isfile( fn ):
            
            print( 'Loading coordinate data from {}.'.format( fn ) )
            data = np.loadtxt(ffn, unpack=0)
            
            if data.shape[-1]==7:                         #IRAF generated coordinate files. TODO: files generated by this script should match this format
                coords, ids = data[:,:2], data[:,6]
            elif data.shape[-1]==2:
                coords = data
                ids = range(data.shape[0])
            
            elif 0 in data.shape:
                warn( 'Coordinate file "{}" contains no data!'.format(fn) )
                return
            else:
                warn( 'Coordinate file "{}" format not understood.'.format(fn) )
            
            if fn==ffn:
                coords -= 1    #Iraf uses 1-base indexing??
            #break
            
        else:
            print( 'No coordinate file exists!' )
            return
    
        return coords, ids
    
    #====================================================================================================
    def undup_coo( self, file_coo, known_coo, tolerance ):
        '''elliminate duplicate coordinates within radial tolerance.'''
        l_dup = [ np.linalg.norm(known_coo - coo, axis=1) < tolerance
                    for coo in file_coo ]
        return l_dup
    
    
    #====================================================================================================
    def load_coo( self ):
        '''load star cordinates from file and plot on image.'''
   
        if not hasattr(self, 'fwhmpsf'):
            warn( 'load_coo: Assuming FWHM PSF of 2.5' )
            #stars.psfaps.DEFAULT_COLOURS[0] = 'r'                      #AND IF YOU DO THE FITTING AFTER THIS???
            fwhm = 2.5
        else:
            fwhm = self.fwhmpsf
        
        txt_offset = 1.5*fwhm
        
        #ffn, phfn = self.filenames.found_coo, self.filenames.phot_coo
        fn = self.filenames.phot_coo
        coords, ids = self.get_coo( fn )
        
        #eliminate coordinate duplication (don't replot apertures already on image)
        if len(self.stars.coords):
            ldup = self.undup_coo( coords, self.stars.coords, fwhm )
            coords = coords[ldup]
            ids = ids[ldup]
        
        for *coo, idd in zip(coords, ids):
            if idd==1:          coo = [coo]     #ApertureCollection needs to know to anticipate extending horizontally
            
            star = self.stars.append( coo=coo[:], fwhm=fwhm, id=idd )         #plots the star apertures on the image assuming the previously measured fwhm for all stars (average)
            #NOTE: you should still do the fitting in the background????
            
            coo = np.squeeze(coo)
            anno = self.ax.annotate(str(int(idd)), coo, coo+txt_offset, color='w',size='small') #transform=self.ax.transData
            self.stars.annotations.append( anno )
            #always add the figure text for id!
        
        #Aperture checks + autocolour
        psfaps, _, skyaps = self.stars.get_apertures()
        skyaps.auto_colour( check='all', edge=self.image_data.shape, cross=psfaps )
        
        #(Re)plot apertures
        psfaps.axadd( self.ax )
        skyaps.axadd( self.ax )
        #self.update_legend( )
       
        self.figure.canvas.draw()
        
        #self.coo_loaded = 1
        
    #===============================================================================================
    @unhookPyQt
    def write_coo( self ):
        print( 'Writing coordinates of selected stars to file: {}'.format(self.filenames.phot_coo) )
        
        if os.path.isfile( self.filenames.phot_coo ):
            yn = input( 'Overwrite {} ([y]/n)?'.format(self.filenames.phot_coo) )
            if not yn in ['y', 'Y', '']:
                print( 'Nothing written.' )
                return
        
        
        print( 'self.filenames.found_coo', self.filenames.found_coo )
        print( 'os.path.exists( self.filenames.found_coo )', os.path.exists( self.filenames.found_coo ) )
        
        
        ipshell()
            
        
        if os.path.exists( self.filenames.found_coo ):
        
            fp_coo = open( self.filenames.found_coo, 'r' )
            
            if os.path.isfile( self.filenames.phot_coo ):
                yn = input( 'Overwrite {} ([y]/n)?'.format(self.filenames.phot_coo) )
            if yn in ['y', 'Y', '']:
                fp_phot = open( self.filenames.phot_coo, 'w' )
            else:
                print( 'Nothing written.' )
                return
            
            coords = np.array( self.stars.coords )
            if self.WCS:
                coords[:,0] = self.image_data.shape[1] - coords[:,0]    #map to original pixel positions (from sky pixel positions) (original image was flipped left-right)
            
            #eliminate duplicates in the coordinates found by daofind to within tolerance
            within = 5.                             #tolerance for star proximity in coordinate file (daofind)
            for line in fp_coo:
                if line.startswith('#'):
                    fp_phot.write( line )
                else:
                    redcoo = np.array( line.split()[:2], dtype=float )
                    r = map(la.norm, coords-redcoo)
                    idx = np.fromiter(r, dtype=float) < within 
                    if any(idx):
                        fp_phot.write( line )
                        coords = coords[~idx]
                        
            fp_coo.close()
            fp_phot.close()
        
        else:
            coords = np.array( self.stars.coords )
            if self.WCS:
                coords[:,0] = self.image_data.shape[1] - coords[:,0]
            
            np.savetxt(self.filenames.phot_coo, coords, fmt='%.3f')
    
    
    #===============================================================================================
    def _on_find_button( self ):
        '''run daofind on image and plot the found stars.'''
        #global daofind, phot                    #THIS IS BAD!! YOU NEED A QUEUE MANAGER
        
        #SHOW THRESHOLD ON COLOUR BAR???????????
        
        print( 'find button!!' )
        
        if os.path.isfile( self.filenames.found_coo ):                                #OPTION
            os.remove(self.filenames.found_coo)
            
        if len(self.stars):
            self.fwhmpsf = self.stars.get_mean_val('fwhm')
            self.sky_sigma = self.stars.get_mean_val('sky_sigma')
        else:
            self.fwhmpsf = 3.0
            self.sky_sigma = 50.
            msg = ("No data available - assuming default paramater values:\n"
                    "FWHM PSF = {}\nSKY SIGMA = {}\nDon't expect miracles...\n" ).format(self.fwhmpsf, self.sky_sigma)
            warn( msg )
        fwhm = self.fwhmpsf
        
        #daofind, phot = queue.get()
        
        starfinder.set_params( self )
        starfinder( self )                              #SUBCLASS???
        
        #self.stars.skyaps.remove()
        #self.stars.psfaps.remove()
        
        self.load_coo( )
        self._write_par = 0                     #not necessary to write parameter file
        
        #self.figure.canvas.draw()
        
    #===============================================================================================
    def _on_phot_button( self, event ):
        
        if not self.status:
            self.finalise( )
            
            print( 'Creating apertures for photometry.' )
            coords = self.stars.coords
            radii = photify.gen_ap_list( string=0 )
            radii = np.array( [radii]*len(coords) )
            coords = np.array( [coords]*(radii.shape[1]) ).reshape( list(radii.shape)+[2] )
            #print( 'RADII', radii.shape, radii )
            #print( 'COORDS', coords.shape, coords )
            #
            photaps = self.stars.photaps = ApertureCollection( coords=coords, radii=radii, gc='c', bc='r' )
            photaps.resize( 0 )
            photaps.addax( self.ax )
            
            #change button colour
            colour = 'g'
            lightcolour = colorConverter.to_rgba( colour, alpha=.7 )
            self.buttons['phot'].color = lightcolour
            self.buttons['phot'].hovercolor = colour
            
            self.status = 1
            self.figure.canvas.draw()
            
        else:
            output = os.path.join(path, 'all.mag')
            photify.set_params( self )
            photify.save_params( )
            
            #embed()
            
            print( 'SkyApertures', SkyApertures.RADII )
            print( 'Phot Aps', photify.phot.photpars.apertures)
            
            photify( fn_im_ls, output )

            self.filenames.mags = output
            self.filenames.image_list = fn_im_ls
        
    #===============================================================================================
    def _on_apcor_button( self ):
        
        apcorrs = ApCorr( self )                #Initialise Iraf task wrapper class
        apcorrs( )
    
    
    #===============================================================================================
    #@unhookPyQt
    def write_pars( self ):
        '''
        write sky & psf parameters to files.
        '''
        #functions = [photify.phot.fitskypars, starfinder.daofind.datapars]
        #fns = ['fitskypars.par', 'datapars.par']
        #for fn, func in zip(fns, functions):
            #if os.path.isfile( path+fn ):
                #yn = input( 'Overwrite {} ([y]/n)?'.format(fn) )
            #else:
                #yn = 'y'
            
            #if yn in ['y', 'Y', '']:
                #print( 'Writing parameter file: {}'.format( fn ) )
                #func.saveParList( path+fn )
        pass
        
    
    
    #===============================================================================================
    def finalise( self ):
        '''
        compute means. write parameter and coordinate files.
        '''
        try:
            self.fwhmpsf = self.stars.get_mean_val( 'fwhm' )
            self.sky_mean = self.stars.get_mean_val( 'sky_mean' )
            self.sky_sigma = self.stars.get_mean_val( 'sky_sigma' )
            print('\nThe following values were determined from %i stars:\nFWHM PSF: %5.3f\nSKY MEAN: %3.3f\nSKY SIGMA: %5.3f\n' %(len(self.stars), self.fwhmpsf, self.sky_mean, self.sky_sigma))
            
            print('The following image parameters hold:\nREADOUT NOISE: {:5.3f}\nSATURATION: {:5.3f}\n\n'.format(self.ron, self.saturation) )
            
            #TODO: PRINT TABLED INFO
            
        except:
            print('NO FITTING DONE!')
        #if self.stars.fitfig:             plt.close( self.stars.fitfig )
        
        if self._write_par:
            self.write_pars()
        self.write_coo()
        
        self.figure.savefig( 'aps.setup.png' )
        #TODO: SAVE AS PICKLED OBJECT!

    
    #===============================================================================================
    def _on_close(self, event):
        '''
        When figure is closed disconnect event listening.
        '''
        self.disconnect()
        
        self.finalise()
        
        plt.close(self.figure)
        print('Closed')


    #===============================================================================================
    @unhookPyQt
    def restart(self):                                                                                    #NAMING
        print('Restarting...')
        for artist in self.apertures.get_all():
            artist.remove()
        
        self.apertures.__init__()                                         #re-initialise Aperture class
        self.stars.__init__()                                             #re-initialise stars
        self.image_data_cleaned = copy(self.image_data)                   #reload image data for fitting

        self.figure.canvas.draw()
            
    #===============================================================================================
    def sky_map(self):
        '''produce logical array which serves as sky map'''
        r,c = self.image_shape
        Xp, Yp = self.pixels                                #the pixel grid
        sbuffer, swidth = self.sbuffer, r/2                         #NOTE: making the swidth paramater large (~half image width) will lead to skymap of entire image without the detected stars
        L_stars, L_sky = [], []
        for x,y in self.stars.found_coo:
            Rp = np.sqrt((Xp-x)**2+(Yp-y)**2)
            l_star = Rp < sbuffer
            l_sky = np.all([~l_star, Rp < sbuffer+swidth],0)          #sky annulus
            L_stars.append(l_star)
            L_sky.append(l_sky)

        L_all_stars = np.any(L_stars,0)
        L_all_sky = np.any(L_sky,0)
        L = np.all([L_all_sky,~L_all_stars],0)

        return L


    #===============================================================================================
    def get_fn(self):
        return self.filenames


        
    
        
        
#################################################################################################################################################
class DaoFind( object ):
    DEFAULT_THRESHOLD = 7.5
    #===============================================================================================
    def __init__(self, datapars=None, findpars=None):
        self.datapars = datapars if datapars else path + 'datapars.par'
        self.findpars = findpars if findpars else path +'findpars.par'
        
        iraf.noao( _doprint=0 )
        iraf.noao.digiphot( _doprint=0  )
        iraf.noao.digiphot.daophot( _doprint=0 )
        self.daofind = iraf.noao.digiphot.daophot.daofind
    
    #===============================================================================================    
    def __call__(self, selector):
        print( 'Finding stars...' )
        datapars, findpars = self.daofind.datapars, self.daofind.findpars
        self.daofind(   selector.filenames.image,
                        fwhmpsf=datapars.fwhmpsf,
                        sigma=datapars.sigma,
                        datamax=datapars.datamax,
                        output=selector.filenames.found_coo )
   
    #===============================================================================================
    def set_params(self, selector, threshold=None, **kwargs):
        
        datapars, findpars = self.daofind.datapars, self.daofind.findpars
        self.daofind.verify = 'no'
        
        #daofind.setParList( ParList='home/hannes/iraf-work/uparm/daofind.par' )
        #daofind.datapars.saveParList( path+'datapars.par' )                 #cannot assign parameter attributes without *.par file existing
        #daofind.datapars.unlearn()
        
        datapars.fwhmpsf =          selector.fwhmpsf                                 #ELSE DEFAULT
        datapars.sigma =            selector.sky_sigma
        datapars.datamax =          selector.saturation
        #datapars.gain
        datapars.readnoise =        selector.ron
        datapars.exposure =         'exposure'                      #Exposure time image header keyword
        datapars.airmass =          'airmass'                       #Airmass image header keyword
        datapars.filter =           'filter'                        #Filter image header keyword
        datapars.obstime =          'utc-obs'                       #Time of observation image header keyword
        
        threshold = threshold if threshold else self.DEFAULT_THRESHOLD
        findpars.threshold =        threshold                       #Threshold in sigma for feature detection
        findpars.nsigma =           1.5                             #Width of convolution kernel in sigma
        findpars.ratio  =           1.                              #Ratio of minor to major axis of Gaussian kernel
        findpars.theta  =           0.                              #Position angle of major axis of Gaussian kernel
        findpars.sharplo=           0.2                             #Lower bound on sharpness for feature detection
        findpars.sharphi=           1.                              #Upper bound on sharpness for  feature detection
        findpars.roundlo=           -1.                             #Lower bound on roundness for feature detection
        findpars.roundhi=           1.                              #Upper bound on roundness for feature detection
        findpars.mkdetec=           'no'                            #Mark detections on the image display ?
    
    #===============================================================================================    
    def save_params( self ):
        datapars, findpars = self.daofind.datapars, self.daofind.findpars
        datapars.saveParList( self.datapars )
        findpars.saveParList( self.finpars )
        
        
#################################################################################################################################################
class Phot( object):
    NAPERTS = 15
    #===============================================================================================
    def __init__(self, centerpars=None, fitskypars=None, photpars=None):
        self.centerpars = centerpars if centerpars else path + 'centerpars.par'
        self.fitskypars = fitskypars if fitskypars else path + 'fitskypars.par'
        self.photpars = photpars if photpars else path + 'photpars.par'
        
        iraf.noao()
        iraf.noao.digiphot()
        iraf.noao.digiphot.daophot()
        self.phot = iraf.noao.digiphot.daophot.phot
    
    #===============================================================================================    
    def __call__(self, image_list, output):
        '''
        image_list - txt list of images to do photometry on
        output - output magnitude filename 
        '''
        
        if os.path.isfile( path+output ):
            os.remove( path+output )

        print('Doing photometry')
        coords = selector.filenames.phot_coo
        image = '@'+image_list
        
        datapars, centerpars, fitskypars, photpars = self.phot.datapars, self.phot.centerpars, self.phot.fitskypars, self.phot.photpars
        self.phot(      image=image,
                        coords=coords,
                        datapars=datapars,
                        centerpars=centerpars,
                        fitskypars=fitskypars,
                        photpars=photpars,
                        output=output)
    
    #===============================================================================================
    def set_params(self, selector):    
        
        datapars, centerpars, fitskypars, photpars = self.phot.datapars, self.phot.centerpars, self.phot.fitskypars, self.phot.photpars
        
        datapars.fwhmpsf = selector.fwhmpsf
        datapars.sigma = selector.sky_sigma
        datapars.datamax = selector.saturation
        #datapars.gain
        datapars.readnoise =           selector.ron
        datapars.exposure =            'exposure'                      #Exposure time image header keyword
        datapars.airmass =             'airmass'                       #Airmass image header keyword
        datapars.filter =              'filter'                        #Filter image header keyword
        datapars.obstime =             'utc-obs'                       #Time of observation image header keyword
        
        
        centerpars.calgorithm = "centroid"                             #Centering algorithm
        centerpars.cbox = selector.cbox                                   #Centering box width in scale units
        centerpars.cthreshold = 3.                                     #Centering threshold in sigma above background
        centerpars.minsnratio = 1.                                     #Minimum signal-to-noise ratio for centering algorithim
        centerpars.cmaxiter = 10                                       #Maximum iterations for centering algorithm
        centerpars.maxshift = 5.                                       #Maximum center shift in scale units
        centerpars.clean = 'no'                                        #Symmetry clean before centering
        centerpars.rclean = 1.                                         #Cleaning radius in scale units
        centerpars.rclip = 2.                                          #Clipping radius in scale units
        centerpars.kclean = 3.                                         #K-sigma rejection criterion in skysigma
        centerpars.mkcenter = 'no'                                     #Mark the computed center
        
        
        #NOTE: The sky fitting  algorithm  to  be  employed.  The  sky  fitting
                #options are:

                #constant                                                                       <----- IF STARS IN SKY
                        #Use  a  user  supplied  constant value skyvalue for the sky.
                        #This  algorithm  is  useful  for  measuring  large  resolved
                        #objects on flat backgrounds such as galaxies or commets.

                #file                                                                           <----- IF PREVIOUSLY KNOWN????
                        #Read  sky values from a text file. This option is useful for
                        #importing user determined sky values into APPHOT.

                #mean                                                                           <----- IF LOW SKY COUNTS
                        #Compute  the  mean  of  the  sky  pixel  distribution.  This
                        #algorithm  is  useful  for  computing  sky values in regions
                        #with few background counts.

                #median                                                                         <----- IF RAPIDLY VARYING SKY / STARS IN SKY
                        #Compute the median  of  the  sky  pixel  distribution.  This
                        #algorithm  is  a  useful for computing sky values in regions
                        #with  rapidly  varying  sky  backgrounds  and  is   a   good
                        #alternative to "centroid".

                #mode                                                                           <----- IF CROWDED FIELD AND STABLE SKY
                        #Compute  the  mode  of  the sky pixel distribution using the
                        #computed  mean  and  median.   This   is   the   recommended
                        #algorithm  for  APPHOT  users  trying  to  measuring stellar
                        #objects in crowded stellar  fields.  Mode  may  not  perform
                        #well in regions with rapidly varying sky backgrounds.

                #centroid                                                                       <----- DEFAULT
                        #Compute  the  intensity-weighted mean or centroid of the sky
                        #pixel histogram.  This  is  the  algorithm  recommended  for
                        #most  APPHOT  users.  It  is  reasonably  robust  in rapidly
                        #varying and crowded regions.

                #gauss
                        #Fit a single Gaussian function to the  sky  pixel  histogram
                        #using non-linear least-squares techniques.

                #ofilter
                        #Compute  the sky using the optimal filtering algorithm and a
                        #triangular weighting function and the histogram of  the  sky
                        #pixels.
        
        swidth = abs(np.subtract( *SkyApertures.RADII ))
        
        fitskypars.salgorithm = "centroid"                             #Sky fitting algorithm
        fitskypars.annulus = SkyApertures.RADII[0]                         #Inner radius of sky annulus in scale units
        fitskypars.dannulus = swidth                                   #Width of sky annulus in scale units
        fitskypars.skyvalue = 0.                                       #User sky value                                 #self.sky_mean
        fitskypars.smaxiter = 10                                       #Maximum number of sky fitting iterations
        fitskypars.sloclip = 3.                                        #Lower clipping factor in percent
        fitskypars.shiclip = 3.                                        #Upper clipping factor in percent
        fitskypars.snreject = 50                                       #Maximum number of sky fitting rejection iterations
        fitskypars.sloreject = 3.                                      #Lower K-sigma rejection limit in sky sigma
        fitskypars.shireject = 3.                                      #Upper K-sigma rejection limit in sky sigma
        fitskypars.khist = 3.                                          #Half width of histogram in sky sigma
        fitskypars.binsize = 0.1                                       #Binsize of histogram in sky sigma
        fitskypars.smooth = 'no'                                       #Boxcar smooth the histogram
        fitskypars.rgrow = 0.                                          #Region growing radius in scale units
        fitskypars.mksky = 'no'                                        #Mark sky annuli on the display
        
        photpars.weighting = "constant"                                #Photometric weighting scheme
        photpars.apertures = self.gen_ap_list()                        #List of aperture radii in scale units
        photpars.zmag = 0.                                             #Zero point of magnitude scale
        photpars.mkapert = 'no'                                        #Draw apertures on the display
        
    #===============================================================================================
    def save_params(self):
        phot = self.phot
        phot.centerpars.saveParList( self.centerpars )
        phot.fitskypars.saveParList( self.fitskypars )
        phot.photpars.saveParList( self.photpars )
    
    #===============================================================================================
    @staticmethod
    def gen_ap_list( rmax=np.ceil(SkyApertures.RADII[1]), naperts=None, string=1):
        '''
        Generate list of parabolically increasing aperture radii.
        '''
        naperts = naperts if naperts else Phot.NAPERTS
        aps = np.linspace(1, rmax, naperts)
        #aps =  np.polyval( (0.085,0.15,2), range(naperts+1) )            #apertures generated by parabola
        if string:
            aps = ', '.join( map(str, np.round(aps,2)) )
        return aps
        
        
#################################################################################################################################################
class ApCorr( object ):
    
    #===============================================================================================
    def __init__(self, selector, mkappars=None, mags=None, ):
        
        self.mkappars = mkappars if mkappars else path+'phlmkapfe.par'
        self.photfile =  selector.filenames.mags
        #self.naperts = selector.naperts
        
        iraf.noao()
        iraf.noao.digiphot()
        iraf.noao.digiphot.photcal()
        self.mkapfile = iraf.noao.digiphot.photcal.mkapfile
    
    #===============================================================================================    
    def __call__( self, apercors='all.mag.cor', bestmagfile='apcor.mag', logfile='apcor.log'):
        
        for fn in [apercors, bestmagfile, logfile]:                             #OVERWRITE FUNCTION
            if os.path.isfile(path+fn):       os.remove(path+fn)
        
        print('Doing aperture corrections...')
        naperts = self.mkapfile.naperts
        self.mkapfile( photfile=self.photfile, naperts=naperts, apercors=apercors)

        selector.filenames.apercors = apercors
        selector.filenames.magfile = bestmagfile
        selector.filenames.logfile = logfile
        
    #===============================================================================================   
    def set_params( self, selector ):
        
        mkapfile = self.mkapfile
        mkapfile.photfiles = selector.filenames.mags                        #The input list of APPHOT/DAOPHOT databases
        mkapfile.naperts = Phot.NAPERTS                                     #The number of apertures to extract
        mkapfile.apercors = apercors                                        #The output aperture corrections file
        mkapfile.smallap = 2                                                #The first aperture for the correction
        mkapfile.largeap = Phot.NAPERTS                                     #The last aperture for the correction
        mkapfile.magfile = magfile                                          #The optional output best magnitudes file
        mkapfile.logfile = logfile                                          #The optional output log file
        mkapfile.plotfile = ''                                              #The optional output plot file
        mkapfile.obsparams = ''                                             #The observing parameters file
        mkapfile.obscolumns = '2 3 4 5'                                     #The observing parameters file format
        mkapfile.append = 'no'                                              #Open log and plot files in append mode
        mkapfile.maglim = 0.1                                               #The maximum permitted magnitude error
        mkapfile.nparams = 3                                                #Number of cog model parameters to fit
        mkapfile.swings = 1.2                                               #The power law slope of the stellar wings
        mkapfile.pwings = 0.1                                               #The fraction of the total power in the stellar wings
        mkapfile.pgauss = 0.5                                               #The fraction of the core power in the gaussian core
        mkapfile.rgescale = 0.9                                             #The exponential / gaussian radial scales
        mkapfile.xwings = 0.                                                #The extinction coefficient
        mkapfile.interactive = 'no'                                         #Do the fit interactively ?
        mkapfile.verify = 'no'                                              #Verify interactive user input ?
    
    #===============================================================================================    
    def save_params( self ):
        self.mkapfile.saveParList( self.mkappars )
        
        
        
#################################################################################################################################################

class Cube(StarSelector):                                                #NEED TO FIGURE OUT HOW TO DO THIS WITH SUPER
    def __init__(self, stars):
        self.filenames = stars.get_fn()
        self.fwhmpsf = stars.fwhmpsf
        self.sky_mean = stars.sky_mean
        self.sky_sigma = stars.sky_sigma
        self.saturation = stars.saturation
        
    #===============================================================================================
    def get_ron(self):
        return float(pyfits.getval(self.filenames.image,'ron'))

    #===============================================================================================
    def sky_cube(self, im_list, stars, n):
        '''Determine the variation of the background sky across the image cube.'''
        '''n - number of images in cube to measure. '''
        print('Determining sky envelope from subsample of %i images...' %n)
        im_list = np.array(im_list)
        N = len(im_list)
        j = N/2                                                                     #use image in middle of cube for sky template
        im_data = pyfits.getdata( im_list[j] )
        sky_envelope = np.zeros( im_data.shape )
        L = stars.sky_map()                         #!!!!!!!!!!!!!!!!!!!!!                                  #sky template
        Bresenham = lambda m, n: np.array([i*n//m + n//(2*m) for i in range(m)])    #select m points from n integers using Bresenham's line algorithm
        ind = Bresenham(n,N)                                                        #sequence of images used in determining sky values
        sky_mean = []; sky_sigma = []
        for im in im_list[ind]:
            im_data = pyfits.getdata( im )
            sky_mean.append( np.mean( im_data[L] ))                                   #USE MEDIAN FOR ROBUSTNESS???? (THROUGHOUT??)
            sky_sigma.append( np.std( im_data[L] ))
            sky_envelope = np.max([sky_envelope,im_data], axis=0)
        #sky_envelope[~L] = 0
        self.sky_mean = np.mean(sky_mean)
        self.sky_sigma = np.mean(sky_sigma)
        return ind, sky_mean, sky_sigma, sky_envelope

#################################################################################################################################################

def val_float(num_str):
    try:
        return isinstance(float(num_str), float)
    except:
        return False

        
def par_from_file( filename ):
    name, typ, mode, default, mn, mx, prompt = np.genfromtxt(filename, dtype=str, delimiter=',', skip_footer=1, unpack=1)
    value_dict = dict( zip(name, default) )
    return value_dict

#################################################################################################################################################


#Initialise autocompletion
#comp = Completer()

#readline.parse_and_bind('tab: complete')                                       #sets the autocomplete function to look for filename matches
#readline.parse_and_bind('set match-hidden-files off')
#readline.parse_and_bind('set skip-completed-text on')
#readline.set_completer(comp.complete)                                          #THIS BREAKS iPYTHON IN SOME CASES....


#start background imports
#queue = mpc.Queue()
#import_process = mpc.Process( target=background_imports, args=(queue,) )
#import_process.start()


#Initialise readout noise table
fn_rnt = '/home/hannes/iraf-work/SHOC_ReadoutNoiseTable_new'
RNT = ReadNoiseTable(fn_rnt)
RUN_DIR = os.getcwd()

#Parse command line arguments
import argparse, glob
parser = argparse.ArgumentParser(description='')
parser.add_argument('-d', '--dir', default=os.getcwd(), dest='dir', help = 'The data directory. Defaults to current working directory.')
parser.add_argument('-x', '--coords', dest='coo', help = 'File containing star coordinates.')
parser.add_argument('-c', '--cubes', nargs='+', default=['cubes.bff.txt'], help = 'Science data cubes to be processed.  Requires at least one argument.  Argument can be explicit list of files, a glob expression, or a txt list.')
#parser.add_argument('-l', '--image-list', default='all.split.txt', dest='ims', help = 'File containing list of image fits files.')

args = parser.parse_args()
path = validity.test(args.dir, os.path.exists, raise_error=1)
path = os.path.abspath(path) + os.sep

#Read the cubes as SHOC_Run
args.cubes = os.path.join( path, args.cubes[0] )
args.cubes = validity.test_ls(args.cubes, validity.trivial, raise_error=1)
cubes = SHOC_Run( filenames=args.cubes, label='sci' )

#generate the unique filenames (as was done when splitting the cube)
reduction_path = os.path.join( path, 'ReducedData' )
cubes.magic_filenames( reduction_path )
fns = []
for cube in cubes:
    fns+= list(cube.filename_gen(1))

#Read input coordinates
if args.coo:
    args.coo = os.path.join( path, args.coo )
    args.coo = validity.test(args.coo, os.path.exists, raise_error=1)


#################################################################################################################################################
#Set global plotting parameters
rc( 'savefig', directory=path )
#rc( 'figure.subplot', left=0.065, right=0.95, bottom=0.075, top=0.95 )    # the left, right, bottom, top of the subplots of the figure
#################################################################################################################################################
    
    
    
#fn_found_coo = args.coo if args.coo else 'all.coo'
#found_coo = os.path.join( path, fn_found_coo )
#phot_coo = os.path.join( path, 'phot.coo' )
    

#Initialise Iraf task wrapper classes
#starfinder = DaoFind( )
#photify = Phot( )


#Check for pre-existing psf measurements
try:
    fitskypars = par_from_file( path+'fitskypars.par' )
    sky_inner = float( fitskypars['annulus'] )
    sky_d = float( fitskypars['dannulus'] )
    sky_outer = sky_inner +  sky_d
    
    datapars = par_from_file( path+'datapars.par' )
    fwhmpsf = float( datapars['fwhmpsf'] )
    sky_sigma = float( datapars['sigma'] )
    ron = float( datapars['readnoise'] )
    saturation = datapars['datamax']
    
    msg = ('Use the following psf + sky values from fitskypars.par and datapars.par? ' 
                '\nFWHM PSF: {:5.3f}'
                '\nSKY SIGMA: {:5.3f}'
                '\nSKY_INNER: {:5.3f}'
                '\nSKY_OUTER: {:5.3f}\n'
                '\nREADOUT NOISE: {:5.3f}'
                '\nSATURATION: {}'.format( fwhmpsf, sky_sigma, sky_inner, sky_outer, ron, saturation ) )
    print( msg )
    yn = input('([y]/n):')
except IOError:
    yn = 'n'
    #mode = 'fit'

    
#Initialise GUI
app = qt.QApplication(sys.argv)
selector = StarSelector( fns, args.coo )
    
    
if yn in ['y','Y','']:
    selector.fwhmpsf, selector.sky_mean, selector.sky_sigma, selector.ron = fwhmpsf, sky_mean, sky_sigma, ron
    #mode = 'select'


#THIS NEEDS TO HAPPEN IN A QUEUE!!!!!!!!!!!! 
#ipshell()


selector.show()
sys.exit( app.exec_() )

#daofind, phot = queue.get()





#L = selector.sky_map()
#sky_data = copy(selector.image_data)
#sky_mean = np.mean(sky_data[L])                                        #USE MEDIAN FOR ROBUSTNESS???? (THROUGHOUT??)
#sky_sigma = np.std(sky_data[L])
#sky_data[~L] = 0

#plt.figure(-1)
#plt.imshow(sky_data)


#Here the sky envelope + skybox
#stack = Cube(selector)
#inds, cube_mean, cube_sigma, sky_env = stack.sky_cube(im_list,selector,100)
#print 'Cube sky mean = ', np.mean(cube_mean)
#print 'Cube sky stdev = ', np.mean(cube_sigma)





#plt.figure(3)
#plt.errorbar(inds, cube_mean, yerr=cube_sigma, fmt='o')
#plt.title( 'Mean Sky Counts' )
#plt.ylabel( 'Counts / pixel' )

#plt.figure(4)
#zlims = plf.zscale_range(sky_env,contrast=1./99)
#plt.imshow(sky_env, origin='lower', vmin=zlims[0], vmax=zlims[1])
#plt.colorbar()
#plt.title('Sky envelope')



#selector.filenames.phot_coo = 'phot.coo'
#selector.filenames.mags = 'allmag'                                 #NOTE: YOU NEED TO DO THIS OUTSIDE THIS METHOD IF YOU WISH TO BE ABLE TO RUN phot AND ap_cor    METHODS INDEPENDANTLY........................



##stars.aperture = [2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 10, 12, 14, 16]
#stack.naperts = 10


#stack.phot('all.split.txt', stars, 'allmag')
#stack.ap_cor()

def goodbye():
    os.chdir(RUN_DIR)
    print('Adios!')

import atexit
atexit.register( goodbye )
