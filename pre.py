# -*- coding: utf-8 -*-

# SHOC pre-reductions (with bells and whistles)

#TODO:

#IO:
    #option for verbosity level
    #option to log to file
    #decorate warnings to contain issuing function info

#IMAGE COMBINE ALGORITHM 
    #----> PIXEL REJECTION / SIGMA CLIPPING

#(optional) MULTIPROCESSING!!!!!!!!!!!!!!!!!!!!!!

#IMPROVE DOCSTRINGS

#MAKE EXTENSIBLE
    #split class definitions into separate scripts

#OPTIONS FOR DISPLAYING COMPUTED IMAGES + STATS

#TODO: check which imports are so bloody slow.... localize them???

from types import ModuleType
def tsktsk(s):
    t2 = time.time()
    if isinstance(s, ModuleType):
        s = s.__name__
    print(s, t2-t1)

print( 'Importing modules...' )
import time
t1 = time.time()

import numpy as np
import astropy.io.fits as pyfits
#import astropy.io.fits as pyfits

#import matplotlib.animation as ani
#import matplotlib.pyplot as plt

import os
import re
import datetime
import subprocess
#from glob import glob
import textwrap

import collections as coll
import itertools as itt

from copy import copy

from astropy.time import TimeDelta
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.coordinates.angles import Angle
from astropy.table import Table as aTable
from astropy.table import Column

from misc import *#interleave, getTerminalSize
from myio import warn, iocheck, parsetolist
from superstring import SuperString, ProgressBar, rreplace
from superstring import Table as sTable

from irafhacks import make_iraf_slicemap

from obstools.airmass   import Young94, altitude

from pySHOC.readnoise   import ReadNoiseTable
from pySHOC.timing      import Time
from pySHOC.io          import Input
from pySHOC.io          import ValidityTests as validity
from pySHOC.io          import Conversion as convert

#import matplotlib.pyplot as plt
#from imagine import supershow


#from decor import profile

from IPython import embed
#from magic.string import banner


tsktsk('modules')
print( 'Done!\n\n' )

#################################################################################################################################################################################################################
#Function definitions
#################################################################################################################################################################################################################

#def in_ipython():
    #try:
        #return __IPYTHON__
    #except:
        #return False

#################################################################################################################################################################################################################
#if not in_ipython():
    #print('\n\nRunning in Python. Defining autocompletion functions.\n\n')
    #import readline
    #from completer import Completer
    
    #comp = Completer()
    #readline.parse_and_bind('tab: complete')
    ##readline.parse_and_bind('set match-hidden-files off')
    #readline.parse_and_bind('set skip-completed-text on')
    #readline.set_completer(comp.complete)                                                   #sets the autocomplete function to look for filename matches
#else:
    #print('\n\nRunning in IPython...\n\n')

def link_to_short_name_because_iraf_sux(filename, count, ext):
    #HACK! BECAUSE IRAF SUX
    linkname = args.dir + '/s{}.{}'.format(count, ext)
    print( 'LINKING:', 'ln -f', os.path.basename(filename), os.path.basename(linkname) )
    subprocess.call( ['ln', '-f', filename, linkname] )


#################################################################################################################################################################################################################
#class definitions
#################################################################################################################################################################################################################

class Date(datetime.date):
    '''We need this so the datetime.date instances print in date format instead of the class representation
    format, when print is called on, for eg. a tuple containing a date_time object.'''
    #====================================================================================================
    def __repr__(self):
        return str(self)


#################################################################################################################################################################################################################
        
class FilenameGenerator(object):
    #====================================================================================================
    def __init__(self, basename, reduction_path='', padwidth=None, sep='.', extension='.fits'):
        self.count = 1
        self.basename = basename
        self.path = reduction_path
        self.padwidth = padwidth
        self.sep = sep
        self.extension = extension
 #====================================================================================================
    def __call__(self, maxcount=None, **kw):
        '''Generator of filenames of unpacked cube.'''
        path            = kw.get('path', self.path)
        sep             = kw.get('sep', self.sep)
        extension       = kw.get('extension', self.extension)
        
        base = os.path.join(path, self.basename)
        
        if maxcount:
            while self.count<=maxcount:
                imnr = '{1:0>{0}}'.format(self.padwidth, self.count)                                                                      #image number string. eg: '0013'
                outname = '{}{}{}{}'.format(base, sep, imnr, extension )                                                                  # name string eg. 'darkstar.0013.fits'
                self.count += 1
                yield outname
        else:
            yield '{}{}'.format(base, self.extension)
        

#################################################################################################################################################################################################################      
class SHOC_Cube( pyfits.hdu.hdulist.HDUList ):          #HACK:  Subclass PrimaryHDU instead???
    '''
    Extend the hdu.hdulist.HDUList class to perform simple tasks on the image stacks.
    '''
    #====================================================================================================
    @classmethod
    def load(cls, fileobj, mode='readonly', memmap=False,
                 save_backup=False, **kwargs):
        do_timing = kwargs.pop('do_timing', False )
        
        hdulist = cls._readfrom(fileobj=fileobj, mode=mode, memmap=memmap,
                                save_backup=save_backup, ignore_missing_end=True, **kwargs)
        
        hdulist.instrumental_setup()
        if do_timing:
            hdulist.time_init()
        
        return hdulist
    
    #====================================================================================================
    def __init__(self, hdus=[], file=None):

        super(SHOC_Cube, self).__init__(hdus, file)
        self._needs_flip        = False
        self._needs_sub         = []
        self._is_master         = False
        self._is_unpacked       = False
        self._is_subframed      = False
        
        self.path, self.basename = os.path.split( self.filename() )
        if self.basename:
            self.basename = self.basename.replace('.fits','')
        
        #self.filename_gen = FilenameGenerator(self.basename)
        self.trigger = None
    #====================================================================================================
    def __repr__(self):
        name, dattrs, values = self.get_instrumental_setup()
        ref = tuple( interleave(dattrs, values) )
        r = name + ':\t' + '%s = %s;\t'*len(values) %ref
        return '{} ==> {}'.format( self.__class__.__name__, r )
    
    #====================================================================================================
    def get_filename(self, with_path=0, with_ext=1, suffix=(), sep='.'):
        if with_path:
            filename = self.filename()
        else:
            _, filename = os.path.split( self.filename() )
        
        *stuff, ext = filename.split(sep)
        ext = [ext] if with_ext else ['']
        suffix = [suffix] if isinstance(suffix,str) else list(suffix)
        suffix = [s.strip(sep) for s in suffix]
        
        return sep.join( filter(None, stuff+suffix+ext) )
    
    #====================================================================================================
    def instrumental_setup(self):
        '''Retrieve the relevant information about the observational setup.  Used for comparitive tests.'''
        header = self[0].header
        
        #exposure time
        self.kct = header.get('kct')               #will only work for internal triggering - i.e. darks & flats.  If external triggering, set explicitly with self.kct =  self.get_kct()
        
        #date
        date, time = header['DATE'].split('T')
        self.date = Date( *map(int, date.split('-')) )          #file creation date
        h = int( time.split(':')[0] )
        self.namedate = self.date - datetime.timedelta( int(h < 12) )                   #starting date of the observing run --> used for naming
        
        #image binning
        self.binning =  tuple(header[s+'BIN'] for s in ['H','V'])
        
        #gain
        self.gain = header.get( 'gain', 0 )
        
        #image dimensions
        self.ndims = header['NAXIS']                                                    #Number of image dimensions
        self.shape = tuple( header['NAXIS'+str(i)] for i in range(1, self.ndims+1) )
        self.dimension = self.shape[:2]                                                      #Pixel dimensions for 2D images
        
        
        #sub-framing
        xb,xe,ye,yb = map( int, header['SUBRECT'].split(',') )
        self.subrect = xb,xe,ye,yb
        xb //= self.binning[0]; xe //= self.binning[0]
        yb //= self.binning[1]; ye //= self.binning[1]
        self.sub = xb,xe,yb,ye
        self._is_subframed = (xe, ye)!=self.dimension
                
        
        #CCD mode
        speed = 1./header['READTIME']
        speed_MHz = int(round(speed/1.e6))

        preamp  = header['PREAMP']
        mode    = header['OUTPTAMP']
        self.acqmode = header['ACQMODE']
        mode_abrv = 'CON' if mode.startswith('C') else 'EM'
        self.mode = '{} MHz, PreAmp @ {}, {}'.format( speed_MHz, preamp, mode_abrv )
        self.mode_trim = '{}MHz{}{}'.format( speed_MHz, preamp, mode_abrv )
        #orientation
        self.flip_state = tuple(header['FLIP'+s] for s in ['X','Y'])
        
        #Time Triggering mode
        self.trigger_mode = header['trigger']
        
        #instrument
        serno = header['SERNO']
        if serno==5982:
            self.instrument = 'SHOC 1'
        elif serno==6448:
            self.instrument = 'SHOC 2'
        else:
            self.instrument = 'unknown!'
            
        #telescope
        #self.telescope = header['telescope']
        
    #====================================================================================================
    def get_instrumental_setup(self, attrs=None):                                                                                                  #YOU CAN MAKE THIS __REPR__????????
        attrs =  attrs  or ['binning', 'dimension', 'mode', 'gain', 'trigger_mode', 'kct']
        dattrs = [at.replace('_',' ').upper() for at in attrs]        #for display
        vals = [ getattr(self, attr, '??') for attr in attrs ]
    
        name = self.get_filename() or 'Unsaved'
        
        return name, dattrs, vals
    
    #====================================================================================================
    def get_pixel_scale(self, telescope):
        '''get pixel scale in arcsec '''
        pixscale = {'1.9'     :       0.076,
                    '1.9+'    :       0.163,          #with focal reducer
                    '1.0'     :       0.167,
                    '0.75'    :       0.218   }
        
        tel = rreplace( telescope, ('focal reducer','with focal reducer'), '+')
        tel = tel.replace('m', '').strip()
        
        return np.array( self.binning ) * pixscale[ tel ]
                    
    #====================================================================================================
    def get_fov(self, telescope):
        '''get fov in arcmin'''
        fov = { '1.9'     :       (1.29, 1.29),
                '1.9+'    :       (2.79, 2.79),        #with focal reducer
                '1.0'     :       (2.85, 2.85),
                '0.75'    :       (3.73, 3.73)   }
        
        tel = rreplace( telescope, ('focal reducer','with focal reducer'), '+')
        tel = tel.replace('m', '').strip()
        
        return fov[ tel ]
        
    #====================================================================================================
    def check(self, frame2, key, raise_error=0):
        '''check fits headers in this image agains frame2 for consistency of key attribute 
        Parameters
        ----------
        key : The attribute to be checked (binning / instrument mode / dimensions / flip state)
        frame2 : SHOC_Cube Objects to check agains
        
        Returns
        ------
        flag : Do the keys match?
        '''
        flag = getattr(self,key) == getattr(frame2,key)
            
        if not flag and raise_error:
            raise ValueError
        else:
            return flag

    #====================================================================================================
    def flip(self):
        flipx, flipy = self.flip_state
        data = self[0].data 
        header = self[0].header
        if flipx:
            print( 'Flipping {} in X.'.format(self.get_filename()) )
            self[0].data = np.fliplr(self[0].data)
            header['FLIPX'] = int(not flipx)
            self.flip_state = tuple(header['FLIP'+s] for s in ['X','Y'])
        if flipy:
            print( 'Flipping {} in Y.'.format(self.get_filename()) )
            data = np.flipud(data)
            header['FLIPY'] = int(not flipy)
            self.flip_state = tuple(header['FLIP'+s] for s in ['X','Y'])

    #====================================================================================================
    def subframe(self, subreg, write=1):
        if self._is_subframed:
            raise TypeError( '{} is already sub-framed!'.format(self.filename()) )
        
        embed()
        
        cb,ce, rb,re = subreg
        print( 'subframing {} to {}'.format( self.filename(), [rb,re,cb,ce]) )
        
        data = self[0].data[rb:re,cb:ce]
        header = self[0].header
        #header['SUBRECT']??
        
        print( '!'*8,  self.sub )
        
        subext = 'sub{}x{}'.format(re-rb, ce-cb)
        outname = self.get_filename(1,1, subext )
        fileobj = pyfits.file._File(outname, mode='ostream', clobber=True)
        
        hdu = pyfits.hdu.PrimaryHDU( data=data, header=header  )
        #embed()
        
        stack = SHOC_Cube(hdu, fileobj)
        stack.instrumental_setup()
        
        #stack._is_subframed = 1
        #stack._needs_sub = []
        #stack.sub = subreg
        
        if write:
            stack.writeto( outname, output_verify='warn' )
        
        return stack

    #====================================================================================================
    def combine(self, func):
        ''' Mean/Median combines the stack using pyfits.
        
        "Median combining can completely remove cosmic ray hits and radioactive decay trails
        from the result, which cannot be done by standard mean combining. However, median 
        combining achieves an ultimate signal to noise ratio about 80% that of mean combining
        the same number of frames. The difference in signal to noise ratio can be compensated 
        by median combining 57% more frames than if mean combining were used. In addition, 
        variants on mean combining, such as sigma clipping, can remove deviant pixels while 
        improving the S/N somewhere between that of median combining and ordinary mean 
        combining. In a nutshell, if all images are "clean", use mean combining. If the 
        images have mild to severe contamination by radiation events such as cosmic rays, 
        use the median or sigma clipping method." - Newberry
        '''

        imnr = '001'                                                                                  #HMMMMMMMMMMMMMMM       #THIS WILL NEED TO CHANGE FOR MULTIPLE SINGLE IMAGES AS INPUT
        header = copy( self[0].header )
        data = func(self[0].data, 0)                                                             #median across images              MULTIPROCESSING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        
        ncomb = header.pop('NUMKIN') if 'numkin' in  header else 0                  #Delete the NUMKIN header keyword
        if 'naxis3' in header:          header.remove( 'NAXIS3' )
        header['NAXIS'] = 2
        header['NCOMBINE'] = ( ncomb, 'Number of images combined' )
        header['ICMB'+imnr] = ( self.filename(), 'Contributors to combined output image' )                               #THIS WILL NEED TO CHANGE FOR MULTIPLE SINGLE IMAGES AS INPUT
        
        #Load the stack as a SHOC_Cube
        hdu = pyfits.PrimaryHDU( data, header )
        outname = next( self.filename_gen() )        #generate the filename
        fileobj = pyfits.file._File(outname, mode='ostream', clobber=True)
        stack = SHOC_Cube(hdu, fileobj)                 #initialise the Cube with target file
        stack.instrumental_setup()
        
        return stack
    
    #====================================================================================================
    def unpack(self, count=1, padw=None, dryrun=0, w2f=1):                              #MULTIPROCESSING!!!!!!!!!!!!
        '''Unpack (split) a 3D cube of images along the 3rd axis. 
        Parameters
        ----------
        outpath : The directory where the images will be unpacked
        count : A running file count
        padw : The number of place holders for the number suffix in filename
        dryrun: Whether to actually unpack the stack
        
        Returns
        ------
        count
        '''
        start_time = time.time()

        stack = self.get_filename()
        header = copy( self[0].header )
        naxis3 = self.shape[-1]
        self.filename_gen.padwidth = padw if padw else len(str(naxis3))                                                 #number of digits in filename number string
        self.filename_gen.count = count

        if not dryrun:
            #edit header
            header.remove( 'NUMKIN' )
            header.remove( 'NAXIS3' ) #Delete this keyword so it does not propagate into the headers of the split files
            header['NAXIS'] = 2       #Number of axes becomes 2
            header.add_history( 'Split from %s' %stack )
            
            #open the txt list for writing
            if w2f:
                basename = self.get_filename(1,0)
                self.unpacked = basename + '.split'
                fp = open(self.unpacked, 'w')
            
            print( '\n\nUnpacking the stack {} of {} images...\n\n'.format(stack ,naxis3) )
            
            #split the cube
            filenames = self.filename_gen( naxis3+count-1 )
            bar.create( naxis3 )
            for j, im, fn in zip( range(naxis3), self[0].data, filenames ):
                bar.progress(count-1)             #count instead of j in case of sequential numbering for multiple cubes

                self.time_worker( j )           #set the timing values in the header for frame j
                
                pyfits.writeto(fn, data=im, header=header, clobber=True)                              #MULTIPROCESSING!!!!!!!!!!!!
                
                if w2f:
                    fp.write(fn+'\n')         #OR outname???????????
                count += 1
            
            if w2f:
                fp.close()
            
            #how long did the unpacking take
            end_time = time.time()
            print('Time taken: %f' %(end_time - start_time))
            
        self._is_unpacked = True
        
        return count
    
    #====================================================================================================
    def set_name_dict(self):
        header = self[0].header
        obj = header.get( 'OBJECT', '' )
        filter = header.get( 'FILTER', 'WL' )
        
        kct = header.get('kct', 0 )
        if int(kct/10):
            kct = str(round(kct))+'s'
        else:
            kct = str(round(kct*1000))+'ms'
        
        self.name_dict = dict(  sep             = '.',
                                obj             = obj,
                                basename        = self.get_filename(0,0),
                                date            = str(self.namedate).replace('-',''),
                                filter          = filter,
                                binning         = '{}x{}'.format(*self.binning),
                                mode            = self.mode_trim,
                                kct             = kct                            )
    
    #################################################################################################################################################################################################################
    # Timing
    #################################################################################################################################################################################################################
    #====================================================================================================
    def get_kct(self):
        
        stack_header = self[0].header
        
        if self.trigger_mode=='Internal':
            t_kct = stack_header['KCT']                                                      #kinetic cycle time between start of subsequent exposures in sec.  i.e. exposure time + readout time
            t_exp = stack_header['EXPOSURE']
            #In internal triggering mode EXPOSURE stores the actual correct exposure time.
            #                       and KCT stores the Kinetic cycle time (dead time + exposure time)
        
        #GPS Triggering (External or External Start)
        elif self.trigger_mode.startswith('External'):
            
            t_d = 0.00676
            #dead (readout) time between exposures in s 
            #NOTE: (deadtime should always the same value unless the user has 
            # (foolishly) changed the vertical clock speed). #MAYBE CHECK stack_header['VSHIFT'] 
            #EDGE CASE WARNING: THE DEADTIME MAY BE LARGER IF WE'RE NOT OPERATING IN FRAME TRANSFER MODE!
            
            if self.trigger_mode.endswith( 'Start' ):       # External Start
                t_exp = stack_header['EXPOSURE']            #exposure time in sec as in header
                t_kct = t_d + t_exp                         #Kinetic Cycle Time
            else:
                t_kct = float(args.kct)                     #kct provided by user at terminal through -k
                t_exp = t_kct - t_d                         #set the 'EXPOSURE' header keyword
        
        return t_exp, t_kct
    
    #====================================================================================================           
    def time_init(self, dryrun=0):
        ''' Do timing corrections on SHOC_Cube.
        UTC : Universal Coordinate Time (array)
        LMST : Local mean sidereal time
        JD : Julian Date
        BJD : Baryocentric julian day
        
        Parameters
        ----------
        None
        
        Returns
        ------
        '''
        sutherland = EarthLocation( lat=-32.376006, lon=20.810678, height=1771 )
        
        stack_header = self[0].header
        trigger_mode = self.trigger_mode
        #datetime_str - observation date and time in image header:
        #For 'Internal' this is :       time at the end of the first exposure (file creation timestamp)
        #                       :       #NOTE: The time here is rounded to the nearest second of computer clock ==> no absolute timing accuracy
        #For 'External Start'   :       
        
        utf = Time( stack_header['DATE'], format='isot', scale='utc', precision=9, location=sutherland )                   #  datetime_str.split('T')
        date_str = stack_header['DATE'].split('T')[0]
        utdate = Time( date_str )     #this should at least be the correct date!
        t_exp, t_kct = self.get_kct()
        td_kct = TimeDelta(t_kct, format='sec')
        
        if self.trigger_mode == 'Internal':
            #Initial time set to middle of first frame exposure        #NOTE: this hardly matters for sub-second t_exp, as the time recorded in header FRAME is rounded to the nearest second
            t0 = utf - 0.5*td_kct                                       #mid time of first frame
            
        if self.trigger_mode.startswith('External'):
            if self.trigger:                                        #trigger provided by user at terminal through -g  (args.gps)
                #NOTE: GPS triggers are provided in SAST!!!!  If they are provided in UT, comment out the following lines
                warn( 'Assuming GPS triggers provided in SAST' )
                tz = -2
                tzd = TimeDelta(tz*3600, format='sec')
                
                trigsec = Angle(self.trigger, 'h').to('arcsec').value / 15
                ttrig = TimeDelta( trigsec, format='sec' )
                ttrig += tzd                                    #trigger now in UTC

                if ttrig.value < 0:
                    ttrig += TimeDelta( 1, format='jd' )        #adjust to positive value -- this needs to be done so we don't accidentally shift the date!

                t0 = utdate + ttrig
                #datetime_str = 'T'.join([str(utdate), self.trigger])     #Correct the time in the datetime_str
                t0 = Time( t0.isot, format='isot', scale='utc', 
                           precision=9, location=sutherland)
                t0 += 0.5*td_kct                                        #set t0 to mid time of first frame
                
            else:
                raise ValueError( 'No GPS triggers provided for {}!'.format(
                                    self.filename()) )
                    #datetime_str = utdate    
        
            if not dryrun:
                stack_header['KCT'] = ( t_kct, 'Kinetic Cycle Time' )                       #Set KCT in header
                stack_header['EXPOSURE'] = ( t_exp, 'Integration time' )                       #Set KCT in header
                
        if not dryrun:
            stack_header['FRAME'] = ( str(t0), 'Start of Frame Exposure' )         #Set correct (GPS triggered) start time in header
        #stack_hdu.flush(output_verify='warn', verbose=1)
        #IF TIMECORR --> NO NEED FOR GPS TIMES TO BE GIVEN EXPLICITLY
        
        print( '{} : TRIGGER is {}. t0 = {}; KCT = {} sec'.format(self.get_filename(), trigger_mode.upper(), t0, t_kct) )
        
        return t0, td_kct
    
    #====================================================================================================
    #def get_timing_array(self, t0):
        #t_kct = round(self.get_kct()[1], 9)                     #numerical kinetic cycle time in sec (rounded to nanosec)
        #td_kct = td_kct * np.arange( self.shape[0], dtype=float )
        #t = t0 + td_kct                         #Time object containing times for all framesin the cube
        #return t
    
    #====================================================================================================
    def set_times(self, t0, td_kct, iers_a=None, coords=None):
        #Sutherland lattitude, longitude
        lat, lon = -32.376006, 20.810678
        
        texp, t_kct = self.get_kct()
        t_kct = round(t_kct, 9)                     #numerical kinetic cycle time in sec (nanosec precision)
        td_kct = td_kct * np.arange( self.shape[-1], dtype=float )
        t = t0 + td_kct                                         #Time object containing time stamps for all frames in the cube
        
        delta, status = t.get_delta_ut1_utc( iers_a, return_status=True )   #set leap second offset from most recent IERS table
        if np.any(status==-2):
            warn( 'Using predicted leap-second values from IERS.' )
        
        t.delta_ut1_utc = delta
        
        #initialize array for timing data
        self.timedata = timedata = np.recarray((len(t)), 
                                               dtype=[('utdate', 'U20'),
                                                      ('uth', float), 
                                                      ('utsec', float),
                                                      ('utc', 'U30'),
                                                      ('utstr', 'U20'),
                                                      ('lmst', float),
                                                      ('jd', float),
                                                      ('gjd', float),
                                                      ('bjd', float),
                                                      ('altitude', float),
                                                      ('airmass', float)])
        #compute timestamps for various scales
        timedata.texp           = texp
        timedata.utc            = t.utc.isot
        timedata.uth            = t.utc.hours                   #UTC in decimal hours for each frame
        timedata.utsec          = timedata.uth * 3600.              #td_kct.value
        utdata = np.fromiter(map(lambda x: tuple(x.split('T')), timedata.utc),
                                                 [('utdate','U20'),('utc','U20')])
        timedata.utdate         = utdata['utdate']
        timedata.utstr          = utdata['utc']
        lmst                    = t.sidereal_time('mean', longitude=lon)  #LMST for each frame
        timedata.lmst           = lmst
        #timedata.last          = t.sidereal_time('apparent', longitude=lon)
        
        timedata.jd             = t.jd
        #timedata.ljd           = np.floor(timedata.jd)
        timedata.gjd            = t.tcg.jd                  #geocentric julian date
        
        self.has_coords = not coords is None
        if self.has_coords:
            bjd_offset = t[:1].bjd( coords, precess=1, abcorr=None ) - timedata.jd[0]
            
            timedata.bjd[:] = timedata.jd +  bjd_offset               #barycentric julian date
            timedata.altitude = altitude(coords.ra.radian,
                                        coords.dec.radian,
                                        lmst.radian,
                                        np.radians(lat))
            timedata.airmass = Young94( np.pi/2 - timedata.altitude )
                
        #print( 'Updating the starting times for datacube {} ...'.format(self.get_filename()) )
        self.time_worker( 0 )
        self.flush( output_verify='warn', verbose=1 )
    
    #====================================================================================================    
    def export_times(self, with_slices=True, count=0):  #single_file=True,
        '''write the timing data for the stack to file(s).'''
        def make_header_line(info, fmt, delimiter):
            import re
            matcher = re.compile( '%-?(\d{1,2})' )
            padwidths = [int(matcher.match(f).groups()[0]) for f in fmt]
            padwidths[0] -= 2
            colheads = [ s.ljust(p) for s,p in zip(info, padwidths) ]
            return delimiter.join( colheads )
        
        #print( 'Writing timing data to file...' )
        TKW = ['utdate', 'uth', 'utsec', 'lmst', 'altitude', 'airmass', 'jd', 'gjd', 'bjd']
        fmt = ('%-10s', '%-12.9f', '%-12.6f', '%-12.9f', '%-12.9f', '%-12.9f', '%-18.9f', '%-18.9f', '%-18.9f')
        formats = dict(zip(TKW, fmt))
        
        table = aTable( self.timedata[TKW] )
        
        if with_slices:
            #TKW     = ['filename'] + TKW
            #fmt     = ('%-35s',) + fmt
            slices = np.fromiter( map(os.path.basename, self.real_slices), 'U35' )
            formats['filename'] = '%-35s'
            table.add_column( Column(slices, 'filename'), 0 )
        
        delimiter = '\t'
        timefile = self.get_filename(1, 0, 'time')
        table.write( timefile ,
                     delimiter='\t',
                     format='ascii.commented_header', 
                     formats=formats )
        
        #if single_file:
        #Write all timing data to a single file
        #delimiter = ' '
        #timefile = self.get_filename(1, 0, 'time')
        #header = make_header_line( TKW, fmt, delimiter )
        #np.savetxt(timefile, T, fmt=fmt, header=header )
        
        #HACK! BECAUSE IRAF SUX
        link_to_short_name_because_iraf_sux(timefile, count, 'time')
            
            
        #else:
            #for i, tkw in enumerate(TKW):
                ##write each time sequence to a separate file...
                #fn = '{}.{}'.format(self.get_filename(1,0), tkw)
                #if tkw in TKW_sf:
                    #if fn.endswith('uth'): fn.replace('uth', 'utc')
                    #np.savetxt( fn, T[i], fmt='%.10f' )
    
    #====================================================================================================
    def time_worker(self, j):
        
        header = self[0].header

        header['utc-obs'] = ( self.timedata.uth[j], 'Start of frame exposure in UTC' )                       #imutil.hedit(imls[j], 'utc-obs', ut.hours(), add=1, ver=0)                       # update timestamp in header
        header['LMST'] = ( self.timedata.lmst[j], 'Local Mean Sidereal Time' )                              #imutil.hedit(imls[j], 'LMST', lmst, add=1, ver=0)                           
        header['UTDATE'] = ( self.timedata.utdate[j], 'Universal Time Date' )                           #imutil.hedit(imls[j], 'UTDATE', ut.iso.split()[0], add=1, ver=0)

        header['JD'] = ( self.timedata.jd[j], 'Julian Date (UTC)' )
        #header['LJD'] = ( self.timedata.ljd[j], 'Local Julian Date' )
        header['GJD'] = ( self.timedata.gjd[j], 'Geocentric Julian Date (TCG)' )
        
        if self.has_coords:
            header['BJD'] = ( self.timedata.bjd[j], 'Barycentric Julian Date (TDB)' )
            header['AIRMASS'] = self.timedata.airmass[j]
        
        #elif j!=0:
            #warn( 'Airmass not yet set for {}!\n'.format( self.get_filename() ) )
        
        #header['TIMECORR'] = ( True, 'Timing correction done' )        #imutil.hedit(imls[j], 'TIMECORR', True, add=1, ver=0)                                       #Adds the keyword 'TIMECORR' to the image header to indicate that timing correction has been done
        #header.add_history('Timing information corrected at %s' %str(datetime.datetime.now()), before='HEAD' )            #Adds the time of timing correction to header HISTORY
    
    
    #====================================================================================================
    #def set_airmass( self, coords=None, lat=-32.376006 ):
        #'''Airmass'''
        #if coords is None:
            #header = self[0].header
            #ra, dec = header['ra'], header['dec']
            #coords = SkyCoord( ra, dec, unit=('h', 'deg') )     #, system='icrs'
        
        #ra_r, dec_r = coords.ra.radian, coords.dec.radian
        #lmst_r = self.timedata.lmst.radian
        #lat_r = np.radians(lat)
        
        #self.timedata.altitude = altitude( coords.ra.radian,
                                           #coords.dec.radian,
                                           #self.timedata.lmst.radian,
                                           
                                          #dec_r, lmst_r, lat_r)
        #z = np.pi/2 - self.altitude
        #self.timedata.airmass = Young94(z)
    
    #====================================================================================================
    def make_slices(self, suffix, i):
        #HACK! BECAUSE IRAF SUX
        #generate file with cube name and slices in iraf slice syntax.  This is a clever way of
        #sidestepping splitting the cubes and having thousands of fits files to deal with, but it
        #remains an ungly, unncessary, and outdated way of doing things.  IMHO (BRAAK!!!)
        #NOTE: This also means that the airmass correction done by phot is done with the airmass of
        #the first frame only... One has to generate an obsparams file for mkapfile COG with the 
        #airmasses and observation times etc...
        
        source = self.get_filename(1, 0, (suffix,'fits'))
        link_to_short_name_because_iraf_sux(source, i, 'fits')
        
        unpacked = self.get_filename(1, 0, (suffix, 'slice'))
        naxis3 = self.shape[-1]
        
        self.real_slices = make_iraf_slicemap( source, naxis3, unpacked )
        
        linkname = args.dir + '/s{}{}'.format(i, '.fits')
        slicefile = args.dir + '/s{}{}'.format(i, '.slice')
        slices = make_iraf_slicemap( linkname, naxis3, slicefile )
        #link_to_short_name_because_iraf_sux(unpacked, i, 'slice')
        self.link_slices = np.array( slices )
        
        #filename = self.get_filename(0, 0, (suffix,'fits'))
        #self.real_slices = make_iraf_slicemap( filename, naxis3 )
    
    #====================================================================================================
    def make_obsparams_file(self, suffix, count):
        slices = np.fromiter( map(os.path.basename, self.real_slices), 'U23' )
        texp = np.ones( self.shape[-1] ) * self.timedata.texp
        Filt = np.empty( self.shape[-1], 'U2')
        Filt[:] = head_info.filter
        data = np.array([slices, Filt, texp, self.timedata.airmass, self.timedata.utstr], object).T
        fmt = ('%-23s', '%-2s', '%-9.6f', '%-9.6f', '%s')
        filename = self.get_filename(1, 0, (suffix, 'obspar'))
        np.savetxt( filename, data, fmt, delimiter='\t' )       
        
        #HACK! BECAUSE IRAF SUX
        slices = np.fromiter( map(os.path.basename, self.link_slices), 'U23' )
        data = np.array([slices, Filt, texp, self.timedata.airmass, self.timedata.utstr], object).T
        filename = args.dir + '/s{}.obspar'.format(count)
        np.savetxt( filename, data, fmt, delimiter='\t' )
        
        #link_to_short_name_because_iraf_sux(filename, count, 'obspar')
        
#################################################################################################################################################################################################################
class SHOC_Run( object ):
    '''
    Class to perform comparitive tests between cubes to see if they are compatable.
    '''
    #====================================================================================================
    MAX_CHECK = 25                                                         #Maximal length for input list if validity checks is to be performed
    MAX_LS = 25                                                            #Maximal number of files to list in dir_info
    
    #Naming convension defaults
    NAMES = type('Names', (), 
                {'flat' :   'f{date}{sep}{binning}[{sep}sub{sub}][{sep}filt{filter}]',
                 'bias' :   'b{date}{sep}{binning}[{sep}m{mode}][{sep}t{kct}]',
                 'sci'  :   '{basename}' })

    
    #====================================================================================================
    def __init__(self, hdus=None, filenames=None, label=None, sep_by=None):
        
        #WARNING:  filenames may contain None as well as duplicate entries.....??????
                    #not sure if duplicates are desireable wrt efficiency.....
        
        self.cubes = list(filter(None, hdus)) if hdus else []
        
        self.sep_by = sep_by
        self.label = label or 'unlabelled'
        
        if not filenames is None:
            #filter null filenames and convert potential Path to str
            self.filenames = list(map(str, filter(None, filenames)))   
            self.load(self.filenames)
        elif not hdus is None:
            self.filenames =  [hdulist.filename() for hdulist in self]
        
    #====================================================================================================
    def __len__(self):
        return len(self.cubes)
    
    #====================================================================================================
    def __repr__(self):
        #self.__class__.__name__
        if self.label is None:
            name = 'SHOC Run   '
        elif 'sci' in self.label:
            name = 'Science Run'
        elif 'bias' in self.label:
            name = 'Bias Run   '
        elif 'flat' in self.label:
            name = 'Flat Run   '
        
        return '{} : {}'.format( name, ' | '.join( self.get_filenames() ) )
    
    #====================================================================================================
    def __getitem__(self, key):
        if isinstance( key, int ):
            if key >= len( self ) :
                raise IndexError( "The index (%d) is out of range." %key )
            return self.cubes[key]
        
        if isinstance(key, slice):
            rl = self.cubes[key]
        if isinstance(key, tuple):
            assert len(key)==1
            key = key[0]                #this should be an array...
        if isinstance(key, (list, np.ndarray)):
            if isinstance(key[0], (bool, np.bool_)):
                assert len(key)==len(self)
                rl = [ self.cubes[i] for i in np.where(key)[0] ]
            elif isinstance(key[0], (int, np.int0)):      #NOTE: be careful bool isa int
                rl = [ self.cubes[i] for i in key ]
        
        return SHOC_Run( rl, label=self.label, sep_by=self.sep_by )
    
    #====================================================================================================
    def __add__(self, other):
        return self.join( other )
    
    #====================================================================================================
    #def __eq__(self, other):
        #return vars(self) == vars(other)
    
    #====================================================================================================
    def pullattr(self, attr, return_as=list):
        return return_as([getattr(item, attr) for item in self])
    
    #====================================================================================================
    def pop(self, i):
        return self.cubes.pop(i)
    
    #====================================================================================================
    def check_rollover_state(self):
        '''Check whether the filenames contain ',_X' an indicator for whether the datacube reached the
        2GB windows file size limit on the shoc server, and was consequently split into a sequence of fits cubes.
        '''
        return np.any(['_X' in _ for _ in self.get_filenames()])
    
    #====================================================================================================
    def join(self, *runs):
        
        runs = list(filter(None, runs))       #Filter empty runs (None)
        labels = [r.label for r in runs]
        hdus = sum([r.cubes for r in runs], self.cubes )
        
        if np.all( self.label == np.array(labels) ):
            label = self.label
        else:
            warn( "Labels {} don't match {}!".format(labels, self.label))
            label = None
        
        return SHOC_Run( hdus, label=label )
        
    #====================================================================================================
    def load(self, filenames, mode='update', memmap=True, save_backup=False, **kwargs):
        '''
        Load data from file. populate data for instrumental setup from fits header.
        '''
        self.filenames = filenames
        
        label = kwargs.pop('label', kwargs.get('label', self.label))
        
        print( '\nLoading data for {} run...'.format(label) )
        
        cubes = []
        for i, fileobj in enumerate(filenames):
            hdu = SHOC_Cube.load(fileobj, mode=mode, memmap=memmap, save_backup=save_backup, **kwargs)           #YOU CAN BYPASS THIS INTERMEDIATE STORAGE IF YOU MAKE THE PRINT OPTION A KEYWORD ARGUMENT FOR THE SHOC_Cube __init__ me
            self.cubes.append(hdu)
    
    #====================================================================================================
    def print_instrumental_setup(self):
        '''Print the instrumental setup for this run as a table.'''
        names, dattrs, vals = zip(*(stack.get_instrumental_setup()
                                    for stack in self))
        
        bgcolours       = {'flat'       : 'cyan', 
                           'bias'       : 'magenta', 
                           'science'    : 'green' }
        label = self.label or 'unlabelled'
        table = sTable( vals, 
                        title = 'Instrumental Setup: {} frames'.format(label.title()),
                        title_props = {'text':'bold',
                                       'bg': bgcolours.get(self.label, 'default')},
                        col_headers = dattrs[0], 
                        row_headers = ['filename'] + list(names), 
                        enumera=True)
        
        print( table )
        #print( 'Instrumental setup ({} cubes):'.format(len(self)) )     #SPANNING DATES?
        #for i, stack in enumerate(self):
            #if i < self.MAX_LS:
                #print( stack )
            #else:
                #print( '.\n'*3 )
                #break
    
    #====================================================================================================
    def reload(self, filenames=None, mode='update', memmap=True, save_backup=False, **kwargs):
        if len(self):
            self.cubes = []
        self.load(filenames, mode, memmap, save_backup, **kwargs)
    
    #====================================================================================================
    def set_times(self, coords=None):
        
        times = [ stack.time_init() for stack in self ]
        t0s, tds = zip(*times)
        t_test = Time( t0s )
        
        #check whether IERS tables are up to date given these cube starting times
        status = t_test.check_iers_table()
        #Cached IERS table will be used if only some of the times are outside the range of the current
        #table.  For the remaining values the predicted leap second values will be used.  If all times
        #are beyond the current table a new table will be grabbed online.
        #cache = np.any(status) and not np.all(status)
        try:
            iers_a = t_test.get_updated_iers_table( cache=True )        #update the IERS table and set the leap-second offset
        except Exception as err:
            warn( 'Unable to update IERS table.' )
            print( err )
            iers_a = None
            
        msg = '\n\nCalculating timing arrays for datacube(s):'
        lm = len(msg)
        print( msg, )
        for i, stack in enumerate(self):
            print( ' '*lm + stack.get_filename()  )
            t0, td = times[i]
            stack.set_times(t0, td, iers_a, coords)

    #====================================================================================================
    def export_times(self, with_slices=True):
        for i, stack in enumerate(self):
            stack.export_times(with_slices, i)
    
    #====================================================================================================
    def export_headers(self):
        '''save fits headers as a text file'''
        for stack in self:
            headfile = stack.get_filename( with_path=1, with_ext=0, suffix='.head' )
            print('\nWriting header to file: {}'.format(os.path.basename(headfile)) )
            #TODO: remove existing!!!!!!!!!!
            stack[0].header.totextfile( headfile, clobber=True )
        
    #====================================================================================================
    def make_slices(self, suffix):
        for i, cube in enumerate(self):
            cube.make_slices(suffix, i)
    
    #====================================================================================================
    def make_obsparams_file(self, suffix):
        for i,cube in enumerate(self):
            cube.make_obsparams_file(suffix, i)
            
    #====================================================================================================
    def set_gps_triggers(self, triggers):
        #if len(triggers)!=len(self):
        if self.check_rollover_state():           #single trigger OK, can infer the remaining ones
            triggers = self.get_rolled_triggers( triggers )
            print( ("\nA single GPS trigger was provided. Run contains auto-split cubes (filesystem rollover due to 2Gb threshold). " +\
                        "Start time for rolled over cubes will be inferred from the length of the preceding cube(s).\n") )
            
        #at this point we expect one trigger time per cube
        if len(self) != len(triggers):
            raise ValueError( ('Only {} GPS trigger given. Please provide {} for {}'
                                ).format( len(triggers), len(self), self ) )
        
        for j, stack in enumerate(self):
            stack.trigger = triggers[j]
            
    #====================================================================================================
    def get_rolled_triggers(self, first_trigger):
        '''
        If the cube rolled over while the triggering mode was 'External' or
        'External Start', determine the start times (inferred triggers) of the 
        rolled over cube(s).
        '''
        slints = [cube.shape[-1] for cube in self]              #stack lengths
        #sorts the file sequence in the correct order
        matcher = re.compile('_X([0-9]+)')                      #re pattern to find the roll-over number (auto_split counter value in filename)
        fns, slints, idx = sorter( self.get_filenames(), slints, range(len(self)),
                                   key=matcher.findall )
        
        print( 'WORK NEEDED HERE!' )
        embed()
        #WARNING: This assumes that the run only contains cubes from the run that rolled-over.  
        #         This should be ok for present purposes but might not always be the case
        idx0 = idx[0]
        self[idx0].trigger = first_trigger
        t0, td_kct = self[idx0].time_init( dryrun=1 )      #dryrun ==> don't update the headers just yet (otherwise it will be done twice!)
    
        d = np.roll( np.cumsum(slints), 1 )
        d[0] = 0
        t0s = t0 + d*td_kct
        triggers = [t0.isot.split('T')[1] for t0 in t0s]
        
        #resort the triggers to the order of the original file sequence
        #_, triggers = sorter( idx, triggers )

        return triggers
    
    #====================================================================================================
    def needs_kct(self):
        return np.any( [stack.trigger_mode=='External' for stack in args.cubes] )
    
    #====================================================================================================
    def that_need_triggers(self):
        #embed()
        return self[[stack.trigger_mode.startswith('External') for stack in self]]
    
    #====================================================================================================
    def magic_filenames( self, reduction_path='', sep='.', extension='.fits' ):
        '''Generates a unique sequence of filenames based on the name_dict.'''
        
        self.set_name_dict()
        
        #re pattern matchers
        opt_pattern = '\[[^\]]+\]'                      #matches the optional keys sections (including square brackets) in the format specifier string from the args.names namespace
        opt_matcher = re.compile( opt_pattern )
        key_pattern = '(\{(\w+)\})'
        key_matcher = re.compile( key_pattern )         #matches the key (between curly brackets) and key (excluding curly brackets) for each section of the format string
        
        #get format specification string from label
        for label in ('sci', 'bias', 'flat'):
            if label in self.label:
                fmt_str = getattr(self.NAMES, label)         #gets the naming format string from the argparse namespace
                break
        
        #check which keys help in generating unique set of filenames - these won't be used
        #print( 'Check that this function is actually producing minimal unique filenames!!' )
        non_unique_keys = [ key for key in self[0].name_dict.keys() 
                        if all([self[0].name_dict[key] == stack.name_dict[key] for stack in self]) ]
        non_unique_keys.pop( non_unique_keys.index('sep') )
        
        
        filenames = [fmt_str]*len(self)
        nfns = []
        for cube, fn in zip(self, filenames):
            nd = copy(cube.name_dict)
            
            badoptkeys = [ key for _, key in key_matcher.findall( fmt_str ) if not (key in nd and nd[key]) ]
            #This checks whether the given key in the name format specifier should be used 
            #(i.e. if it has a corresponding entry in the SHOC_Cube instance's name_dict.
            #If one of the keys are unset in the name_dict, this optional key will be eliminated 
            #when generating the filename below.
            
            for opt_sec in opt_matcher.findall( fmt_str ):
                if (any(key in opt_sec for key in badoptkeys)
                    or any(key in opt_sec for key in non_unique_keys)):
                    fn = fn.replace( opt_sec, '' )
                    #replace the optional sections which contain keywords that are not in the corresponding name_dict
                    #and replace the optional sections which contain keywords that don't contribute to the uniqueness of the filename set
            nfns.append( fn.format( **nd ) )
        
        filenames = [fn.replace('[','').replace(']','') for fn in nfns]  #eliminate square brackets
        #last resort append numbers to the non-unique filenames
        if len(set(filenames)) < len(set(self.filenames)):
            unique_fns, idx = np.unique(filenames, return_inverse=1)
            nfns = []
            for basename in unique_fns:
                count = filenames.count(basename)       #number of files with this name
                if count>1:
                    padwidth = len(str(count))
                    g = FilenameGenerator( basename, padwidth=padwidth, sep='_', extension='' )
                    fns = list( g(count) )
                else:
                    fns = [basename]
                nfns += fns
            
            _, filenames = sorter( idx, nfns )               #sorts by index. i.e. restores the order of the original filename sequence
        
        #create a FilenameGenerator for each stack
        for stack, fn in zip(self, filenames):
            padwidth = len( str(stack.shape[-1]) )
            
            stack.filename_gen = FilenameGenerator( fn, reduction_path, padwidth, sep, extension )
        
        return filenames
    
    #====================================================================================================
    def genie(self, i=None):
        ''' returns list of generated filename tuples for cubes up to file number i'''
        return list(itt.zip_longest(*[cube.filename_gen(i) for cube in self]))
    
    #====================================================================================================
    def get_filenames(self, with_path=0, with_ext=1):
        '''filenames of run constituents.'''
        return [stack.get_filename(with_path, with_ext) for stack in self]
    
    #====================================================================================================
    def export_filenames(self, fn):
        
        if not fn.endswith('.txt'):                     #default append '.txt' to filename for positive identification by dir_info method
            fn += '.txt'
        
        print( '\nWriting names of {} run to file {}...\n'.format(self.label, fn) )
        with open(fn, 'w') as fp:
            for f in self.filenames:
                fp.write( f+'\n' )
                
    #====================================================================================================
    def writeout(self, suffix, dryrun=0): #TODO:  INCORPORATE FILENAME GENERATOR
        fns = []
        for stack in self:
            fn_out = stack.get_filename(1,0, (suffix, 'fits') )      #FILENAME GENERATOR????
            fns.append(fn_out)
            
            if not dryrun:
                print('\nWriting to file: {}'.format(os.path.basename(fn_out)) )
                stack.writeto(fn_out, output_verify='warn', clobber=True)
                
                #save the header as a text file
                headfile = stack.get_filename(1,0), (suffix, 'head' )
                print('\nWriting header to file: {}'.format(os.path.basename(headfile)) )
                stack[0].header.totextfile( headfile )
        
        return fns
    #====================================================================================================
    def zipper(self, keys):
        
        if isinstance(keys, str):
            return keys, [getattr(s, keys) for s in self]
        elif len(keys)==1:
            return keys[0], [getattr(s, keys[0]) for s in self]
        else:
            return tuple(keys), list(zip( *([getattr(s, key) for s in self] for key in keys) ))
    
    #====================================================================================================
    def attr_sep(self, *keys):
        ''' 
        Seperate a run according to the attribute given in keys. 
        keys can be a tuple of attributes (str), in which case it will seperate into runs with a unique combination 
        of these attributes.
        
        Returns
        -------
        atdict : dictionary containing attrs, run pairs where attrs are the attributes of run given by keys
        flag :  1 if attrs different for any cube in the run, 0 all have the same attrs
        '''

        keys, attrs = self.zipper( keys )
            
        if self.sep_by == keys:                                 #is already separated by this key
            SR = StructuredRun( zip([attrs[0]], [self]) )
            SR.sep_by = keys
            SR.label = self.label
            return SR, 0
        
        atset = set(attrs)                                       #unique set of key attribute values
        atdict = {}

        if len(atset)==1:                                        #all input files have the same attribute (key) value(s)
            flag = 0
            atdict[ attrs[0] ] = self
            self.sep_by = keys
        else:                                                     #binning is not the same for all the input cubes
            flag = 1
            for ats in atset:                                    #map unique attribute values to slices of run with those attributes
                l = np.array([attr==ats for attr in attrs])      #list for-loop needed for tuple attrs
                eq_run = self[l]                                   #SHOC_Run object of images with equal key attribute
                eq_run.sep_by = keys
                atdict[ ats ] = eq_run                          #put into dictionary
                
        SR = StructuredRun( atdict )
        SR.sep_by = keys
        SR.label = self.label
        return SR, flag
     
        
    #====================================================================================================
    def combine( self, func ):
        ''' Median combines a list of frames (with equal binning) using pyfits
        
        Returns
        ------
        outname : user defined output name for combined frame (for this binning)
        master_flag : binary flag indicating whether user has input a master flat.
        '''
        def single_combine(ims):        #TODO MERGE WITH SHOC_Cube.combine????
            '''Combine a run consisting of single images.'''
            header = copy( ims[0][0].header )
            data = func([im[0].data for im in ims], 0)
                    
            header.remove('NUMKIN')
            header['NCOMBINE'] = ( len(ims), 'Number of images combined' )
            for i, im in enumerate(ims):
                imnr = '{1:0>{0}}'.format(3, i+1)         #Image number eg.: 001
                comment = 'Contributors to combined output image' if i==0 else ''
                header['ICMB'+imnr] = ( im.get_filename(), comment )
            
            #outname = next( ims[0].filename_gen() )  #uses the FilenameGenerator of the first image in the SHOC_Run
            
            #pyfits.writeto(outname, data, header=header, output_verify='warn', clobber=True)
            return pyfits.PrimaryHDU( data, header )#outname
            
        #sep_by = self.sep_by            #how the run is separated
        
        #check for single frame inputs (user specified master frames)
        dim_dict, _ = self.attr_sep('ndims')           #separate the images by number of dimensions
        ims = dim_dict[2]       if 2 in dim_dict        else []
        stacks = dim_dict[3]    if 3 in dim_dict        else []
        
        master_flag = 0
        combined = []
        if len(ims)==1:
            master_flag = 1
            outname = ims[0].filename()
            msg = ( '\n\nYou have input a single image named: {}. '
                    'This image will be used as the master {} frame for {} binning.'
                    ).format(outname, self.label, self[0].binning)
            print( msg )
            combined.append( ims[0] )
        
        elif len(ims):
            if args.combine=='daily':           #THIS SHOULD BE FED AS AN ARGUMENT (ALONG WITH SIGMA)
                input( 'ERROR!! Ambiguity in combine.  The run should be date separated!' )
            else:
                combined.append( single_combine( ims ) )
            
        for stack in stacks:
            combined.append( stack.combine(func) )
        
        if len(combined)>1:
            msg = '''Ambiguity in combine! Cannot discriminate between given files based on header info alone.
                Please select the most appropriate option among the following: 
                '''
            warn( textwrap.dedent(msg) )
            self.print_instrumental_setup()
            i = Input.str( 'Ix?', 0, check=lambda x: x<len(combined), convert=int )
        else:
            i = 0
        
        return combined[i], master_flag
    
    #unpack datacube(s) and assign 'outname' to output images
    #if more than one stack is given 'outname' is appended with 'n_' where n is the number of the stack (in sequence) 
    #====================================================================================================
    def unpack(self, sequential=0, dryrun=0, w2f=1):
        if not dryrun:
            outpath = self[0].filename_gen.path
            print( "The following image stack(s) will be unpacked into {}:\n{}".format( outpath, '\n'.join(self.get_filenames()) ) )
        
        count = 1
        naxis3 = [stack.shape[-1] for stack in self]
        tot_num = sum(naxis3)
        padw = len(str(tot_num)) if sequential else None                          #if padw is None, unpack_stack will determine the appropriate pad width for the cube
        
        if dryrun:
            if len(args.timing)!=tot_num:      #WARNING - NO LONGER VALID                                    #check whether the numer of images in the timing stack are equal to the total number of frames in the cubes.
                raise ValueError( 'Number of images in timing list ({}) not equal to total number in given stacks ({}).'.format( len(args.timing), tot_num ) )
        
        for i,stack in enumerate(self):
            count = stack.unpack( count, padw, dryrun, w2f )
            
            if not sequential:
                count = 1
        
        if not dryrun:
            print('\n'*3 + 'A total of %i images where unpacked' %tot_num + '\n'*3) 
            if w2f:
                catfiles([stack.unpacked for stack in self], 'all.split'  )          #RENAME??????????????????????????????????????????????????????????????????????????????????????????????
    
    #====================================================================================================
    def check(self, run2, keys, raise_error=0, match=0):
        '''check fits headers in this run agains run2 for consistency of key (binning / instrument mode / dimensions / flip state /) 
        Parameters
        ----------
        keys :          The attribute(s) to be checked
        run2 :          SHOC_Run Object to check against
        raise_error :   Should an error be raised upon mismatch
        match   :       If set, method returns boolean array that can be used to
                        filter unnecessary cubes. If unset method returns single boolean - False for complete match, True for any mismatch
        
        Returns
        ------
        flag :  key mismatch?
        '''
        
        keys, attr1 = self.zipper( keys )       ;        fn1 = np.array( self.get_filenames() )                              #THIS IS NO GOOD.  BETTER TO CREATE SOME SORT OF WEAK REFERENCE TO THE OBJECT????????????
        keys, attr2 = run2.zipper( keys )       ;        fn2 = np.array( run2.get_filenames() )                           #lists of attribute values (key) for given input lists
        
        match1 = np.array([attr in attr2 for attr in attr1])    #which of 1 occur in 2
        
        if set(attr1).issuperset(set(attr2)):   #All good, run2 contains all the cubes with matching attributes
            if match:
                return match1                    #use this to select out the minimum set of cubes needed (filter out unneeded cubes)
            else:
                return False                    #NOTE: this returns the opposite of match. i.e. False --> the checked attributes all match (YOU SHOULD PROBABLY FIX THIS)
        else:            #some attribute mismatch occurs
            match2 = np.array([attr in attr1 for attr in attr2])      #which of 2 occur in 1 (at this point we know that one of these values are False)
            
            if any(~match2):
                #FIXME:   ERRONEOUS ERROR MESSAGES!
                fns = ',\n\t'.join(fn1[~match1])
                badfns = ',\n\t'.join(fn2[~match2])
                mmvals = ' or '.join(set(np.fromiter(map(str, attr2), 'U64')[~match2]))                       #set of string values for mismatched attributes
                keycomb = ('{} combination' if isinstance(keys, tuple) else '{}').format(keys)
                operation = 'de-biasing' if 'bias' in self.label else 'flat fielding'
                desc = ('Incompatible %s in'                % keycomb,
                        '\t%s'                              % badfns,
                        'No %s frames with %s %s for %s'    %(self.label, mmvals, keycomb, operation),
                        '\t%s'                              % fns,
                        '\n')
                msg = '\n'.join(desc)
                
                if raise_error==1:
                    raise ValueError('\n\nERROR! %s' %msg)                       #obviates the for loop
                elif raise_error==0:
                    warn(msg) 
            
            if match:
                return match1       #np.array([attr in attr2 for attr in attr1])
            else:
                return True

    #====================================================================================================
    def flag_sub(self, science_run, raise_error=0):
        return
        #print( 'flag' )
        #embed()
        #dim_mismatch = self.check(science_run, 'dimension', 0)
        #if dim_mismatch:
            #keys = 'binning', 'dimension'
            #sprops = set( science_run.zipper(keys)[1] )
            #cprops = set( self.zipper(keys)[1] )
            #missing_props = sprops - cprops
            #for binning, dim in zip(*missing_props):
                
        
        dim_mismatch = self.check(science_run, 'dimension', 0)
        if dim_mismatch:
            is_subframed = np.array( [s._is_subframed for s in science_run] )
            if np.any( is_subframed ):
                #flag the cubes that need to be subframed
                sf_bins = [s.binning for s in science_run if s._is_subframed]           #those binnings that have subframed cubes
                for cube in self:
                    if cube.binning in sf_bins:
                        cube._needs_sub = 1
        
    #====================================================================================================
    #def set_airmass(self, coords=None):
        #for stack in self:
            #stack.set_airmass(coords)
            
    #====================================================================================================
    def set_name_dict(self):
        for stack in self:
            stack.set_name_dict()

    #====================================================================================================
    def close(self):
        [stack.close() for stack in self]
    

#################################################################################################################################################################################################################
#def frame_input(simsnls, what): 
    #'''Check whether the given frames are compatible with the given science images.'''
    #warn = 1
    #msg =       'Please enter the filename(s) of the raw' + what + ' field(s) / datacube(s).   Or enter the *.txt file containing the list.  ' + \
                #'All these will be combined into a single master '+ what + '.\n' + \
                #'If you already have a normalised master ' + what + ', you may simply enter its name.\n'
    #while warn:
        #inls = Input.list(msg, imaccess, None, 1)         #If empty ----------> No flat fielding / bias subtraction will be done
        #if not len(inls):
            #print('You have elected not to do ' + what + ' processing.  Note that this will reduce the quality of the photometry!')
            #warn = 0
        #else:
            #for fn in inls:
                #print_instrumental_setup(fn)
        
            #warn = check(['binning', 'dimension', 'mode', 'flip_state'], inls, simsnls, what, 'science')
    
    #return inls        

    
#################################################################################################################################################################################################################
class StructuredRun(coll.MutableMapping):
    """ 
    Emulates dict to hold multiple SHOC_Run instances indexed by their attributes. 
    The attribute names given in sep_by are the ones by which the run is separated 
    into unique segments (also runs).
    """
    #====================================================================================================
    def __init__(self, *args, **kwargs):
        self.data = dict()
        self.update( dict(*args, **kwargs) )  # use the free update to set keys
    
    #====================================================================================================    
    def __getitem__(self, key):
        return self.data[key]
    
    #====================================================================================================
    def __setitem__(self, key, value):
        self.data[key] = value
    
    #====================================================================================================
    def __delitem__(self, key):
        del self.data[key]
    
    #====================================================================================================
    def __iter__(self):
        return iter(self.data)
    
    #====================================================================================================
    def __len__(self):
        return len(self.data)
    
    #====================================================================================================
    def __repr__(self):
        print( 'REWORK REPR' )
        #embed()
        
        return '\n'.join( [ ' : '.join(map(str, x)) for x in self.items() ] )
    
    #====================================================================================================
    def flatten(self):
        if isinstance( list(self.values())[0], SHOC_Cube ):
            run = SHOC_Run( list(self.values()), label=self.label, sep_by=self.sep_by )
        else:
            run = SHOC_Run(label=self.label).join( *self.values() )
        
        #eliminate duplicates
        _, idx = np.unique(run.get_filenames(), return_index=1)
        dup = np.setdiff1d( range(len(run)), idx )
        for i in reversed(dup):
            run.pop( i )
        
        return run
    
    #====================================================================================================
    def writeout(self, suffix):
        return self.flatten().writeout( suffix )
    
    #====================================================================================================
    def attr_sep(self, *keys):
        if self.sep_by == keys:
            return self
        
        return self.flatten().attr_sep( *keys )
        
    #====================================================================================================
    def magic_filenames(self, reduction_path='', sep='.', extension='.fits'):
        return self.flatten().magic_filenames( reduction_path, sep, extension )
    
    #====================================================================================================
    def compute_master(self, mbias=None, load=0, w2f=1):
        '''
        Compute the master image(s) (bias / flat field)
        
        Parameters
        ----------
        mbias :    A StructuredRun instance of master biases (optional)
        load  :    If set, the master frames will be loaded as SHOC_Cubes.  If unset kept as filenames
        
        Returns
        -------
        SR:     A StructuredRun of master frames separated by the same keys as self
        '''
        
        if mbias:
            #print( 'self.sep_by, mbias.sep_by', self.sep_by, mbias.sep_by )
            assert self.sep_by == mbias.sep_by
        
        keys = self.sep_by
        masters = {}                                    #master bias filenames
        datatable = []
        for attrs, run in self.items():
            
            #print( 'attrs', attrs )
            #print( 'run', run )
            
            #if isinstance(run, SHOC_Run):
            if run is None:                                     #Unmatched!
                masters[attrs] = None
                continue
            
            master, master_flag = run.combine( args.fcombine )                #master bias/flat frame for this attr val
            
            if not master_flag:
                stack = run[0]                        #~!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                master._needs_flip = stack._needs_flip

                if master._needs_flip:                 #NEED TO FORMAT THE STRUCTURE OF THIS!!!!!!!!!!!!!
                    master.flip()
                    master._needs_flip = not master._needs_flip
                
                #FIXME: MEGA NEST!!?
                if 'flat' in self.label:
                    if mbias:
                        masterbias = mbias[attrs]
                        if masterbias:
                            print( ('Subtracting master bias {} for'
                                    '\n\t{} {} from'
                                    '\n\tmaster flat {}.').format(masterbias.get_filename(),
                                                                  attrs,
                                                                  keys,
                                                                  master.basename) )
                            master[0].data -= masterbias[0].data
                        else:
                            print( 'No master bias for\n\t{} {}.'.format(attrs, keys) )
                                        
                    ffmean = np.mean( master[0].data )                                                      #flat field mean
                    print( 'Normalising flat field...' )
                    master[0].data /= ffmean                                                              #flat field normalization
                
                master.flush(output_verify='warn', verbose=1)      #writes full frame master
                
                #writes subframed master
                #submasters = [master.subframe(sub) for sub in stack._needs_sub] 
                
                #master.close()
            
            masters[attrs] =  master
            datatable.append( (master.get_filename(0,1),) + attrs )
            
        print(   )
            
        #Table for master frames
        bgcolours       = {'flat' : 'cyan', 'bias' : 'magenta', 'sci' : 'green' }
        title           = 'Master {} frames:'.format(self.label)
        title_props     = { 'text':'bold', 'bg': bgcolours[self.label] }
        col_head        = ('Filename',) + tuple( map(str.upper, keys) )
        table = sTable( datatable, title, title_props, col_head )
        
        print( table )
        #TODO: STATISTICS????????
        
        if load:
            #this creates a run of all the master frames which will be split into individual 
            #SHOC_Cube instances upon the creation of the StructuredRun at return
            label = 'master {}'.format( self.label )
            mrun = SHOC_Run( hdus=masters.values(), label=label )
        
        if w2f:
            fn = label.replace(' ','.')
            outname = os.path.join(args.output_dir, fn)
            mrun.export_filenames( outname )
        
        #NOTE:  The dict here is keyed on the closest matching attributes in self!
        SR = StructuredRun( masters )           
        SR.sep_by = self.sep_by
        SR.label = self.label
        return SR
    
    #====================================================================================================
    def subframe(self, c_sr):
        #Subframe
        print( 'sublime subframe' )
        i=0
        substacks = []
        #embed()
        for attrs, r in self.items():
            #_, csub = c_sr[attrs].sub   #what are the existing dimensions for this binning, mode combo
            stack = c_sr[attrs]
            #_, ssub = r.zipper('sub')
            _, srect = r.zipper('subrect')
            #if i==0:    
                #i=1
            missing_sub = set(srect) - set([(stack.subrect)])
            print( stack.get_filename(0,1), r )         
            print( stack.subrect, srect )
            print( missing_sub )
            for sub in missing_sub:
                #for stack in c_sr[attrs]:
                substacks.append( stack.subframe( sub ) )
        
        #embed()
        
        print( 'substacks', [s.sub for s in substacks] )
        
        b = c_sr.flatten()
        print( 'RARARARARRAAAR!!!', b.zipper('sub') )
        
        newcals = SHOC_Run(substacks, label=c_sr.label) + b
        return newcals
    
    
    #def subframe(self, c_sr):
        ##Subframe
        #i=0
        #substacks = []
        #for attrs, r in self.items():
            #_, csub = c_sr[attrs].zipper('sub')   #what are the existing dimensions for this binning, mode combo
            #_, ssub = r.zipper('sub')
            #if i==0:
                #embed()
                #i=1
            #missing_sub = set(csub) - set(ssub)
            #for sub in missing_sub:
                #for stack in c_sr[attrs]:
                    #substacks.append( stack.subframe( sub ) )
        
        #newcals = SHOC_Run(substacks) + c_sr.flatten()
        
        #return newcals
    
#################################################################################################################################################################################################################          
class SHOCHeaderCard( pyfits.card.Card ):
    '''Extend the pyfits.card.Card class for interactive user input'''
    
    #====================================================================================================
    def __init__(self, keyword=None, value=None, comment=None, **kwargs):
        
        self.example = kwargs.pop( 'example', '' )                      #Additional display information         #USE EXAMPLE OF inp.str FUNCTION
        self.askfor = kwargs.pop( 'askfor', True )                      #should the user be asked for input
        self.check = kwargs.pop( 'check', validity.trivial )             #validity test
        self.conversion = kwargs.pop('conversion', convert.trivial )    #string conversion
        self.default = ''
        
        if self.askfor:         #prompt the user for input
            value = Input.str( comment+self.example+': ', self.default, check=self.check, verify=False, what=comment, convert=self.conversion )             #USE EXAMPLE OF _input.str FUNCTION
        elif self.check( value ):
            value = self.conversion( value )
        else:
            raise ValueError('Invalid value {} for {}. '.format(repr(value), keyword) )
        
        #print(value)
        pyfits.card.Card.__init__(self, keyword, value, comment, **kwargs)
        
        
class SHOCHeader( pyfits.Header ):
    RNT = ReadNoiseTable()
    '''Extend the pyfits.Header class for interactive user input'''
    #====================================================================================================
    def __init__(self, cards=[], txtfile=None):
        pyfits.Header.__init__(self, cards, txtfile)
        
        #self.not_asked_for = [card.keyword for card in cards]
    
    
    #====================================================================================================
    def set_defaults(self, header):
        '''
        Set the default values according to those found in the given header.  
        Useful in interteractive mode to avoid loads of typing.
        '''
        for card in self.cards:
            key = card.keyword
            if key in header.keys():
                card.default = header[key]              #If these keywords already exist in the header, make their value the default
            else:
                card.default = ''
        #YOU CAN SUPERCEDE THE DEFAULTS IF YOU LIKE.
        
        
    #====================================================================================================
    def set_ron_sens_date(self, header):    #GET A BETTER NAME!!!!!!!!!!!!
        '''set Readout noise, sensitivity, observation date in header.'''
        # Readout noise and Sensitivity as taken from ReadNoiseTable
        ron, sens, saturation = self.RNT.get_readnoise( header )
        
        self['RON']             = ron,                          'CCD Readout Noise'
        self['SENSITIV']        = sens,                         'CCD Sensitivity'                                                    # Images taken at SAAO observatory
        self['OBS-DATE']        = header['DATE'].split('T')[0], 'Observation date'
        #self['SATURATION']??
        
        return ron, sens, saturation
        
    #====================================================================================================
    def check(self):
        '''check which keys actually need to be updated'''
        pop = []                                                                                                            #list of indeces of header that will not need updating
        for card in self.cards:
            #print( card )
            #print( type(card) )
            if card.value == '':                                                                           #Empty string exception. i.e. if no input given and no default
                print("%s will not be updated" %card.comment)
                self.pop(card.keyword)
            if hasattr(card, 'default'):
                if card.value == card.default:
                    print("%s kept as default %s\n" %(card.comment, card.default))
                    self.pop(card.keyword)                                                                  #remove popped key indeces from update list
    
    #def take_from(self, header):                                        #OR TEXT FILE or args!!!!!!!!!!!!!!
        #'''take header information from hdu of another fits file.'''
        #for j in range(len(self.headkeys)):
            ##if self.headkeys[j] in header.keys():
            #self.headinfo[j] = header[self.headkeys[j]]
            ##else:
                ##'complain!'
        
    #====================================================================================================
    def update_to(self, hdu):                                                        #OPTIONAL REFERENCE FITS FILE?.......
        ''' Updating header information
            INPUT: hdu - header data unit of the fits image
            OUTPUTS: NONE
        '''
        print( 'Updating header for {}.'.format(hdu.get_filename()) )
        header = hdu[0].header
  
        for card in self.cards:
            #print( card )
            header.set(card.keyword, card.value, card.comment)                                                               #If  the  field  already  exists  it  is edited
        
        hdu.flush(output_verify='warn', verbose=1)                      #SLOW!! AND UNECESSARY!!!!
         
            
        #hdu.close()                                                             #HMMMMMMMMMMMMMMM
        #UPDATE HISTORY / COMMENT KEYWORD TO REFLECT CHANGES
        
        
    
#################################################################################################################################################################################################################  
#Bias reductions
#################################################################################################################################################################################################################
def bias_proc(m_bias_dict, sb_dict):             #THIS SHOULD BE A METHOD OF THE SHOC_RUN CLASS
    #NOTE: Bias frames here are technically dark frames taken at minimum possible exposure time.  SHOC's dark current is (theoretically) constant with time,so these may be used as bias frames.  
    #	   Also, there is therefore no need to normalised these to the science frame exposure times.
    
    '''
    Do the bias reductions on sciene data
    Parameters
    ----------
    mbias_dict : Dictionary with binning,filename key,value pairs for master biases
    sb_dict : Dictionary with binning,run key,value pairs for science data
    
    Returns
    ------
    Bias subtracted SHOC_Run
    '''
    
    fn_ls = []
    for attrs, master in m_bias_dict.items():
        if master is None:
            continue
        
        stacks = sb_dict[attrs]              #the cubes (as SHOC_Run) for this attrs value
       
        stacks_b = []
        msg = '\nDoing bias subtraction on the stack: '
        lm = len(msg)
        print( msg )
        for stack in stacks:
            print( ' '*lm, stack.get_filename() )
            
            header = stack[0].header
            header['BIASCORR'] = ( True, 'Bias corrected' )					 #Adds the keyword 'BIASCORR' to the image header to indicate that bias correction has been done
            hist = 'Bias frame {} subtracted at {}'.format( master.get_filename(), datetime.datetime.now())
            header.add_history(hist, before='HEAD' ) 	#Adds the filename and time of bias subtraction to header HISTORY
            
            stack[0].data -= master[0].data
            
    sb_dict.label = 'science frames (bias subtracted)'
    return sb_dict
    
  
#################################################################################################################################################################################################################  
#Flat field reductions
#################################################################################################################################################################################################################

def flat_proc(mflat_dict, sb_dict):             #THIS SHOULD BE A METHOD OF THE SHOC_RUN CLASS
    '''Do the flat field reductions
    Parameters
    ----------
    mflat_dict : Dictionary with binning,run key,value pairs for master flat images
    sb_dict : Dictionary with binning,run key,value pairs
    
    Returns
    ------
    Flat fielded SHOC_Run
    '''
    
    fn_ls = []
    for attrs, masterflat in mflat_dict.items():
        
        if isinstance( masterflat, SHOC_Run):
            masterflat = masterflat[0]                  #HACK!!
        
        mf_data = masterflat[0].data            #pyfits.getdata(masterflat, memmap=True)
        
        if round(np.mean(mf_data), 1) != 1:
            raise ValueError('Flat field not normalised!!!!')
        
        stacks = sb_dict[attrs]                                       #the cubes for this binning value
        
        msg = '\nDoing flat field division on the stack: '
        lm = len(msg)
        print( msg, )
        for stack in stacks:
            print( ' '*lm + stack.get_filename() )
            
            header = stack[0].header
            header['FLATCORR'] = (True, 'Flat field corrected')					#Adds the keyword 'FLATCORR' to the image header to indicate that flat field correction has been done
            hist = 'Flat field {} subtracted at {}'.format( masterflat.get_filename(), datetime.datetime.now() )
            header.add_history(hist, before='HEAD' ) 	#Adds the filename used and time of flat field correction to header HISTORY
            
            stack[0].data /= mf_data                                                                            #flat field division

    sb_dict.label = 'science frames (flat fielded)'
    return sb_dict
    
#################################################################################################################################################################################################################  
# Headers
#################################################################################################################################################################################################################  

def header_proc( run, _pr=True ):             #THIS SHOULD BE A METHOD OF THE SHOC_RUN CLASS
    '''update the headers where necesary'''
    
    section_header( 'Updating Fits headers' )
    
    update_head = SHOCHeader( )
    update_head.set_defaults( run[0][0].header )        #set the defaults for object info according to those (if available) in the header of the first cube
    
    table = [
    ('OBJECT',       'Object name/alias',   '',                                     validity.trivial,  convert.trivial),
    ('RA',           'RA',                   "(eg: '03:14:15' or '03 14 15')",       validity.RA,       convert.RA),
    ('DEC',          'Dec',                  "(eg: '+27:18:28.1' or '27 18 28.1')",  validity.DEC,      convert.DEC),
    ('EPOCH',        'Epoch of RA and Dec',  'eg: 2000',                             validity.float,    convert.trivial),
    ('FILTER',       'Filter',               '(WL for clear)',                       validity.trivial,  convert.trivial),
    ('OBSERVAT',     'Observatory',          '',                                     validity.trivial,  convert.trivial),
    #('RON',          'CCD Readout Noise',    '',                                     validity.trivial,  convert.trivial),
    #('SENSITIVITY',  'CCD Sensitivity',      '',                                     validity.trivial,  convert.trivial),
    ('TELESCOP',     'Telescope',             '',                                    validity.trivial,  convert.trivial) 
    ]

    keywords ,description, example, check, conversion = zip(*table)
    #convert.ws2ds

    if args.update_headers:
        info = []
        askfor = []
        for kw in keywords:
            match = [key for key in head_info.__dict__.keys() if kw.lower().startswith(key)]    #match the terminal (argparse) input arguments with the keywords
            #print(match)
            if len(match):
                inf = getattr( head_info, match[0] )    #terminal input value (or default) for this keyword
                if inf:
                    info.append( inf )
                    askfor.append( 0 )
                else:
                    info.append( '' )                   #if terminal input info is empty this item will be asked for explicitly
                    askfor.append( 1 )
            else:               #the keywords that don't have corresponding terminal input arguments
                info.append( '' )
                askfor.append( 0 )
        
    else:    
        info = ['']*len(table)  #['test', '23 22 22.33', '44 33 33.534', '2000', '2012 12 12', 'WL','saao', '','']    #
        askfor = [1,1,1,1,1,0,1]
    
    if sum(askfor):
        print("\nPlease enter the following information about the observations to populate the image header. If you enter nothing that item will not be updated.")  #PRINT FORMAT OPTIONS
        
    update_cards = [ SHOCHeaderCard(key, val, comment, example=ex, check=vt, conversion=conv, askfor=ask) 
                    for key, val, comment, ex, vt, conv, ask 
                    in zip(keywords, info, description, example, check, conversion, askfor) ]
    
    update_head.extend( update_cards )
    #update_head['OBSERVAT'] = 

    print( '\n\nUpdating all headers with the following information:\n' )
    print( repr(update_head) )
    print( )
    
    table = []
    for hdu in run:
        # Readout noise and Sensitivity as taken from ReadNoiseTable
        details = update_head.set_ron_sens_date( hdu[0].header ) #set RON, SENS, OBS-DATE (the RON and SENS may change through the run)
        
        update_head.check()                                #Check if updating observation info is necessary                            #NOT WORKING!!!!!!!!!!!!!
        update_head.update_to( hdu )                       #copy the cards in update_head to the hdu header and flush
        #hdu.set_name_dict()
        
        table.append( details )
    
    if _pr:             
        #TODO:  CAN BENEFIT HERE FROM STRUCTURED RUN, TO PRESENT INFO MORE CONSICELY
        table = sTable(table, 
                       title = 'Readout Noise', 
                      col_headers=('RON', 'SENSITIVITY', 'SATURATION'), 
                      row_headers=run.get_filenames())
        print( table )
    
    return update_head #this will have the RON and SENSITIVITY of the last updated
    
#################################################################################################################################################################################################################  
# Science image pre-reductions
#################################################################################################################################################################################################################

def match_closest(sci_run, calib_run, exact, closest=None, threshold_warn=7, _pr=1):
    '''
    Match the attributes between sci_run and calib_run.
    Matches exactly to the attributes given in exact, and as closely as possible to the  
    attributes in closest. Separates sci_run by the attributes in both exact and 
    closest, and builds an index dictionary for the calib_run which can later be used to
    generate a StructuredRun instance.
    Parameters
    ----------
    sci_run :           The SHOC_Run to which comparison will be done
    calib_run :         SHOC_Run which will be trimmed by matching
    exact :       tuple or str. keywords to match exactly         NOTE: No checks run to ensure calib_run forms a subset of sci_run w.r.t. these attributes
    closest:      tuple or str. keywords to match as closely as possible

    Returns
    ------
    s_sr :              StructuredRun of science frames separated 
    out_sr

    '''
    #====================================================================================================
    def str2tup(keys):
        if isinstance(keys, str):
            keys = keys,          #a tuple
        return keys
    #====================================================================================================
    
    msg = ('\nMatching {} frames to {} frames by:\tExact {};\t Closest {}\n'
            ).format( calib_run.label.upper(), sci_run.label.upper(), exact, repr(closest) )
    print( msg )
    
    #create the StructuredRun for science frame and calibration frames
    exact, closest = str2tup(exact), str2tup(closest)
    sep_by = tuple( key for key in flatten([exact, closest]) if not key is None )   
    s_sr, sflag = sci_run.attr_sep( *sep_by )
    c_sr, bflag = calib_run.attr_sep( *sep_by )
    
    #Do the matching - map the science frame attributes to the calibration StructuredRun element with closest match
    lme = len(exact)                      #NOTE AT THE MOMENT THIS ONLY USES THE FIRST KEYWORD IN closest TO DETERMINE THE CLOSEST MATCH 
    _, sciatts = sci_run.zipper( sep_by )                       #sep_by key attributes of the sci_run
    _, calibatts = calib_run.zipper( sep_by )
    ssciatts = np.array( list(set(sciatts)), object )           #a set of the science frame attributes
    calibatts = np.array( list(set(calibatts)), object )
    sss = ssciatts.shape
    where_thresh = np.zeros((2*sss[0], sss[1]+1))       #state array to indicate where in data threshold is exceeded (used to colourise the table)
    
    runmap, attmap = {}, {}
    datatable = []
    
    for i, attrs in enumerate(ssciatts):
        lx = np.all( calibatts[:,:lme]==attrs[:lme], axis=1 )       #those calib cubes with same attrs (that need exact matching)
        delta = abs( calibatts[:,lme] - attrs[lme] )
        
        if ~lx.any():   #NO exact matches
            threshold_warn = False                                  #Don't try issue warnings below
            cattrs = (None,)*len(sep_by)
            crun = None
        else:
            lc = delta==delta[lx].min()
            l = lx & lc
            cattrs = tuple( calibatts[l][0] )
            crun = c_sr[cattrs]
        
        tattrs = tuple(attrs)                                           #array to tuple
        attmap[tattrs] = cattrs
        runmap[tattrs] = crun

        datatable.append( (str(s_sr[tattrs]), ) + tattrs )
        datatable.append( (str(crun),)          + cattrs )
        
        #Threshold warnings
        #FIXME:  MAKE WARNINGS MORE READABLE
        if threshold_warn:
            deltatype = type(delta[0])
            threshold = deltatype(threshold_warn)  #type cast the attribute for comparison (datetime.timedelta for date attribute, etc..)
            if np.any( delta[l] > deltatype(0) ):
                where_thresh[2*i:2*(i+1), lme+1] += 1
                
            if np.any( delta[l] > threshold ):
                fns = ' and '.join( c_sr[cattrs].get_filenames() )
                sci_fns = ' and '.join( s_sr[tattrs].get_filenames() )
                msg = ( 'Closest match of {} {} in {}\n'
                        '\tto {} in {}\n'
                        '\texcedees given threshold of {}!!\n\n'        
                        ).format( tattrs[lme], closest[0].upper(), fns, cattrs[lme], sci_fns, threshold_warn )
                warn( msg )
                where_thresh[2*i:2*(i+1), lme+1] += 1
        
    out_sr = StructuredRun( runmap )
    out_sr.label = calib_run.label
    out_sr.sep_by = s_sr.sep_by
    
    if _pr:
        #Generate data table of matches
        col_head = ('Filename(s)',) + tuple( map(str.upper, sep_by) )
        where_row_borders = range(0, len(datatable)+1, 2)
        
        table = sTable( datatable,
                       title='Matches',
                       title_props=dict(text='bold', bg='light blue'),
                       col_headers=col_head,
                       where_row_borders=where_row_borders )
        
        #colourise           #TODO: highlight rows instead of colourise??
        unmatched = [None in row for row in datatable]
        unmatched = np.tile( unmatched, (len(sep_by)+1,1) ).T
        states = where_thresh
        states[unmatched] = 3
        table.colourise( states, 'default', 'yellow', 202, {'bg':'red'} )
        
        print( '\nThe following matches have been made:' )
        print( table )
        
    return s_sr, out_sr

#====================================================================================================
def sciproc(run):                               #WELL THIS CAN NOW BE A METHOD OF SHOC_Run CLASS
    
    section_header( 'Science frame processing' )
    
    run.print_instrumental_setup()
    
    if args.w2f:
        outname = os.path.join( args.output_dir, 'cubes.txt' )
        run.export_filenames( outname )
    
    if args.update_headers:
        updated_head = header_proc(run)
    
    #====================================================================================================
    # Timing
    if args.timing or args.split:
        section_header( 'Timing' )
        
    if args.gps:
        args.cubes.that_need_triggers().set_gps_triggers( args.gps )
    
    if args.timing or args.split:
        run.set_times( head_info.coords )
        #run.set_airmass( head_info.coords )
    
    #====================================================================================================
    # Debiasing
    suffix = ''
    if args.bias:
        
        section_header( 'Debiasing' )
        
        suffix += 'b'           #the string to append to the filename base string to indicate bias subtraction
        s_sr, b_sr = match_closest( run, args.bias, 
                                    ('binning','mode'), 'kct',
                                    threshold_warn=None)
        
        b_sr.magic_filenames( args.output_dir )
        
        masterbias = b_sr.compute_master( load=1, w2f=args.w2f )      #StructuredRun for master biases separated by 'binning','mode', 'kct' 
        
        #s_sr.subframe( b_sr )

        b_sr = bias_proc(masterbias, s_sr)
        
        #s_sr.writeout( suffix )
        
        #run.reload( bls )
        #s_sr, _ = run.attr_sep( 'binning' )     #'filter'
    else:
        s_sr, _ = run.attr_sep( 'binning' )     #'filter'
        masterbias = masterbias4flats = None
    
    #====================================================================================================
    # Flat fielding
    if args.bias or args.flats:         section_header( 'Flat field processing' )
    
    if args.flats:
        suffix += 'ff'          #the string to append to the filename base string to indicate flat fielding
        
        matchdate = 'date'      if args.combine=='daily'     else None
        threshold_warn = 1      if matchdate                 else None
        
        s_sr, f_sr = match_closest(run, args.flats, 'binning', matchdate, threshold_warn)
        f_sr.magic_filenames( args.output_dir )
        
        #embed()
        
        if args.bias:
            if args.bias4flats:
                f_sr, b4f_sr = match_closest(f_sr.flatten(), args.bias4flats, ('binning','mode'), 'kct')
                b4f_sr.magic_filenames( args.output_dir, extension='.b4f.fits' )
                
                #TODO: CROSS CHECK THE bias4flats SO WE DON'T RECOMPUTE unncessary...
                
                masterbias4flats = b4f_sr.compute_master( load=1 )                        #StructuredRun for master biases separated by 'binning','mode', 'kct' 
                #else:
                    #masterbias4flats = masterbias           #masterbias contains frames with right mode to to flat field debiasing
            else:
                masterbias4flats = None
        
        masterflats = f_sr.compute_master( masterbias4flats, load=1 )
        #q = s_sr.subframe( masterflats )
        #embed()
        #raise Exception
        
        masterflats, _ = masterflats.attr_sep( 'binning' )
        s_sr, _ = s_sr.attr_sep( 'binning' )
        
        s_sr = flat_proc( masterflats, s_sr)                       #returns StructuredRun of flat fielded science frames
    
    if args.bias or args.flats:
        
        nfns = s_sr.writeout( suffix )
        
        #Table for Calibrated Science Run
        title = 'Calibrated Science Run'
        title_props = { 'text':'bold', 'bg': 'green' }
        keys, attrs = run.zipper( ('binning', 'mode', 'kct') )
        col_head = ('Filename',) + tuple( map(str.upper, keys) )
        datatable = [(os.path.split(fn)[1],) + attr for fn, attr in zip(nfns, attrs)]
        table = sTable( datatable, title, title_props, col_head )
        
        print( table )
        
        if args.w2f:
            outname = os.path.join(args.output_dir, 'cubes.bff.txt')
            run.export_filenames( outname )
            
    #animate(sbff[0])
    
    #raise ValueError
    #====================================================================================================
    #splitting the cubes
    if args.split:
        #User input for science frame output designation string
        if args.interactive:    
            print('\n'*2)
            names.science = Input.str('You have entered %i science cubes with %i different binnings (listed above).\nPlease enter a naming convension:\n' %(len(args.cubes), len(sbff)),
                                '{basename}', example='s{sep}{filter}{binning}{sep}',  check=validity.trivial)
            
            nm_option = Input.str('Please enter a naming option:\n1] Sequential \n2] Number suffix\n', '2', check=lambda x: x in [1,2] )
            sequential = 1  if nm_option==1 else 0
        else:
            sequential = 0
        
        run.magic_filenames( args.output_dir )
        run.unpack( sequential, w2f=args.w2f )								#THIS FUNCTION NEEDS TO BE BROADENED IF YOU THIS PIPELINE AIMS TO REDUCE MULTIPLE SOURCES....
    else:
        #One can do photometry without splitting the cubes!!
        #TODO: check w2f???
        run.make_slices( suffix )
        run.export_times( with_slices=True )
        run.make_obsparams_file( suffix )
        run.export_headers()
        
    return run
  
#################################################################################################################################################################################################################
#Misc Function definitions
#################################################################################################################################################################################################################

#################################################################################################################################################################################################################    
def imaccess( filename ):
    return True #i.e. No validity test performed!
    #try:
        #pyfits.open( filename )
        #return True
    #except BaseException as err:
        #print( 'Cannot access the file {}...'.format(repr(filename)) )
        #print( err )
        #return False

###########################################################################q######################################################################################################################################      
#TODO: externalise
def get_coords( obj_name ):
    ''' Attempts a SIMBAD Sesame query with the given object name. '''
    from astropy.coordinates.name_resolve import get_icrs_coordinates
    try: 
        print( '\nQuerying SIMBAD database for {}...'.format(repr(obj_name)) )
        coo = get_icrs_coordinates( obj_name )
        ra = coo.ra.to_string( unit='h', precision=2, sep=' ', pad=1 )
        dec = coo.dec.to_string( precision=2, sep=' ', alwayssign=1, pad=1 )
        
        print( 'The following ICRS J2000.0 coordinates were retrieved:\nRA = {}, DEC = {}\n'.format(ra, dec) )
        return coo, ra, dec
        
    except Exception as err:     #astcoo.name_resolve.NameResolveError
        #ipshell()
        try:
            from obstools.RKCat import Jparser
            coo = Jparser( [obj_name] ).to_SkyCoords()[0]
            ra = coo.ra.to_string( unit='h', precision=2, sep=' ', pad=1 )
            dec = coo.dec.to_string( precision=2, sep=' ', alwayssign=1, pad=1 )
            
            print( 'The following ICRS J2000.0 coordinates were retrieved:\nRA = {}, DEC = {}\n'.format(ra, dec) )
            return coo, ra, dec
            
        except Exception as err:
            print( err )
    
        print( 'ERROR in retrieving coordinates...' )
        #print( err )
        return None, None, None

#################################################################################################################################################################################################################      

def catfiles(infiles, outfile):
    '''used to concatenate large files.  Works where bash fails due to too large argument lists'''
    with open(outfile, 'w') as outfp:
        for fname in infiles:
            with open(fname) as infile:
                for line in infile:
                    outfp.write(line)   

    
#################################################################################################################################################################################################################

def section_header( msg, swoosh='=', _print=True ):
    width = getTerminalSize()[0]
    swoosh = swoosh * width
    msg = SuperString( msg ).center( width )
    info = '\n'.join( ['\n', swoosh, msg, swoosh, '\n' ] )
    if _print:
        print( info )
    return info

################################################################################################################
#MAIN
#TODO: PRINT SOME INTRODUCTORY STUFFS
################################################################################################################

#Parse sys.argv arguments from terminal input
def setup():
    global args, head_info, names
    
    # exit clause for script parser.exit(status=0, message='')
    from sys import argv
    import argparse
    
    #Main parser
    main_parser = argparse.ArgumentParser(description='Data reduction pipeline for SHOC.')
    
    #group = parser.add_mutually_exclusive_group()
    #main_parser.add_argument('-v', '--verbose', action='store_true')
    #main_parser.add_argument('-s', '--silent', action='store_true')
    main_parser.add_argument('-i', '--interactive', action='store_true', default=False, dest='interactive', 
                help='Run the script in interactive mode.  You will be prompted for input when necessary')

    main_parser.add_argument('-d', '--dir', default=os.getcwd(),
                help = 'The data directory. Defaults to current working directory.')
    main_parser.add_argument('-o', '--output-dir',
                help = ('The data directory where the reduced data is to be placed.' 
                        'Defaults to input directory'))
    main_parser.add_argument('-w', '--write-to-file', nargs='?', const=True, default=True, dest='w2f', 
                help = ('Controls whether the script creates txt list of the files created.' 
                        'Requires -c option. Optionally takes filename base string for txt lists.'))
    main_parser.add_argument('-c', '--cubes', nargs='+', type=str, 
                help = ('Science data cubes to be processed.  Requires at least one argument.'  
                        'Argument can be explicit list of files, a glob expression, a txt list, or a directory.'))
    main_parser.add_argument('-b', '--bias', nargs='+', default=False, 
                help = ('Bias subtraction will be done. Requires -c option. Optionally takes argument(s) which indicate(s)'
                        'filename(s) that can point to' 
                           ' master bias / '
                            'cube of unprocessed bias frames / '
                            'txt list of bias frames / '
                            'explicit list of bias frames.'))
    main_parser.add_argument('-f', '--flats', nargs='+', default=False, 
                help = ('Flat fielding will be done.  Requires -c option.  Optionally takes an argument(s) which indicate(s)'
                        ' filename(s) that can point to either '
                            'master flat / '
                            'cube of unprocessed flats / '
                            'txt list of flat fields / '
                            'explicit list of flat fields.'))
    #main_parser.add_argument('-u', '--update-headers',  help = 'Update fits file headers.')
    main_parser.add_argument('-s', '--split', nargs='?', const=True, default=False, 
                help = 'Split the data cubes. Requires -c option.')
    main_parser.add_argument('-t', '--timing', nargs='?', const=True, default=True, 
                help = ('Calculate the timestamps for data cubes. Note that time-stamping is '
                        'done by default when the cubes are split.  The timing data will be '
                        'written to a text files with the cube basename and extention ".time"') )
    main_parser.add_argument('-g', '--gps', nargs='+', default=None, 
                help = 'GPS triggering times. Explicitly or listed in txt file')
    main_parser.add_argument('-k', '--kct', default=None, 
                help = 'Kinetic Cycle Time for External GPS triggering.')
    main_parser.add_argument('-q', '--combine', nargs='+', default=['daily', 'mean'], 
                help = "Specifies how the bias/flats will be combined. Options are daily/weekly mean/median.")
    args = argparse.Namespace()
    
    #mx = main_parser.add_mutually_exclusive_group                           #NEED PYTHON3.3 AND MULTIGROUP PATCH FOR THIS...  OR YOUR OWN ERROR ANALYSIS???
    
    #Header update parser
    head_parser = argparse.ArgumentParser()
    head_parser.add_argument('update-headers', nargs='?', help='Update fits file headers.') #action=store_const?
    head_parser.add_argument('-obj', '--object', nargs='*', help='', default=[''])
    head_parser.add_argument('-ra', '--right-ascension', nargs='*', default=[''], dest='ra', help='')
    head_parser.add_argument('-dec', '--declination', nargs='*', default=[''], dest='dec', help='')
    head_parser.add_argument('-epoch', '--epoch', default=None, help='')
    head_parser.add_argument('-date', '--date', nargs='*', default=[''], help='')
    head_parser.add_argument('-filter', '--filter', default=None, help='')
    head_parser.add_argument('-tel', '--telescope', dest='tel', default=None, type=str, help='')
    head_parser.add_argument('-obs', '--observatory', dest='obs', default=None, help='')
    #head_parser.add_argument('-em', '--em-gain', dest='em', default=None, help='Electron Multiplying gain level')
    head_info = argparse.Namespace()
    
    #Name convension parser
    name_parser = argparse.ArgumentParser()
    name_parser.add_argument( 'name', nargs='?',
                help = ("template for naming convension of output files.  "
                        "eg. 'foo{sep}{basename}{sep}{filter}[{sep}b{binning}][{sep}sub{sub}]' where"
                        "the options are: "
                        "basename - the original filename base string (no extention); "
                        "name - object designation (if given or in header); "
                        "sep - separator character(s); filter - filter band; "
                        "binning - image binning. eg. 4x4; "
                        "sub - the subframed region eg. 84x60. "
                        "[] brackets indicate optional parameters.") ) #action=store_const?
    name_parser.add_argument('-fl', '--flats', nargs=1, default='f[{date}{sep}]{binning}[{sep}sub{sub}][{sep}filt{filter}]')
    name_parser.add_argument('-bi', '--bias', default='b{date}{sep}{binning}[{sep}m{mode}][{sep}t{kct}]', nargs=1) 
    name_parser.add_argument('-sc', '--science-frames', nargs=1, dest='sci', default='{basename}')
    names = argparse.Namespace()
    
    parsers = [main_parser, head_parser, name_parser]
    namespaces = [args, head_info, names]
    
    valid_commands = ['update-headers', 'names']
    #====================================================================================================
    def groupargs(arg, currentarg=[None]):
        '''Groups the arguments in sys.argv for parsing.'''
        if arg in valid_commands:
            currentarg[0] = arg
        return currentarg[0]
    
    commandlines = [ list(args) for cmd, args in itt.groupby(argv, groupargs) ]   #Groups the arguments in sys.argv for parsing
    for vc in valid_commands:
        setattr(args, vc.replace('-','_'), vc in argv)
        if not vc in argv:
            commandlines.append( [''] )                                         #DAMN HACK!
    
    for cmds, parser, namespace in zip(commandlines, parsers, namespaces):
        parser.parse_args( cmds[1:], namespace=namespace )
    
    #TODO:  SPLIT SANITY CHECK FUNCTION
    
    #Sanity checks for mutually exclusive keywords
    args_dict = args.__dict__
    disallowedkw = {'interactive': set(args_dict.keys()) - set(['interactive']),
                    }                                                                   #no other keywords allowed in interactive mode
    if args.interactive:
        for key in disallowedkw['interactive']:
            if args_dict[key]:
                raise KeyError( '%s (%s) option not allowed in interactive mode' %(main_parser.prefix_chars+key[0], main_parser.prefix_chars*2+key) )

    #Sanity checks for mutually inclusive keywords. Potentailly only one of the listed keywords required => or
    requiredkw = {      'cubes':                ['timing', 'flats', 'bias', 'update-headers'],
                        'bias':                 ['cubes'],
                        'flats':                ['cubes'],
                        'update-headers':       ['cubes'],
                        'split':                ['cubes'],
                        'timing':               ['cubes', 'gps'],
                        'ra':                   ['update-headers'],
                        'dec':                  ['update-headers'],
                        'obj':                  ['update-headers'],
                        'epoch':                ['update-headers', 'ra', 'dec'],
                        'date':                 ['update-headers'],
                        'filter':               ['update-headers'],
                        'kct':                  ['gps'],
                        'combine':              ['bias', 'flats']       }
    for key in args_dict:
        if args_dict[key] and key in requiredkw:                                        #if this option has required options
            if not any( rqk in args_dict for rqk in requiredkw[key] ):                    #if none of the required options for this option are given
                ks = main_parser.prefix_chars*2+key                                          #long string for option which requires option(s)
                ks_desc = '%s (%s)' %(ks[1:3], ks)
                rqks = [main_parser.prefix_chars*2+rqk for rqk in requiredkw[key]]           #list of long string for required option(s)
                rqks_desc = ' / '.join( ['%s (%s)' %(rqk[1:3], rqk) for rqk in rqks] )   
                raise KeyError( 'One or more of the following option(s) required with option {}: {}'.format(ks_desc, rqks_desc) )
        
    #print( '!'*100 )
    #print( args )
    #print( '!'*100 )
    
    #Sanity checks for non-interactive mode
    any_rqd = ['cubes', 'bias', 'flats', 'split', 'timing', 'update_headers', 'names']  #any of these need to be specified for an action
    if not any([getattr(args, key) for key in any_rqd]):
        raise ValueError('No action specified!\n Please specify one or more commands: -s,-t, update-headers, name, or -i for interactive mode')
    #====================================================================================================
    if args.dir:
        args.dir = iocheck(args.dir, os.path.exists, 1)

        args.dir = os.path.abspath(args.dir)
        #os.chdir(args.dir)                                        #change directory  ??NECESSARY??
    
    if args.output_dir:
        args.output_dir = iocheck(args.output_dir, os.path.exists, 1)
    else:
        args.output_dir = args.dir
    
    #====================================================================================================
    if args.cubes is None:
        args.cubes = args.dir       #no cubes explicitly provided will use list of all files in input directory
    
    if args.cubes:
        args.cubes = parsetolist(args.cubes, 
                                 os.path.exists, 
                                 path=args.dir, 
                                 raise_error=1)
        
        if not len(args.cubes):
            raise ValueError( 'File {} contains no data!!'.format('?') )
        
        args.cubes = SHOC_Run( filenames=args.cubes, label='science' )
        
        for cube in args.cubes:             #DO YOU NEED TO DO THIS IN A LOOP?
            cube._needs_flip = not cube.check( args.cubes[0], 'flip_state' )                                #self-consistency check for flip state of cubes #NOTE: THIS MAY BE INEFICIENT IF THE FIRST CUBE IS THE ONLY ONE WITH A DIFFERENT FLIP STATE...
        
    if args.split:
        if args.output_dir[0]:                                          #if an output directory is given
            args.output_dir = os.path.abspath( args.output_dir[0] )
            if not os.path.exists(args.output_dir):                              #if it doesn't exist create it
                print( 'Creating reduced data directory {}.\n'.format(args.output_dir) )
                os.mkdir( args.output_dir )

    #====================================================================================================
    if args.gps:
        args.timing = True              #Do timing if gps info given
        
        if len(args.gps)==1:         #triggers give either as single trigger time string or filename of trigger list
            valid_gps = iocheck( args.gps[0], validity.RA, raise_error=-1 )         #if valid single time this will return that same str else None
            if not valid_gps:
                args.gps = parsetolist( args.gps, validity.RA, 
                                        path=args.dir, 
                                        abspath=0, 
                                        sort=0, 
                                        raise_error=1 )
        
        #at ths point args.gps is list of explicit time strings.  
        #Check if they are valid representations of time
        args.gps = [iocheck( g, validity.RA, raise_error=1, convert=convert.RA ) for g in args.gps]
        
        #if cubes are GPS triggered on each individual frame
        if args.cubes.needs_kct() and args.kct is None:
            msg = textwrap.dedent('''
                In 'External' triggering mode EXPOSURE stores the total accumulated exposure timewhich is utterly useless.
                I need the actual exposure time - i hope you've written it down somewhere!!
                Please specify KCT (Exposure time + Dead time):
                ''')
            args.kct = Input.str(msg, 0.04, check=validity.float, what='KCT')
    
    #====================================================================================================        
    try:
        #set the kct for the cubes
        for stack in args.cubes:
            _, stack.kct = stack.get_kct()
    
    except AttributeError as err:       #error catch for gps triggering
        #Annotate the traceback
        msg = section_header( 'Are these GPS triggered frames??', swoosh='!', _print=False )
        err = type(err)( '\n\n'.join((err.args[0], msg)) )
        raise err.with_traceback( sys.exc_info()[2] )
    
    #====================================================================================================        
    if args.flats or args.bias:
        args.combine = list(map(str.lower, args.combine))
        when = 'day', 'daily', 'week', 'weekly'
        how = 'mean', 'median'
        understanding = when + how
        transmap = dict(grouper(when, 2))
        understood, misunderstood = map(list, 
                            partition(understanding.__contains__, args.combine))
        if any(misunderstood):
            raise ValueError('Argument(s) {} for combine not understood.'
                             ''.format(misunderstood))
        else:
            understood = [transmap.get(u,u) for u in understood]
            how, when = next(zip(*partition(how.__contains__, understood)))
            args.combine = when
            args.fcombine = getattr(np, how)
            print('\nBias/Flat combination will be done by {} {}.'
                  ''.format(when, how))
        
    #====================================================================================================        
    if args.flats:
        args.flats = parsetolist(args.flats, imaccess, path=args.dir, raise_error=1)
        args.flats = SHOC_Run( filenames=args.flats, label='flat' )
        
        match = args.flats.check(args.cubes, 'binning', 1, 1)
        args.flats = args.flats[match]                          #isolates the flat fields that match the science frames --> only these will be processed
        
        for flat in args.flats:
            flat._needs_flip = not flat.check( args.cubes[0], 'flip_state' )                            #NOTE: THIS MAY BE INEFICIENT IF THE FIRST CUBE IS THE ONLY ONE WITH A DIFFERENT FLIP STATE...
        
        args.flats.flag_sub( args.cubes )                       #flag the flats that need to be subframed, based on the science frames which are subframed
        
        args.flats.print_instrumental_setup()
        
    #else:
        #print( 'WARNING: No flat fielding will be done!' )
    
    #====================================================================================================    
    if args.bias:
        args.bias = parsetolist(args.bias, imaccess, path=args.dir, raise_error=1)
        args.bias = SHOC_Run( filenames=args.bias, label='bias' )
        
        
        #match the biases for the flat run
        if args.flats:
            match4flats = args.bias.check( args.flats, ['binning', 'mode'], 0, 1)
            args.bias4flats = args.bias[match4flats]
            for bias in args.bias4flats:
                bias._needs_flip = bias.check( args.flats[0], 'flip_state' )
            
            print( 'Biases for flat fields: ')
            args.bias4flats.print_instrumental_setup()
            
        #match the biases for the science run
        match4sci = args.bias.check( args.cubes, ['binning', 'mode'], 0, 1)
        args.bias = args.bias[match4sci]
        for bias in args.bias:
            bias._needs_flip = bias.check( args.cubes[0], 'flip_state' )                        #NOTE: THIS MAY BE INEFICIENT IF THE FIRST CUBE IS THE ONLY ONE WITH A DIFFERENT FLIP STATE...
        
        args.bias.flag_sub( args.cubes )
    
        print( 'Biases for science frames: ' )
        args.bias.print_instrumental_setup()
    
    #else:
        #warn( 'No de-biasing will be done!' )
    
    #====================================================================================================
    delattr( head_info, 'update-headers')
    head_info.coords = None
    if args.update_headers:
        #print( head_info )
        #for attr in ['object', 'ra', 'dec', 'date']:
        head_info.object = ' '.join( head_info.object )
        head_info.ra = ' '.join( head_info.ra )
        head_info.dec = ' '.join( head_info.dec )
        head_info.date = ' '.join( head_info.date )
        
        if head_info.ra and head_info.dec:
            iocheck( head_info.ra, validity.RA, 1 )
            iocheck( head_info.dec, validity.DEC, 1 )
            head_info.coords = SkyCoord(ra=head_info.ra, dec=head_info.dec, unit=('h','deg'))  #, system='icrs'
        else:
            head_info.coords, head_info.ra, head_info.dec = get_coords( head_info.object )
        
        if not head_info.date:
            #head_info.date = args.cubes[0].date#[c.date for c in args.cubes]
            warn( 'Dates will be assumed from file creation dates.' )
        
        if not head_info.filter:
            warn( 'Filter assumed as WL.' )
            head_info.filter = 'WL'
        
        if head_info.epoch:
            iocheck( head_info.epoch, validity.epoch, 1)
        else:    
            warn( 'Assuming epoch J2000' )
            head_info.epoch = 2000
        
        if not head_info.obs:
            warn( '\nAssuming SAAO Sutherland observatory!\n' ) 
            head_info.obs = 'SAAO'
        
        if not head_info.tel:
            warn( '\nAssuming SAAO 1.9m telescope!\n' ) 
            head_info.tel = '1.9m'
    
    elif args.timing or args.split:
        #Need target coordinates for Barycentrization! Check the headers
        try:
            head_info.ra = args.cubes[0][0].header['ra']
            head_info.dec = args.cubes[0][0].header['dec']
            head_info.coords = SkyCoord(ra=head_info.ra, dec=head_info.dec, unit=('h','deg'))
        except KeyError:
            warn( '''Object coordinates not found in header! 
                     Barycentrization cannot be done without knowing target coordinates!''' )
        
        #iocheck( head_info.date, validity.DATE, 1 )
    #else:
        #warn( 'Headers will not be updated!' )
    
    #====================================================================================================
    #if args.timing and not head_info.coords:
        #Target coordinates not provided / inferred from 
       # warn( 'Barycentrization cannot be done without knowing target coordinates!' )
    
    
    if args.names:
        SHOC_Run.NAMES.flats = names.flats
        SHOC_Run.NAMES.bias = names.bias
        SHOC_Run.NAMES.sci = names.sci

    #ANIMATE
    

    #WARN IF FILE HAS BEEN TRUNCATED -----------> PYFITS DOES NOT LIKE THIS.....WHEN UPDATING:  ValueError: total size of new array must be unchanged
    
    
################################################################################################################
if __name__ == '__main__':
    RUN_DIR = os.getcwd()
    #RNT_filename = '/home/SAAO/hannes/work/SHOC_ReadoutNoiseTable_new'              


    #filenames, dirnames, txtnames = ['rar'],[],[]   #dir_info(args.dir)

    #class initialisation
    bar = ProgressBar()                     #initialise progress bar
    bar2 = ProgressBar(nbars=2)             #initialise the double progress bar
    setup()

    #raise ValueError( 'STOPPING' )

    run = sciproc( args.cubes )


    def goodbye():
        os.chdir(RUN_DIR)
        print('Adios!')

    import atexit
    atexit.register( goodbye )

#main()




    
