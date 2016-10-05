#__version__ = '2.12'

import os
import re
import time
import datetime
#import warnings
import textwrap
import subprocess
import collections as col
import itertools as itt
from copy import copy


import numpy as np
#import matplotlib.animation as ani
#import matplotlib.pyplot as plt

#WARNING: THESE IMPORT ARE MEGA SLOW!! ~10s  (localize to mitigate)
import astropy.io.fits as pyfits
from astropy.time import TimeDelta
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.coordinates.angles import Angle
from astropy.table import Column, Table as aTable

from recipes.list import sorter
from recipes.iter import interleave
from recipes.string import rreplace
from ansi.table import Table as sTable

from myio import warn
from obstools.airmass   import Young94, altitude

from pySHOC.readnoise   import ReadNoiseTable
#TODO: choose which to use for timing: spice or astropy
from pySHOC.timing      import Time, get_updated_iers_table, light_time_corrections
from pySHOC.io          import (ValidityTests as validity,
                                Conversion as convert,
                                Input)

#debugging
from IPython import embed

#def warn(message, category=None, stacklevel=1):
    #return warnings.warn('\n'+message, category=None, stacklevel=1)
    

#TODO:  class SHOC Timing
    #specify required timing accuracy --> decide which algorithms to use based on this!
    
#TODO:            
#class Trigger (maybe Time subclass??)
    
    #def is_external
    #is_gps
    #is_internal
    #value


#import matplotlib.pyplot as plt
#from imagine import supershow


#from decor import profile

#from IPython import embed
#from magic.string import banner


#tsktsk('modules')
#print( 'Done!\n\n' )

################################################################################
#Function definitions
################################################################################

def get_coords(name, verbose=True):
    '''Attempts a SIMBAD Sesame query with the given object name'''
    
    try: 
        if verbose:
            print('\nQuerying SIMBAD database for {}...'.format(repr(name)))
        
        coo = SkyCoord.from_name(name)
        
    except Exception as err:     #astcoo.name_resolve.NameResolveError
        warn('Object {} not found in SIMBAD database'.format(repr(name)))
        #FIXME, arbitrary exception does not mean 'not in database!'
        
        try:
            from obstools.jparser import Jparser
            coo = Jparser(name).skycoord()
            
        except Exception as err:
            print(err)  #TODO traceback
            print('ERROR in retrieving coordinates...')
            return
    
    if verbose:
        fmt = dict(precision=2, sep=' ', pad=1)
        ra = coo.ra.to_string(unit='h', **fmt)
        dec = coo.dec.to_string(alwayssign=1, **fmt)
        print('The following ICRS J2000.0 coordinates were retrieved:\n'
              'RA = {}, DEC = {}\n'.format(ra, dec))
    
    return coo


def get_coords_ra_dec(name, verbose=True, **fmt):
    '''return SkyCoords and str rep for ra and dec'''
    coords = get_coords(name, verbose=True)
    if coords is None:
        return None, None, None
    
    default_fmt = dict(precision=2, sep=' ', pad=1)
    fmt.update(default_fmt)
    ra = coords.ra.to_string(unit='h', **fmt)
    dec = coords.dec.to_string(alwayssign=1, **fmt)
    
    return coords, ra, dec



def link_to_short_name_because_iraf_sux(filename, count, ext):
    #HACK! BECAUSE IRAF SUX
    linkname = args.dir + '/s{}.{}'.format(count, ext)
    print('LINKING:', 'ln -f', os.path.basename(filename), os.path.basename(linkname))
    subprocess.call(['ln', '-f', filename, linkname])


################################################################################
#class definitions
################################################################################
class Date(datetime.date):
    '''
    We need this so the datetime.date instances print in date format instead
    of the class representation format, when print is called on, for eg. a tuple
    containing a date_time object.
    '''
    #===========================================================================
    def __repr__(self):
        return str(self)


################################################################################
class FilenameGenerator(object):
    #===========================================================================
    def __init__(self, basename, reduction_path='', padwidth=None, sep='.', 
                 extension='.fits'):
        self.count = 1
        self.basename = basename
        self.path = reduction_path
        self.padwidth = padwidth
        self.sep = sep
        self.extension = extension
 
    #===========================================================================
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
        

################################################################################
class shocCube(pyfits.hdu.hdulist.HDUList):        #Rename shocCube
    #HACK:  Subclass PrimaryHDU instead???
    #TODO: location as class attribute??
    '''
    Extend the hdu.hdulist.HDUList class to perform simple tasks on the image stacks.
    '''
    #===========================================================================
    @classmethod
    def load(cls, fileobj, mode='readonly', memmap=False, save_backup=False,
             **kwargs):
        do_timing = kwargs.pop('do_timing', False)
        
        hdulist = cls._readfrom(fileobj=fileobj, mode=mode, memmap=memmap,
                                save_backup=save_backup, ignore_missing_end=True,
                                #do_not_scale_image_data=True,
                                **kwargs)
                                
        hdulist.instrumental_setup()
        if do_timing:
            hdulist.time_init()
        
        return hdulist
    
    #===========================================================================
    def __init__(self, hdus=[], file=None):

        super(shocCube, self).__init__(hdus, file)
        self._needs_flip        = False
        self._needs_sub         = []
        self._is_master         = False
        self._is_unpacked       = False
        self._is_subframed      = False
        
        self.path, self.basename = os.path.split(self.filename())
        if self.basename:
            self.basename = self.basename.replace('.fits','')
        
        #self.filename_gen = FilenameGenerator(self.basename)
        #self.trigger = None
    
    #===========================================================================
    #WARNING: this will break things
    #def __len__(self):
        #return self.shape[-1] #NOTE: will not work before instrumental setup is run
    
    #===========================================================================
    def __repr__(self):
        name, dattrs, values = self.get_instrumental_setup()
        ref = tuple( interleave(dattrs, values) )
        r = name + ':\t' + '%s = %s;\t'*len(values) %ref
        return '{} ==> {}'.format( self.__class__.__name__, r )
    
    #===========================================================================
    def get_filename(self, with_path=0, with_ext=1, suffix=(), sep='.'):
        if with_path:
            filename = self.filename()
        else:
            _, filename = os.path.split( self.filename() )
        
        *stuff, ext = filename.split(sep)
        ext = [ext] if with_ext else ['']
        suffix = [suffix] if isinstance(suffix, str) else list(suffix)
        suffix = [s.strip(sep) for s in suffix]
        
        return sep.join( filter(None, stuff+suffix+ext) )
    
    #===========================================================================
    def instrumental_setup(self):
        '''
        Retrieve the relevant information about the observational setup. Used 
        for comparitive tests.
        '''
        header = self.header
        
        #instrument
        serno = header['SERNO']
        self.instrument = 'SHOC ' + str([5982, 6448].index(serno) + 1)
        #else: self.instrument = 'unknown!'
        
        #date
        date, time = header['DATE'].split('T')
        self.date = Date(*map(int, date.split('-')))          #file creation date
        h = int(time.split(':')[0])
        self.namedate = self.date - datetime.timedelta(int(h < 12))  #starting date of the observing run --> used for naming
        
        #image binning
        self.binning = tuple(header[s+'BIN'] for s in ['H','V'])
        
        #gain
        self.gain = header.get('gain', 0)
        
        #image dimensions
        self.ndims = header['NAXIS']                                                    #Number of image dimensions
        self.shape = tuple(header['NAXIS'+str(i)] 
                                for i in range(1, self.ndims+1))
        self.dimension = self.shape[:2]                                                      #Pixel dimensions for 2D images
        
        
        #sub-framing
        xb,xe, ye,yb = map(int, header['SUBRECT'].split(','))
        self.subrect = xb,xe, ye,yb
        xb //= self.binning[0]
        xe //= self.binning[0]
        yb //= self.binning[1]
        ye //= self.binning[1]
        self.sub = xb,xe, yb,ye
        self._is_subframed = (xe, ye) != self.dimension
                
        
        #readout speed
        speed = 1./header['READTIME']
        speed_MHz = int(round(speed/1.e6))
        
        #CCD mode
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
        self.trigger = header.get('gpsstart')       #None for older SHOC data
        
        #Date (midnight UT)
        date_str = header['date'].split('T')[0]    #or FRAME
        self.utdate = Time(date_str)       #this should at least be the correct date!
        
        #exposure time
        self.exp, self.kct = self.get_kct()   
        #NOTE: The attributes `trigger` and `kct` will be None only if we expect
        #the user to provide them explicitly (older SHOC data)
        
        if self.kct:
            self.duration = self.shape[-1] * self.kct
        
        
        #object name
        self.target = header.get('object')
        
        #coordinates
        self.coords = self.get_coords()
        
        #header keywords that need to be updated
        #if self.trigger_mode.startswith('External') and is old:
            #'KCT', 'EXPOSURE', 'GPSSTART'
        
        #self.kct = header.get('kct')   #will only work for internal triggering.
        #self.exp = header.get('exposure')
        
        #telescope
        #self.telescope = header['telescope']
    
    #===========================================================================
    def _get_coords(self, ra, dec):
        if ra and dec:
            try:
                return SkyCoord(ra=ra, dec=dec, unit=('h','deg'))
            except ValueError as err:
                warn('SHIT')
    
    #===========================================================================
    #TODO: property?????????        NOTE: consequences for head_proc
    def get_coords(self):
        header = self[0].header
        
        ra, dec = header.get('objra'), header.get('objdec')
        coords = self._get_coords(ra, dec)
        if coords:
            return coords
        
        if self.target:
            #No / bad coordinates available, but object name available - try resolve
            coords = get_coords(self.target, verbose=False)
        
        if coords:
            return coords
        
        #No header coordinates / Name resolve failed / No object name available
        
        #LAST resort use TELRA, TELDEC!!!! (Will only work for new SHOC data!!)
        #NOTE: These will lead to slightly less accurate timing solutions (Quantify?)
        ra, dec = header.get('telra'), header.get('teldec')
        coords = self._get_coords(ra, dec)
        
        #TODO: optionally query for named sources in this location
        if coords:
            warn('USING TELESOPE POINTING COORDINATES.')
        
        return coords
        
        
    #===========================================================================
    @property
    def has_coords(self):
        return self.coords is not None
    
    #===========================================================================
    @property
    def has_old_keys(self):
        from pySHOC.convert_keywords import KEYWORDS as kw_old_to_new
        header = self[0].header
        return any(map(header.__contains__, next(zip(*kw_old_to_new))))
    
    #===========================================================================
    def get_instrumental_setup(self, attrs=None):
        #YOU CAN MAKE THIS __REPR__????????
        
        defaults = ['binning', 'dimension', 'mode', 'gain', 'trigger_mode', 
                    'kct', 'duration']
        attrs =  attrs  or defaults
        dattrs = [at.replace('_',' ').upper() for at in attrs]        #for display
        vals = [getattr(self, attr, '??') for attr in attrs]
    
        name = self.get_filename() or 'Unsaved'
        
        return name, dattrs, vals
    
    #===========================================================================
    def get_pixel_scale(self, telescope):
        '''get pixel scale in arcsec '''
        pixscale = {'1.9'     :       0.076,
                    '1.9+'    :       0.163,          #with focal reducer
                    '1.0'     :       0.167,
                    '0.75'    :       0.218   }
        
        tel = rreplace( telescope, ('focal reducer','with focal reducer'), '+')
        tel = tel.replace('m', '').strip()
        
        return np.array( self.binning ) * pixscale[ tel ]
                    
    #===========================================================================
    def get_field_of_view(self, telescope):
        '''get fov in arcmin'''
        fov = { '1.9'     :       (1.29, 1.29),
                '1.9+'    :       (2.79, 2.79),        #with focal reducer
                '1.0'     :       (2.85, 2.85),
                '0.75'    :       (3.73, 3.73)   }
        
        tel = rreplace( telescope, ('focal reducer','with focal reducer'), '+')
        tel = tel.replace('m', '').strip()
        
        return fov[tel]
     
    fov = get_field_of_view
     
    #===========================================================================
    def check(self, frame2, key, raise_error=0):
        '''check fits headers in this image agains frame2 for consistency of key attribute 
        Parameters
        ----------
        key : The attribute to be checked (binning / instrument mode / dimensions / flip state)
        frame2 : shocCube Objects to check agains
        
        Returns
        ------
        flag : Do the keys match?
        '''
        flag = getattr(self,key) == getattr(frame2,key)
            
        if not flag and raise_error:
            raise ValueError
        else:
            return flag

    #===========================================================================
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

    #===========================================================================
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
        
        stack = shocCube(hdu, fileobj)
        stack.instrumental_setup()
        
        #stack._is_subframed = 1
        #stack._needs_sub = []
        #stack.sub = subreg
        
        if write:
            stack.writeto( outname, output_verify='warn' )
        
        return stack

    #===========================================================================
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
        
        #Load the stack as a shocCube
        hdu = pyfits.PrimaryHDU( data, header )
        outname = next( self.filename_gen() )        #generate the filename
        fileobj = pyfits.file._File(outname, mode='ostream', clobber=True)
        stack = shocCube(hdu, fileobj)                 #initialise the Cube with target file
        stack.instrumental_setup()
        
        return stack
    
    #===========================================================================
    def unpack(self, count=1, padw=None, dryrun=0, w2f=1):                              #MULTIPROCESSING!!!!!!!!!!!!
        '''Unpack (split) a 3D cube of images along the 3rd axis. 
        Parameters
        ----------
        outpath : The directory where the imags will be unpacked
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
            header.remove('NUMKIN')
            header.remove('NAXIS3') #Delete this keyword so it does not propagate into the headers of the split files
            header['NAXIS'] = 2       #Number of axes becomes 2
            header.add_history('Split from %s' %stack)
            
            #open the txt list for writing
            if w2f:
                basename = self.get_filename(1,0)
                self.unpacked = basename + '.split'
                fp = open(self.unpacked, 'w')
            
            print( '\n\nUnpacking the stack {} of {} images...\n\n'.format(stack ,naxis3) )
            
            #split the cube
            filenames = self.filename_gen( naxis3+count-1 )
            bar.create(naxis3)
            for j, im, fn in zip( range(naxis3), self[0].data, filenames ):
                bar.progress(count-1)             #count instead of j in case of sequential numbering for multiple cubes

                self.timestamp_header(j)           #set the timing values in the header for frame j
                
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
    
    #===========================================================================
    def set_name_dict(self):
        header = self[0].header
        obj = header.get('OBJECT', '')
        filter = header.get('FILTER', 'WL')
        
        kct = header.get('kct', 0 )
        if int(kct/10):
            kct = str(round(kct))+'s'
        else:
            kct = str(round(kct*1000))+'ms'
        
        self.name_dict = dict(sep       = '.',
                              obj       = obj,
                              basename  = self.get_filename(0,0),
                              date      = str(self.namedate).replace('-',''),
                              filter    = filter,
                              binning   = '{}x{}'.format(*self.binning),
                              mode      = self.mode_trim,
                              kct       = kct                            )
    
    ############################################################################
    # Timing #TODO: separate class here
    ############################################################################
    #===========================================================================           
    @property
    def needs_timing_fix(self):
        '''check for date-obs keyword to determine if header information needs updating'''
        return not ('date-obs' in self[0].header)     #FIXME: is this good enough???
    
    #===========================================================================           
    def time_init(self, location):
        '''
        Do timing corrections on shocCube.
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
        #TODO: double check this crap!!!!!!!!!!!!!
        # NOTE:
        # Older SHOC data (pre 2016)
        # --------------------------
        # Times recorded in FITS header are as follows:
        #   Mode: 'Internal'        : (FRAME, DATE) Time at the end of the first exposure (file creation timestamp)
        #                           : The time here is rounded to the nearest second of computer clock
        #                           :   ==> uncertainty of +- 0.5 sec (for absolute timing)
        #                           : DATE-OBS not recorded 
        #   Mode: 'External Start'  : EXPOSURE - exposure time (sec)
        #                             WARNING: KCT, DATE-OBS not recorded 
        #                           : WARNING: start time not recorded
        #   Mode: 'External'        : WARNING: KCT, DATE-OBS not recorded
        #                           : EXPOSURE stores the total accumulated exposure time
        
        # Recent SHOC data (post software upgrade)
        # ----------------------------------------
        #   Mode: 'Internal'        : DATE-OBS  - start time accurate to microsec
        #   Mode: 'External [Start]': GPSSTART  - GPS start time (UTC; external)
        #                           : KCT       - Kinetic Cycle Time
        #   Mode: 'External'        : GPS-INT   - GPS trigger interval (msec)
        
        
        stack_header = self[0].header
        trigger_mode = self.trigger_mode
    
        if self.needs_timing_fix:
            return self.get_time_data_old(location)
        else:
            return self.get_time_data_new(location)
    
    #===========================================================================
    def get_kct(self):
        if self.needs_timing_fix:
            return self.get_kct_old()
        else:
            return self[0].header['exposure'], self[0].header['kct']
    
    #===========================================================================
    def get_kct_old(self):      #FIXME: in a subclass!  God object anti-pattern
        
        stack_header = self[0].header
        
        if self.trigger_mode=='Internal':
            #kinetic cycle time between start of subsequent exposures in sec.  i.e. exposure time + readout time
            t_kct = stack_header['KCT']
            t_exp = stack_header['EXPOSURE']
            #In internal triggering mode EXPOSURE stores the actual correct exposure time.
            #                       and KCT stores the Kinetic cycle time (dead time + exposure time)
        
        #GPS Triggering (External or External Start)
        elif self.trigger_mode.startswith('External'):
            
            t_dead = 0.00676
            #dead (readout) time between exposures in s 
            #NOTE: (deadtime should always the same value unless the user has 
            # (foolishly) changed the vertical clock speed). 
            #MAYBE CHECK stack_header['VSHIFT'] 
            #WARNING: EDGE CASE: THE DEADTIME MAY BE LARGER IF WE'RE NOT OPERATING 
            #IN FRAME TRANSFER MODE!
            
            if self.trigger_mode.endswith('Start'):         #External Start
                t_exp = stack_header['EXPOSURE']            #exposure time in sec as in header
                t_kct = t_dead + t_exp                         # Kinetic Cycle Time
            else:
                #trigger mode external - exposure and kct needs to be provided at terminal through -k
                return None, None
                #
                #t_kct = float(args.kct)                     #kct provided by user at terminal through -k
                #t_exp = t_kct - t_dead                      #set the 'EXPOSURE' header keyword
        
        return t_exp, t_kct
    
    #===========================================================================           
    def get_time_data_new(self, location):
        #new / fixed data!  Rejoice!
        header = self[0].header
        ts = header['DATE-OBS']         # NOTE: This keyword is confusing (UTC-OBS would be better), but since it is now in common use, we (reluctantly do the same)
        t0 = Time(ts, format='isot', scale='utc', 
                    precision=9, location=location)    #time for start of first frame
        td_kct = TimeDelta(self.kct, format='sec')
        tmid = t0 +  0.5 * td_kct       #NOTE: TimeDelta has higher precision than Quantity
        return tmid, td_kct
    
    #===========================================================================           
    def get_time_data_old(self, location):      #FIXME SUBCLASS OldshocCube
        '''
        Extract ralavent time data from the FITS header.
        
        Returns
        ------
        First frame mid time
        
        '''
        header = self[0].header
        
        #date_str = header['DATE'].split('T')[0]
        #utdate = Time(date_str)     #this should at least be the correct date!
        
        t_exp, t_kct = self.exp, self.kct
        td_kct = TimeDelta(t_kct, format='sec')     #NOTE: TimeDelta has higher precision than Quantity
        #td_kct = t_kct * u.sec
        
        if self.trigger_mode == 'Internal':
            #Initial time set to middle of first frame exposure
            #NOTE: this hardly matters for sub-second t_exp, as the time recorded 
            #       in header FRAME is rounded to the nearest second
            utf = Time(header['DATE'],                #or FRAME
                       format='isot', scale='utc', 
                       precision=9, location=location)    #time for end of first frame    
            tmid = utf - 0.5 * td_kct          #mid time of first frame
            #return  tmid, td_kct
        
        if self.trigger_mode.startswith('External'):
            if self.trigger:
                t0 = self.utdate + self.trigger
                t0 = Time(t0.isot, format='isot', scale='utc', 
                          precision=9, location=location)
                tmid = t0 + 0.5 * td_kct                                        #set t0 to mid time of first frame
                #return tmid, td_kct
            
            else:
                raise ValueError('No GPS triggers provided for {}!'.format(self.filename()))
        
        #stack_hdu.flush(output_verify='warn', verbose=1)
        #IF TIMECORR --> NO NEED FOR GPS TIMES TO BE GIVEN EXPLICITLY
        
        print('{} : TRIGGER is {}. tmid = {}; KCT = {} sec'
              ''.format(self.get_filename(), self.trigger_mode.upper(), tmid, t_kct))
        
        return tmid, td_kct
    
    #===========================================================================
    #def get_timing_array(self, t0):
        #t_kct = round(self.get_kct()[1], 9)                     #numerical kinetic cycle time in sec (rounded to nanosec)
        #td_kct = td_kct * np.arange( self.shape[0], dtype=float )
        #t = t0 + td_kct                         #Time object containing times for all framesin the cube
        #return t
    
    #===========================================================================
    def set_times(self, t0, td_kct, iers_a=None, coords=None, location=None,):
                 #TODO: corrections):

        #t_kct = round(t_kct, 9)    #numerical kinetic cycle time in sec (nanosec precision)
        
        #Time object containing time stamps for all frames in the cube
        td_kct = td_kct * np.arange(self.shape[-1], dtype=float)
        t = t0 + td_kct
        
        #set leap second offset from most recent IERS table
        delta, status = t.get_delta_ut1_utc(iers_a, return_status=True)  
        if np.any(status==-2): #TODO: verbose?
            warn('Using predicted leap-second values from IERS.')
        t.delta_ut1_utc = delta
        
        #initialize array for timing data
        #TODO: external control for which are calculated
        self.timedata = timedata = np.recarray(len(t), 
                                               dtype=[('utdate',    'U20'),
                                                      ('uth',       float), 
                                                      ('utsec',     float),
                                                      ('utc',       'U30'),
                                                      ('utstr',     'U20'),
                                                      ('lmst',      float),
                                                      ('jd',        float),
                                                      ('gjd',       float),
                                                      ('bjd',       float),
                                                      ('altitude',  float),
                                                      ('airmass',   float)])
        #compute timestamps for various scales
        #timedata.texp           = texp
        timedata.utc            = t.utc.isot
        #UTC in decimal hours for each frame
        uth = (t.utc - self.utdate).to('hour')
        timedata.uth            = uth.value
        timedata.utsec          = uth.to('s').value
        
        #split UTDATE and UTC time
        utdata = t.isosplit()
        timedata.utdate         = utdata['utdate']
        timedata.utstr          = utdata['utc']
        
        #if location is None:
        
        #else:
        lat = location.latitude.radian
        lon = location.longitude
        
        #LMST for each frame
        lmst                    = t.sidereal_time('mean', longitude=lon)
        timedata.lmst           = lmst
        #timedata.last          = t.sidereal_time('apparent', longitude=lon)
        
        #Julian Dates
        timedata.jd             = t.jd
        #timedata.ljd           = np.floor(timedata.jd)
        timedata.gjd            = t.tcg.jd                #geocentric julian date
        
        #do barycentrization
        if coords is None and self.has_coords:
            coords = self.coords
        
        if not None in (coords, location):
            #barycentric julian date (with light travel time corrections)
            bjd = light_time_corrections(t, coords, precess='first', abcorr=None)
            timedata.bjd = bjd
            
            #altitude and airmass
            timedata.altitude = altitude(coords.ra.radian,
                                         coords.dec.radian,
                                         lmst.radian,
                                         np.radians(lat))
            timedata.airmass = Young94(np.pi/2 - timedata.altitude)
            
        self.timestamp_header(0, t0, verbose=True)
        self.flush(output_verify='warn', verbose=True)
    
    #===========================================================================    
    def export_times(self, with_slices=False, count=0):  #single_file=True,
        '''write the timing data for the stack to file(s).'''
        def make_header_line(info, fmt, delimiter):
            import re
            matcher = re.compile('%-?(\d{1,2})')
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
            slices = np.fromiter(map(os.path.basename, self.real_slices), 'U35')
            formats['filename'] = '%-35s'
            table.add_column(Column(slices, 'filename'), 0)
        
        delimiter = '\t'
        timefile = self.get_filename(1, 0, 'time')
        table.write(timefile,
                    delimiter=delimiter,
                    format='ascii.commented_header', 
                    formats=formats)
        
        #if single_file:
        #Write all timing data to a single file
        #delimiter = ' '
        #timefile = self.get_filename(1, 0, 'time')
        #header = make_header_line( TKW, fmt, delimiter )
        #np.savetxt(timefile, T, fmt=fmt, header=header )
        
        #HACK! BECAUSE IRAF SUX
        use_iraf = False
        if use_iraf:
            link_to_short_name_because_iraf_sux(timefile, count, 'time')
            
            
        #else:
            #for i, tkw in enumerate(TKW):
                ##write each time sequence to a separate file...
                #fn = '{}.{}'.format(self.get_filename(1,0), tkw)
                #if tkw in TKW_sf:
                    #if fn.endswith('uth'): fn.replace('uth', 'utc')
                    #np.savetxt( fn, T[i], fmt='%.10f' )
    
    #===========================================================================
    def timestamp_header(self, j, t0=None, verbose=False):
        ''' '''
        if verbose:
            print('Updating the starting times for datacube {} ...'   #FIXME: repeat print not necessary
                  ''.format(self.get_filename()))
        
        header = self[0].header
        timedata = self.timedata
        
        if self.needs_timing_fix:
            header['KCT'] = (self.kct, 'Kinetic Cycle Time')         #set KCT in header
            header['EXPOSURE'] = (self.exp, 'Integration time')      #set in header
            if t0:
                #also set DATE-OBS keyword
                header['DATE-OBS'] = str(t0)
                if self.trigger_mode.startswith('External'):    
                    #Set correct (GPS triggered) start time in header
                    header['GPSSTART'] = (str(t0), 'GPS start time (UTC; external)')  
                    #TODO: OR set empty?

        
        header['utc-obs']   = (timedata.uth[j], 'Start of frame exposure in UTC')                       #imutil.hedit(imls[j], 'utc-obs', ut.hours(), add=1, ver=0)                       # update timestamp in header
        header['LMST']      = (timedata.lmst[j], 'Local Mean Sidereal Time')                              #imutil.hedit(imls[j], 'LMST', lmst, add=1, ver=0)                           
        header['UTDATE']    = (timedata.utdate[j], 'Universal Time Date')                           #imutil.hedit(imls[j], 'UTDATE', ut.iso.split()[0], add=1, ver=0)

        header['JD']        = (timedata.jd[j], 'Julian Date (UTC)' )
        #header['LJD']      = ( timedata.ljd[j], 'Local Julian Date' )
        header['GJD']       = (timedata.gjd[j], 'Geocentric Julian Date (TCG)')
        
        if self.has_coords:
            header['BJD']   = (timedata.bjd[j], 'Barycentric Julian Date (TDB)')
            header['AIRMASS'] = (timedata.airmass[j], 'Young 1994 model')
        
        #elif j!=0:
            #warn( 'Airmass not yet set for {}!\n'.format( self.get_filename() ) )
        
        #header['TIMECORR'] = ( True, 'Timing correction done' )        #imutil.hedit(imls[j], 'TIMECORR', True, add=1, ver=0)                                       #Adds the keyword 'TIMECORR' to the image header to indicate that timing correction has been done
        #header.add_history('Timing information corrected at %s' %str(datetime.datetime.now()), before='HEAD' )            #Adds the time of timing correction to header HISTORY
    
    
    #===========================================================================
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
    
    #===========================================================================
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
    
    #===========================================================================
    def make_obsparams_file(self, suffix, count):
        slices = np.fromiter( map(os.path.basename, self.real_slices), 'U23' )
        texp = np.ones(self.shape[-1]) * self.texp
        Filt = np.empty(self.shape[-1], 'U2')
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
        
################################################################################
class shocRun( object ):           #rename shocRun
    #TODO: merge method?
    '''
    Class to perform comparitive tests between cubes to see if they are compatable.
    '''
    #===========================================================================
    MAX_CHECK = 25  #Maximal length for input list if validity checks is to be performed
    MAX_LS = 25     #Maximal number of files to list in dir_info
    
    #Naming convension defaults
    NAMES = type(
        'Names', (), 
        {'flat' :   'f{date}{sep}{binning}[{sep}sub{sub}][{sep}filt{filter}]',
         'bias' :   'b{date}{sep}{binning}[{sep}m{mode}][{sep}t{kct}]',
         'sci'  :   '{basename}'}
                )

    
    #===========================================================================
    def __init__(self, hdus=None, filenames=None, label=None, sep_by=None,
                 location='sutherland'):
        
        #WARNING:  filenames may contain None as well as duplicate entries.....??????
                    #not sure if duplicates is desireable wrt efficiency.....
        
        self.cubes = list(filter(None, hdus))   if hdus     else []
        
        self.sep_by = sep_by
        self.label = label
        
        if not filenames is None:
            self.filenames = list(filter(None, filenames))            #filter None
            self.load(self.filenames)
        elif not hdus is None:
            self.filenames =  [hdulist.filename() for hdulist in self]
        
        #set location
        #self.location = EarthLocation.of_site(location)  
        #TODO: HOW SLOW IS THIS COMPARED TO ABOVE LINE???
        sutherland = EarthLocation(lat=-32.376006, lon=20.810678, height=1771)
        self.location = sutherland
        
        
    #===========================================================================
    def __len__(self):
        return len(self.cubes)
    
    #===========================================================================
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
        
        return '{} : {}'.format( name, ' | '.join(self.get_filenames()))
    
    #===========================================================================
    def __getitem__(self, key):
        
        if isinstance( key, int ):
            if key >= len(self) :
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
                
            elif isinstance(key[0], (int, np.int0)): #NOTE: be careful bool isa int
                rl = [ self.cubes[i] for i in key ]
        
        return shocRun( rl, label=self.label, sep_by=self.sep_by )
    
    #===========================================================================
    def __add__(self, other):
        return self.join( other )
    
    #===========================================================================
    #def __eq__(self, other):
        #return vars(self) == vars(other)
    
    #===========================================================================
    #def pullattr(self, attr, return_as=list):
        #return return_as([getattr(item, attr) for item in self])
    
    #===========================================================================
    def pop(self, i):       #TODO: OR SUBCLASS LIST?
        return self.cubes.pop(i)
    
    #===========================================================================
    def join(self, *runs):
        
        runs = list( filter(None, runs) )       #Filter empty runs (None)
        labels = [r.label for r in runs]
        hdus = sum([r.cubes for r in runs], self.cubes )
        
        if np.all(self.label == np.array(labels)):
            label = self.label
        else:
            warn( "Labels {} don't match {}!".format(labels, self.label) )
            label = None
        
        return shocRun(hdus, label=label)
        
    #===========================================================================
    def load(self, filenames, mode='update', memmap=False, save_backup=False,
             **kwargs):
        '''
        Load data from file. populate data for instrumental setup from fits header.
        '''
        self.filenames = filenames
        
        label = kwargs.pop('label') if 'label' in kwargs else self.label 
        print( '\nLoading data for {} run...'.format(label) )   #TODO: use logging
        
        #cubes = []
        for i, fileobj in enumerate(filenames):
            hdu = shocCube.load(fileobj, mode=mode, memmap=memmap, 
                                 save_backup=save_backup, **kwargs)           
            #YOU CAN BYPASS THIS INTERMEDIATE STORAGE IF YOU MAKE THE PRINT OPTION
            #A KEYWORD ARGUMENT FOR THE shocCube __init__ me
            self.cubes.append(hdu)
    
    #===========================================================================
    def print_instrumental_setup(self):
        '''Print the instrumental setup for this run as a table.'''
        names, dattrs, vals = zip(*(stack.get_instrumental_setup()
                                    for stack in self))
        
        bgcolours = {'flat'       : 'cyan', 
                     'bias'       : 'magenta', 
                     'science'    : 'green'}       #TODO: move to configuration
        bgc = bgcolours.get(self.label, 'default')
        title = 'Instrumental Setup: {} frames'.format(self.label.title())
        table = sTable(vals, 
                       title = title,
                       title_props = dict(text='bold', bg=bgc),
                       col_headers = dattrs[0], 
                       row_headers = ['filename'] + list(names), 
                       number_rows = True )
        
        print(table)
        #print( 'Instrumental setup ({} cubes):'.format(len(self)) )     #SPANNING DATES?
        #for i, stack in enumerate(self):
            #if i < self.MAX_LS:
                #print( stack )
            #else:
                #print( '.\n'*3 )
                #break
    
    #===========================================================================
    def reload(self, filenames=None, mode='update', memmap=True, 
               save_backup=False, **kwargs):
        if len(self):
            self.cubes = []
        self.load(filenames, mode, memmap, save_backup, **kwargs)
    
    ############################################################################
    # Timing
    ############################################################################
    #===========================================================================
    def set_times(self, coords=None):
        
        times = [stack.time_init(self.location) for stack in self]
        t0s, tds = zip(*times)
        t_test = Time(t0s)
        
        #check whether IERS tables are up to date given these cube starting times
        status = t_test.check_iers_table()
        # Cached IERS table will be used if only some of the times are outside 
        # the range of the current table. For the remaining values the predicted
        # leap second values will be used.  If all times are beyond the current 
        # table a new table will be grabbed online.
        #cache = np.any(status) and not np.all(status)
        try:
            #update the IERS table and set the leap-second offset
            iers_a = get_updated_iers_table(cache=True)
        except Exception as err:    #FIXME:  unspecified exception catch = bad!
            warn('Unable to update IERS table.')
            print(err)      #TODO with traceback?
            iers_a = None
            
        msg = '\n\nCalculating timing arrays for datacube(s):'
        lm = len(msg)
        print(msg,)
        for i, stack in enumerate(self):
            print( ' '*lm + stack.get_filename()  )
            t0, td = times[i]
            stack.set_times(t0, td, iers_a, coords, self.location)

    #===========================================================================
    def export_times(self, with_slices=False):
        for i, stack in enumerate(self):
            stack.export_times(with_slices, i)
    
    #===========================================================================
    def gen_need_kct(self):
        for stack in self:
            yield (stack.needs_timing_fix() and 
                   stack.trigger_mode == 'External')
    
    #===========================================================================
    def that_need_kct(self):
        return self[list(self.gen_need_kct())]
    
    #===========================================================================
    def gen_need_triggers(self):
        for stack in self:
            yield (stack.needs_timing_fix() and 
                   stack.trigger_mode.startswith('External'))       
            #TODO: stack.trigger.is_gps()
    
    #===========================================================================
    def that_need_triggers(self):
        return self[list(self.gen_need_triggers())]
    
    #===========================================================================
    def set_gps_triggers(self, times, triggers_in_local_time=True):
        #trigger provided by user at terminal through -g  (args.gps)
        #if len(times)!=len(self):
        #check if single trigger OK (we can infer the remaining ones)
        if self.check_rollover_state():
            times = self.get_rolled_triggers(times)
            print(
            '\nA single GPS trigger was provided. Run contains auto-split ' 
            'cubes (filesystem rollover due to 2Gb threshold on old windows pc).' 
            ' Start time for rolled over cubes will be inferred from the length' 
            ' of the preceding cube(s).\n'
            )
            
        #at this point we expect one trigger time per cube
        if len(self) != len(times):
            raise ValueError('Only {} GPS trigger given. Please provide {} for '
                             '{}'.format(len(times), len(self), self))
        
        #NOTE: GPS triggers are assumed to be in SAST.  If they are provided in
        # UT pass triggers_in_local_time = False
        #get time zone info
        timezone = -7200    if triggers_in_local_time   else 0
        #warn('Assuming GPS triggers provided in local time (SAST)')
        for j, stack in enumerate(self):
            #convert trigger time to seconds from midnight UT
            trigsec = Angle(self.trigger, 'h').to('arcsec').value / 15.
            trigsec += timezone
            #trigger now in sec UTC from midnight
            
            #adjust to positive value -- this needs to be done (since tz=-2) so 
            #we don't accidentally shift the date. (since we are measuring time 
            #from midnight on the previous day DATE/FRAME in header)
            if ttrig.value < 0:
                ttrig += 86400
            
            stack.trigger = TimeDelta(ttrig, format='sec')
            
    #===========================================================================
    def check_rollover_state(self):
        '''
        Check whether the filenames contain ',_X' an indicator for whether the 
        datacube reached the 2GB windows file size limit on the shoc server, and
        was consequently split into a sequence of fits cubes. 
        
        Notes: 
        -----
        This applies for older SHOC data only
        '''
        return np.any(['_X' in _ for _ in self.get_filenames()])
    
    #===========================================================================
    def get_rolled_triggers(self, first_trigger_time):
        '''
        If the cube rolled over while the triggering mode was 'External' or
        'External Start', determine the start times (inferred triggers) of the 
        rolled over cube(s).
        '''
        slints = [cube.shape[-1] for cube in self]              #stack lengths
        #sorts the file sequence in the correct order
         #re pattern to find the roll-over number (auto_split counter value in filename)
        matcher = re.compile('_X([0-9]+)')
        fns, slints, idx = sorter( self.get_filenames(), slints, range(len(self)),
                                   key=matcher.findall )
        
        print( 'WORK NEEDED HERE!' )
        embed()
        #WARNING: This assumes that the run only contains cubes from the run that rolled-over.  
        #         This should be ok for present purposes but might not always be the case
        idx0 = idx[0]
        self[idx0].trigger = first_trigger_time
        t0, td_kct = self[idx0].time_init(dryrun=1)      
        #dryrun ==> don't update the headers just yet (otherwise it will be done twice!)
    
        d = np.roll(np.cumsum(slints), 1)
        d[0] = 0
        t0s = t0 + d*td_kct
        triggers = [t0.isot.split('T')[1] for t0 in t0s]
        
        #resort the triggers to the order of the original file sequence
        #_, triggers = sorter( idx, triggers )

        return triggers
    
    #===========================================================================
    def export_headers(self):
        '''save fits headers as a text file'''
        for stack in self:
            headfile = stack.get_filename(with_path=1, with_ext=0,
                                          suffix='.head')
            print('\nWriting header to file: {}'.format(os.path.basename(headfile)))
            #TODO: remove existing!!!!!!!!!!
            stack[0].header.totextfile( headfile, clobber=True )
        
    #===========================================================================
    def make_slices(self, suffix):
        for i, cube in enumerate(self):
            cube.make_slices(suffix, i)
    
    #===========================================================================
    def make_obsparams_file(self, suffix):
        for i,cube in enumerate(self):
            cube.make_obsparams_file(suffix, i)

    #===========================================================================
    #TODO: as Mixin ???
    def magic_filenames( self, reduction_path='', sep='.', extension='.fits' ):
        '''Generates a unique sequence of filenames based on the name_dict.'''
        
        self.set_name_dict()
        
        #re pattern matchers
        #matches the optional keys sections (including square brackets) in the 
        #format specifier string from the args.names namespace
        opt_pattern = '\[[^\]]+\]'                      
        opt_matcher = re.compile(opt_pattern)
        #matches the key (including curly brackets) and key (excluding curly 
        #brackets) for each section of the format string
        key_pattern = '(\{(\w+)\})'
        key_matcher = re.compile(key_pattern)
        
        #get format specification string from label
        for label in ('sci', 'bias', 'flat'):
            if label in self.label:
                #get the naming format string from the argparse namespace
                fmt_str = getattr(self.NAMES, label)
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
            
            badoptkeys = [key for _, key in key_matcher.findall(fmt_str)
                          if not (key in nd and nd[key])]
            #This checks whether the given key in the name format specifier should be used 
            #(i.e. if it has a corresponding entry in the shocCube instance's name_dict.
            #If one of the keys are unset in the name_dict, this optional key will be eliminated 
            #when generating the filename below.
            
            for opt_sec in opt_matcher.findall( fmt_str ):
                if (any(key in opt_sec for key in badoptkeys)
                    or any(key in opt_sec for key in non_unique_keys)):
                    fn = fn.replace(opt_sec, '')
                    #replace the optional sections which contain keywords that 
                    #are not in the corresponding name_dict and replace the 
                    #optional sections which contain keywords that don't 
                    #contribute to the uniqueness of the filename set
            nfns.append( fn.format( **nd ) )
        
        #eliminate square brackets
        filenames = [fn.replace('[','').replace(']','') for fn in nfns]
        #last resort append numbers to the non-unique filenames
        if len(set(filenames)) < len(set(self.filenames)):
            unique_fns, idx = np.unique(filenames, return_inverse=1)
            nfns = []
            for basename in unique_fns:
                count = filenames.count(basename) #number of files with this name
                if count>1:
                    padwidth = len(str(count))
                    g = FilenameGenerator(basename, padwidth=padwidth, sep='_',
                                          extension='' )
                    fns = list( g(count) )
                else:
                    fns = [basename]
                nfns += fns
            
            #sorts by index. i.e. restores the order of the original filename sequence
            _, filenames = sorter(idx, nfns)
        
        #create a FilenameGenerator for each stack
        for stack, fn in zip(self, filenames):
            padwidth = len( str(stack.shape[-1]) )
            
            stack.filename_gen = FilenameGenerator(fn, reduction_path, padwidth,
                                                   sep, extension )
        
        return filenames
    
    #===========================================================================
    def genie(self, i=None):
        ''' returns list of generated filename tuples for cubes up to file number i'''
        return list(itt.zip_longest(*[cube.filename_gen(i) for cube in self]))
    
    #===========================================================================
    def get_filenames(self, with_path=0, with_ext=1):
        '''filenames of run constituents.'''
        return [stack.get_filename(with_path, with_ext) for stack in self]
    
    #===========================================================================
    def export_filenames(self, fn):
        
        if not fn.endswith('.txt'): #default append '.txt' to filename
            fn += '.txt'
            
            
        print('\nWriting names of {} run to file {}...\n'.format(self.label, fn))
        with open(fn, 'w') as fp:
            for f in self.filenames:
                fp.write( f+'\n' )
                
    #===========================================================================
    def writeout(self, suffix, dryrun=0): #TODO:  INCORPORATE FILENAME GENERATOR
        fns = []
        for stack in self:
            fn_out = stack.get_filename(1,0, (suffix, 'fits')) #FILENAME GENERATOR????
            fns.append(fn_out)
            
            if not dryrun:
                print('\nWriting to file: {}'.format(os.path.basename(fn_out)) )
                stack.writeto(fn_out, output_verify='warn', clobber=True)
                
                #save the header as a text file
                headfile = stack.get_filename(1,0), (suffix, 'head' )
                print('\nWriting header to file: {}'.format(os.path.basename(headfile)) )
                stack[0].header.totextfile( headfile )
        
        return fns
    #===========================================================================
    def zipper(self, keys):
        if isinstance(keys, str):
            return keys, [getattr(s, keys) for s in self] #s.__dict__[keys]??
        elif len(keys)==1:
            return keys[0], [getattr(s, keys[0]) for s in self]
        else:
            return (tuple(keys), 
                    list(zip(*([getattr(s, key) for s in self] for key in keys))))
    
    #===========================================================================
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
            atdict[attrs[0]] = self
            self.sep_by = keys
        else:                                                     #binning is not the same for all the input cubes
            flag = 1
            for ats in atset:                                    #map unique attribute values to slices of run with those attributes
                l = np.array([attr==ats for attr in attrs])      #list for-loop needed for tuple attrs
                eq_run = self[l]                                   #shocRun object of images with equal key attribute
                eq_run.sep_by = keys
                atdict[ats] = eq_run                          #put into dictionary
                
        SR = StructuredRun(atdict)
        SR.sep_by = keys
        SR.label = self.label
        return SR, flag
     
        
    #===========================================================================
    def combine(self, func):
        ''' Median combines a list of frames (with equal binning) using pyfits
        
        Returns
        ------
        outname : user defined output name for combined frame (for this binning)
        master_flag : binary flag indicating whether user has input a master flat.
        '''
        def single_combine(ims):        #TODO MERGE WITH shocCube.combine????
            '''Combine a run consisting of single images.'''
            header = copy( ims[0][0].header )
            data = func([im[0].data for im in ims], 0)
                    
            header.remove('NUMKIN')
            header['NCOMBINE'] = ( len(ims), 'Number of images combined' )
            for i, im in enumerate(ims):
                imnr = '{1:0>{0}}'.format(3, i+1)         #Image number eg.: 001
                comment = 'Contributors to combined output image' if i==0 else ''
                header['ICMB'+imnr] = ( im.get_filename(), comment )
            
            #outname = next( ims[0].filename_gen() )  #uses the FilenameGenerator of the first image in the shocRun
            
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
            msg = ('\n\nYou have input a single image named: {}. '
                   'This image will be used as the master {} frame for {} binning.'
                  ).format(outname, self.label, self[0].binning)
            print( msg )
            combined.append(ims[0])
        
        elif len(ims):
            if args.combine == 'daily':   #THIS SHOULD BE FED AS AN ARGUMENT (ALONG WITH SIGMA)
                input('ERROR!! Ambiguity in combine.  The run should be date separated!')
            else:
                combined.append(single_combine(ims))
            
        for stack in stacks:
            combined.append(stack.combine(func))
        
        if len(combined) > 1:
            msg = ('Ambiguity in combine! Cannot discriminate between given '
                   'files based on header info alone. Please select the most '
                   'appropriate file among the following:')
            warn(msg)
            self.print_instrumental_setup()
            i = Input.str('Ix?', 0, check=lambda x: int(x) < len(combined), 
                          convert=int)
        else:
            i = 0
        
        return combined[i], master_flag
    
    #unpack datacube(s) and assign 'outname' to output images
    #if more than one stack is given 'outname' is appended with 'n_' where n is the number of the stack (in sequence) 
    #===========================================================================
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
    
    #===========================================================================
    def check(self, run2, keys, raise_error=0, match=0):
        '''
        check fits headers in this run agains run2 for consistency of key 
        (binning / instrument mode / dimensions / flip state /) 
        Parameters
        ----------
        keys :          The attribute(s) to be checked
        run2 :          shocRun Object to check against
        raise_error :   Should an error be raised upon mismatch
        match   :       If set, method returns boolean array that can be used to
                        filter unnecessary cubes. If unset method returns single 
                        boolean - False for complete match, True for any mismatch
        
        Returns
        ------
        flag :  key mismatch?
        '''
        
        #lists of attribute values (key) for given input lists
        keys, attr1 = self.zipper(keys)
        keys, attr2 = run2.zipper(keys)
        fn1 = np.array(self.get_filenames())
        fn2 = np.array(run2.get_filenames())
        
        #which of 1 occur in 2
        match1 = np.array([attr in attr2 for attr in attr1])    
        
        if set(attr1).issuperset( set(attr2) ):  
            #All good, run2 contains all the cubes with matching attributes
            if match:
                #use this to select out the minimum set of cubes needed (filter out unneeded cubes)
                return match1
            else:
                return False
                #NOTE: this returns the opposite of match. i.e. False --> the 
                # checked attributes all match #FIXME
        else:            #some attribute mismatch occurs
            #which of 2 occur in 1 (at this point we know that one of these values are False)
            match2 = np.array([attr in attr1 for attr in attr2])      
            if any(~match2):
                #FIXME:   ERRONEOUS ERROR MESSAGES!
                fns = ',\n\t'.join(fn1[~match1])
                badfns = ',\n\t'.join(fn2[~match2])
                mmvals = ' or '.join(set(np.fromiter(map(str, attr2), 'U64')[~match2]))                       #set of string values for mismatched attributes
                keycomb = ('{} combination' if isinstance(keys, tuple) else '{}').format(keys)
                operation = 'de-biasing' if 'bias' in self.label else 'flat fielding'
                desc = ('Incompatible {} in'
                        '\n\t{}'
                        '\nNo {} frames with {} {} for {}'
                        '\n\t{}'
                        '\n\n').format(keycomb, badfns, self.label, mmvals, 
                                       keycomb, operation, fns)
                #msg = '\n\n{}: {}\n'
                
                if raise_error==1:
                    raise ValueError('\n\nERROR! %s' %desc)
                elif raise_error==0:
                    warn( desc ) 
            
            if match:
                return match1       #np.array([attr in attr2 for attr in attr1])
            else:
                return True

    #===========================================================================
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
                sf_bins = [s.binning for s in science_run if s._is_subframed]
                #those binnings that have subframed cubes
                for cube in self:
                    if cube.binning in sf_bins:
                        cube._needs_sub = 1
        
    #===========================================================================
    #def set_airmass(self, coords=None):
        #for stack in self:
            #stack.set_airmass(coords)
            
    #===========================================================================
    def set_name_dict(self):
        for stack in self:
            stack.set_name_dict()

    #===========================================================================
    def close(self):
        [stack.close() for stack in self]
    

################################################################################
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

    
################################################################################
class StructuredRun(col.MutableMapping):
    """ 
    Emulates dict to hold multiple shocRun instances indexed by their attributes. 
    The attribute names given in sep_by are the ones by which the run is separated 
    into unique segments (also runs).
    """
    #===========================================================================
    def __init__(self, *args, **kwargs):
        self.data = dict()
        self.update( dict(*args, **kwargs) )  # use the free update to set keys
    
    #===========================================================================    
    def __getitem__(self, key):
        return self.data[key]
    
    #===========================================================================
    def __setitem__(self, key, value):
        self.data[key] = value
    
    #===========================================================================
    def __delitem__(self, key):
        del self.data[key]
    
    #===========================================================================
    def __iter__(self):
        return iter(self.data)
    
    #===========================================================================
    def __len__(self):
        return len(self.data)
    
    #===========================================================================
    def __repr__(self):
        print( 'REWORK REPR' )
        #embed()
        
        return '\n'.join( [ ' : '.join(map(str, x)) for x in self.items() ] )
    
    #===========================================================================
    def flatten(self):
        if isinstance( list(self.values())[0], shocCube ):
            run = shocRun( list(self.values()), label=self.label, sep_by=self.sep_by )
        else:
            run = shocRun(label=self.label).join( *self.values() )
        
        #eliminate duplicates
        _, idx = np.unique(run.get_filenames(), return_index=1)
        dup = np.setdiff1d( range(len(run)), idx )
        for i in reversed(dup):
            run.pop( i )
        
        return run
    
    #===========================================================================
    def writeout(self, suffix):
        return self.flatten().writeout( suffix )
    
    #===========================================================================
    def attr_sep(self, *keys):
        if self.sep_by == keys:
            return self
        
        return self.flatten().attr_sep( *keys )
        
    #===========================================================================
    def magic_filenames(self, reduction_path='', sep='.', extension='.fits'):
        return self.flatten().magic_filenames( reduction_path, sep, extension )
    
    #===========================================================================
    def compute_master(self, how_combine, mbias=None, load=0, w2f=1, outdir=None):
        '''
        Compute the master image(s) (bias / flat field)
        
        Parameters
        ----------
        mbias :    A StructuredRun instance of master biases (optional)
        load  :    If set, the master frames will be loaded as shocCubes.  If unset kept as filenames
        
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
            
            #if isinstance(run, shocRun):
            if run is None:                                     #Unmatched!
                masters[attrs] = None
                continue
            
            master, master_flag = run.combine(how_combine)                #master bias/flat frame for this attr val
            
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
                                        
                    ffmean = np.mean(master[0].data)                                                      #flat field mean
                    print('Normalising flat field...')
                    master[0].data /= ffmean                                                              #flat field normalization
                
                master.flush(output_verify='warn', verbose=1)      #writes full frame master
                
                #writes subframed master
                #submasters = [master.subframe(sub) for sub in stack._needs_sub] 
                
                #master.close()
            
            masters[attrs] =  master
            datatable.append( (master.get_filename(0,1),) + attrs )
            
        print()
            
        #Table for master frames
        bgcolours       = {'flat' : 'cyan', 'bias' : 'magenta', 'sci' : 'green'}
        title           = 'Master {} frames:'.format(self.label)
        title_props     = {'text':'bold', 'bg': bgcolours[self.label]}
        col_head        = ('Filename',) + tuple(map(str.upper, keys))
        table = sTable(datatable, title, title_props, col_headers=col_head)
        
        print(table)
        #TODO: STATISTICS????????
        
        if load:
            #this creates a run of all the master frames which will be split into individual 
            #shocCube instances upon the creation of the StructuredRun at return
            label = 'master {}'.format( self.label )
            mrun = shocRun(hdus=masters.values(), label=label)
        
        if w2f:
            fn = label.replace(' ','.')
            outname = os.path.join(outdir, fn)
            mrun.export_filenames(outname)
        
        #NOTE:  The dict here is keyed on the closest matching attributes in self!
        SR = StructuredRun( masters )           
        SR.sep_by = self.sep_by
        SR.label = self.label
        return SR
    
    #===========================================================================
    def subframe(self, c_sr):
        #Subframe
        print( 'sublime subframe' )
        #i=0
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
        
        newcals = shocRun(substacks, label=c_sr.label) + b
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
        
        #newcals = shocRun(substacks) + c_sr.flatten()
        
        #return newcals
    
################################################################################
class shocHeaderCard(pyfits.card.Card):
    '''Extend the pyfits.card.Card class for interactive user input'''
    
    #===========================================================================
    def __init__(self, keyword=None, value=None, comment=None, **kwargs):
        
        self.example = kwargs.pop('example', '')                      #Additional display information         #USE EXAMPLE OF inp.str FUNCTION
        self.askfor = kwargs.pop('askfor', True)                      #should the user be asked for input
        self.check = kwargs.pop('check', validity.trivial)             #validity test
        self.conversion = kwargs.pop('conversion', convert.trivial)    #string conversion
        self.default = ''
        
        if self.askfor:         #prompt the user for input
            value = Input.str(comment + self.example+': ', self.default, 
                              check=self.check, verify=False, what=comment, 
                              convert=self.conversion )#USE EXAMPLE OF _input.str FUNCTION
        elif self.check(value):
            value = self.conversion(value)
        else:
            raise ValueError('Invalid value %r for %s.' %(value, keyword))
        
        #print(value)
        pyfits.card.Card.__init__(self, keyword, value, comment, **kwargs)
        
        
class shocHeader(pyfits.Header):
    RNT = ReadNoiseTable()
    '''Extend the pyfits.Header class for interactive user input'''
    #===========================================================================
    def __init__(self, cards=[], txtfile=None):
        pyfits.Header.__init__(self, cards, txtfile)
        
        #self.not_asked_for = [card.keyword for card in cards]
    
    
    #===========================================================================
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
        
        
    #===========================================================================
    def set_ron_sens_date(self, header):    #GET A BETTER NAME!!!!!!!!!!!!
        '''set Readout noise, sensitivity, observation date in header.'''
        # Readout noise and Sensitivity as taken from ReadNoiseTable
        ron, sens, saturation = self.RNT.get_readnoise(header)
        
        self['RON']         = ron,                          'CCD Readout Noise'
        self['SENSITIV']    = sens,                         'CCD Sensitivity'                                                    # Images taken at SAAO observatory
        self['OBS-DATE']    = header['DATE'].split('T')[0], 'Observation date'
        #self['SATURATION']??
        
        return ron, sens, saturation
        
    #===========================================================================
    def check(self):
        '''check which keys actually need to be updated'''
        #pop = []                                                                                                            #list of indeces of header that will not need updating
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
        
    #===========================================================================
    def update_to(self, hdu):                                                        #OPTIONAL REFERENCE FITS FILE?.......
        ''' Updating header information
            INPUT: hdu - header data unit of the fits image
            OUTPUTS: NONE
        '''
        print('Updating header for {}.'.format(hdu.get_filename()))
        header = hdu[0].header
  
        for card in self.cards:
            #print( card )
            header.set(card.keyword, card.value, card.comment)                                                               #If  the  field  already  exists  it  is edited
        
        hdu.flush(output_verify='warn', verbose=1)                      #SLOW!! AND UNECESSARY!!!!
         
            
        #hdu.close()                                                             #HMMMMMMMMMMMMMMM
        #UPDATE HISTORY / COMMENT KEYWORD TO REFLECT CHANGES
        
    #===========================================================================
    #def convert_old_new(self, forward=True):
        #from pySHOC.convert_keywords import KEYWORDS
        
        #for old, new in KEYWORDS:
            #try:
                #if forward:
                    #hdu.header.rename_keyword(old, new)
                #else:
                    #hdu.header.rename_keyword(new, old)
            #except ValueError as e:
                #print(e)
            