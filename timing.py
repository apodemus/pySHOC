from astropy.time import Time
import numpy as np

#from myio import warn
from warnings import warn
#from superstring import ProgressBar

from decor.profile import profile
profiler = profile()


def get_updated_iers_table(cache=True, verbose=True):
    '''Get updated IERS data'''
    if verbose:
        print( 'Updating IERS table...', )
    
    #import IERS data class
    from astropy.utils.iers import IERS_A, IERS_A_URL
    from astropy.utils.data import download_file
    
    #get IERS data tables from cache / download
    iers_a_file = download_file(IERS_A_URL, cache=cache)
    iers_a = IERS_A.open(iers_a_file)                       #load data tables
    
    if verbose:
        print( 'Done' )
    
    return iers_a
        

import spiceypy as spice
from astropy.coordinates import ICRS, FK5
from astropy.constants import c, G, M_sun
#M_sun = M_sun.value     #Solar mass (kg)


#====================================================================================================
#@profiler.histogram

#TODO: Check the accuracy of these routines against astropy, utc2bjd, etc....

def light_time_corrections(t, coords, precess=False, abcorr=None):
    '''
    Barycentric julian day.  Corrections done for Romer, Einstein and Shapiro
    delays.
    
    Params
    ------
    coords - SkyCoord
    
    precess - whether or not to precess coordinates
    
    abcorr - how (if) aberation corrections should be done
    
    Aberation corrections: (from the spice spice.spkpos readme)
    
    "Reception" case in which photons depart from the target's location at 
    the light-time corrected epoch et-lt and *arrive* at the observer's 
    location at `et'
    "LT"    Correct for one-way light time (also called "planetary aberration")
            using a Newtonian formulation.
    LT+S"   Correct for one-way light time and stellar aberration using a 
            Newtonian formulation.
    "CN"    Converged Newtonian light time correction.  In solving the light
            time equation, the "CN" correction iterates until the solution 
            converges
    CN+S"   Converged Newtonian light time and stellar aberration corrections.'
    
    "Transmission" case in which photons *depart* from the observer's location
    at `et' and arrive at the target's location at the light-time corrected 
    epoch et+lt ---> prepend 'X' to the description strings as given above.
    
    Neither special nor general relativistic effects are accounted for in 
    the aberration corrections applied by this routine.
    
    '''

    tabc = ['NONE', 'LT', 'LT+S', 'CN', 'CN+S']
    rabc = ['X'+s for s in tabc]
    allowed_abc = tabc+rabc
    abcorr = str(abcorr).replace(' ','').upper()
    if abcorr in allowed_abc:
        ABCORR = abcorr
    else:
        warn('Aberation correction specifier {} not understood. '
             'Next time use one of {}.\n'
             'No aberration correction(s) will be done. ' 
             '(Abberation effects should be negligable anyway!)'
             ''.format(abcorr, allowed_abc) ) 
        

    #TODO: do this when loading the module
    #load kernels:
    SPK = '/home/hannes/repos/SpiceyPy/kernels/de430.bsp'          #ephemeris kernel : https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/
    #TODO: automate finding latest kernels:  http://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/
    LSK = '/home/hannes/repos/SpiceyPy/kernels/naif0012.tls'       #leap second kernel
    spice.furnsh(SPK)
    spice.furnsh(LSK)
    
    xyzObj = get_Obj_coords(t, coords, precess=precess)
    xyzSun, xyzEarth = get_Earth_Sun_coords(t[:1], ABCORR)  #HACK for speed!!!!
    
    rd = romer_delay(xyzEarth, xyzObj)
    ed = einstein_delay(t.tt.jd)
    shd = shapiro_delay(xyzEarth, xyzSun, xyzObj)
    
    #print( 'Romer delay', rd )
    #print( 'Einstein delay', ed )
    #print( 'Shapiro delya', shd )
    
    return t.tdb.jd - rd - ed - shd
    
    
#====================================================================================================
#@profile()
def get_Earth_Sun_coords(t, ABCORR):
    '''
    Get Earth (Geocentric) and solar (Heliocentric) ephemeris (km) relative to solar
    system Barycenter at times t
    '''
    FRAME = 'J2000'
    OBSERVER = '0'                            #Solar System Barycenter (SSB)
    
    N = len(t)
    #TODO: vectorize??
    xyzEarth = np.empty((N, 3), np.float128)
    xyzSun = np.empty((N, 3), np.float128)
    ltEarth = np.empty(N)
    ltSun = np.empty(N)
    for i, t in enumerate(t):
        #Ephemeris time (seconds since J2000 TDB)
        et = spice.utc2et(str(t))
        #Earth geocenter wrt SSB in J2000 coordiantes
        xyzEarth[i], ltEarth[i] = spice.spkpos(OBSERVER, et, FRAME, ABCORR, 'earth')
        #Sun heliocenter wrt SSB in J2000 coordiantes
        xyzSun[i], ltSun[i]   = spice.spkpos(OBSERVER, et, FRAME, ABCORR, '10')

    return xyzSun, xyzEarth


#def precess(coords, t, every):
    

#====================================================================================================
def get_Obj_coords(t, coords, precess=False):
    ''' '''
    if not precess:
        return coords.cartesian.xyz
    
    if isinstance(precess, str):
        if precess.lower() == 'first':
            precessed = coords.transform_to(FK5(equinox=t[0]))
            return precessed.cartesian.xyz
        
        #TODO: EVERY
        #TODO: INTERPOLATION?
    
    N = len(t)
    xyzObj = np.empty((N, 3), np.float128)
    
    
    if precess is True:
        #TODO: MULTIPROCESS FOR SPEED!!!!!?????
        for i, t in enumerate(t):
            #bar.progress(i)
            #precess to observation date and time and then transform back to FK5 (J2000)
            coonew = coords.transform_to(FK5(equinox=t))
            xyzObj[i] = coonew.cartesian.xyz

        return xyzObj
    
    else:
        raise ValueError

#====================================================================================================
def romer_delay(xyzEarth, xyzObj):
    '''
    Calculate Rømer delay (classical light travel time correction) in units of days
    
    Notes:
    ------
    https://en.wikipedia.org/wiki/Ole_R%C3%B8mer
    https://en.wikipedia.org/wiki/R%C3%B8mer%27s_determination_of_the_speed_of_light
    
    '''
    #ephemeris units is in km / s .   convert to m / (julian) day
    convf = c.to('km/day').value
    delay = (xyzEarth * xyzObj).sum(1) / convf
    
    return delay

#====================================================================================================    
def einstein_delay(jd_tt):
    '''
    Calculate Eistein delay in units of days
    '''
    red_jd_tt = jd_tt - 2451545.0
    g = np.radians(357.53 + 0.98560028 * red_jd_tt)       #mean anomaly of Earth                  
    L_Lj = np.radians(246.11 + 0.90251792*red_jd_tt)      #Difference in mean ecliptic longitudea of the Sun and Jupiter
    delay = 0.001657 * np.sin(g) + 0.000022 * np.sin(L_Lj)
    
    return delay / 86400.

#====================================================================================================
def shapiro_delay(xyzEarth, xyzSun, xyzObj):
    '''
    Calculate Shapiro delay in units of days
    
    https://en.wikipedia.org/wiki/Shapiro_delay
    '''
    Earth_Sun = xyzEarth - xyzSun                           #Earth to Sun vector
    d_Earth_Sun = np.linalg.norm(Earth_Sun, axis=1)[None].T   #Earth to Sun distance
    u_Earth_Sun = Earth_Sun / d_Earth_Sun               #Earth to Sun unit vector
    
    #dot product gives cosine of angle between Earth and Sun
    cosTheta = (u_Earth_Sun * xyzObj).sum(1)
    
    #Approximate for Shapiro delay
    delay = (2*G*M_sun/c**3) * np.log(1-cosTheta)
    
    #print(cosTheta)
    #print(2*G*M_sun/c**3)
    
    return delay.to('day').value

#====================================================================================================   
#NOTE:  Despite these corrections, there is still a ~0.02s offset between the BJD_TDB computed by this code and the IDL code http://astroutils.astronomy.ohio-state.edu/time/
       



class Time(Time):
    #TODO: HJD:
    
    #TODO: phase method...
    
    '''
    Extends the astropy.time.core.Time class to include method that returns the time in hours.
    
    The astropy.time package provides functionality for manipulating times and dates. Specific emphasis
    is placed on supporting time scales (e.g. UTC, TAI, UT1) and time representations (e.g. JD, MJD,
    ISO 8601) that are used in astronomy. It uses Cython to wrap the C language ERFA time and calendar
    routines. All time scale conversions are done by Cython vectorized versions of the ERFA routines 
    and are fast and memory efficient.

    All time manipulations and arithmetic operations are done internally using two 64-bit floats to 
    represent time. Floating point algorithms from [1] are used so that the Time object maintains 
    sub-nanosecond precision over times spanning the age of the universe.
    
    [1]     Shewchuk, 1997, Discrete & Computational Geometry 18(3):305-363
    '''
    #====================================================================================================
    #@property
    def isosplit(self):
        '''Split ISO time between date and time (from midnight)'''
        splitter = lambda x: tuple(x.split('T'))
        dtype = [('utdate','U20'), ('utc','U20')]
        utdata = np.fromiter(map(splitter, self.utc.isot), dtype)
        return utdata
    
    ##====================================================================================================
    ##@property
    #def hours(self):
        #'''Converts time to hours since midnight.'''
        #vals = np.atleast_1d( self.iso )
        #hours = np.empty(vals.shape)
        #for j,val in enumerate(vals):
            #_, hms = val.split()
            #h,m,s = map( float, hms.split(':') )
            #hours[j] = h + m/60. + s/3600.
        
        ##if len(hours)==1:
            ##return hours[0]
        ##else:
        #return hours
    
    ##====================================================================================================
    #@property
    #def sec(self):
        #return self.hours * 3600.
    
    ##====================================================================================================
    #@property
    #def radians(self):
        #return np.radians( self.hours*15. )

    #====================================================================================================
    def check_iers_table(self):
        '''
        For the UT1 to UTC offset, one has to interpolate in observed values provided by the International Earth Rotation and Reference Systems Service. By default, astropy is shipped with the final values provided in Bulletin B, which cover the period from 1962 to shortly before an astropy release, and these will be used to compute the offset if the delta_ut1_utc attribute is not set explicitly. For more recent times, one can download an updated version of IERS B or IERS A (which also has predictions), and set delta_ut1_utc as described in get_delta_ut1_utc
        '''
        from astropy.utils.iers import (TIME_BEFORE_IERS_RANGE,
                                        TIME_BEYOND_IERS_RANGE)
        delta, status = self.get_delta_ut1_utc(return_status=True)
        beyond = status == TIME_BEYOND_IERS_RANGE
        if np.any( beyond ):
            warn('{} / {} times are outside of range covered by IERS table.'
                ''.format(beyond.sum(), len(beyond)))
        return beyond
    
    #====================================================================================================
    #@profile()
    #def bjd(self, coords, precess='first', abcorr=None): #TODO:  OO

 
        #xyzObj = get_Obj_coords(self, coords, precess=precess)
        #xyzSun, xyzEarth = get_Earth_Sun_coords(self)
        
        #rd = romer_delay(xyzEarth, xyzObj)
        #ed = einstein_delay(self.tt.jd)
        #shd = shapiro_delay(xyzEarth, xyzSun)
        
        ##print( 'Romer delay', rd )
        ##print( 'Einstein delay', ed )
        ##print( 'Shapiro delya', shd )
        
        #return self.tdb.jd - rd - ed - shd
        
    #====================================================================================================
    #def __add__(self, other):
        #print( 'INTERCEPTING ADD' )
        #ts = super(Time, self).__add__(other)
        #format = ts.format
        #scale = ts.scale
        #precision = ts.precision
        
        #embed()
        
        #t = Time(ts.jd1, ts.jd2, format=ts.format, scale=scale, precision=precision, copy=False, location=self.location)  #gives it the additional methods defined above
        #return getattr(t.replicate(format=format), scale)
        
    ##====================================================================================================
    #def __sub__(self, other):
        #print( 'INTERCEPTING SUB!' )
        #ts = super(Time, self).__sub__(other)
        #format = ts.format
        #scale = ts.scale
        #precision = ts.precision
        #t = Time(ts.jd1, ts.jd2, format=ts.format, scale=scale, precision=precision, copy=False, location=self.location)  #gives it the additional methods defined above
        #return getattr(t.replicate(format=format), scale)
        
    #====================================================================================================    


def utc2bjd(times, coords):
    '''use the html form at the url to convert to BJD_TDB'''
    
    import urllib.request, urllib.parse
    import re
    
    urlbase = 'http://astroutils.astronomy.ohio-state.edu/time/utc2bjd'
    html = urlbase + '.html'
    form = urlbase + '.php?'
    
    #data for the php form
    payload = dict(observatory = 'saao', 
                   raunits = 'hours', 
                   spaceobs = 'none')
    payload['ra'] = coords.ra.to_string('h', sep=' ')
    payload['dec'] = coords.dec.to_string(sep=' ')
    
    #encode form data
    params = urllib.parse.urlencode(payload)
    
    #web form can only handle 1e4 times simultaneously
    if len(t) > 1e4:
        'TODO' #split time array into chunks
        
    #encode the times to url format str (urllib does not success convert newlines appropriately)
    newline, spacer, joiner = '%0D%0A', '+',  '&'         #html POST code translations
    fixes = ('-', spacer), (':', spacer), (' ', spacer)
    ts = newline.join(t.iso)
    for fix in fixes:
        ts = ts.replace(*fix)
    
    #append times to url data
    params = (params + '&jds=' + ts).encode()
    
    #submit the form
    req = urllib.request.Request(form)
    req.add_header('Referer', html)
    r = urllib.request.urlopen(req, params)

    #parse the returned data
    jdpatr = '(\d{7}\.\d{9})\n'
    matcher = re.compile(jdpatr)
    jds = matcher.findall(raw.decode())
    
    #convert back to time object
    bjd_tdb = Time(np.array(jds, float), format='jd')
    
    return bjd_tdb    
        