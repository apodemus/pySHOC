from astropy.time import Time
import numpy as np

from myio import warn
from superstring import ProgressBar

#from decor import profile

class Time(Time):
    #TODO: HJD:
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
    @property
    def hours(self):
        '''Converts time to hours since midnight.'''
        vals = np.atleast_1d( self.iso )
        hours = np.empty(vals.shape)
        for j,val in enumerate(vals):
            _, hms = val.split()
            h,m,s = map( float, hms.split(':') )
            hours[j] = h + m/60. + s/3600.
        
        #if len(hours)==1:
            #return hours[0]
        #else:
        return hours
    
    #====================================================================================================
    @property
    def sec(self):
        return self.hours * 3600.
    
    #====================================================================================================
    @property
    def radians(self):
        return np.radians( self.hours*15. )
    
    #====================================================================================================
    @staticmethod
    def get_updated_iers_table( cache=True ):
        '''Get updated IERS data'''
        print( 'Updating IERS table...', )
        from astropy.utils.iers import IERS_A, IERS_A_URL       #import IERS data class
        from astropy.utils.data import download_file
        iers_a_file = download_file(IERS_A_URL, cache=cache)     #get IERS data tables from cache / download
        iers_a = IERS_A.open(iers_a_file)                       #load data tables
        print( 'Done' )
        return iers_a
        
    #====================================================================================================
    def check_iers_table(self):
        '''
        For the UT1 to UTC offset, one has to interpolate in observed values provided by the International Earth Rotation and Reference Systems Service. By default, astropy is shipped with the final values provided in Bulletin B, which cover the period from 1962 to shortly before an astropy release, and these will be used to compute the offset if the delta_ut1_utc attribute is not set explicitly. For more recent times, one can download an updated version of IERS B or IERS A (which also has predictions), and set delta_ut1_utc as described in get_delta_ut1_utc
        '''
        from astropy.utils.iers import TIME_BEFORE_IERS_RANGE, TIME_BEYOND_IERS_RANGE
        delta, status = self.get_delta_ut1_utc( return_status=True )
        beyond = status == TIME_BEYOND_IERS_RANGE
        if np.any( beyond ):
            warn( '{} / {} times are outside of range covered by IERS table.'.format(beyond.sum(), len(beyond)) )
        return beyond
    
    #====================================================================================================
    #@profile()
    def bjd(self, coo, precess=1, abcorr=None): #TODO:  OO
        '''
        Barycentric julian day.  Corrections done for Romer, Einstein and Shapiro delays.
        '''
        from spice import furnsh, utc2et, spkpos
        from astropy.coordinates import ICRS, FK5
        from scipy.constants import c                                   #Speed of light
        from scipy.constants import gravitational_constant as G
        from astropy.constants import M_sun                             #Solar mass (kg)
        M_sun = M_sun.value
        
        #Aberation corrections: (from the spice spkpos readme)
        #"Reception" case in which photons depart from the target's location at the light-time corrected epoch 
        #           et-lt and *arrive* at the observer's location at `et'
        #"LT"       Correct for one-way light time (also called "planetary aberration") using a Newtonian formulation.
        #LT+S"      Correct for one-way light time and stellar aberration using a Newtonian formulation.
        #"CN"       Converged Newtonian light time correction.  In solving the light time equation, the "CN" correction iterates until the solution converges
        #CN+S"      Converged Newtonian light time and stellar aberration corrections.'
        #"Transmission" case in which photons *depart* from the observer's location at `et' and arrive at the 
        #       target's location at the light-time corrected epoch et+lt ---> prepend 'X' to the description strings as given above.
        #Neither special nor general relativistic effects are accounted for in the aberration corrections applied by this routine.
        tabc = ['NONE', 'LT', 'LT+S', 'CN', 'CN+S']
        rabc = ['X'+s for s in tabc]
        allowed_abc = tabc+rabc
        abcorr = str(abcorr).replace(' ','').upper()
        if abcorr in allowed_abc:
            ABCORR = abcorr
        else:
            warn( 'Aberation correction specifier {} not understood. Next time use one of {}.'.format(abcorr, allowed_abc) )
            warn( "No aberration correction(s) will be done. (Abberation effects should be negligable anyway!)" ) 
            
        FRAME = 'J2000'
        OBSERVER = '0'                                                  #Solar System Barycenter (SSB)
        
        #load kernels
        SPK = "/home/hannes/Downloads/SPICE/kernels/de430.bsp"          #ephemeris kernel
        LSK = "/home/hannes/Downloads/SPICE/kernels/naif0010.tls"       #leap second kernel
        furnsh( SPK )
        furnsh( LSK )
        
        #====================================================================================================
        #@profile()
        def get_Earth_Sun_coords():
            '''
            Get Earth (Geocentric) and solar (Heliocentric) ephemeris (km) relative to solar
            system Barycenter
            '''
            N = len(self)
            xyzEarth, ltEarth = np.empty( (N, 3), np.float128 ), np.empty( N )
            xyzSun, ltSun = np.empty( (N, 3), np.float128 ), np.empty( N )
            for i, t in enumerate(self):
                et = utc2et( str(t) )                                                            #Ephemeris time (seconds since J2000 TDB)
                xyzEarth[i], ltEarth[i] = spkpos( OBSERVER,  et,  FRAME,  ABCORR,  'earth' )     #Earth geocenter wrt SSB in J2000 coordiantes
                xyzSun[i]  , ltSun[i]   = spkpos( OBSERVER,  et,  FRAME,  ABCORR,  '10' )       #Sun heliocenter wrt SSB in J2000 coordiantes

            return xyzSun, xyzEarth

        #====================================================================================================
        def get_Obj_coords(coo, precess=0):

            N = len(self)
            xyzObj = np.empty( (N, 3), np.float128 )
            if precess:         #TODO: OPTION FOR FIRST, or ALL
                bar = ProgressBar()
                bar.create( len(self) )
                #TODO: MULTIPROCESS!!!!!?????
                for i, t in enumerate(self):
                    bar.progress(i)
                    coonew = coo.transform_to( FK5(equinox=t) )              #precess to observation date and time and then transform back to FK5 (J2000)
                    xyzObj[i] = coonew.cartesian.xyz
            else:
                xyzObj = coo.cartesian.xyz

            return xyzObj

        #====================================================================================================
        def romer_delay():

            convf = 1e3 / 86400. / c                                                 #ephemeris units is in km / s -->convert to m / (julian) day
            delay = convf*(xyzEarth * xyzObj).sum(1)
            return delay

        #====================================================================================================    
        def einstein_delay(jd_tt) :
            red_jd_tt = jd_tt - 2451545.0
            g = np.radians( 357.53 + 0.98560028*red_jd_tt )                         #mean anomaly of Earth
            L_Lj = np.radians( 246.11 + 0.90251792*red_jd_tt )                            #Difference in mean ecliptic longitudea of the Sun and Jupiter
            delay = 0.001657 * np.sin(g) + 0.000022 * np.sin( L_Lj )
            return delay / 86400.

        #====================================================================================================
        def shapiro_delay():
            Earth_Sun = xyzEarth - xyzSun                       #Earth to Sun vector
            d_Earth_Sun = np.linalg.norm( Earth_Sun, axis=1 )[None].T   #Earth to Sun distance
            u_Earth_Sun = Earth_Sun / d_Earth_Sun               #Earth to Sun unit vector

            cosTheta = (u_Earth_Sun * xyzObj).sum(1)            #dot product gives cosine of angle between 

            delay = 2*(G*M_sun/c**3)*np.log(1-cosTheta)                    #Approximation for Shapiro delay
            return delay / 86400.

        #====================================================================================================   
        #NOTE:  Despite these corrections, there is still a ~0.02s offset between the BJD_TDB computed by this code and the IDL code http://astroutils.astronomy.ohio-state.edu/time/
        
        xyzObj = get_Obj_coords( coo, precess=precess )
        xyzSun, xyzEarth = get_Earth_Sun_coords( )
        
        rd = romer_delay()
        ed = einstein_delay( self.tt.jd )
        shd = shapiro_delay()
        
        #print( 'Romer delay', rd )
        #print( 'Einstein delay', ed )
        #print( 'Shapiro delya', shd )
        
        return self.tdb.jd - rd - ed - shd
        
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
    
        