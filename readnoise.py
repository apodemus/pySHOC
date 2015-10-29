import numpy as np
import pyfits
from io import BytesIO
import textwrap
import collections as coll

from IPython import embed

#****************************************************************************************************
class ReadNoiseTable( np.ndarray ):
    '''Readout Noise Table for SHOC.'''
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __new__(cls):
        
        data = cls.get_table()
        obj = np.asarray(data).view(cls)
        
        return obj
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @classmethod
    def get_table(cls, form='m'):
        #table in human readable form
        table = '''
            #SerNo      Freq    Scope   Mode                    PreAmp  Sensitivity     ReadNoise       ReadTime        Saturation
            6448        10MHz   14-bit  Electron Multiplying    2.5     21.97           52.07           1.0E-07         3603
            6448        10MHz   14-bit  Electron Multiplying    5.2     10.05           46.33           1.0E-07         7877
            6448        5MHz    14-bit  Electron Multiplying    1       44.92           69.63           5.0E-06         1762
            6448        5MHz    14-bit  Electron Multiplying    2.5     19.24           45.02           5.0E-06         4114
            6448        5MHz    14-bit  Electron Multiplying    5.2     8.6             35.69           5.0E-06         9205
            6448        3MHz    14-bit  Conventional            1       9.92            13.99           3.0E-06         7980
            6448        3MHz    14-bit  Conventional            2.5     3.96            10.85           3.0E-06         16384
            6448        3MHz    14-bit  Conventional            5.2     1.77            9.79            3.0E-06         16384
            6448        3MHz    14-bit  Electron Multiplying    1       44.32           51.85           3.0E-06         1786
            6448        3MHz    14-bit  Electron Multiplying    2.5     19.07           33.18           3.0E-06         4151
            6448        3MHz    14-bit  Electron Multiplying    5.2     8.59            26.29           3.0E-06         9216
            6448        1MHz    16-bit  Electron Multiplying    1.      18.61           32.20           1.0E-06         4254
            6448        1MHz    16-bit  Electron Multiplying    2.5     7.43            19.62           1.0E-06         10655
            6448        1MHz    16-bit  Electron Multiplying    5.2     3.39            16.54           1.0E-06         23353
            6448        1MHz    16-bit  Conventional            1       3.79            8.22            1.0E-06         20888
            6448        1MHz    16-bit  Conventional            2.5     1.53            6.52            1.0E-06         51744
            6448        1MHz    16-bit  Conventional            5.2     0.68            6.03            1.0E-06         65536
            5982        10MHz   14-bit  Electron Multiplying    2.4     24.35           60.14           1.0E-07         5907
            5982        10MHz   14-bit  Electron Multiplying    4.9     11.57           51.14           1.0E-07         12431
            5982        5MHz    14-bit  Electron Multiplying    1       52.4            82.27           5.0E-06         2745
            5982        5MHz    14-bit  Electron Multiplying    2.4     20.74           48.53           5.0E-06         6935
            5982        5MHz    14-bit  Electron Multiplying    4.9     9.54            39.59           5.0E-06         15077
            5982        3MHz    14-bit  Conventional            1       10.98           15.81           3.0E-06         13100
            5982        3MHz    14-bit  Conventional            2.4     4.23            11.59           3.0E-06         16384
            5982        3MHz    14-bit  Conventional            4.9     1.82            10.19           3.0E-06         16384
            5982        3MHz    14-bit  Electron Multiplying    1       51.4            61.17           3.0E-06         2798
            5982        3MHz    14-bit  Electron Multiplying    2.4     19.71           34.3            3.0E-06         7297
            5982        3MHz    14-bit  Electron Multiplying    4.9     9.49            29.99           3.0E-06         15156
            5982        1MHz    16-bit  Electron Multiplying    1       19.14           33.69           1.0E-06         7515
            5982        1MHz    16-bit  Electron Multiplying    2.4     7.48            19.75           1.0E-06         19229
            5982        1MHz    16-bit  Electron Multiplying    4.9     3.60            18.0            1.0E-06         39955
            5982        1MHz    16-bit  Conventional            1       4.06            9.3             1.0E-06         35428
            5982        1MHz    16-bit  Conventional            2.4     1.69            7.49            1.0E-06         65536
            5982        1MHz    16-bit  Conventional            4.9     0.63            5.84            1.0E-06         65536'''
        
        if form=='h':
            return table
        else:
            #Convert table to machine readible form
            buffer = BytesIO()
            for line in textwrap.dedent(table).split('\n'):
                linedata = filter(None, line.split( '  ' ))
                outline = '\t'.join(linedata)
                if outline:
                    buffer.write( str.encode(outline+'\n') )
            buffer.seek(0)
            
            return np.genfromtxt( buffer, dtype=None, names=True, delimiter='\t' )
            
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def get_readnoise(self, fn_or_header):
        '''get the read out noise and sensitivity from the read noise table given the fits header'''
        if isinstance(fn_or_header, pyfits.header.Header):
            header = fn_or_header
        else:
            with open(fn_or_header,'rb') as fp:
                header = pyfits.Header.fromfile( fp )

        #CCD acquisition mode
        mode = header['OUTPTAMP']
        lmode = self['Mode'] == mode.encode()

        #Readout clock frequency
        freq = 1. / header['READTIME']
        freq_MHz = str(round(freq/1.e6)) +'MHz'                     #The integer frequency in MHz as a string
        lfreq = self['Freq'] == freq_MHz.encode()

        #Preamp gain setting
        preamp = header['PREAMP']
        lpreamp = self['PreAmp'] == preamp

        #serial number (SHOC 1 or 2)
        serno = header['SERNO']
        lserno = self['SerNo'] == serno
        
        return self._from_bools(lmode, lfreq, lpreamp, lserno)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def get_readnoise_from_kw(self, **kw):
        
        raise NotImplementedError
    
        mode_conversion = { 'EM' : 'Electron Multiplying',
                            'CON' : 'Conventional' }
        serno_conv = { 1 : 6448,
                       2 : 5982 }
        #TODO: Convert keywords to lower case
        mode    = kw.get('mode')
        if mode in mode_conversion:
            mode = mode_conversion[mode].encode()
        
        serno           =       serno_conv[ kw.get('shoc') ]
        
        preamp          =       kw.get('preamp')
        
        freq            =       kw.get('freq')
        freq_MHz        =       str(int(freq)) + 'MHz'
        
        lfreq   = self['Freq']   == freq_MHz.encode()
        lpreamp = self['PreAmp'] == preamp
        lserno  = self['SerNo']  == serno
        
        return self._from_bools(lfreq, lpreamp, lserno)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _from_bools(self, *ls):
        l = np.all(ls, 0)          #Boolean array used to determine readout noise value from the table
        if sum(l) > 1:
            raise ValueError('WARNING!!!!!   Read noise value not uniquely determined!!!')                 #this should never happen

        saturation = self['Saturation'][l][0]
        ron = self['ReadNoise'][l][0]
        sens = self['Sensitivity'][l][0]

        return ron, sens, saturation
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def get_saturation(self, fn_fits):
        return self.get_readnoise(fn_fits)[-1]
        
#****************************************************************************************************