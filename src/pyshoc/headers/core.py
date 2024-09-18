"""
Functions for working with FITS headers
"""


from loguru import logger
from astropy.io.fits import Header


# ---------------------------------------------------------------------------- #
HEADER_KEYS_MISSING_OLD = \
    [
        'OBJECT',
        'OBJEPOCH',
        # 'OBJEQUIN',  # don't need both 'OBJEPOCH' and 'OBJEQUIN'
        'OBJRA',
        'OBJDEC',
        'OBSERVER',
        'OBSTYPE',
        'DATE-OBS',

        # 'TELESCOP',
        # 'TELFOCUS',
        # 'TELRA',
        # 'TELDEC',
        # 'INSTRUME',
        # 'INSTANGL',
        #
        # 'WHEELA',
        # 'WHEELB',
        # 'DATE-OBS',
        # 'GPS-INT',
        # 'GPSSTART',
        #
        # 'HA',
        # 'AIRMASS',
        # 'ZD',

        # 'DOMEPOS',  # don't know post facto

        # # Spectrograph stuff
        # 'ESHTMODE',
        # 'INSTSWV',
        # 'NSETHSLD',
        # 'RAYWAVE',
        # 'CALBWVNM',

    ]


# ---------------------------------------------------------------------------- #

class Header(Header):
    """Extend the pyfits.Header class for interactive user input"""

    def __init__(self, cards=(), copy=False):
        super().__init__(cards, copy)

    def needs_update(self, info):
        """check which keys actually need to be updated"""
        to_update = {}
        for key, val in info.items():
            if self.get(key) != val:
                to_update[key] = val
            else:
                logger.debug('{!r} will not be updated.', key)
        return to_update

    # def get_readnoise(self):
    #     """
    #     Readout noise, sensitivity, saturation as taken from ReadNoiseTable
    #     """
    #     from pyshoc import readNoiseTable
    #     return readNoiseTable.get_readnoise(self)
    #
    # def get_readnoise_dict(self, with_comments=False):
    #     """
    #     Readout noise, sensitivity, saturation as taken from ReadNoiseTable
    #     """
    #     data = self.get_readnoise()
    #     keywords = 'RON', 'SENSITIV', 'SATURATE'
    #     if with_comments:
    #         comments = ('CCD Readout Noise', 'CCD Sensitivity',
    #                     'CCD saturation counts')
    #         data = zip(data, comments)
    #     return dict(zip(keywords, data))
    #
    # def set_readnoise(self):
    #     """set Readout noise, sensitivity, observation date in header."""
    #     # Readout noise and Sensitivity as taken from ReadNoiseTable
    #     ron, sensitivity, saturation = self.readNoiseTable.get_readnoise(self)
    #
    #     self['RON'] = (ron, 'CCD Readout Noise')
    #     self['SENSITIV'] = sensitivity, 'CCD Sensitivity'
    #     # self['OBS-DATE'] = header['DATE'].split('T')[0], 'Observation date'
    #     # self['SATURATION']??
    #     # Images taken at SAAO observatory
    #
    #     return ron, sensitivity, saturation
