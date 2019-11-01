from obstools.fastfits import parse_header
import re

from recipes.logging import LoggingMixin

from .header import HEADER_KEYS_MISSING_OLD
from astropy.io.fits import Header, BITPIX2DTYPE
import numpy as np

# Regex to match end of FITS header
SRE_END = re.compile(rb'END {77}\s*')
# TODO: check if there is any performance diff in using the astropy fits regex

# Regex to identify type of observation bias / dark / flats / science
SRE_OBSTYPE = re.compile(r"""
(?:OBSTYPE)     # keywords (non-capture group if `capture_keywords` False)
\s*?=\s+        # (optional) whitespace surrounding value indicator '='
'?([^'\s/]*)'?  # value associated with any of the keys (un-quoted)
""", re.VERBOSE)


def _get_hdu_class(hdr):
    m = SRE_OBSTYPE.search(hdr)
    if m is None:
        return shocOldHDU

    obs_type = m.group(1)#.decode()
    if 'bias' in obs_type:
        return shocBiasHDU

    if 'flat' in obs_type:
        return shocFlatHDU

    for key in HEADER_KEYS_MISSING_OLD:
        if key not in hdr:
            return shocOldHDU

    return shocNewHDU


class shocHDU(LoggingMixin):
    """
    Base for faster initialization and data read access for SHOC fits files
    """

    @classmethod
    def load(cls, filename):
        """
        Load data from file(s).

        Parameters
        ----------
        filename

        Returns
        -------

        """

        hdr, data_start_bytes, mm = parse_header(filename)
        return _get_hdu_class(hdr)(filename, hdr, data_start_bytes)

    def __init__(self, filename, hdr=None, data_start_bytes=None):

        if (hdr is None) and (data_start_bytes is None):
            hdr, data_start_bytes, mm = parse_header(filename)

        # read header
        self.filename = str(filename)
        self.header = hdr = Header.fromstring(hdr)

        # check if data is 3D
        n_dim = hdr['NAXIS']
        if n_dim not in (2, 3):
            raise TypeError('%r only accepts 2D or 3D data!'
                            % self.__class__.__name__)

        # figure out the size of a data block
        bits_per_pixel = hdr['BITPIX']
        dtype = np.dtype(BITPIX2DTYPE[bits_per_pixel]).newbyteorder('>')
        shape = (hdr.get('NAXIS3', 1), hdr['NAXIS2'], hdr['NAXIS1'])
        # NOTE: the order of axes on an numpy array are opposite of the order
        #  specified in the FITS file.

        # self.image_start_bytes = abs(bits_per_pixel) * nax1 * nax2 // 8
        self.bzero = hdr.get('BZERO', 0)
        self.data = np.memmap(filename, dtype, 'r', data_start_bytes, shape)

    def __getitem__(self, key):
        # NOTE: adding a float here converts from np.memmap to np.array
        return self.data[key] + self.bzero

    def __len__(self):
        return len(self.data)


class shocOldHDU(shocHDU):
    pass


class shocNewHDU(shocHDU):
    pass


class shocBiasHDU(shocHDU):
    def get_coords(self):
        return


class shocFlatHDU(shocBiasHDU):
    pass
