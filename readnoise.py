import os
import inspect
from io import BytesIO

import numpy as np
from astropy.io.fits import Header

__all__ = ['ReadNoiseTable']


def _as_array(raw):
    # Convert table to machine readible form
    buffer = BytesIO()
    for line in raw.split('\n'):  # textwrap.dedent(table)
        linedata = filter(None, line.split('  '))
        outline = '\t'.join(linedata)
        if outline:
            buffer.write(str.encode(outline + '\n'))
    buffer.seek(0)

    return np.genfromtxt(buffer, dtype=None, names=True, delimiter='\t')


# ****************************************************************************************************
class ReadNoiseTable(np.ndarray):
    """Readout Noise Table for SHOC."""

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __new__(cls):
        """load the data table"""
        path = inspect.getfile(cls)
        path, _ = os.path.split(path)
        filename = os.path.join(path, 'ReadNoiseTable.txt')  # table in human readable form

        data = cls.get_table(filename)
        obj = np.asarray(data).view(cls)

        return obj

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @classmethod
    def get_table(cls, filename, form='m'):
        """
        return the table, either as a nice human readable string (form='h')
        or as a numpy recarray (form='m')
        """
        with open(filename) as fp:
            raw = fp.read()

        if form == 'h':
            return raw
        else:
            return _as_array(raw)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def get_readnoise(self, fn_or_header):
        """
        get the read out noise and sensitivity from the table given the fits header
        """
        if isinstance(fn_or_header, Header):
            header = fn_or_header
        else:
            with open(fn_or_header, 'rb') as fp:
                header = Header.fromfile(fp)

        # CCD acquisition mode
        mode = header['OUTPTAMP']
        lmode = self['Mode'] == mode.encode()

        # Readout clock frequency
        freq = 1. / header['READTIME']
        freq_MHz = round(freq / 1.e6)
        # The integer frequency in MHz as a string
        lfreq = self['FreqMHz'] == freq_MHz

        # Preamp gain setting
        preamp = header['PREAMP']
        lpreamp = self['PreAmp'] == preamp

        # serial number (SHOC 1 or 2)
        serno = header['SERNO']
        lserno = self['SerNo'] == serno

        return self._from_bools(lmode, lfreq, lpreamp, lserno)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def get_readnoise_from_kw(self, **kw):

        raise NotImplementedError

        mode_conversion = {'EM': 'Electron Multiplying',
                           'CON': 'Conventional'}
        serno_conv = {1: 6448,
                      2: 5982}
        # TODO: Convert keywords to lower case
        mode = kw.get('mode')
        if mode in mode_conversion:
            mode = mode_conversion[mode].encode()

        serno = serno_conv[kw.get('shoc')]

        preamp = kw.get('preamp')

        freq = kw.get('freq')
        freq_MHz = int(freq)

        lfreq = self['FreqMHz'] == freq_MHz
        lpreamp = self['PreAmp'] == preamp
        lserno = self['SerNo'] == serno

        return self._from_bools(lfreq, lpreamp, lserno)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _from_bools(self, *ls):
        l = np.all(ls, 0)
        # Boolean array used to determine readout noise value from the table
        if sum(l) > 1:
            # this should never happen
            raise ValueError('Read noise value not uniquely determined!!!')


        saturation = self['Saturation'][l][0]
        ron = self['ReadNoise'][l][0]
        sens = self['Sensitivity'][l][0]

        return ron, sens, saturation

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def get_saturation(self, fn_fits):
        return self.get_readnoise(fn_fits)[-1]