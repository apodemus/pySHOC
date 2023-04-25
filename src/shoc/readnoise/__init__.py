"""
This module provides access to  camera specs

`readNoiseTables` is a dict keyed on camera serial nrs.  Each value is a dict
keyed on a tuple on (readoutFrq, outAmpMode, preAmpGain) that uniquely
determine the ron, read_time, saturation, sensitivity and bit_depth
"""
import os
from astropy.table import Table

#            SHOC1, SHOC2
SERIAL_NRS = [5982, 6448]


def tbl2tpl(tbl, i0, i1):
    return map(tuple, tbl[tbl.colnames[i0:i1]])


path, _ = os.path.split(__file__)
readNoiseTables = dict()
for i in (1, 2):
    fn = '%s/data/SHOC%i.txt' % (path, i)
    tbl = Table.read(fn, format='ascii')
    readNoiseTables[SERIAL_NRS[i - 1]] = dict(zip(tbl2tpl(tbl, 0, 3),
                                                  tbl2tpl(tbl, 3, None)))
