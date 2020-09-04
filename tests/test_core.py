import more_itertools as mit
import functools as ftl
from recipes.testing import Expect
from astropy.io.fits.hdu.base import _BaseHDU
from pathlib import Path
from pySHOC import shocCampaign, shocHDU, shocNewHDU, shocBiasHDU, shocFlatHDU
import pytest
import numpy as np
import os
import tempfile as tmp

# TODO: old + new data all modes!!!
# TODO: all combinations of science, bias, dark, flats (+ masters)
# TODO:

# pylint: disable=C0111     # Missing %s docstring
# pylint: disable=R0201     # Method could be a function

# pretty sample images here:
DATA = Path(__file__).parent / 'data'
EX1 = DATA / 'AT2020hat'
CAL = DATA / 'calibration'

#
np.random.seed(12345)


# ---------------------------------- Helpers --------------------------------- #

def list_of_files():
    # create text file with list of filenames for test load
    fp, filename = tmp.mkstemp('.txt')
    for name in EX1.glob('*.fits'):
        os.write(fp, f'{name}{os.linesep}'.encode())
    os.close(fp)
    return filename

# --------------------------------- Fixtures --------------------------------- #


@pytest.fixture
def run():
    return shocCampaign.load(EX1)
# run = shocCampaign.load(EX1)

# ----------------------------------- Tests ---------------------------------- #


class TestCampaign:
    @pytest.mark.parametrize(
        'pointer',
        (  # single file as a str
            f'{EX1}/SHA_20200731.0001.fits',
            # single file as a Path object
            EX1 / 'SHA_20200731.0001.fits',
            # file list
            [f'{EX1}/SHA_20200731.0001.fits',
             f'{EX1}/SHA_20200731.0002.fits'],
            # globbing patterns
            f'{EX1}/SHA_20200731.000[12].fits',
            f'{EX1}/SHA_20200731.000*.fits',
            # directory
            EX1, str(EX1),
            # pointer to text file with list of filenames
            f'@{list_of_files()}'
        )
    )
    def test_load(self, pointer):
        run = shocCampaign.load(pointer)

    def test_file_helper(self, run):
        run.files
        run.files.names
        run.files.stems
        run.files.nrs

    @pytest.mark.parametrize(
        'index',
        (  # simple indexing
            0,
            -1,
            # by filename
            'SHA_20200731.0007.fits',
            'SHA_20200731.0007',  # both should work
        )
    )
    def test_single_index(self, run, index):
        print(run[index].file.name)
        assert isinstance(run[index], shocHDU)

    @pytest.mark.parametrize(
        'index,expected',
        [        # slice
            (slice(0, 4, 2),
             ['SHA_20200731.0001.fits', 'SHA_20200731.0003.fits']),

            # sequences of ints
            ([0, 1, 3, -1],
             ['SHA_20200731.0001.fits', 'SHA_20200731.0002.fits',
              'SHA_20200731.0004.fits', 'SHA_20200731.0022.fits']),

            # array of ints
            (np.arange(3),
             ['SHA_20200731.0001.fits', 'SHA_20200731.0002.fits',
              'SHA_20200731.0003.fits']),

            # boolean array
            (np.random.randint(0, 2, 22).astype(bool),
             ['SHA_20200731.0002.fits', 'SHA_20200731.0003.fits', 
             'SHA_20200731.0004.fits', 'SHA_20200731.0006.fits', 
             'SHA_20200731.0009.fits', 'SHA_20200731.0011.fits', 
             'SHA_20200731.0012.fits', 'SHA_20200731.0014.fits', 
             'SHA_20200731.0015.fits', 'SHA_20200731.0017.fits', 
             'SHA_20200731.0018.fits', 'SHA_20200731.0019.fits']),

            # by list of filenames
            (('SHA_20200731.0007.fits', 'SHA_20200731.0008.fits'),
             ['SHA_20200731.0007.fits', 'SHA_20200731.0008.fits']),

            # by globbing pattern
            ('SHA*[78].fits',
             ['SHA_20200731.0007.fits', 'SHA_20200731.0008.fits',
              'SHA_20200731.0017.fits', 'SHA_20200731.0018.fits']),

            # by brace expansion
            ('SHA*{7,8}.fits',
             ['SHA_20200731.0007.fits', 'SHA_20200731.0008.fits',
              'SHA_20200731.0017.fits', 'SHA_20200731.0018.fits']),

            # by filename sequence slice
            ('*0731.00[10:22].*',
             ['SHA_20200731.0010.fits', 'SHA_20200731.0011.fits',
              'SHA_20200731.0012.fits', 'SHA_20200731.0013.fits',
              'SHA_20200731.0014.fits', 'SHA_20200731.0015.fits',
              'SHA_20200731.0016.fits', 'SHA_20200731.0017.fits',
              'SHA_20200731.0018.fits', 'SHA_20200731.0019.fits',
              'SHA_20200731.0020.fits', 'SHA_20200731.0021.fits'])
        ]
    )
    def test_multi_index(self, run, index, expected):
        sub = run[index]
        assert isinstance(sub, shocCampaign)
        assert sub.files.names == expected

    def test_pprint(self, run):
        print(run, run.table(run), sep='\n\n')



# @pytest.mark.parametrize(
# 'filename,expected',
#     [(CAL/'SHA_20200822.0005.fits', shocBiasHDU),
#      (CAL/'SHA_20200801.0001.fits', shocFlatHDU),
#      (EX1/'SHA_20200731.0022.fits', shocNewHDU)]
#     )
# def test_hdu_type(filename, expected):
#     obj = _BaseHDU.readfr

# @expected(
#     (CAL/'SHA_20200822.0005.fits', shocBiasHDU,
#      CAL/'SHA_20200801.0001.fits', shocFlatHDU,
#      EX1/'SHA_20200731.0022.fits', shocNewHDU)
# )
def hdu_type(filename):
    return _BaseHDU.readfrom(filename).__class__
    # print('....', filename)
    # print(obj)
    # return obj


Expect(hdu_type)(
    {CAL/'SHA_20200822.0005.fits': shocBiasHDU,
     CAL/'SHA_20200801.0001.fits': shocFlatHDU,
     EX1/'SHA_20200731.0022.fits': shocNewHDU},
    globals())

# TODO: shocOldHDU, shocMasterBias, shocMasterFlat

# TODO
# def test_select
