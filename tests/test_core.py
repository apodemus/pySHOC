import more_itertools as mit
import functools as ftl
from recipes.testing import  expected, Expect, mock
from astropy.io.fits.hdu.base import _BaseHDU
from pathlib import Path
from shoc.core import (shocCampaign, shocHDU,  shocDarkHDU, shocFlatHDU,
                       shocDarkMaster, shocOldDarkHDU)
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


test_hdu_type = Expect(_BaseHDU.readfrom)(
    {
        CAL/'SHA_20200822.0005.fits':                       shocDarkHDU,
        CAL/'SHA_20200801.0001.fits':                       shocFlatHDU,
        EX1/'SHA_20200731.0022.fits':                       shocHDU,
        CAL/'bias-20200822-8x8-1MHz-2.4-CON.fits':          shocDarkMaster,
        mock(CAL/'20121212.001.fits', obstype='dark'):      shocOldDarkHDU
    },
    left_transform=type
)
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
             'SHA_20200731.0004.fits', 'SHA_20200731.0006.fits', 
              'SHA_20200731.0004.fits', 'SHA_20200731.0006.fits',
              'SHA_20200731.0009.fits', 'SHA_20200731.0011.fits',
             'SHA_20200731.0009.fits', 'SHA_20200731.0011.fits', 
              'SHA_20200731.0009.fits', 'SHA_20200731.0011.fits',
              'SHA_20200731.0012.fits', 'SHA_20200731.0014.fits',
             'SHA_20200731.0012.fits', 'SHA_20200731.0014.fits', 
              'SHA_20200731.0012.fits', 'SHA_20200731.0014.fits',
              'SHA_20200731.0015.fits', 'SHA_20200731.0017.fits',
             'SHA_20200731.0015.fits', 'SHA_20200731.0017.fits', 
              'SHA_20200731.0015.fits', 'SHA_20200731.0017.fits',
              'SHA_20200731.0018.fits', 'SHA_20200731.0019.fits']),

            # list of filenames
            (('SHA_20200731.0007.fits', 'SHA_20200731.0008.fits'),
             ['SHA_20200731.0007.fits', 'SHA_20200731.0008.fits']),

            # list of filenames without extensions
            (('SHA_20200731.0007', 'SHA_20200731.0008'),
             ['SHA_20200731.0007.fits', 'SHA_20200731.0008.fits']),

            # globbing pattern
            ('SHA*[78].fits',
             ['SHA_20200731.0007.fits', 'SHA_20200731.0008.fits',
              'SHA_20200731.0017.fits', 'SHA_20200731.0018.fits']),

            # globbing pattern
            ('SHA*0[7..8].fits',
             ['SHA_20200731.0007.fits', 'SHA_20200731.0008.fits']),

            # brace expansion
            ('SHA*{7,8}.fits',
             ['SHA_20200731.0007.fits', 'SHA_20200731.0008.fits',
              'SHA_20200731.0017.fits', 'SHA_20200731.0018.fits']),

            # brace expansion with range
            ('*0731.00{10..21}.*',
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
        assert set(sub.files.names) == set(expected)

    @pytest.mark.parametrize(
        'run',
        [shocCampaign.load(x) for x in (EX1, EX2, EX3)]
    )
    def test_pprint(self, run):
        print(run)
        print(run.table(run))
        print(run[:1])
        # print()
        # print()




# @pytest.mark.parametrize(
# 'filename,expected',
#     [(CAL/'SHA_20200822.0005.fits', shocDarkHDU),
#      (CAL/'SHA_20200801.0001.fits', shocFlatHDU),
#      (EX1/'SHA_20200731.0022.fits', shocNewHDU)]
#     )
# def test_hdu_type(filename, expected):
#     obj = _BaseHDU.readfr

# @expected(
#     (CAL/'SHA_20200822.0005.fits', shocDarkHDU,
#      CAL/'SHA_20200801.0001.fits', shocFlatHDU,
#      EX1/'SHA_20200731.0022.fits', shocNewHDU)
# )


# TODO: shocOldHDU, shocMasterBias, shocMasterFlat

# TODO
# def test_select
