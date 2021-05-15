from shoc.timing import TimeDelta
from astropy import time

import more_itertools as mit
import functools as ftl
# from recipes.testing import Expect
from recipes.containers import is_property
from recipes.testing import  expected, Expect, mock
from astropy.io.fits.hdu.base import _BaseHDU
from pathlib import Path
from shoc.core import (shocCampaign, shocHDU,  shocDarkHDU, shocFlatHDU,
                       shocDarkMaster, shocOldDarkHDU)
import pytest
import numpy as np
import os
import tempfile as tmp
import inspect

# TODO: old + new data all modes!!!
# TODO: all combinations of science, bias, dark, flats (+ masters)
# TODO:

# pylint: disable=C0111     # Missing %s docstring
# pylint: disable=R0201     # Method could be a function

# pretty sample images here:
DATA = Path(__file__).parent / 'data'
EX1 = DATA / 'AT2020hat'
CAL = DATA / 'calibration'

ROOT = Path('/media/Oceanus/work/Observing/data/sources/')
EX2 = ROOT / 'Chariklo/20140429.016{,._X2}.fits'
EX3 = ROOT / 'CVs/polars/CTCV J1928-5001/SHOC/raw'

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


class TestTiming:
    @pytest.mark.parametrize(
        't',
        [
            # TimeDelta(TimeDelta(1)),
            TimeDelta(time.TimeDelta(1)),
            #  TimeDelta(1) * 2,
            #  2 * TimeDelta(1),
            #  TimeDelta(1) / 2
        ])
    def test_type(self, t):
        assert type(t) is TimeDelta
        print(t)


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


class TestHDU:
    def test_str(self, run):
        print(str(run[0]))

    # def test_hdu_type(self, filename, expected):
    #     assert _BaseHDU.readfrom(filename).__class__ == expected
    # @expected([(CAL/'SHA_20200822.0005.fits', shocDarkHDU),
    #            (CAL/'SHA_20200801.0001.fits', shocFlatHDU),
    #            (EX1/'SHA_20200731.0022.fits', shocNewHDU)],
    #            globals_=globals())
    # def hdu_type(self, filename):
    #     return _BaseHDU.readfrom(filename).__class__
    # print('....', filename)
    # print(obj)
    # return obj
    # @pytest.mark.parametrize('hdu', run()[:1])

    def test_timing(self, run):
        hdu = run[0]
        t = hdu.t
        for attr, p in inspect.getmembers(type(t), is_property):
            getattr(t, attr)

    # TODO:
    # test_timing type(obs.t), shocTimingNew, shocTimingOld


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

    # TODO: test_id, test_group, test_combine, test_save
    # steps below
    @pytest.mark.skip()
    def test_masters(self, run):
        from obstools.stats import median_scaled_median
        from shoc import MATCH_FLATS, MATCH_DARKS

        is_flat = np.array(run.calls('pointing_zenith'))
        run[is_flat].set_attrs(obstype='flat')

        grp = run.group_by('obstype')
        gobj, gflats = grp['object'].match(grp['flat'], *MATCH_FLATS)
        needs_debias = gobj.to_list().join(gflats)
        gobs, gbias = needs_debias.match(grp['bias'], *MATCH_DARKS)

        if gbias:
            # all the "dark" stacks we need are here
            mbias = gbias.merge_combine()
            mbias.save(CAL)

            #
            if gflats:
                gflats.group_by(mbias).subtract(mbias)

        if gflats:
            mflats = gflats.merge_combine()
            mflats.save(CAL)


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
