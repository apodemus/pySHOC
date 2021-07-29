
# pylint: disable=C0111     # Missing %s docstring
# pylint: disable=R0201     # Method could be a function


# std libs
import os
import inspect
import tempfile as tmp
from pathlib import Path
from collections import defaultdict

# third-party libs
import pytest
import numpy as np

# local libs
from recipes.testing import Expect, Throws, expected, mock, logger
from shoc.header import Header
from shoc.core import (shocCampaign, shocHDU, shocDarkHDU, shocFlatHDU,
                       shocDarkMaster, shocFlatMaster,
                       shocOldDarkHDU, shocOldFlatHDU)
from shoc.timing import UnknownTimeException

logger.setLevel(10)
# TODO: old + new data all modes!!!
# TODO: all combinations of science, bias, dark, flats (+ masters)
# TODO:


# pretty sample images here:
DATAPATH = Path(__file__).parent / 'data'
_ = Path('/media/Oceanus/work/Observing/data/sources/')
CAL = DATAPATH / 'calibration'
EX1 = DATAPATH / 'AT2020hat'
EX2 = _ / 'Chariklo/20140429.016{,._X2}.fits'
EX3 = _ / 'CVs/polars/CTCV J1928-5001/SHOC/raw'
DATAPATHS = [EX1, EX2, EX3]

# derived HDU classes
# list(shocHDU._shocHDU__shoc_hdu_types.values())[2:]
childHDUs = [shocDarkHDU, shocFlatHDU,
             shocOldDarkHDU, shocOldFlatHDU,
             shocDarkMaster, shocFlatMaster]

#
np.random.seed(12345)


# ---------------------------------- Helpers --------------------------------- #
datasets = defaultdict(shocCampaign.load)


def list_of_files(i):
    # create text file with list of filenames for test load
    fp, filename = tmp.mkstemp('.txt')
    for name in DATAPATHS[i].glob('*.fits'):
        os.write(fp, f'{name}{os.linesep}'.encode())
    os.close(fp)
    return filename


def random_data(head):
    shape = [head[f'NAXIS{i}'] for i in range(1, head['NAXIS'])[::-1]]
    return np.random.rand(*shape)

# --------------------------------- Fixtures --------------------------------- #


@pytest.fixture(scope='session', params=DATAPATHS)
def dataset(request):
    return datasets[request.param]


@pytest.fixture(scope='session')
def header():
    return Header.fromstring((DATAPATH / 'shoc_header.txt').read_text())

# ----------------------------------- Tests ---------------------------------- #


test_hdu_type = Expect(shocHDU.readfrom)(
    {
        CAL/'SHA_20200822.0005.fits':                       shocDarkHDU,
        CAL/'SHA_20200801.0001.fits':                       shocFlatHDU,
        EX1/'SHA_20200731.0022.fits':                       shocHDU,
        CAL/'bias-20200822-8x8-1MHz-2.4-CON.fits':          shocDarkMaster,
        mock(CAL/'20121212.001.fits', obstype='dark'):      shocOldDarkHDU
    },
    left_transform=type
)

# FIXME:  TypeError: missing a required argument: 'header'
# @expected(dict(zip(childHDUs, itt.repeat(ECHO))),
#             right_transform=type)
# def init(kls, header):
#     return kls(random_data(header), header)


# @pytest.skip
class TestHDU:
    def test_str(self, dataset):
        print(str(dataset[0]))

    @pytest.mark.parametrize('kls', childHDUs)
    def test_init(self, kls, header):
        assert isinstance(kls(random_data(header), header), kls)

    @expected({
        EX1: None,
        EX2: Throws(UnknownTimeException),
        EX3: None
    })
    def test_timing(self, dataset):
        hdu = dataset[0]
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
            f'@{list_of_files(0)}'
        )
    )
    def test_load(self, pointer):
        shocCampaign.load(pointer)

    def test_pprint(self, run):
        print(run)
        print(run.table(run))
        print(run[:1])
        print()
        print()

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
        # print(run[index].file.name)
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

    # TODO: test_id, test_group, test_combine, test_save
    # steps below

    @pytest.mark.skip()
    def test_masters(self, run):
        from obstools.stats import median_scaled_median
        from shoc import MATCH_FLATS, MATCH_DARKS, repeat

        is_flat = np.array(run.calls('pointing_zenith'))
        run[is_flat].attrs.set(repeat(obstype='flat'))

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


# TODO
# def test_select
