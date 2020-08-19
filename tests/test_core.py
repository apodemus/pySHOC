from pathlib import Path
from pySHOC import shocCampaign, shocHDU
import pytest
import numpy as np
import os
import tempfile as tmp

# TODO: old + new data all modes!!!
# TODO: all combinations of science, bias, dark, flats (+ masters)
# TODO:

# pylint: disable=C0111     # Missing %s docstring

# pretty sample images here:
datapath = Path(__file__).parent / 'data/AT2020hat'


@pytest.fixture
def run():
    return shocCampaign.load(datapath)


def list_of_files():
    # create text file with list of filenames for test load
    fp, filename = tmp.mkstemp('.txt')
    for name in datapath.glob('*.fits'):
        os.write(fp, f'{name}{os.linesep}'.encode())
    os.close(fp)
    return filename


@pytest.mark.skip
@pytest.mark.parametrize(
    'pointer',
    (  # single file as a str
        f'{datapath}/SHA_20200731.0001.fits',
        # single file as a Path object
        datapath / 'SHA_20200731.0001.fits',
        # file list
        [f'{datapath}/SHA_20200731.0001.fits',
         f'{datapath}/SHA_20200731.0002.fits'],
        # globbing patterns
        f'{datapath}/SHA_20200731.000[12].fits',
        f'{datapath}/SHA_20200731.000*.fits',
        # directory
        datapath, str(datapath),
        # pointer to text file with list of filenames
        f'@{list_of_files()}'
    )
)
def test_load(pointer):
    run = shocCampaign.load(pointer)


@pytest.mark.parametrize(
    'index',
    (  # simple indexing
        0, -1,
        # by filename
        'SHA_20200731.0007.fits', 'SHA_20200731.0007',  # both should work
    )
)
def test_indexing_single(run, index):
    assert isinstance(run[index], shocHDU)


@pytest.mark.parametrize(
    'index',
    (  # slice
        slice(0, 4, 2),
        # sequences of ints
        [0, 1, 3, -1], np.arange(3),
        # boolean array
        np.random.int(0, 2, len(run)).astype(bool),
        # by list of filenames
        ('SHA_20200731.0007.fits', 'SHA_20200731.0008.fits'),
        # by globbing pattern
        'SHA*[78].fits'
    )
)
def test_indexing_multi(run, index):
    assert isinstance(run[index], shocCampaign)

def test_filehelper(run):
    run.files.names
    run.files.stems
    run.files.nrs

# TODO
# def test_select
