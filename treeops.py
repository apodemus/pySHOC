#TODO: auto sort into source directories based on header info + make as dir struct

from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from recipes.array import unique_rows
from obstools.fastfits import fastheader, FitsCube

from .core import shocRun


def with_ext_gen(root, extention):
    """return all files in tree with given extension as list of Paths"""
    ext = extention.strip('.')
    return Path(root).rglob('*.{}'.format(ext))


def with_ext(root, extention):
    return list(with_ext_gen(root, extention))


def get_fits(root, ignore={}):
    """
    Return all fits files in tree as Path object. optionally files can be filtered based on the
    content of their headers. *ignore* is a dict keyed on header keywords
    """
    for fitsfile in with_ext_gen(root, '.fits'):
        header = fastheader(fitsfile)
        skip = False
        for key, ign in ignore.items():
            if header.get(key, '') in ign:
                skip = True
                break
        if not skip:
            yield fitsfile


def unique_modes(root):
    """
    Return an array with rows containing the unique set of SHOC observational modes that comprise the
    fits files in the root directory and all its sub-directories.
     """
    fitsfiles = get_fits(root)
    run = shocRun(filenames=fitsfiles)
    names, dattrs, vals = zip(*(stack.get_instrumental_setup(('binning', 'mode', 'emGain'))
                                for stack in run))

    #Convert to strings so we can compare
    vals = np.array([list(map(str, v)) for v in vals])

    return unique_rows(vals)    # unique_rows(np.array(vals, dtype='U50'))


def get_first_frames(root, ignore={}):
    data = {}
    for fits in get_fits(root, ignore):
        ff = FitsCube(fits)
        data[fits] = ff[0]
    return data


def get_first_frames_png(root, fliplr=True, clobber=False, verbose=True,
                         ignore=dict(obstype=('bias', 'flat'))):
    """
    Pull the first frames from all the fits files in root and its sub-directories
    :param root:
    :param fliplr:
    :param clobber:
    :param verbose:
    :return:
    """
    for fpath, data in get_first_frames(root, ignore).items():

        if fliplr:
            data = np.fliplr(data)

        vmin, vmax = np.percentile(data, (2.25, 99.75))

        #
        fig = plt.figure(figsize=(8,8), frameon=False)
        ax = fig.add_axes([0,0,1,1], frameon=False)
        ax.imshow(data, origin='llc',
                  cmap='gist_earth', vmin=vmin, vmax=vmax)

        # add filename to image
        fig.text(0.01, 0.99, fpath.name,
                 color='w', va='top', size=12, fontweight='bold')

        fpng = fpath.with_suffix('.png')
        if not fpng.exists() or clobber:
            if verbose:
                print('saving', str(fpng))
            fig.savefig(str(fpng))
        else:
            if verbose:
                print('not saving', str(fpng))


def partition_by_source(root, fits_only=False, remove_empty=True):
    """
    Partition the files in the root directory into folders based on the OBSTYPE and OBJECT keyword
    values in their headers. Only the directories named by the default `dddd` convention are
    searched, so this function can be run on the same root path without trouble.

    :param root:
        Name of the root folder to partition
    :param fits_only:
        if False move files with the same basenames (but different extentions) such as those created
         by photometry etc.
    :param remove_empty:
        Remove empty folders after partitioning is done

    Example:
    hannes@prometheus:/data/Feb_2017$ rsync -vazh shocnawe:/data/40in/sha/2017/021[34] .
    receiving incremental file list
    0213/
    0213/SHA_20170213.0001.fits
    0213/SHA_20170213.0002.fits
    0213/SHA_20170213.0003.fits
    0213/SHA_20170213.0004.fits
    0213/SHA_20170213.0005.fits
    0213/SHA_20170213.0006.fits
    0214/
    0214/SHA_20170214.0001.fits
    0214/SHA_20170214.0002.fits
    0214/SHA_20170214.0003.fits
    0214/SHA_20170214.0004.fits
    0214/SHA_20170214.0010.fits
    0214/SHA_20170214.0020.fits


    """
    root = Path(root)
    dateFolderPattern = '[0-9]' * 4
    fitsfiles = root.rglob(dateFolderPattern + '/*.fits')

    # partFunc = lambda f: fastheader(f).get('obstype', None)
    partDict = defaultdict(list)
    for filepath in fitsfiles:
        id_ = fastheader(filepath).get('obstype', None)
        partDict[id_].append(filepath)

    # remove the object files
    objFiles = partDict.pop('object')

    # create bias/flat directories and move flagged files into them
    for name, files in partDict.items():
        folder = root / name
        _mover(files, folder, fits_only)

    # partition the objects into directories
    objDict = defaultdict(list)
    for filename in objFiles:
        name = fastheader(filename).get('object', None)
        name = name.replace(' ', '_')
        objDict[name].append(filename)

    for name, files in objDict.items():
        folder = root / name
        _mover(files, folder, fits_only)

    # finally remove the empty directories
    if remove_empty:
        for folder in root.glob(dateFolderPattern):
            if not len(list(folder.iterdir())):
                folder.rmdir()


def _mover(files, folder, fits_only):
    if not folder.exists():
        folder.mkdir()
    for filename in files:
        filename.rename(folder / filename.name)
        if not fits_only:
            for other in (filename.parent / filename.stem).glob('*'):
                other.rename(folder / other.name)


#def flagger