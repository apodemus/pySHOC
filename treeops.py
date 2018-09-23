
import logging
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from obstools.fastfits import fastheader, FitsCube
from recipes.array import unique_rows

from .core import shocRun


def with_ext_gen(root, extension):
    """return all files in tree with given extension as list of Paths"""
    path = Path(root)
    if not path.exists():
        raise ValueError('Not a valid system path: %s' % str(path))

    ext = extension.strip('.')
    return path.rglob('*.%s' % ext)


def with_ext(root, extension):
    return list(with_ext_gen(root, extension))


def get_tree(root, extension=''):
    """
    Get

    Parameters
    ----------
    root
    extension

    Returns
    -------

    """
    if extension:
        extension = '.' + extension.strip('.')

    tree = defaultdict(list)
    for file in Path(root).glob('*/*%s' % extension):
        tree[file.parent.name].append(file)
    return tree


def get_fits(root, ignore=None):
    """
    Generator that yields all fits files in tree as Path object. Optionally
    files can be filtered based on the content of their headers.

    Parameters
    ----------
    root: str or `pathlib.Path`
        path to the root directory. All files and sub-directories will be
        traversed recursively returning fits files
    ignore: dict, optional
        A dict keyed on header keywords containing a sequence of values to
        ignore. Any file that has header values equaling items in the ignore
        list for that keyword will be filtered.

    Yields
    ------
    fitsfile: `pathlib.Path`

    """
    if ignore is None:
        ignore = {}

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

    # Convert to strings so we can compare
    vals = np.array([list(map(str, v)) for v in vals])

    return unique_rows(vals)  # unique_rows(np.array(vals, dtype='U50'))


def get_data(root, ignore=None, subset=1):
    """
    Get a subset of data from each fits file.  For example, get the first frame
    from each data cube.

    Parameters
    ----------
    root: str or `pathlib.Path`
        path to the root directory. All files and sub-directories will be
    ignore: dict, optional
        A dict keyed on header keywords containing a sequence of values to
        ignore. Any file that has header values equaling items in the ignore
        list for that keyword will be filtered.
    subset: int or slice or array-like of size 2 or 3, optional
        The range of frames to retrieve. The `subset` argument will be mapped to
        a slice, and that slice will be retrieved from each cube. This means
        that if the slice maps to indices that are beyond the size of the cube
        an zero-sized array may be returned. The default is 1. i.e. retrieve
        the first frame from each cube.

    Returns
    -------
    data: dict
        dict keyed on filenames containing data as numpy arrays.

    """
    if 1 < np.size(subset) <= 3:
        subset = slice(*subset)
    else:
        raise ValueError('Invalid subset')

    data = {}
    for fits in get_fits(root, ignore):
        ff = FitsCube(fits)
        data[fits] = ff[subset]
    return data


def get_images(root, ignore=None, subset=1, clobber=False, fliplr=True,
               cmap='gist_earth', plims=(2.25, 99.75), show_filenames=True,
               ext='.png'):
    """
    Pull the subset of frames from all the fits files in root and its
    sub-directories.

    Parameters
    ----------
    root
    ignore
    subset
    clobber
    fliplr
    cmap
    plims
    show_filenames
    ext

    Returns
    -------

    """
    if ignore is None:
        ignore = dict(obstype=('bias', 'flat'))

    ext = ext.strip('.')
    for i, (fpath, data) in enumerate(get_data(root, ignore, subset).items()):
        # create figure
        fig = plt.figure(figsize=(8, 8), frameon=False)
        if i == 0:
            # check if extension is supported
            supported = fig.canvas.get_supported_filetypes().keys()
            if ext not in supported:
                plt.close()
                raise ValueError('Image extension %r not supported. Try one'
                                 'of these instead: %s' %(ext, supported))
        # add axes (full frame)
        ax = fig.add_axes([0, 0, 1, 1], frameon=False)

        # flip image if required
        if fliplr:
            data = np.fliplr(data)

        # set color limits as percentile
        vmin, vmax = np.percentile(data, plims)

        # show image
        ax.imshow(data, origin='llc',
                  cmap=cmap, vmin=vmin, vmax=vmax)

        if show_filenames:
            # add filename to image
            fig.text(0.01, 0.99, fpath.name,
                     color='w', va='top', size=12, fontweight='bold')

        image_path = fpath.with_suffix('.%s' % ext)
        if not image_path.exists() or clobber:
            logging.info('saving %s', image_path)
            fig.savefig(str(image_path))
        else:
            logging.info('not saving %s', image_path)


def partition_by_source(root, fits_only=False, remove_empty=True, dry_run=False):
    """
    Partition the files in the root directory into folders based on the OBSTYPE
    and OBJECT keyword values in their headers. Only the directories named by
    the default `dddd` name convention are searched, so this function can be
    run multiple times on the same root path without trouble.

    Parameters
    ----------
    root: str
        Name of the root folder to partition
    fits_only: bool
        if False also move files with the same stem but different extensions.
    remove_empty: bool
        Remove empty folders after partitioning is done
    dry_run: bool
        if True, return the would-be partition tree as a dict and leave folder
        structure unchanged.


    Examples
    --------
    >>> !tree /data/Jan_2018
    /data/Jan_2018
    ├── 0117
    │   ├── SHA_20180117.0001.fits
    │   ├── SHA_20180117.0002.fits
    │   ├── SHA_20180117.0003.fits
    │   ├── SHA_20180117.0004.fits
    │   ├── SHA_20180117.0010.fits
    │   ├── SHA_20180117.0011.fits
    │   └── SHA_20180117.0012.fits
    ├── 0118
    │   ├── SHA_20180118.0001.fits
    │   ├── SHA_20180118.0002.fits
    │   ├── SHA_20180118.0003.fits
    │   ├── SHA_20180118.0100.fits
    │   └── SHA_20180118.0101.fits
    ├── 0122
    ├── 0123
    │   ├── SHA_20180123.0001.fits
    │   ├── SHA_20180123.0002.fits
    │   ├── SHA_20180123.0003.fits
    │   ├── SHA_20180123.0004.fits
    │   ├── SHA_20180123.0010.fits
    │   ├── SHA_20180123.0011.fits
    │   └── SHA_20180123.0012.fits
    ├── env
    │   └── env20180118.png
    ├── log.odt
    └── shoc-gui-bug.avi

    5 directories, 22 files

    >>> tree = treeops.partition_by_source('/data/Jan_2018')
    >>> !tree /data/Jan_2018
    /data/Jan_2018
    ├── env
    │   └── env20180118.png
    ├── flat
    │   ├── SHA_20180118.0100.fits
    │   ├── SHA_20180118.0101.fits
    │   ├── SHA_20180123.0001.fits
    │   ├── SHA_20180123.0002.fits
    │   ├── SHA_20180123.0003.fits
    │   ├── SHA_20180123.0004.fits
    │   ├── SHA_20180123.0010.fits
    │   ├── SHA_20180123.0011.fits
    │   └── SHA_20180123.0012.fits
    ├── log.odt
    ├── OW_J0652-0150
    │   ├── SHA_20180117.0001.fits
    │   ├── SHA_20180117.0002.fits
    │   ├── SHA_20180117.0003.fits
    │   └── SHA_20180117.0004.fits
    ├── OW_J0821-3346
    │   ├── SHA_20180117.0010.fits
    │   ├── SHA_20180117.0011.fits
    │   ├── SHA_20180117.0012.fits
    │   ├── SHA_20180118.0001.fits
    │   ├── SHA_20180118.0002.fits
    │   └── SHA_20180118.0003.fits
    └── shoc-gui-bug.avi

    4 directories, 22 files

    """
    root = Path(root)
    dateFolderPattern = '[0-9]' * 4
    fitsfiles = root.rglob(dateFolderPattern + '/*.fits')

    # partFunc = lambda f: fastheader(f).get('obstype', None)
    partDict = defaultdict(list)
    for path in fitsfiles:
        id_ = fastheader(path).get('obstype', None)
        # if id_ is not None: we don't know the obstype
        partDict[id_].append(path)

    # pop files with 'object' obstype
    objFiles = partDict.pop('object')

    # partition the objects into directories
    for filename in objFiles:
        name = fastheader(filename).get('object', None)
        name = name.replace(' ', '_')
        partDict[name].append(filename)

    # Remove files that could not be id'd
    unknown = partDict.pop(None, None)

    # create bias/flat/source directories and move collected files into them
    tree = defaultdict(list)
    for name, files in partDict.items():
        folder = root / name
        if not (folder.exists() or dry_run):
            folder.mkdir()
        for file in stem_gen(files, fits_only):
            tree[name].append(file)
            if not dry_run:
                file.rename(folder / file.name)

    # finally remove the empty directories
    if remove_empty:
        for folder in root.glob(dateFolderPattern):
            if not len(list(folder.iterdir())):
                folder.rmdir()

    return tree


def stem_gen(files, fits_only):
    for file in files:
        yield file
        if not fits_only:
            yield from (file.parent / file.stem).glob('*')


# def _move_files(files, folder, fits_only):
#     if not folder.exists():
#         folder.mkdir()
#     # move files into folder
#     for filename in files:
#         filename.rename(folder / filename.name)
#         #
#         if not fits_only:
#             for other in (filename.parent / filename.stem).glob('*'):
#                 other.rename(folder / other.name)

# def flagger
