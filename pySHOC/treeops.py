"""
Operations on SHOC files residing in a nested directory structure (file system 
tree)
"""

from collections import defaultdict

from astropy.io.fits.header import Header

from recipes.array import unique_rows
from obstools import io

from .core import shocCampaign


def iter_ext(files, extensions):
    """
    Yield all the files that exist with the same root and stem but different
    extension(s). 

    Parameters
    ----------
    files : Container or Iterable
        The files to consider
    extensions : str or Container of str
        All file extentions to consider

    Yields
    -------
    Path
        [description]
    """
    if isinstance(extensions, str):
        extensions = (extensions, )

    for file in files:
        yield file

        for ext in extensions:
            new = (file.parent / file.stem).with_suffix(f'.{ext.lstrip(".")}')
            if new.exists:
                yield new


def get_tree(root, extension=''):
    """
    Get the file tree as a dictionary keyed on folder names containing file
    names with each folder

    Parameters
    ----------
    root
    extension

    Returns
    -------

    """
    tree = defaultdict(list)
    for file in io.iter_files(root, extension, True):
        tree[file.parent.name].append(file)
    return tree


def unique_modes(root):
    """
    Return an array with rows containing the unique set of SHOC observational
    modes that comprise the fits files in the root directory and all its
    sub-directories.
    """

    run = shocCampaign.load(root, recurse=True)
    modes = run.attrs('binning', 'readout.mode')

    # Convert to strings so we can compare
    vals = [list(map(str, v)) for v in modes]
    return unique_rows(vals)


def partition_by_source(root, extensions=('fits',), remove_empty=True,
                        dry_run=False):
    """
    Partition the files in the root directory into folders based on the OBSTYPE
    and OBJECT keyword values in their headers. Only the directories named by
    the default `dddd` name convention are searched, so this function can be
    run multiple times on the same root path without trouble.

    Parameters
    ----------
    root: str
        Name of the root folder to partition
    extensions: tuple
        if given also move files with the same stem but different extensions.
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

    # if 'fits' not in extensions
    fitsfiles = list(io.iter_files(root, 'fits'))
    assert len(fitsfiles) > 0
    root = fitsfiles[0].parent

    partition = defaultdict(list)
    for file in fitsfiles:
        header = Header.fromfile(file)
        key = header.get('obstype', None)
        obj = header.get('object', None)
        if (key == 'object') and obj:
            key = obj.replace(' ', '_')
        # if kind is None: we don't know the obstype
        partition[key].append(file)

    # Remove files that could not be id'd by source name
    partition.pop(None, None)  # unknown

    # create bias/flat/source directories and move collected files into them
    tree = defaultdict(list)
    for name, files in partition.items():
        folder = root / name
        if not (folder.exists() or dry_run):
            folder.mkdir()

        for file in iter_ext(files, extensions):
            tree[name].append(file)
            if not dry_run:
                file.rename(folder / file.name)

    # finally remove the empty directories
    if remove_empty:
        for folder in root.iterdir():
            if folder.is_file():
                continue

            if len(list(folder.iterdir())) == 0:
                folder.rmdir()

    return tree
