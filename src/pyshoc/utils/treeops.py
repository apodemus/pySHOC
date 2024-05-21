"""
Operations on SHOC files residing in a nested directory structure (file system 
tree)
"""

# std
import os
import shutil
import warnings
import itertools as itt
from pathlib import Path
from collections import defaultdict

# third-party
import more_itertools as mit
from astropy.coordinates.jparser import shorten

# local
from recipes import io, pprint as ppr
from recipes.decorators import raises

# relative
from ..core import Campaign


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


def get_common_parent(paths):
    root = Path()
    for folders in itt.zip_longest(*(p.parts for p in paths)):
        folders = set(folders)
        if len(folders) != 1:
            break
        root /= folders.pop()
    return root


def _rename(name):
    return shorten(name).replace(' ', '_')


def get_object_name(obstype, objname):
    if obstype != 'object':
        # calibration data
        return obstype
    
    if objname in ('', None):
        # Ignore files that could not be id'd by source name
        return
    
    # get folder name
    return _rename(objname)


def make_tree(src, dest, grouping, naming, extensions='*'):

    # load data
    if isinstance(src, (str, Path)):
        src = Campaign.load(src, recurse=True)
    elif not isinstance(src, Campaign):
        raise TypeError('Invalid source')

    # make grouping
    partition = src.group_by(*grouping)

    # compute requested folder tree (don't create any folders just yet)
    dest = Path(dest)
    tree = defaultdict(list)
    for gid, sub in partition.items():
        name = naming(*gid)
        if not name:
            warnings.warn(f'Could not get a valid filename for key {gid} with '
                          f'files: {sub.files.names}')
            continue

        folder = dest / name
        for file in io.iter_ext(sub.files.paths, extensions):
            tree[folder].append(file)

    return tree


def partition_by_source(src, dest=None, extensions='*', move=True,
                        overwrite=False, dry_run=False, clean_up=True):
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

    # load data
    if isinstance(src, (str, Path)):
        root = Path(src)
        src = Campaign.load(src, recurse=True)
    elif isinstance(src, Campaign):
        # get common parent folder
        root = get_common_parent(src.files.paths)
    else:
        raise TypeError('Invalid source')

    # default destination same as source
    if dest is None:
        dest = root

    # now get the desired folder structure
    tree = make_tree(src, dest, ('obstype', 'target'), get_object_name,
                     extensions)

    # choose message severity
    emit = warnings.warn if dry_run else raises(OSError)

    # check available space at destination
    if not move:
        check = dest
        while not check.exists():
            check = check.parent

        free_bytes = shutil.disk_usage(check).free
        req_bytes = sum(p.stat().st_size for p in mit.collapse(tree.values()))
        if free_bytes < req_bytes:
            kws = dict(unit='b', significant=1)
            emit(f'Not enough space on disc. New file tree requires '
                 f'{ppr.eng(req_bytes, **kws)}, you only have '
                 f'{ppr.eng(free_bytes, **kws)} available at location '
                 f'{str(check)!r}')

    # if this is a dry run, we are done
    if dry_run:
        return tree

    # Now move / copy the files
    action = os.rename if move else shutil.copy2
    for folder, files in tree.items():
        # create folder
        folder.mkdir(parents=True, exist_ok=True)
        for file in files:

            new = folder / file.name
            if new.exists() and not overwrite:
                emit(f'File {str(new)!r} will be overwritten.')

            # do it!
            action(file, new)

    # finally remove the empty directories
    if clean_up:
        for folder in root.iterdir():
            if folder.is_file():
                continue

            if len(list(folder.iterdir())) == 0:
                folder.rmdir()

    return tree
