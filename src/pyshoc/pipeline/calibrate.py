
"""
Calibrate SHOC observations
"""


# std
import textwrap

# third-party
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger

# local
import motley
from motley.table import Table
from recipes.string import pluralize
from scrawl.image import ImageDisplay

# relative
from .. import config as cfg
from .. import MATCH, Campaign, calDB




# ---------------------------------------------------------------------------- #

def calibrate(run, path=None, overwrite=False):
    """
    Calibrate the observing campaign in `run`. 

    Master calibration frames are search for, first by checking if any of the
    `run`'s constituent `hdu`s are flagged as calibration files in the fits
    headers. Failing that, we search the file system folder in `path` if
    provided. If available, previously computed master calibration frames will
    be preferred unless `overwrite=True`, in which case they will be recomputed
    from raw files (if found). Once the master frames are located/computed,
    calibration frames are set on the `run`'s constituent `hdu`s by this
    routine, arithmetic is done when accessing the frames by slicing
    `hdu.calibrated` attribute.

    Parameters
    ----------
    run : pyshoc.Campaign
        Observing run containing multiple `hdu`s (fits files).
    path : Path-like, optional
        File system location for searching for calibration frames, by default
        None.
    overwrite : bool, optional
        Whether to force new compute, by default False

    Returns
    -------
    gobj: pyshoc.ObsGroup
        Observing campaign grouped by calibration groups.
    master_dark: pyshoc.ObsGroup
        Grouped master dark frames.
    master_flats: pyshoc.ObsGroup
        Grouped master flat frames.
    """

    gobj = run.select_by(obstype='object')
    raw_flats, master_flats = find(run, 'flat', path, overwrite)
    raw_darks, master_dark = find(run.join(raw_flats), 'dark', path, overwrite)

    if raw_darks:
        master_dark.update(compute_masters(raw_darks, 'dark', path, overwrite))

    if raw_flats:
        # Have raw flat field cubes.
        # debias flats
        # check = [hdu.data[0,0,0] for hdu in raw_flats.to_list()]
        raw_flats.group_by(master_dark).set_calibrators(dark=master_dark)
        # raw_flats.group_by(master_dark).subtract(master_dark, handle_missing=warnings.warn)

        # check2 =  [hdu.data[0,0,0] for hdu in raw_flats.to_list()]
        master_flats.update(compute_masters(raw_flats, 'flat', path, overwrite))

    if master_dark or master_flats:
        # enable on-the-fly calibration
        gobj.set_calibrators(master_dark, master_flats)
        logger.info('Calibration frames set.')
    else:
        logger.bind(indent=True).info('No calibration frames found in {:s}.', run)

    return gobj, master_dark, master_flats


def split(run, kind):
    """Split off the calibration frames."""

    # get calibrators in run
    grp = run.group_by('obstype')
    if files := grp[None].join(grp['']).files.names:
        nl = '\n    '
        raise ValueError(textwrap.dedent(f'''\
            Encountered invalid or missing `obstype` value in files:
                {nl.join(files)}
            Please specify the obstype by editing the fits headers directly on the HDU:
                >>> run[0].header['obstype'] = 'object'
            or for the entire run by doing:
                >>> run.attrs.set(repeat(obstype='object'))
            on the appropriate set of files.'''))

    cal = grp.pop(kind, Campaign())

    # drop unnecessary calibration stacks
    if cal:
        need = set(run.missing(kind))
        gcal = cal.group_by(*MATCH[kind][0])
        if unneeded := set(gcal.keys()) - need:
            dropping = Campaign()
            for u in unneeded:
                dropping.join(gcal.pop(u))
            logger.info('Discarding unnecessary calibration frames: {:s}.',
                        dropping)
        cal = gcal.to_list()

    return cal, grp.to_list()


def find(run, kind, path=None, ignore_masters=False):

    attrs = attx, attc = MATCH[kind]
    gid = (*attx, *attc)
    cal, run = split(run, kind)
    gcal = cal.group_by(*attx)
    need = set(run.required_calibration(kind))  # instrumental setups

    # no calibration stacks in run. Look for pre-computed master images.
    masters = run.new_groups()
    masters.group_id = gid, {}
    if need:
        if path:
            logger.info('Searching for calibration frames in path: {!r:}.', path)
            # get calibrators from provided path.
            xcal = Campaign.load(path)
            _, gcal = run.match(xcal, *attrs)

            xcal = gcal.to_list()
            masters = xcal.select_by(ndim=2).group_by(*gid)
            cal = cal.join(xcal.select_by(ndim=3))

            # where = repr(str(path))
            found_in_path = set(xcal.attrs(*attx))
            need -= found_in_path
        elif not ignore_masters:
            # Lookup master calibrators in db, unless overwrite
            # where = 'database'
            masters = calDB.get(run, kind, master=True)

            # found_db_master =
            need -= set(masters.to_list().attrs(*attx))

    # get missing calibration stacks (raw) from db. Here we prefer to use the
    # stacks that are passed in, and suplement from the db. The passed stacks
    # will thus always be computed / used.
    if need:
        colours = cfg.console.colors[kind]
        selection = np.any([run.selection(**dict(zip(attx, _)))
                            for _ in need], 0)
        gcal = calDB.get(run[selection], kind, master=False)

        # found_db_raw = gcal.keys()
        if gcal:
            cal = cal.join(gcal.to_list())
            need -= set(cal.attrs(*attx))

    if need:
        logger.bind(indent=2).warning(textwrap.dedent(
            '''\
            Could not find {:s} for observed data with instrumental {:s}
            {:s}
            in database {!r:}.\
            '''),
            motley.apply(pluralize(kind), colours),
            pluralize('setup', need),
            Table(need, col_headers=attx),
            str(calDB[kind])
        )

    # finally, group for convenience
    matched = run.match(cal.join(masters), *attrs)
    logger.bind(indent=2).info(
        'The following files were matched:\n{:s}\n',
        matched.pformat(title=f'Matched {kind.title()}',
                        g1_style=colours)
    )

    return gcal, masters


def compute_masters(stacks, kind, outpath=None, overwrite=False,
                    png=False, **kws):

    logger.opt(lazy=True).info('Computing master {0[0]:s} for {0[1]:d} stacks.',
                               lambda: (pluralize(kind), len(stacks)))

    # all the stacks we need are here: combine
    masters = stacks.merge_combine()

    # save fits
    masters.save(outpath or calDB.master[kind], overwrite=overwrite)

    # Update database
    calDB.update(masters.to_list())

    # save png images
    if png:
        defaults = dict(figsize=[7.14, 5.55],
                        plims=(0, 100))
        for obs in masters.values():
            img = ImageDisplay(obs.data, **{**defaults, **kws})
            img.figure.savefig(obs.file.path.with_suffix('.png'),
                               bbox_inches='tight', pad_inches=0)
            plt.close(img.figure)

    # calDB
    return masters
