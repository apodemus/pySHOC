
"""
Calibrate SHOC observations
"""


# std
import textwrap
import warnings

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
from .. import CONFIG, MATCH, calDB, shocCampaign


COLOURS = CONFIG.console.colors


def calibrate(run, path=None, overwrite=False):

    gobj = run.select_by(obstype='object')
    raw_flats, master_flats = find_cal(run, 'flat', path, overwrite)
    raw_darks, master_dark = find_cal(run.join(raw_flats), 'dark', path, overwrite)

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
        logger.info('No calibration frames found in {:s}', run)

    return gobj, master_dark, master_flats


def split_cal(run, kind):
    """split off the calibration frames"""

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

    cal = grp.pop(kind, shocCampaign())

    # drop unnecessary calibration stacks
    if cal:
        need = set(run.missing(kind))
        gcal = cal.group_by(*MATCH[kind][0])
        if unneeded := set(gcal.keys()) - need:
            dropping = shocCampaign()
            for u in unneeded:
                dropping.join(gcal.pop(u))
            logger.info('Discarding unnecessary calibration frames: {:s}.',
                        dropping)
        cal = gcal.to_list()

    return cal, grp.to_list()


def find_cal(run, kind, path=None, ignore_masters=False):

    attrs = attx, attc = MATCH[kind]
    gid = (*attx, *attc)
    cal, run = split_cal(run, kind)
    gcal = cal.group_by(*attx)
    need = set(run.required_calibration(kind))  # instrumental setups

    # no calibration stacks in run. Look for pre-computed master images.
    masters = run.new_groups()
    masters.group_id = gid, {}
    if need:
        if path:
            logger.info('Searching for calibration frames in path: {!r:}', path)
            # get calibrators from provided path.
            xcal = shocCampaign.load(path)
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
        selection = np.any([run.selection(**dict(zip(attx, _)))
                            for _ in need], 0)
        gcal = calDB.get(run[selection], kind, master=False)

        # found_db_raw = gcal.keys()
        if gcal:
            cal = cal.join(gcal.to_list())
            need -= set(cal.attrs(*attx))

    if need:
        logger.warning(textwrap.dedent(
            '''\
            Could not find {:s} for observed data with instrumental {:s}
            {:s}
            in database {!r:}.\
            '''),
            motley.apply(pluralize(kind), COLOURS[kind]),
            pluralize('setup', need),
            Table(need, col_headers=attx),
            str(calDB[kind])
        )

    # finally, group for convenience
    matched = run.match(cal.join(masters), *attrs)
    logger.info('The following files were matched:\n{:s}\n',
                matched.pformat(title=f'Matched {kind.title()}',
                                g1_style=COLOURS[kind]))

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
