
"""
Calibrate SHOC observations
"""


# std
import logging

# third-party
import numpy as np
import matplotlib.pyplot as plt

# local
import motley
from motley.table import Table
from scrawl.imagine import ImageDisplay
from recipes.logging import logging, get_module_logger
from recipes.string import plural

# relative
from .. import calDB, shocCampaign, MATCH, COLOURS


# module level logger
logger = get_module_logger()
logging.basicConfig()
logger.setLevel(logging.INFO)


def calibrate(run, path=None, overwrite=False):

    gobj = run.select_by(obstype='object')
    fstax, mflat = find_cal(run, 'flat', path, overwrite)
    dstax, mdark = find_cal(run.join(fstax), 'dark', path, overwrite)

    if dstax:
        mdark.update(compute_masters(dstax, 'dark', path, overwrite))

    if fstax:
        # debias flats
        # check = [hdu.data[0,0,0] for hdu in fstax.to_list()]
        fstax.group_by(mdark).subtract(mdark)
        # check2 =  [hdu.data[0,0,0] for hdu in fstax.to_list()]
        mflat.update(compute_masters(fstax, 'flat', path, overwrite))

    if mdark or mflat:
        # enable on-the-fly calibration
        gobj.set_calibrators(mdark, mflat)
        logger.info('Calibration frames set.')
    else:
        logger.info('No calibration frames found in %s', run)

    return gobj, mdark, mflat


def split_cal(run, kind):
    """split off the calibration frames"""

    # get calibrators in run
    grp = run.group_by('obstype')
    cal = grp.pop(kind, shocCampaign())

    # drop unnecessary calibration stacks
    if cal:
        need = set(run.missing(kind))
        gcal = cal.group_by(*MATCH[kind][0])
        unneeded = set(gcal.keys()) - need
        if unneeded:
            dropping = shocCampaign()
            for u in unneeded:
                dropping.join(gcal.pop(u))
            logger.info('Discarding unneeded calibration frames: %s', dropping)
        cal = gcal.to_list()

    return cal, grp.to_list()


def find_cal(run, kind, path=None, ignore_masters=False):

    attrs = attx, attc = MATCH[kind]
    gid = (*attx, *attc)
    cal, run = split_cal(run, kind)
    gcal = cal.group_by(*attx)
    need = set(run.required_calibration(kind))  # instrumental setups

    # found_in_run = set(cal.attrs(*attx))
    # found_db_master = found_db_raw = found_in_path = set()
    # logger.info('Found %i calibration files in the run.', len(cal))

    # no calibration stacks in run. Look for pre-computed master images.
    masters = run.new_groups()
    masters.group_id = gid, {}
    if need:
        if path:
            logger.info('Searching for calibration frames in path: %r', path)
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
        logger.warning(
            'Could not find %s for observed data with %s\n%s\n in '
            'database %r',
            motley.apply(plural(kind), COLOURS[kind]),
            plural('setup', need), Table(need, col_headers=attx),
            str(calDB[kind])
        )

    # finally, group for convenience
    matched = run.match(cal.join(masters), *attrs)
    logger.info('The following files were matched:\n%s\n',
                matched.pformat(title=f'Matched {kind.title()}',
                                g1_style=COLOURS[kind]))

    return gcal, masters


def compute_masters(stacks, kind, outpath=None, overwrite=False,
                    png=False, **kws):

    logger.info('Computing master %s for %i stacks.',
                plural(kind), len(stacks))

    # all the stacks we need are here: combine
    masters = stacks.merge_combine()  # get_combine_func(kind)

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
