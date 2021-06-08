
from matplotlib import rc
from pathlib import Path
from obstools.stats import median_scaled_median
from .. import shocCampaign, MATCH
from ..caldb import RAW, MASTER
import numpy as np
import logging
from recipes.logging import logging, get_module_logger
from scrawl.imagine import ImageDisplay
import matplotlib.pyplot as plt

from recipes.sets import OrderedSet

from IPython.display import display

# module level logger
logger = get_module_logger()
logging.basicConfig()
logger.setLevel(logging.INFO)



def plural(s):
    if s.endswith('s'):
        return s + 'es'
    return s + 's'


def contains_fits(path, recurse=False):
    glob = path.rglob if recurse else path.glob
    return bool(next(glob('*.fits'), False))


def identify(run):
    # identify
    # is_flat = np.array(run.calls('pointing_zenith'))
    # run[is_flat].set_attrs(obstype='flat')

    g = run.guess_obstype()



def calibrate(run, path=None, overwrite=False):

    gobj = run.select_by(obstype='object')
    fstacks, mflat = find_cal(run, path, 'flat', overwrite)
    dstacks, mdark = find_cal(run.join(fstacks), path, 'dark', overwrite)
    mdark = mdark.join(compute_masters(dstacks, path, overwrite, 'dark'))
    mflat = mflat.join(compute_masters(fstacks, path, overwrite, 'flat', mdark))

    if mdark or mflat:
        # enable on-the-fly calibration
        gobj.set_calibrators(mdark, mflat)
        logger.info('Calibration frames set.')
    else:
        logger.info('No calibration frames found in %s', run)

    #
    return gobj, mdark, mflat




def split_cal(run, kind):
    grp = run.group_by('obstype')
    robj = grp['object']

    # get calibrators in run
    cal = grp.get(kind, shocCampaign())

    # drop unnecessary calibration stacks
    if cal:
        need = set(run.missing(kind))
        gcal = cal.group_by(MATCH[kind][0])
        unneeded = set(gcal.keys()) - need
        if unneeded:
            dropping = shocCampaign()
            for u in unneeded:
                dropping.join(gcal.pop(u))
            logger.info('Discarding unneeded calibration frames: %s', dropping)
        cal = gcal.to_list()

    return cal, robj


def find_cal(run, path, kind, overwrite):

    cal, run = split_cal(run, kind)

    attrs = MATCH[kind]
    need = set(run.missing(kind))

    # lookup pre-computed master images
    masters = None
    if need and not overwrite:
        # no calibration stacks in run. Lookup master calibrators in db
        # TODO: optimize this, should not need to load all these files
        xcal = shocCampaign.load(path or MASTER[kind])
        gcal, masters = run.match(xcal, *attrs)
        need = set(gcal.keys()) - set(masters.keys())

        tmp = masters.to_list()
        logger.info('Found %i master %s in database: \n%s',
                    len(tmp), plural(kind), tmp.pfromat())

    # get missing calibration stacks from db. Here we prefer to use the stacks
    # that are passed in, and suplement from the db. The passed stacks will
    # thus always be computed / used.
    if need:
        xcal = shocCampaign.load(RAW[kind], obstype=kind, recurse=True)
        # TODO: optimize this, should not need to load all these files
        gobj, gcal = run.match(xcal, *attrs)  # TODO: report=True
        cal = cal.join(list(filter(None, map(gcal.get, need))))
        need -= set(gobj.keys())

        logger.info('Found %i %s stacks:\n%s', len(cal), kind, cal.pformat())

    if need:
        logger.warning('Could not find %s for observed data with modes\n%s',
                       plural(kind), '\n'.join(map(str, need)))

    return cal, masters


def get_master(run, path, overwrite, kind, dark=None):

    # first check if the run contains calibration frames

    # group by obstype
    grp = run.group_by('obstype')
    gcal = grp.get(kind, shocCampaign())
    gobj, gcal = grp['object'].match(gcal, *MATCH[kind])

    masters = None
    action = 'Found'
    have_files = contains_fits(path, recurse=True)
    if gcal:

        if overwrite or not have_files:
            masters = compute_masters(gcal, path, overwrite, kind, dark)
            action = 'Computed'

    if (masters is None) and have_files:
        # no calibration stacks in run. Lookup master calibrators in db
        # TODO: optimize this, should not need to load all these files
        cal = shocCampaign.load(path)
        gobj, masters = run.match(cal, *MATCH[kind])
        # FIXME: not all may be matched here

    if masters is None:
        # could not find master calibration frames. Look for the raw stacks
        # that we can compute from
        cal = shocCampaign.load(RAW[kind], obstype=kind, recurse=True)
        # TODO: optimize this, should not need to load all these files
        gobj, gcal = grp['object'].match(cal, *MATCH[kind])
        # TODO: report=True
        masters = compute_masters(gcal, path, overwrite, kind, dark)

    if not masters:
        logger.warning('Could not find %s!', plural(kind))
        return

    # group
    masters = masters.group_by(gcal)
    # make sure items in group are shocHDU not shocCampaign
    masters.update({k: v[0] for k, v in masters.items()})

    tmp = masters.to_list()
    # attrs = OrderedSet(tmp.table.attrs) - {'target', 'timing.duration', 'nframes'}
    logger.info('%s master %s: \n%s', action, plural(kind), tmp.pformat())

    return masters


def get_combine_func(kind):
    if kind in ('dark', 'bias'):
        return np.median

    if kind == 'flat':
        return median_scaled_median

    raise ValueError(f'Unknown calibration type {kind}')


def compute_masters(stacks, outpath, overwrite, kind, dark_master=None,
                    png=True, **kws):

    logger.info('Computing master %s for %i stacks.',
                plural(kind), len(stacks))

    # debias flats
    if dark_master:
        try:
            g = stacks.group_by(dark_master).subtract(dark_master)
        except Exception as err:
            from IPython import embed
            import textwrap
            import traceback
            embed(header=textwrap.dedent(
                f"""\
                    Caught the following {type(err).__name__}:
                    %s
                    Exception will be re-raised upon exiting this embedded interpreter.
                    """) % traceback.format_exc())
            raise

    # all the stacks we need are here: combine
    masters = stacks.merge_combine(get_combine_func(kind))

    # save fits
    masters.save(outpath, overwrite=overwrite)

    # save png images
    if png:
        defaults = dict(figsize=[7.14, 5.55],
                        plims=(0, 100))
        for obs in masters.values():
            img = ImageDisplay(obs.data, **{**defaults, **kws})
            img.figure.savefig(obs.file.path.with_suffix('.png'),
                               bbox_inches='tight', pad_inches=0)
            plt.close(img.figure)

    return masters


def pre(run, folder, overwrite):

    # do pre reductions
    gobj, *masters = calibrate(run, folder, overwrite=False)

    # create image grid
    for i, m in enumerate(masters):
        if not m:
            continue

        title = ('{filters!s}; ', '{readout}; {binning}')[(not i):]
        logger.info('Master %s Images:', ('dark', 'flat')[i].title())
        thumbs = m.to_list().thumbnails(title=''.join(title))
        display(thumbs.figure)


def main(folder):
    #
    root_folder = Path(folder)
    output_folder = root_folder
    fig_folder = Path('/home/hannes/Documents/papers/dev/J1928/figures')
    rc('savefig', directory=fig_folder)

    # Load data
    run = shocCampaign.load(root_folder)

    # identify
    identify(run)

    # calibrate
    gobj, mbias, mflat = calibrate(run, output_folder, overwrite=False)

    # TODO group by source

    # register
    gobj.coalign_

    # track
    # photomerty
    # decorrelate
    # spectral analysis
