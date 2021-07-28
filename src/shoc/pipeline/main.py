

# std libs
import logging
import logging.config
import multiprocessing as mp

# third-party libs
import cmasher as cmr
from matplotlib import rc

# local libs
from obstools.phot.core import PhotInterface
from recipes.logging import logging, get_module_logger

# relative libs
from .calibrate import calibrate
from .. import shocCampaign, shocHDU
from . import logs, WELCOME_BANNER, FolderTree
from pyxides.vectorize import repeat

# TODO group by source

# track
# photomerty
# decorrelate
# spectral analysis

# rc('savefig', directory=FIGPATH)
# rc('text', usetex=False)
rc('font', size=14)
rc('axes', labelweight='bold')


# module level logger
logger = get_module_logger()
logging.basicConfig()
logger.setLevel(logging.INFO)


def contains_fits(path, recurse=False):
    glob = path.rglob if recurse else path.glob
    return bool(next(glob('*.fits'), False))


def identify(run):
    # identify
    # is_flat = np.array(run.calls('pointing_zenith'))
    # run[is_flat].attrs.set(repeat(obstype='flat'))

    g = run.guess_obstype()


# def get_sample_image(hdu)

def setup(root):
    # setup results folder
    paths = FolderTree(root)
    paths.create()
    # update cache locations
    sample_cache = shocHDU.get_sample_image.__cache__
    sample_cache.filename = paths.sample / sample_cache.path.name

    return paths


def main(path, target=None):

    # say hello
    print(WELCOME_BANNER)

    # ------------------------------------------------------------------------ #
    # setup
    paths = setup(path)
    target = target or paths.base.name

    # -------------------------------------------------------------------------#
    # logger.info('Creating log listener')
    # logQ = mp.Queue()  # The logging queue for workers
    # # TODO: open logs in append mode if resume
    # config_main, config_listener, config_worker = logs.config(paths.logs, logQ)
    # #
    # logging.config.dictConfig(config_main)

    # # create log listener process
    # stop_logging_event = mp.Event()
    # logListener = mp.Process(name='logListener',
    #                          target=logs.listener_process,
    #                          args=(logQ, stop_logging_event, config_listener))
    # logListener.start()
    # logger.info('Log listener active')

    try:
        # pipeline main work
        _main(paths, target)

    except Exception:
        # catch errors so we can safely shut down the listeners
        logger.exception('Exception during pipeline execution.')
        # plot_diagnostics = False
        # plot_lightcurves = False
    else:
        # Workers all done, listening can now stop.
        # logger.info('Telling listener to stop ...')
        # stop_logging_event.set()
        # logListener.join()
        pass


def _main(paths, target):
    # ------------------------------------------------------------------------ #
    # Load data
    run = shocCampaign.load(paths.input, obstype='object')
    run.attrs.set(repeat(telescope='74in',
                  target=target))
    # HACK
    # run['202130615*'].calls('header.remove', 'DATE-OBS')

    #
    daily = run.group_by('date')
    daily.pprint(titled=repr)

    # Sample thumbnails (before calibration)
    thumbnail_kws = dict(statistic='median',
                         figsize=(9, 7.5),
                         title_kws={'size': 'xx-small'})
    thumbs = run.thumbnails(**thumbnail_kws)
    thumbs.fig.savefig(paths.plots / 'thumbs.png')

    # ------------------------------------------------------------------------ #
    # Calibrate

    # Compute/retrieve master dark/flat. Point calibration images to science stacks
    gobj, mdark, mflat = calibrate(run, overwrite=False)

    # Sample thumbnails (after calibration)
    thumbs = run.thumbnails(**thumbnail_kws)
    thumbs.fig.savefig(paths.plots / 'thumbs-cal.png')

    # Image Registration
    reg = run.coalign_dss(deblend=True)
    mos = reg.mosaic(cmap=cmr.chroma,
                     # regions={'alpha': 0.35}, labels=False
                     )
    mos.fig.savefig('mosaic.png')

    # txt, arrows = mos.mark_target(
    #     run[0].target,
    #     arrow_head_distance=2.,
    #     arrow_size=10,
    #     arrow_offset=(1.2, -1.2),
    #     text_offset=(4, 5), size=12, fontweight='bold'
    # )

    # %%

    pi = PhotInterface()

    for date, obs in daily.items():
        filename = paths.phot / f'{date!r}-ragged.txt'
        pi.ragged(obs, filename)
