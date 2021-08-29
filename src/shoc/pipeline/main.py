

# std
import logging

# third-party
import cmasher as cmr
from matplotlib import rc
from pyxides.vectorize import repeat

# local
from recipes.logging import logging

# relative
from . import FolderTree
from .calibrate import calibrate
from .. import shocCampaign, shocHDU


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
# logger = get_module_logger()
# logging.basicConfig()
logger = logging.getLogger()
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

def reset_cache_paths(mapping):
    for func, folder in mapping.items():
        cache = func.__cache__
        cache.filename = folder / cache.path.name


def setup(root):
    # setup results folder
    paths = FolderTree(root)
    paths.create()

    # interactive gui save directory
    rc('savefig', directory=paths.plots)

    # update cache locations
    reset_cache_paths({shocHDU.get_sample_image: paths.output,
                       shocHDU.detect: paths.output})

    return paths


def main(path, target=None):

    # ------------------------------------------------------------------------ #
    # setup
    paths = setup(path)
    target = target or paths.input.name

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
    from obstools.phot import PhotInterface

    # ------------------------------------------------------------------------ #
    # This is needed because rcParams['savefig.directory'] doesn't work for
    # fig.savefig
    def savefig(fig, name, **kws):
        return fig.savefig(paths.plots / name, **kws)

    # ------------------------------------------------------------------------ #
    # Load data
    run = shocCampaign.load(paths.input, obstype='object')
    run.attrs.set(repeat(telescope='74in',
                         target=target))
    # HACK
    # run['202130615*'].calls('header.remove', 'DATE-OBS')

    #
    daily = run.group_by('date')
    logger.info('\n%s', daily.pformat(titled=repr))

    # Sample thumbnails (before calibration)
    thumbnail_kws = dict(statistic='median',
                         figsize=(9, 7.5),
                         title_kws={'size': 'xx-small'})
    thumbs = run.thumbnails(**thumbnail_kws)
    savefig(thumbs.fig, 'thumbs.png')

    # ------------------------------------------------------------------------ #
    # Calibrate

    # Compute/retrieve master dark/flat. Point science stacks to calibration
    # images.
    gobj, mdark, mflat = calibrate(run, overwrite=False)

    # Sample thumbnails (after calibration)
    thumbs = run.thumbnails(**thumbnail_kws)
    savefig(thumbs.fig, 'thumbs-cal.png')

    # Image Registration
    reg = run.coalign_dss(deblend=True)
    mosaic = reg.mosaic(cmap=cmr.chroma,
                        # regions={'alpha': 0.35}, labels=False
                        )
    savefig(mosaic.fig, 'mosaic.png', bbox_inches='tight')

    # txt, arrows = mos.mark_target(
    #     run[0].target,
    #     arrow_head_distance=2.,
    #     arrow_size=10,
    #     arrow_offset=(1.2, -1.2),
    #     text_offset=(4, 5), size=12, fontweight='bold'
    # )

    # %%

    phot = PhotInterface(run, reg, paths.phot)
    ts = phot.ragged()
