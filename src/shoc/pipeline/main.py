

# third-party
import cmasher as cmr
from matplotlib import rc
from loguru import logger

# local
from recipes import op
from pyxides.vectorize import repeat

# relative
from .. import shocCampaign, shocHDU
from . import FolderTree, logging
from .calibrate import calibrate



# t0 = time.time()
# TODO group by source

# track
# photomerty
# decorrelate
# spectral analysis

# rc('savefig', directory=FIGPATH)
# rc('text', usetex=False)
rc('font', size=14)
rc('axes', labelweight='bold')

# ---------------------------------------------------------------------------- #


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
    if not paths.root.exists():
        raise NotADirectoryError(str(root))
    #
    paths.create()

    # setup logging for pipeline
    logging.config(paths.logs / 'main.log')

    # interactive gui save directory
    rc('savefig', directory=paths.plots)

    # update cache locations
    reset_cache_paths({
        shocHDU.get_sample_image: paths.output,
        shocHDU.detect: paths.output
    })

    return paths


def main(path, target=None):

    # ------------------------------------------------------------------------ #
    # setup
    paths = setup(path)
    target = target or paths.root.name

    # -------------------------------------------------------------------------#
    #

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
    logger.info('\n{:s}', daily.pformat(titled=repr))

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
                        # regions={'alpha': 0.35}, labels=Falseyou
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

    from scrawl.imagine import ImageDisplay

    # plot ragged apertures
    # TODO: move to phot once caching works
    dilate = 2
    for im, seg, hdu in zip(thumbs.images, reg.detections[1:], run):
        seg.dilate(dilate)
        img = ImageDisplay(im, cmap=cmr.voltage_r)
        seg.show_contours(img.ax, cmap='hot', lw=1.5)
        seg.show_labels(img.ax, color='w', size='xx-small')

def basename(path):
    return path.name.rsplit('.', 2)[0]


# def _gdp(hdu, paths):
#     def _get(path, suffix):
#         trial = path.with_suffix('.' + suffix.lstrip('.'))
#         return trial if trial.exists() else None

#     products = TreeLike()

#     base = hdu.file.basename
#     stem = hdu.file.stem
#     products['FITS headers'] =   _get(paths.headers / stem, '.txt')
#     products['Image Samples']  = _get(paths.image_samples / stem, '.png')

#     lcx = ('txt', 'npy')

#     individual = [_get(paths.phot / base, ext) for ext in lcx]
#     combined = [_get(paths.phot, ext) for ext in lcx]

#     products['Light Curves']['Raw'] = [individual, combined]


def get_data_products(run, paths):

    products = TreeLike()
    timestamps = run.attrs('t.t0')
    _, stems = cosort(timestamps, run.files.stems)
    bases = sorted(repr(date).replace('-', '') for date in run.attrs.date)

    def row_assign(filenames, reference=stems, empty=''):
        incoming = {base: list(vals) for base, vals in
                    itt.groupby(sorted(filenames), basename)}
        for base in reference:
            yield incoming.get(base, empty)

        # return (incoming.get(base, ()) for base in basenames)

    # ['Spectral Estimates', ]
    # 'Periodogram','Spectrogram'
    products['FITS']['files'] = run.files.paths
    products['FITS']['headers'] = list(paths.headers.iterdir())

    # Images
    images = sorted(paths.image_samples.iterdir())
    if images:
        products['Images']['Samples'] = dict(zip(('start', 'mid', 'end'),
                                                 zip(*row_assign(images))))

    products['Images']['Source Regions'] = \
        list(row_assign(paths.source_regions.iterdir()))

    products['Images']['Overview'] = [
        next(paths.plots.glob(f'{name}.png'), '')
        for name in ('thumbs', 'thumbs-cal', 'mosaic')]
    # [name] =

    # Light curves
    # cmb_txt = iter_files(paths.phot, 'txt')
    individual = iter_files(paths.phot, '*.*.{txt,npy}', recurse=True)
    combined = iter_files(paths.phot, '*.{txt,npy}')
    products['Light Curves']['Raw'] = [
        (*indiv, *cmb) for indiv, cmb in
        zip(row_assign(individual, empty=['']),
            row_assign(combined, bases))
    ]

    return products


def write_data_products_xlsx(run, paths, filename=None):
    def hyperlink_ext(path):
        return f'=HYPERLINK("{path}", "{path.suffix[1:]}")'

    def hyperlink_path(path):
        return f'=HYPERLINK("{path}", "{path.name}")'

    #
    products = get_data_products(run, paths)
    # duplicate Overview images so that they get merged below
    products['Images']['Overview'] = [products['Images']['Overview']] * len(run)

    # create table
    tbl = Table.from_dict(products,
                          title='Data Products',
                          convert={Path: hyperlink_ext,
                                   'files': hyperlink_path,
                                   'Overview': hyperlink_path},
                          split_nested={tuple, list},
                          formatter=';;;[Blue]@')
    # write
    return tbl.to_xlsx(filename or paths.products,
                       widths={'Overview': 4,
                               'files': 23,
                               'headers': 10,
                               'Source Regions': 10,
                               ...: 7},
                       align={'files': '<',
                              'Overview': dict(horizontal='center',
                                               vertical='center',
                                               text_rotation=90),
                              ...: dict(horizontal='center',
                                        vertical='center')},
                       merge_unduplicate=('data', 'headers'))


def intervals(hdu, n_intervals):
    n = hdu.nframes
    yield from mit.pairwise(range(0, n + 1, n // n_intervals))


def get_sample_image(hdu, stat, min_depth, interval, save_as='png', path='.'):
    image = hdu.get_sample_image(stat, min_depth, interval)
    if save_as:
        im = ImageDisplay(image)
        i, j = interval
        im.save(path / f'{hdu.file.stem}.{i}-{j}.{save_as}')
        plt.close(im.figure)


def get_image_samples(run, stat='median', min_depth=5, n_intervals=3,
                      save_as='png', path='.'):

    # sample = delayed(get_sample_image)
    # with Parallel(n_jobs=1) as parallel:
    # return parallel
    return [
        hdu.get_sample_image(stat, min_depth, (i, j), save_as, path)
        for hdu in run
        for i, j in intervals(hdu, n_intervals)
    ]


def compute_preview_products(run, paths):
    # get results from previous run
    previous = get_data_products(run, paths)

    # write fits headers to text
    if not previous['FITS']['headers']:
        logger.info('Writing fits headers to text for {:d} files.', len(run))
        for hdu in run:
            hdu.header.totextfile(paths.headers / f'{hdu.file.stem}.txt')

    # plot image samples
    if not previous['Images']:
        sample_images = get_image_samples(run, path=paths.image_samples)

    # plot thumbnails for sample image from first portion of each data cube
    if not previous['Images']['Overview'][0]:
        portion = mit.chunked(sample_images, len(run))

        thumbs = plot_image_grid(next(portion), titles=run.files.names,
                                 title_kws=thumbnail_kws)
        thumbs.fig.savefig(paths.plots / 'thumbs.png')
