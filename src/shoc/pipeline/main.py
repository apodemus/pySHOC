"""
Photometry pipeline for the Sutherland High-Speed Optical Cameras.
"""


# std
import sys
import itertools as itt
from pathlib import Path

# third-party
import click
import cmasher as cmr
import more_itertools as mit
import matplotlib.pyplot as plt
from matplotlib import rc
from loguru import logger

# local
from motley.table import Table
from pyxides.vectorize import repeat
from obstools.image.registration import ImageRegister
from scrawl.imagine import ImageDisplay, plot_image_grid
from recipes.lists import cosort
from recipes.string import most_similar
from recipes.dicts import TreeLike, groupby
from recipes.io import iter_files, show_tree

# relative
from .. import shocCampaign, shocHDU
from . import (APPERTURE_SYNONYMS, SUPPORTED_APERTURES, WELCOME_BANNER,
               FolderTree, logging)
from .calibrate import calibrate


# setup logging for pipeline
logging.config()

#
ImageRegister.refining = False

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

# Sample thumbnails plot config
thumbnail_kws = dict(statistic='median',
                     figsize=(9, 7.5),
                     title_kws={'size': 'xx-small'})


# ---------------------------------------------------------------------------- #


# def contains_fits(path, recurse=False):
#     glob = path.rglob if recurse else path.glob
#     return bool(next(glob('*.fits'), False))


# def identify(run):
#     # identify
#     # is_flat = np.array(run.calls('pointing_zenith'))
#     # run[is_flat].attrs.set(repeat(obstype='flat'))

#     g = run.guess_obstype()


def enable_local_caching(mapping):
    for func, folder in mapping.items():
        cache = func.__cache__
        cache.enable(folder / cache.path.name)


def get_root(files_or_folder, _level=0):
    files_or_folder = map(Path, files_or_folder)
    folders = groupby(files_or_folder, Path.is_dir)
    parent, *ambiguous = {*folders.get(True, ()),
                          *map(Path.parent.fget, folders.get(False, ()))}
    if not ambiguous:
        return parent

    # Files with multiple parents.
    if _level < 2:
        return get_root(ambiguous, _level + 1)

    raise ValueError('Please provide an output folder for results '
                     'eg: -o path/to/results. ')


def resolve_output(output, root):
    if Path(output).is_absolute():
        return output
    return (root / output).resolve()


def resolve_aperture(_ctx, _param, value):
    match = value.lower()
    match = APPERTURE_SYNONYMS.get(match, match)
    if match in SUPPORTED_APERTURES:
        return match

    match = most_similar(match, SUPPORTED_APERTURES)
    if match:
        logger.info('Interpreting aperture name {!r} as {!r}.', value, match)
        return match

    raise click.BadParameter(
        f'{value!r}. Valid choices are: {SUPPORTED_APERTURES}')


def check_target(run):
    targets = set(run.attrs.target)
    if invalid := targets.intersection({None, ''}):
        raise ValueError(f'Invalid target {invalid.pop()!r}')

    if len(targets) > 1:
        raise NotImplementedError(
            f'Targets are: {targets}.Running the pipeline for multiple targets'
            f' simultaneously is currently not supported.'
        )

    return targets.pop()


def setup(root, output):
    """Setup results folder."""
    root = Path(root)
    if not (root.exists() and root.is_dir()):
        raise NotADirectoryError(str(root))

    #
    paths = FolderTree(root, output)
    paths.create()

    # add log file sink
    logger.add(paths.logs / 'main.log', level='DEBUG', format=logging.formatter,
               colorize=False)

    # interactive gui save directory
    rc('savefig', directory=paths.plots)

    # update cache locations
    enable_local_caching({
        shocHDU.get_sample_image: paths.output,
        shocHDU.detect: paths.output
    })

    return paths


def welcome(banner):
    def wrapper(func):
        print(banner)
        return func
    return wrapper

# CLI


@welcome(WELCOME_BANNER)    # say hello
@click.command()
@click.argument('files_or_folder', nargs=-1, type=click.Path())
# required=True)
#
@click.option('-o', '--output', type=click.Path(),
              default='./.pyshoc', show_default=True,
              help='Output folder for data products. Default creates the '
                   '".pyshoc" folder under the root input folder.')
#
@click.option('-t', '--target',
              help='Name of the target. Will be used to retrieve object '
              'coordinates.')
#
@click.option('-tel', '--telescope',
              help='Name of the telescope that the observations where done with'
                   ' eg: "40in", "1.9m", "lesedi" etc. It is sometimes '
                   'necessary to specify this if the fits header information is'
                   ' missing or incorrect, but can otherwise be ignored.')
#
@click.option('--top', type=int, default=5,
              help='Number of brightest sources to do photometry on.')
#
@click.option('-aps', '--apertures',
              #   type=click.Choice(SUPPORTED_APERTURES, case_sensitive=False),
              default='ragged', show_default=True,
              callback=resolve_aperture,
              help='The type(s) of apertures to use. If multiple'
                   'types are specified, photometry will be done for each type'
                   'concurrently.')
#
@click.option('--sub',  type=click.IntRange(),
              help='For single file mode, the slice of data cube to consider. '
              'Useful for debugging. Ignored in multi-file mode.')
#
# @click.option('--timestamps', type=click.Path(),
#               help='Location of the gps timestamp file for observation trigger '
#                    'time. Necessary for older SHOC data where this information '
#                    'is not available in the fits headers. File should have one '
#                    'timestamp per line in chronological order of the filenames.'
#                    ' The default is to look for a file named `gps.sast` or '
#                    '`gps.utc` in the processing root folder.')
#
@click.option('--overwrite/--xx', default=False,
              help='Overwite pre-existing data products. Default is False => '
              'Don\'t re-compute anything unless explicitly asked to. This is '
              'safer and can save time on multiple re-runs of the pipeline.')
#
# @click.option('--gui')
@click.version_option()
def main(files_or_folder, output, target, telescope, top, apertures, sub,
         overwrite):
    """
    Main entry point for pyshoc pipeline.
    """
    if not files_or_folder:
        sys.exit('Please provide a folder or filename(s) for reduction.')

    # ------------------------------------------------------------------------ #
    # resolve & check inputs
    root = get_root(files_or_folder)
    output = resolve_output(output, root)

    # setup
    paths = setup(root, output)

    #
    single_file_mode = (len(files_or_folder) == 1 and
                        root.exists() and
                        root.is_file() and
                        root.name.lower().endswith('fits'))
    if not single_file_mode and sub:
        logger.info('Ignoring sub {} for multi-file run.', sub)

    # -------------------------------------------------------------------------#
    try:
        # pipeline main routine
        _main(paths, target, telescope, top, overwrite)

    except Exception as err:
        # catch errors so we can safely shut down any remaining processes
        from better_exceptions import format_exception

        logger.exception('Exception during pipeline execution.\n{}',
                         '\n'.join(format_exception(*sys.exc_info())))


def _main(paths, target, telescope, top,  overwrite):
    from obstools.phot import PhotInterface

    # ------------------------------------------------------------------------ #
    # This is needed because rcParams['savefig.directory'] doesn't work for
    # fig.savefig
    def savefig(fig, name, **kws):
        return fig.savefig(paths.plots / name, **kws)

    # ------------------------------------------------------------------------ #
    # Load data
    run = shocCampaign.load(paths.root, obstype='object')
    # update info if given
    run.attrs.set(repeat(telescope=telescope,
                         target=target))
    # HACK
    # run['202130615*'].calls('header.remove', 'DATE-OBS')

    # Print summary table
    daily = run.group_by('date')
    logger.info('\n{:s}\n', daily.pformat(titled=repr))

    # write summary spreadsheet
    run.tabulate.to_xlsx(paths.summary)
    logger.info("The table above is available in spreadsheet format at: "
                "'{!s:}'", paths.summary)

    # sample images and header info
    compute_preview_products(run, paths)

    # ------------------------------------------------------------------------ #
    # Calibrate

    # Compute/retrieve master dark/flat. Point science stacks to calibration
    # images.
    gobj, mdark, mflat = calibrate(run, overwrite=False)

    # Sample thumbnails (after calibration)
    previous = get_data_products(run, paths)
    if not previous['Images']['Overview'][1]:
        thumbs = run.thumbnails(**thumbnail_kws)
        savefig(thumbs.fig, paths.plots / 'thumbs-cal.png')

    # have to ensure we have single target here
    target = check_target(run)

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
    if not previous['Images']['Regions']:
        ui = reg.plot_detections()
        ui.save(filenames=[f'{hdu.file.stem}.regions.png' for hdu in run],
                path=paths.plots)

    # phot = PhotInterface(run, reg, paths.phot)
    # ts = mv phot.ragged() phot.regions()

    # Write data products spreadsheet
    write_data_products_xlsx(run, paths)

    logger.info('The following data products were created:\n{}',
                show_tree(paths.output))


def plot_detections(run, segments, images, loc, dilate=2):

    # plot ragged apertures
    # TODO: move to phot once caching works

    for im, seg, hdu in zip(images, segments, run):
        seg.dilate(dilate)
        img = ImageDisplay(im, cmap=cmr.voltage_r)
        seg.show_contours(img.ax, cmap=cmr.pride, lw=1.5)
        seg.show_labels(img.ax, color='w', size='xx-small')
        img.save(loc / f'{hdu.file.stem}.regions.png')


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
