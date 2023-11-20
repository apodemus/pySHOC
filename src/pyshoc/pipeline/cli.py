

# std
import atexit
from pathlib import Path

# third-party
import click
from loguru import logger
from matplotlib import rcParams

# local
from recipes import io
from recipes.dicts import groupby
from recipes.string import most_similar

# relative
from .. import CONFIG, shocHDU
from ..core import get_tel
from ..config import PathConfig, _prefix_paths
from . import APPERTURE_SYNONYMS, SUPPORTED_APERTURES, logging, main as pipeline


def check_files_exist(files_or_folder):
    for path in files_or_folder:
        if not Path(path).exists():
            raise click.BadParameter(f'File does not exist: {path!s}')
            # raise FileNotFoundError(path)


def _resolve_files(files):
    if not isinstance(files, str) and len(files) == 1:
        files = files[0]

    if not files:
        return

    if ',' in files:
        files = list(map(Path, map(str.strip, files.split(','))))
        check_files_exist(files)
        return files

    return list(io.iter_files(files))


def resolve_files(ctx, param, files):

    if files := _resolve_files(files):
        return files

    click.echo(f'Could not resolve any files for input {files}')
    while True:
        try:
            return click.prompt(
                'Please provide a folder or filename(s) for reduction',
                value_proc=_resolve_files
            )
        except ValueError as err:
            click.echo(str(err))


def get_root(files_or_folder, _level=0):

    files_or_folder = iter(files_or_folder)

    folders = groupby(Path.is_dir, files_or_folder)
    parent, *ambiguous = {*folders.get(True, ()),
                          *map(Path.parent.fget, folders.get(False, ()))}
    if not ambiguous:
        logger.info('Input root: {}', parent)
        return parent

    # Files with multiple parents.
    if _level < 2:
        return get_root(ambiguous, _level + 1)

    raise ValueError(
        "Since the input files are from different system folders, I'm not "
        'sure where to put the results directory (normally the default is the '
        'input folder). Please provide an output folder for results '
        'eg: -o /path/to/results'
    )


def resolve_output(output, root):
    out = _prefix_paths(output, root)
    logger.info('Output root: {}', out)
    return out


def resolve_aperture(_ctx, _param, value):
    match = value.lower()
    match = APPERTURE_SYNONYMS.get(match, match)
    if match in SUPPORTED_APERTURES:
        return match

    # match
    if match := most_similar(match, SUPPORTED_APERTURES):
        logger.info('Interpreting aperture name {!r} as {!r}.', value, match)
        return match

    raise click.BadParameter(
        f'{value!r}. Valid choices are: {SUPPORTED_APERTURES}')


def resolve_tel(_ctx, param, value):
    if value is not None:
        return get_tel(value)


def resolve_target(_ctx, _param, value):
    if value == 'arget':
        raise click.BadParameter('Did you mean `--target`? (with 2x "-")')
    return value


def setup(root, output, use_cache):
    """Setup results folder tree."""

    root = Path(root).resolve()
    if not (root.exists() and root.is_dir()):
        raise NotADirectoryError(str(root))

    # path helper
    paths = PathConfig.from_config(root, output, CONFIG)
    paths.create(ignore='calibration')

    # add log file sink
    logfile = paths.files.logging
    logger.add(logfile, colorize=False, **CONFIG.logging.file)
    atexit.register(logging.cleanup, logfile)

    # matplotlib interactive gui save directory
    rcParams['savefig.directory'] = paths.folders.plotting

    # set detection algorithm
    if algorithm := CONFIG.detection.pop('algorithm', None):
        shocHDU.detection.algorithm = algorithm

    # update cache locations
    # shocHDU.get_sample_image.__cache__.disable()
    # shocHDU.detection.__call__.__cache__.disable()
    if use_cache:
        enable_local_caching({
            # get_hdu_image_products: paths.folders.cache / 'image-samples.pkl'
            shocHDU.get_sample_image:              paths.folders.cache / 'sample-images.pkl',
            shocHDU.detection._algorithm.__call__: paths.folders.cache / 'source-regions.pkl'
        })

    return paths


def enable_local_caching(mapping):
    for func, filename in mapping.items():
        func.__cache__.enable(filename)


# CLI
# ---------------------------------------------------------------------------- #

@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.argument('files_or_folder', nargs=-1, callback=resolve_files)
# required=True)
#
@click.option('-o', '--output', type=click.Path(),
              default='./.pyshoc',  # show_default=True,
              help='Output folder for data products. Default creates the '
                   '".pyshoc" folder under the root input folder.')
#
@click.option('-t', '--target',
              callback=resolve_target,
              help='Name of the target. Will be used to retrieve object '
                   'coordinates and identify target in field.')
# @click.option('--target_is_folder',
#
@click.option('-tel', '--telescope',
              metavar='[74|40|lesedi]',  # TODO salt
              callback=resolve_tel,
              help='Name of the telescope that the observations where done with'
                   ' eg: "40in", "1.9m", "lesedi" etc. It is necessary to '
                   'specify this if multiple files are being reduced and the '
                   'fits header information is missing or incorrect. If input '
                   'files are from multiple telescopes, update the headers '
                   'before running the pipeline.')
#
@click.option('-top', type=int, default=5,
              help='Number of brightest sources to do photometry on.')
#
@click.option('-aps', '--apertures',
              type=click.Choice(SUPPORTED_APERTURES, case_sensitive=False),
              #   metavar=f'[{"|".join(SUPPORTED_APERTURES)}]',
              default='ragged', show_default=True,
              callback=resolve_aperture,
              help='The type(s) of apertures to use. If multiple '
              'types are specified, photometry will be done for each type '
              'concurrently. Abbreviated names are understood.')
#
@click.option('--sub', type=click.IntRange(),
              help='For single file mode, the slice of data cube to consider. '
                   'Useful for debugging. Ignored if processing multiple fits '
                   'files.')
#
# @click.option('--timestamps', type=click.Path(),
#               help='Location of the gps timestamp file for observation trigger '
#                    'time. Necessary for older SHOC data where this information '
#                    'is not available in the fits headers. File should have one '
#                    'timestamp per line in chronological order of the filenames.'
#                    ' The default is to look for a file named `gps.sast` or '
#                    '`gps.utc` in the processing root folder.')
#
@click.option('-w', '--overwrite', flag_value=True,  default=False,
              help='Overwite pre-existing data products. Default is False => '
                   "Don't re-compute anything unless explicitly asked to. This "
                   'is safer and can save time on multiple re-runs of the '
                   'pipeline for the same data, but with a different kind of '
                   'aperture, etc.')
#
@click.option('--cache/--no-cache', default=True,
              help='Enable/Disable caching.')
#
@click.option('--plot/--no-plot', default=True,
              help='Show figure windows.')
@click.option('--cutouts/--no-cutouts', default=True,
              help='Display source cutouts in terminal.')
@click.version_option()
def main(files_or_folder, output='./.pyshoc',
         target=None, telescope=None,
         top=5, apertures='ragged', sub=...,
         overwrite=False, cache=None,
         plot=True, cutouts=True):
    """
    Main entry point for pyshoc pipeline command line interface.
    """

    # ------------------------------------------------------------------------ #
    # resolve & check inputs
    logger.section('Setup')
    root = get_root(files_or_folder)
    output = resolve_output(output, root)
    logger.debug('--overwrite is {}', overwrite)
    logger.info('Previous results will be {}.',
                'overwritten' if overwrite else 'used if available')

    # setup
    cache = bool(not overwrite if cache is None else cache)
    paths = setup(root, output, cache) # 

    # check if multiple input
    single_file_mode = (len(files_or_folder) == 1 and
                        root.exists() and
                        root.is_file() and
                        root.name.lower().endswith('fits'))
    if not single_file_mode and sub:
        logger.info('Ignoring option sub {} for multi-file run.', sub)

    # -------------------------------------------------------------------------#
    # try:

    # pipeline main routine
    pipeline.main(paths, target, telescope, top, plot, cutouts, overwrite)

    # except Exception as err:
    #     # catch errors so we can safely shut down any remaining processes
    #     from better_exceptions import format_exception

    #     logger.exception('Exception during pipeline execution.\n{}',
    #                      '\n'.join(format_exception(*sys.exc_info())))
