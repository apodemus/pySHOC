

# std
import sys
import atexit
import shutil
from pathlib import Path

# third-party
import click
from loguru import logger
from matplotlib import rcParams

# local
from obstools.sites.saao import telescopes
from recipes import io
from recipes.string import most_similar
from recipes.containers.dicts import groupby

# relative
from .. import HDU, config as cfg
from .._version import version as VERSION
from . import APPERTURE_SYNONYMS, SUPPORTED_APERTURES, logging, main as pipeline


# ---------------------------------------------------------------------------- #
PYENV = sys.version.split(' ', 1)[0]

# ---------------------------------------------------------------------------- #

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
        logger.info('Input root: {}.', parent)
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
    output = Path(output)
    if not output.is_absolute():
        output = root / output
    logger.info('Output root: {}.', output)
    return output


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
        return telescopes.get_name(value)


def resolve_target(_ctx, _param, value):
    if value == 'arget':
        raise click.BadParameter('Did you mean `--target`? (with 2x "-")')
    return value


def setup(root, output, overwrite, use_cache, config, verbose):
    """Setup results folder tree."""

    root = Path(root).resolve()
    if not (root.exists() and root.is_dir()):
        raise NotADirectoryError(str(root))

    # search for local config
    filename = 'config.yaml'
    if config := config or next((file for folder in (root, output)
                                 if (file := folder / filename).exists()), None):
        logger.info('Using local config: {!s}.', config)
        config = cfg.load(config)

        level = ['info', 'debug'][min(verbose, 1)].upper()
        config.logging.console['level'] = level
        logger.remove(logging._sink_ids[0])
        logging._sink_ids = logging.config()

    else:
        # use global config
        config = cfg.CONFIG
        # make a local copy
        shutil.copy(cfg.user_config_path, output / filename)

    # ------------------------------------------------------------------------ #
    # check for previous version stamp
    logger.debug('--overwrite is {}.', overwrite)
    vcf = output / '.version'
    if overwrite is None:
        if vcf.exists():
            old_version = vcf.read_text().strip()
            if overwrite := (old_version != str(VERSION)):
                logger.info(
                    'Pipeline results are from an older version of the '
                    'pipeline: {}. Current version is {}. Previous results will'
                    ' be overwritten with new results.', old_version, VERSION
                )
        else:
            overwrite = False
            logger.info('No verion file available at output {!s}. Overwrite is '
                        'False.', vcf)
    else:
        logger.info('Previous results will be {}.',
                    'overwritten' if overwrite else 'used if available')

    if overwrite:
        vcf.write_text(VERSION)

    #
    use_cache = bool(not overwrite if use_cache is None else use_cache)

    # ------------------------------------------------------------------------ #
    # path helper
    paths = cfg.PathManager(root, output, config)
    paths.create(ignore='calibration')

    # add log file sink
    logfile = paths.files.logging
    logger.add(logfile, colorize=False, **config.logging.file)
    atexit.register(logging.cleanup, logfile)

    # matplotlib interactive gui save directory
    rcParams['savefig.directory'] = output

    # set detection algorithm
    if algorithm := config.detection.get('algorithm', None):
        HDU.detection.algorithm = algorithm

    # update cache locations
    # HDU.get_sample_image.__cache__.disable()
    # HDU.detection.__call__.__cache__.disable()
    if use_cache:
        cache_path =  paths.folders.cache / PYENV
        enable_local_caching({
            # get_hdu_image_products: paths.folders.cache / 'image-samples.pkl'
            HDU.get_sample_image:              cache_path / 'sample-images.pkl',
            HDU.detection._algorithm.__call__: cache_path / 'source-regions.pkl'
        })

    return paths, overwrite


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
              default='./pyshoc',  # show_default=True,
              help='Output folder for data products. Default creates the '
                   '"pyshoc" folder under the root input folder.')
#
@click.option('-cfg', '--config',
              type=click.Path(),
              help='Path to pyshoc configuration yaml file. If not given, search '
                   'through input, output, user config folders in that order.')
@click.option('-t', '--target',
              callback=resolve_target,
              help='Name of the target. Will be used to retrieve object '
                   'coordinates and identify target in field.')
# @click.option('--target_is_folder',
#
@click.option('-tel', '--telescope',
              metavar='[74|40|lesedi]',  # TODO salt
              callback=resolve_tel,
              help='Name of the telescope that the observations were done with'
                   ' eg: "40in", "1.9m", "lesedi" etc. It is necessary to '
                   'specify this if multiple files are being reduced and the '
                   'fits header information is incomplete or incorrect. If input '
                   'files are from multiple telescopes, you should update the headers '
                   'before running the pipeline.')
@click.option('-top', type=int, default=5,
              help='Number of brightest sources to do photometry on.')
#
# @click.option('-aps', '--apertures',
#               type=click.Choice(SUPPORTED_APERTURES, case_sensitive=False),
#               #   metavar=f'[{"|".join(SUPPORTED_APERTURES)}]',
#               default='ragged', show_default=True,
#               callback=resolve_aperture,
#               help='The type(s) of apertures to use. If multiple '
#               'types are specified, photometry will be done for each type '
#               'concurrently. Abbreviated names are understood.')
# #
@click.option('--sub', type=click.IntRange(),
              help='For single file mode, the slice of data cube to consider. '
                   'Useful for debugging. Ignored if processing multiple fits '
                   'files.')
@click.option('-j', '--njobs', default=-1, show_default=True, type=click.INT)
# @click.option('--timestamps', type=click.Path(),
#               help='Location of the gps timestamp file for observation trigger '
#                    'time. Necessary for older SHOC data where this information '
#                    'is not available in the fits headers. File should have one '
#                    'timestamp per line in chronological order of the filenames.'
#                    ' The default is to look for a file named `gps.sast` or '
#                    '`gps.utc` in the processing root folder.')
#
@click.option('-w', '--overwrite', flag_value=True, default=None,
              help='Overwite pre-existing data products. Default is False => '
                   "Don't re-compute anything unless explicitly asked to. This "
                   'is safer and can save time on multiple re-runs of the '
                   'pipeline for the same data, but with different '
                   'configurations, for example.')
#
@click.option('--cache/--no-cache', default=True,
              help='Enable/Disable persistent caching. Cache location is '
                   'configurable in `config.yaml`')
#
@click.option('--plot/--no-plot', default=True,
              help='Switch plotting on or off.')
@click.option('--gui/--no-gui', default=True,
              help='Use mpl-multitab gui to embed interactive plots.')
@click.option('--cutouts/--no-cutouts', default=False,
              help='Display source cutouts in terminal.')
@click.option('-v', '--verbose', count=True)
@click.version_option()
def main(files_or_folder, output='./pyshoc', config=None,
         target=None, telescope=None,
         top=5,  # apertures='ragged',
         sub=..., njobs=-1,
         overwrite=False, cache=None,
         plot=True, gui=True, cutouts=True, verbose=0):
    """
    Main entry point for pyshoc pipeline command line interface.
    """

    # ------------------------------------------------------------------------ #
    # resolve & check inputs
    logger.section('Setup')
    root = get_root(files_or_folder)
    output = resolve_output(output, root)

    # setup
    paths, overwrite = setup(root, output, overwrite, cache, config, verbose)

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
    pipeline.main(paths, target, telescope, top, njobs, plot, gui, cutouts, overwrite)

    # except Exception as err:
    #     # catch errors so we can safely shut down any remaining processes
    #     from better_exceptions import format_exception

    #     logger.exception('Exception during pipeline execution.\n{}',
    #                      '\n'.join(format_exception(*sys.exc_info())))
