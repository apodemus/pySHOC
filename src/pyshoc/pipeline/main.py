"""
Photometry pipeline for the Sutherland High-Speed Optical Cameras.
"""


# std
import sys
import itertools as itt
from pathlib import Path
from collections import defaultdict

# third-party
import numpy as np
import more_itertools as mit
from loguru import logger
from matplotlib import rcParams
from mpl_multitab import MplMultiTab, QtWidgets

# local
import motley
from motley.table import Table
from pyxides.vectorize import repeat
from scrawl.image import plot_image_grid
from obstools.image import SkyImage
from obstools.modelling import int2tup
from recipes.dicts import DictNode
from recipes import cosort, io, pprint as pp
from recipes.decorators.reporting import trace

# relative
from .. import CONFIG, shocCampaign
from . import logging
from .calibrate import calibrate


# ---------------------------------------------------------------------------- #
# setup logging for pipeline
logging.config()


CONSOLE_CUTOUTS_TITLE = motley.stylize(CONFIG.console.cutouts.pop('title'))

# ---------------------------------------------------------------------------- #


# t0 = time.time()
# TODO group by source

# track
# photomerty
# decorrelate
# spectral analysis

# rc('savefig', directory=FIGPATH)
# rc('text', usetex=False)
rcParams.update({'font.size': 14,
                 'axes.labelweight': 'bold',
                 'image.cmap': CONFIG.plotting.cmap})

# ---------------------------------------------------------------------------- #


# def contains_fits(path, recurse=False):
#     glob = path.rglob if recurse else path.glob
#     return bool(next(glob('*.fits'), False))


# def identify(run):
#     # identify
#     # is_flat = np.array(run.calls('pointing_zenith'))
#     # run[is_flat].attrs.set(repeat(obstype='flat'))

#     g = run.guess_obstype()


# ---------------------------------------------------------------------------- #

def basename(path):
    return path.name.rsplit('.', 2)[0]


def check_single_target(run):

    targets = set(run.attrs('target'))
    if invalid := targets.intersection({None, ''}):
        raise ValueError(f'Invalid target {invalid.pop()!r}')

    if len(targets) > 1:
        raise NotImplementedError(
            f'Targets are: {targets}. Running the pipeline for multiple targets'
            f' simultaneously is currently not supported.'
        )

    return targets.pop()


def check_required_info(run, telescope, target):
    info = {}
    # check if info required
    if telescope is None:  # default
        if None in set(run.attrs.telescope):
            run.pprint()
            raise ValueError('Please provide telescope name, eg: -tel 74')
    else:
        info['telescope'] = telescope

    if target is None:
        targets = [*(set(run.attrs('target')) - {None})]
        if len(targets) > 1:
            raise ValueError(
                f'Fits headers indicate multiple targets: {targets}. Only '
                'single target campaigns are currently supported by this data '
                'reduction pipeline. If the fits headers are incorrect, you '
                'may provide the target name eg: --target "HU Aqr".'
            )

        target, = targets

    if (len(run) > 1) and not target:
        raise ValueError(
            'Could not find target name in fits headers. Please provide '
            'this eg via: --target HU Aqr.'
        )
    else:
        info['target'] = target

    return info


# def _gdp(hdu, paths):
#     def _get(path, suffix):
#         trial = path.with_suffix('.' + suffix.lstrip('.'))
#         return trial if trial.exists() else None

#     products = TreeLike()

#     base = hdu.file.basename
#     stem = hdu.file.stem
#     products['FITS headers'] =   _get(paths.headers / stem, '.txt')
#     products['Image Samples']  = _get(paths.sample_images / stem, '.png')

#     lcx = ('txt', 'npy')

#     individual = [_get(paths.phot / base, ext) for ext in lcx]
#     combined = [_get(paths.phot, ext) for ext in lcx]

#     products['Light Curves']['Raw'] = [individual, combined]


def row_assign(run, filenames, reference=None, empty=''):

    incoming = {base: list(vals) for base, vals in
                itt.groupby(sorted(mit.collapse(filenames)), basename)}

    if reference is None:
        _, reference = cosort(run.attrs('t.t0'), run.files.basenames)

    for base in reference:
        yield incoming.get(base, empty)


def get_previous_data_products(run, paths):

    products = DictNode()
    timestamps = run.attrs('t.t0')
    _, stems = cosort(timestamps, run.files.stems)
    bases = sorted(repr(date).replace('-', '') for date in run.attrs.date)

    # ['Spectral Estimates', ]
    # 'Periodogram','Spectrogram'
    products['FITS']['files'] = run.files.paths
    products['FITS']['headers'] = list(paths.headers.iterdir())

    # Images
    products['Images']['Samples'] = list(
        row_assign(run, sorted(paths.sample_images.iterdir())))

    products['Images']['Source Regions'] = list(row_assign(run, paths.source_regions.iterdir()))

    products['Images']['Overview'] = [
        file for name in (CONFIG.files.thumbs, CONFIG.files.thumbs_cal, CONFIG.files.mosaic)
        if (file := paths.plots / name).exists()
    ]
    # [name] =

    # Light curves
    # cmb_txt = io.iter_files(paths.phot, 'txt')
    individual = io.iter_files(paths.phot, '*.*.{txt,npy}', recurse=True)
    combined = io.iter_files(paths.phot, '*.{txt,npy}')
    products['Light Curves']['Raw'] = [
        (*indiv, *cmb) for indiv, cmb in
        zip(row_assign(run, individual, empty=['']),
            row_assign(run, combined, bases))
    ]

    logger.opt(lazy=True).debug(
        'Found previous data products: \n{}',
        lambda: products.pformat(rhs=_show_products(paths.root))
    )

    return products


class _show_products:
    def __init__(self, root):
        self.root = root

    def __call__(self, files):

        if isinstance(files, (list, tuple)):
            if any(files):
                return pp.collection(type(files)(map(self, files)),
                                     sep=',\n ', fmt=_qstr)
            return str(files)

        if isinstance(files, Path):
            return str(files.relative_to(self.root)
                       if self.root in files.parents
                       else files)
        return files


def _qstr(s):
    return s if '\n' in s else repr(s)


def write_data_products_xlsx(run, paths, filename=None):
    def hyperlink_ext(path):
        return f'=HYPERLINK("{path}", "{path.suffix[1:]}")'

    def hyperlink_path(path):
        return f'=HYPERLINK("{path}", "{path.name}")'

    #
    products = get_previous_data_products(run, paths)
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


def get_intervals(hdu, subset, n_intervals):
    n = hdu.nframes
    if subset:
        yield slice(*int2tup(subset)).indices(n)[:2]
        return

    yield from mit.pairwise(range(0, n + 1, n // n_intervals))


def _get_hdu_samples(hdu, detection, show_cutouts):

    stat = CONFIG.samples.stat
    min_depth = CONFIG.samples.min_depth
    n_intervals = CONFIG.samples.n_intervals
    subset = CONFIG.samples.subset

    for i, (j, k) in enumerate(get_intervals(hdu, subset, n_intervals)):
        # Source detection. Reporting happens below.
        # NOTE: caching enabled for line below in `setup`
        image = SkyImage.from_hdu(hdu, stat, min_depth, (j, k),
                                  **{**detection, 'report': False})

        if show_cutouts and i == 0 and image.seg:
            logger.opt(lazy=True).info(
                'Source images:\n{}',
                lambda: image.seg.show.console.format_cutouts(
                    image.data, title=CONSOLE_CUTOUTS_TITLE.format(hdu=hdu),
                    **CONFIG.console.cutouts)
            )

        yield (j, k), image


def get_sample_images(run, detection=True, show_cutouts=False):

    # sample = delayed(get_sample_image)
    # with Parallel(n_jobs=1) as parallel:
    # return parallel

    # Get params from config
    detection = detection or {}
    if detection:
        logger.section('Source Detection')

        if detection is True:
            detection = CONFIG.detection

    samples = defaultdict(dict)
    for hdu in run:
        for interval, image in _get_hdu_samples(hdu, detection, show_cutouts):
            samples[hdu.file.name][interval] = image

    return samples


def plot_sample_images(run, samples, path, ui=None, thumbs=CONFIG.files.thumbs,
                       overwrite=True):

    if ui:
        logger.info('Adding sample images to ui: {}', ui)
        _plot_sample_images(run, samples, path,  ui, overwrite)

    # plot thumbnails for sample image from first portion of each data cube
    if thumbs:
        if not (thumbs := Path(thumbs)).is_absolute():
            thumbs = path / thumbs

        # portion = mit.chunked(sample_images, len(run))
        images,  = zip(*map(dict.values, samples.values()))

        # filenames, images = zip(*(map(dict.items, samples.values())))
        image_grid = plot_image_grid(images,
                                     titles=list(samples.keys()),
                                     **CONFIG.plotting.thumbnails)

        if not thumbs.exists() or overwrite:
            image_grid.figure.savefig(thumbs)  # image_grid.save ??

        if ui:
            ui.add_tab('Overview', thumbs.name, fig=image_grid.figure)


def _plot_sample_images(run, samples, path, ui, overwrite):
    n_intervals = CONFIG.samples.n_intervals
    subset = CONFIG.samples.subset

    filename_template = ''
    if save_as := CONFIG.samples.save_as:
        _j_k = '.{j}-{k}' if (n_intervals > 1) or subset else ''
        filename_template = f'{{hdu.file.stem}}{_j_k}.{save_as}'

    for hdu in run:
        images = samples[hdu.file.name]
        # grouping
        year, day = str(hdu.t.date_for_filename).split('-', 1)

        for (j, k), image in images.items():
            # add tab to ui
            key = [year, day, hdu.file.nr,
                   *([_j_k.format(j=j, k=k)] if (n_intervals > 1) else [])]

            filename = path / filename_template.format(hdu=hdu, j=j, k=k)
            if filename.exists() and not overwrite:
                filename = None

            #
            logger.info('Plottng sample {}', key)
            fig = ui.add_tab('Sources', *key,)
            # callback=ftl.partial(plot_image, image, filename))
            plot_image(fig, image=image, save_as=filename)

# @caching.cached(typed={'hdu': _hdu_hasher}, ignore='save_as')


def plot_image(fig, *indices, image, save_as=None):

    # image = samples[indices]
    art = image.plot(regions=CONFIG.plotting.segments.contours,
                     labels=CONFIG.plotting.segments.labels)

    if save_as:
        art.image.axes.figure.save(save_as)  # ['image-regions']
    # return im.figure


def compute_preview_products(run, paths, ui, overwrite,
                             thumbs=CONFIG.files.thumbs, show_cutouts=False):
    # get results from previous run
    products = get_previous_data_products(run, paths)

    # write fits headers to text
    # if not products['FITS']['headers']:

    showfile = str
    if str(paths.headers).startswith(str(paths.root)):
        def showfile(h): return h.relative_to(paths.root)

    for hdu, headfile in itt.zip_longest(run, products['FITS']['headers']):
        if not headfile:
            headfile = paths.headers / f'{hdu.file.stem}.txt'
            logger.info('Writing fits header to text at {}.', showfile(headfile))
            hdu.header.totextfile(headfile)

    # thumbs = ''
    # if overwrite or not products['Images']['Overview']:
    samples = get_sample_images(run, detection=False, show_cutouts=show_cutouts)

    plot_sample_images(run, samples, paths.plots, ui, thumbs=thumbs,
                       overwrite=overwrite)

    products['Images']['Samples'] = list(
        row_assign(run, sorted(paths.sample_images.iterdir())))

    return products

    # source regions
    # if not any(products['Images']['Source Regions']):
    #     sample_images = products['Images']['Samples']
    #     from IPython import embed
    #     embed(header="Embedded interpreter at 'src/shoc/pipeline/main.py':641")


@trace
def main(paths, target, telescope, top, plot, show_cutouts, overwrite):
    #
    # from obstools.phot import PhotInterface

    # GUI
    ui = None
    if plot:
        app = QtWidgets.QApplication(sys.argv)
        ui = MplMultiTab(title=CONFIG.gui.title, pos=CONFIG.gui.pos)

    # ------------------------------------------------------------------------ #
    # Load data
    run = shocCampaign.load(paths.root, obstype='object')

    # update info if given
    info = check_required_info(run, telescope, target)
    logger.debug('User input from CLI: {}', info)
    if info:
        if info['target'] == '.':
            info['target'] = paths.root.name.replace('_', ' ')
            logger.info('Using target name from root directory name: {!r}',
                        info['target'])

        missing_telescope_info = run[~np.array(run.attrs.telescope, bool)]
        missing_telescope_info.attrs.set(repeat(telescope=info.pop('telescope')))
        run.attrs.set(repeat(info))

    # ------------------------------------------------------------------------ #
    # Preview
    logger.section('Overview')

    # Print summary table
    daily = run.group_by('date')
    logger.info('Observations of {} by date:\n{:s}\n', info['target'],
                daily.pformat(titled=repr))

    # write observation log latex table
    paths.obslog.write_text(
        run.tabulate.to_latex(
            style='table',
            caption=f'SHOC observations of {info["target"]}.',
            label=f'tbl:obs-log:{info["target"]}'
        )
    )

    # write summary spreadsheet
    run.tabulate.to_xlsx(paths.summary)
    logger.info('The table above is available in spreadsheet format at:\n'
                '{!s:}', paths.summary)

    # Sample images prior to calibration and header info
    products = compute_preview_products(run, paths, ui, overwrite)

    # ------------------------------------------------------------------------ #
    # Calibrate
    logger.section('Calibration')

    # Compute/retrieve master dark/flat. Point science stacks to calibration
    # images.
    gobj, mdark, mflat = calibrate(run, overwrite=overwrite)

    # Sample images (after calibration)
    # if overwrite or CONFIG.files.thumbs_cal.exists():
    # sample_images_cal, segments = \
    samples_cal = get_sample_images(run, show_cutouts=show_cutouts)
    #
    plot_sample_images(run, samples_cal, paths.plots, None,
                       CONFIG.files.thumbs_cal, overwrite)

    # have to ensure we have single target here
    target = check_single_target(run)

    # Image Registration
    logger.section('Image Registration')
    # #
    if paths.reg.exists() and not overwrite:
        logger.info('Loading image register from file: {}.', paths.reg)
        reg = io.deserialize(paths.reg)
        reg.params = np.load(paths.reg_params)
    else:
        # align
        # note: source detections were reported above in `get_sample_images`
        reg = run.coalign_survey(**CONFIG.register,
                                 **{**CONFIG.detection, 'report': False})

        # save
        logger.info('Saving image register at: {}.', paths.reg)
        reg.params = np.load(paths.reg_params)
        io.serialize(paths.reg, reg)
        # np.save(paths.reg_params, reg.params)

    # from IPython import embed
    # embed(header="Embedded interpreter at 'src/shoc/pipeline/main.py':490")

    # if not any(products['Images']['Source Regions']):
    #     ui = reg.plot_detections()
    #     ui.save(filenames=[f'{hdu.file.stem}.regions.png' for hdu in run],
    #             path=paths.plots)

    # reg.plot_clusters(nrs=True)

    # mosaic
    # mosaic = reg.mosaic(names=run.files.stems, **CONFIG.plotting.mosaic)
    # ui.add_tab('Overview', 'Mosaic',)  # fig=mosaic.fig
    # savefig(mosaic.fig, CONFIG.files.mosaic, bbox_inches='tight')

    # txt, arrows = mos.mark_target(
    #     run[0].target,
    #     arrow_head_distance=2.,
    #     arrow_size=10,
    #     arrow_offset=(1.2, -1.2),
    #     text_offset=(4, 5), size=12, fontweight='bold'
    # )

    # %%
    # if not products['Images']['Source Regions']:
    #     ui = reg.plot_detections()
    #     ui.save(filenames=[f'{hdu.file.stem}.regions.png' for hdu in run],
    #             path=paths.plots)

    # phot = PhotInterface(run, reg, paths.phot)
    # ts = mv phot.ragged() phot.regions()

    # Write data products spreadsheet
    write_data_products_xlsx(run, paths)

    logger.info('The following data products were created:\n{}',
                io.show_tree(paths.output))

    # logger.section('Source Tracking')
    # logger.section('Quick Phot')

    # for hdu, (img, labels) in zip(run, itt.islice(zip(reg, reg.labels_per_image), 1, None)):
    #     # back transform to image coords
    #     coords = (reg.xy[sorted(labels)] - reg.params[1, :2]) / reg.rscale[1]

    #     tracker = SourceTracker(coords, img.seg.circularize().dilate(2))
    #     tracker.init_memory(hdu.nframes, DATAPATH / 'shoc/phot/' / hdu.file.stem, overwrite=True)
    #
    # tracker.run(hdu.calibrated)

    # logger.section('Photometry')

    # Launch the GUI
    if plot:
        ui.show()
        app.exec_()
        # sys.exit(app.exec_())

# %%
