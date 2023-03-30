"""
Photometry pipeline for the Sutherland High-Speed Optical Cameras.
"""


# std
import sys
import atexit
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
from pyxides.vectorize import repeat
from scrawl.image import plot_image_grid
from obstools.image import SkyImage
from obstools.modelling import int2tup
from recipes import io
from recipes.decorators.reporting import trace

# relative
from .. import CONFIG, shocCampaign
from . import logging, products
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

# ---------------------------------------------------------------------------- #


def compute_preview(run, paths, ui, overwrite,
                    thumbs=CONFIG.files.thumbs, show_cutouts=False):
    # get results from previous run
    overview, data_products = products.get_previous(run, paths)

    showfile = str
    if str(paths.headers).startswith(str(paths.root)):
        def showfile(h): return h.relative_to(paths.root)

    # write fits headers to text
    for hdu in run:
        stem = hdu.file.stem
        txtfile = f'{stem}.txt'
        headfile = data_products[stem]['info/headers'].get(txtfile, '')
        if not headfile or overwrite:
            headfile = paths.headers / txtfile
            logger.info('Writing fits header to text at {}.', showfile(headfile))
            hdu.header.totextfile(headfile, overwrite=overwrite)

    # thumbs = ''
    # if overwrite or not products['Images']['Overview']:
    samples = get_sample_images(run, detection=False, show_cutouts=show_cutouts)

    thumbs = paths.output / thumbs
    image_grid, = plot_sample_images(run, samples, paths, None, thumbs,
                                     overwrite=overwrite)
    if ui:
        ui.add_tab('Overview', thumbs.name, fig=image_grid.figure)

    return data_products

    # source regions
    # if not any(products['Images']['Source Regions']):
    #     sample_images = products['Images']['Samples']


def plot_sample_images(run, samples, paths, ui=None, thumbs=CONFIG.files.thumbs,
                       overwrite=True, delay=False):
    returns = []

    if ui:
        logger.info('Adding sample images to ui: {}', ui)
        delay = False if ui is None else delay
        figures = _plot_sample_images(run, samples, paths.sample_images,
                                      ui, overwrite, delay)
        returns.append(list(figures))

    # plot thumbnails for sample image from first portion of each data cube
    if thumbs:
        if not (thumbs := Path(thumbs)).is_absolute():
            thumbs = paths.output / thumbs

        returns.append(plot_thumbnails(samples, ui, thumbs, overwrite))

    return returns


def plot_thumbnails(samples, ui, thumbs, overwrite):

    # portion = mit.chunked(sample_images, len(run))
    images, = zip(*map(dict.values, samples.values()))

    # filenames, images = zip(*(map(dict.items, samples.values())))
    image_grid = plot_image_grid(images,
                                 titles=list(samples.keys()),
                                 **CONFIG.plotting.thumbnails)

    if not thumbs.exists() or overwrite:
        image_grid.figure.savefig(thumbs)  # image_grid.save ??

    if ui:
        ui.add_tab('Overview', thumbs.name, fig=image_grid.figure)

    return image_grid


def _plot_sample_images(run, samples, path, ui, overwrite, delay=False):
    n_intervals = CONFIG.samples.n_intervals
    subset = CONFIG.samples.subset

    filename_template = ''
    if save_as := CONFIG.samples.save_as:
        _j_k = '.{j}-{k}' if (n_intervals > 1) or subset else ''
        filename_template = f'{{hdu.file.stem}}{_j_k}.{save_as}'
        # logger

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
            fig = ui.add_tab('Samples', *key)

            if delay:
                logger.info('Plotting delayed: Adding plot callback for {}', key)
                ui['Samples'].add_callback(plot_image, image=image)
                atexit.register(save_image, image, filename)
            else:
                logger.info('Plotting sample image {}', key)
                plot_image(fig, image=image, save_as=filename)

            yield fig

# @caching.cached(typed={'hdu': _hdu_hasher}, ignore='save_as')


def plot_image(fig, *indices, image, save_as=None):

    # image = samples[indices]
    logger.debug('Plotting {}', image)
    display, art = image.plot(fig=fig.figure,
                              regions=CONFIG.plotting.segments.contours,
                              labels=CONFIG.plotting.segments.labels,
                              coords='pixel')

    save_image(art.image, save_as)

    return art


def save_image(image, filename):
    if filename:
        logger.info('Saving image: {}', filename)
        image.axes.figure.savefig(filename)

# ---------------------------------------------------------------------------- #


def get_intervals(hdu, subset, n_intervals):
    n = hdu.nframes
    if subset:
        yield slice(*int2tup(subset)).indices(n)[:2]
        return

    yield from mit.pairwise(range(0, n + 1, n // n_intervals))


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
    data_products = compute_preview(run, paths, ui, overwrite)

    # ------------------------------------------------------------------------ #
    # Calibrate
    logger.section('Calibration')

    # Compute/retrieve master dark/flat. Point science stacks to calibration
    # images.
    gobj, mdark, mflat = calibrate(run, overwrite=overwrite)

    # if overwrite or CONFIG.files.thumbs_cal.exists():
    # sample_images_cal, segments = \

    # have to ensure we have single target here
    target = check_single_target(run)

    # Image Registration
    logger.section('Image Registration (WCS)')
    #
    if do_reg := (overwrite or not paths.reg.exists()):
        # Sample images (after calibration)
        samples_cal = get_sample_images(run, show_cutouts=show_cutouts)
    else:
        logger.info('Loading image register from file: {}.', paths.reg)
        reg = io.deserialize(paths.reg)
        reg.params = np.load(paths.reg_params)
        # retrieve samples from register
        samples_cal = {
            fn: {('', ''): im}
            for fn, im in zip(run.files.names, list(reg)[1:])
        }
        # samples_cal = get_sample_images(run, show_cutouts=show_cutouts)

    if plot:
        plot_sample_images(run, samples_cal, paths, ui,
                           CONFIG.files.thumbs_cal, overwrite)

    # align
    if do_reg:
        # note: source detections were reported above in `get_sample_images`
        reg = run.coalign_survey(**CONFIG.register,
                                 **{**CONFIG.detection, 'report': False})

        # save
        logger.info('Saving image register at: {}.', paths.reg)
        reg.params = np.load(paths.reg_params)
        io.serialize(paths.reg, reg)
        # np.save(paths.reg_params, reg.params)

    # reg.plot_clusters(nrs=True)

    # mosaic
    if plot:
        mosaic = reg.mosaic(names=run.files.stems, **CONFIG.plotting.mosaic)
        ui.add_tab('Overview', 'Mosaic', fig=mosaic.fig)
        mosaic.fig.savefig(paths.output / CONFIG.files.mosaic, bbox_inches='tight')

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
    products.write_xlsx(run, paths)

    # ------------------------------------------------------------------------ #
    # logger.section('Quick Phot')
    logger.section('Source Tracking')

    from obstools.phot.tracking import SourceTracker

    image_labels = itt.islice(zip(reg, reg.labels_per_image), 1, None)
    for i, (hdu, (img, labels)) in enumerate(zip(run, image_labels)):
        # back transform to image coords
        coords = reg._trans_to_image(i).transform(reg.xy[sorted(labels)])

        tracker = SourceTracker(coords, img.seg.circularize().dilate(2))
        tracker.init_memory(hdu.nframes, paths.phot / hdu.file.stem,
                            overwrite=overwrite)
        break

    #
    tracker.run(hdu.calibrated)

    # logger.section('Photometry')

    # Launch the GUI
    if plot:
        logger.debug('Launching GUI')
        ui.show()
        # app.exec_()
        sys.exit(app.exec_())

    # from IPython import embed
    # embed(header="Embedded interpreter at 'src/pyshoc/pipeline/main.py':676")

# %%
