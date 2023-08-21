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
import aplpy as apl
import more_itertools as mit
from loguru import logger
from astropy.io import fits
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
from recipes.string import remove_prefix, shared_prefix

# relative
from .. import CONFIG, shocCampaign
from . import products
from .calibrate import calibrate
from .logging import config as config_logging, logger


# ---------------------------------------------------------------------------- #
# logging config
config_logging()

# ---------------------------------------------------------------------------- #
# plot config
rcParams.update({'font.size': 14,
                 'axes.labelweight': 'bold',
                 'image.cmap': CONFIG.plotting.cmap})
# rc('text', usetex=False)


# ---------------------------------------------------------------------------- #

CONSOLE_CUTOUTS_TITLE = motley.stylize(CONFIG.console.cutouts.pop('title'))

# ---------------------------------------------------------------------------- #


# t0 = time.time()
# TODO group by source

# track
# photomerty
# decorrelate
# spectral analysis

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
# utils

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
# data


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
        headfile = data_products[stem].get(txtfile, '')
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

    return samples, overview, data_products

    # source regions
    # if not any(products['Images']['Source Regions']):
    #     sample_images = products['Images']['Samples']


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


# ---------------------------------------------------------------------------- #
# plotting

def plot_sample_images(run, samples, paths, ui=None, thumbs=CONFIG.files.thumbs,
                       overwrite=True, delay=CONFIG.plotting.delay):
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
    image_grid = plot_image_grid(images, use_blit=False,
                                 titles=list(samples.keys()),
                                 **CONFIG.plotting.thumbnails)

    if not thumbs.exists() or overwrite:
        image_grid.figure.savefig(thumbs)  # image_grid.save ??

    if ui:
        ui.add_tab('Overview', thumbs.name, fig=image_grid.figure)

    return image_grid


def get_filename_template(ext):
    cfg = CONFIG.samples
    if ext := ext.lstrip('.'):
        _j_k = '.{j}-{k}' if (cfg.n_intervals > 1) or cfg.subset else ''
        return f'{{stem}}{_j_k}.{ext}'
    return ''


def _plot_sample_images(run, samples, path, ui, overwrite,
                        delay=CONFIG.plotting.delay):

    filename_template = get_filename_template(CONFIG.samples.save_as)
    #
    for hdu in run:
        # grouping
        year, day = str(hdu.t.date_for_filename).split('-', 1)

        for (j, k), image in samples[hdu.file.name].items():

            filename = path / filename_template.format(stem=hdu.file.stem, j=j, k=k)

            # add tab to ui
            key = [year, day, hdu.file.nr]

            if j and k and (j, k) != (0, hdu.nframes):
                key.append('{j}-{k}'.format(j=j, k=k))

            fig = ui.add_tab('Samples', *key)

            # plot
            if filename.exists() and not overwrite:
                filename = None

            if delay:
                logger.info('Plotting delayed: Adding plot callback for {}', key)
                ui['Samples'].add_callback(plot_image, image=image)
                atexit.register(save_fig, fig.figure, filename)
            else:
                logger.info('Plotting sample image {}', key)
                plot_image(fig, image=image, save_as=filename)

            yield fig

# @caching.cached(typed={'hdu': _hdu_hasher}, ignore='save_as')


def plot_image(fig, *indices, image, save_as=None):

    # image = samples[indices]
    logger.debug('Plotting image {}', image)
    display, art = image.plot(fig=fig.figure,
                              regions=CONFIG.plotting.segments.contours,
                              labels=CONFIG.plotting.segments.labels,
                              coords='pixel',
                              use_blit=False)

    save_fig(art.image.axes.figure, save_as)
    return art


def plot_drizzle(fig, *indices, ff, save_as):
    logger.info('Plotting drizzle image.')

    if not indices or indices[-1] == 'Drizzle':
        ff.show_colorscale(cmap=CONFIG.plotting.cmap)
        fig.tight_layout()

    save_fig(fig, save_as)


def save_fig(fig, filename):
    if filename:
        logger.info('Saving image: {}', filename)
        fig.savefig(filename)


def save_samples_fits(samples, wcss, path, overwrite):
    # save samples as fits with wcs
    filename_template = get_filename_template('fits')
    for (stem, subs), wcs in zip(samples.items(), wcss):
        path = path / stem
        for (j, k), image in subs.items():
            filename = path / filename_template.format(stem=stem, j=j, k=k)

            if filename.exists() or overwrite:
                header = fits.Header(image.meta)
                header.update(wcs.to_header())
                fits.writeto(filename, image.data, header)

# ---------------------------------------------------------------------------- #


def write_rsync_script(run, paths, username=CONFIG.remote.username,
                       server=CONFIG.remote.server):

    remotes = list(map(str, run.calls.get_server_path(None)))
    prefix = f'{server}:{shared_prefix(remotes)}'
    filelist = paths.info / 'remote-files.txt'
    io.write_lines(filelist, [remove_prefix(_, prefix) for _ in remotes])

    outfile = paths.info / 'rsync-remote.sh'
    outfile.write_text(
        f'sudo rsync -arvzh --info=progress2 '
        f'--files-from={filelist!s} --no-implied-dirs '
        f'{username}@{prefix} {paths.output!s}'
    )

# ---------------------------------------------------------------------------- #
# main


def init(paths, telescope, target, overwrite):
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

    # write script for remote data retrieval
    if (CONFIG.remote.write_rsync_script and
            (overwrite or not (paths.remote.exists() or paths.rsync_script.exists()))):
        write_rsync_script(run, paths)

    return run, info


def preview(run, paths, info, ui, overwrite):
    logger.section('Overview')

    # Print summary table
    daily = run.group_by('date') # 't.date_for_filename'
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
    return compute_preview(run, paths, ui, overwrite)


def calibration(run, overwrite):
    logger.section('Calibration')

    # Compute/retrieve master dark/flat. Point science stacks to calibration
    # images.
    gobj, mdark, mflat = calibrate(run, overwrite=overwrite)

    # TODO: logger.info('Calibrating sample images.')
    # from IPython import embed
    # embed(header="Embedded interpreter at 'src/pyshoc/pipeline/main.py':534")
    # samples

    # if overwrite or CONFIG.files.thumbs_cal.exists():
    # sample_images_cal, segments = \


def registration(run, paths, ui, plot, show_cutouts, overwrite):
    logger.section('Image Registration (WCS)')
    #
    # None in [hdu.wcs for hdu in run]
    if do_reg := (overwrite or not paths.reg.exists()):
        # Sample images (after calibration)
        samples_cal = get_sample_images(run, show_cutouts=show_cutouts)
    else:
        logger.info('Loading image registry from file: {}.', paths.reg.file)
        reg = io.deserialize(paths.reg.file)
        reg.params = np.load(paths.reg.params)

        # retrieve samples from register
        # TODO: get from fits?
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
        reg = register(run, paths, samples_cal, overwrite)

    if plot:
        plot_overview(reg, run, ui, paths)
    return reg


def register(run, paths, samples_cal, overwrite):

    # note: source detections were reported above in `get_sample_images`
    reg = run.coalign_survey(
        **CONFIG.registration, **{**CONFIG.detection, 'report': False}
    )

    # TODO region files

    # save
    logger.info('Saving image registry at: {}.', paths.reg.file)
    reg.params = np.load(paths.reg.params)
    io.serialize(paths.reg.file, reg)
    # np.save(paths.reg.params, reg.params)

    # reg.plot_clusters(nrs=True)

    # Build image WCS
    wcss = reg.build_wcs(run)
    # save samples fits
    save_samples_fits(samples_cal, wcss, paths.sample_images, overwrite)

    # Drizzle image
    reg.drizzle(paths.drizzle, CONFIG.drizzle.pixfrac)

    return reg


def plot_overview(reg, run, ui, paths):

    # mosaic
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

    # drizzle
    ff = apl.FITSFigure(str(paths.drizzle))
    ui.add_tab('Overview', 'Drizzle', fig=ff._figure)
    filename = paths.plots / paths.drizzle.with_suffix('.png').name

    if CONFIG.plotting.delay:
        logger.info('Plotting delayed: Adding plot callback for drizzle.')
        ui['Overview'].add_callback(plot_drizzle, ff=ff, save_as=None)
        atexit.register(save_fig, ff._figure, filename)
    else:
        plot_drizzle(ff._figure, ff=ff, save_as=filename)


def _track(hdu, seg, labels, coords, path, dilate=CONFIG.tracking.dilate,
           njobs=CONFIG.tracking.dilate):

    logger.info(motley.stylize('Launching tracker for {:|darkgreen}.'), 
                hdu.file.name)

    if CONFIG.tracking.circularize:
        seg = seg.circularize()

    tracker = SourceTracker(coords, seg.dilate(dilate), labels=labels)
    tracker.init_memory(hdu.nframes, path, overwrite=overwrite)
    tracker.run(hdu.calibrated, njobs=njobs)

    # plot
    if CONFIG.tracking.plot:
        def get_filename(name, folder=path / 'plots'):
            return folder / f'positions-{name.lower().replace(" ", "")}'

        ui, art = tracker.plot.positions(ui=ion)
        if ion:
            ui.tabs.save(get_filename)
        else:
            for i, j in enumerate(tracker.use_labels):
                ui[i].savefig(get_filename(f'source{j}'))

        # plot individual
        fig = plot_positions_individual(tracker)

    return tracker, ui


def track(run, reg):

    logger.section('Source Tracking')
    spanning = sorted(set.intersection(*map(set, reg.labels_per_image)))
    logger.info('Sources: {} span all observations.', spanning)

    image_labels = itt.islice(zip(reg, reg.labels_per_image), 1, None)
    for i, (hdu, (img, labels)) in enumerate(zip(run, image_labels)):
        # back transform to image coords
        coords = reg._trans_to_image(i).transform(reg.xy[sorted(labels)])

        logger.info('Launching tracker for {}.', hdu.file.name)

        tracker, ui = _track(hdu, img.seg, spanning, coords,
                             products.resolve_path(paths.tracking, hdu.file.stem))

        # return ui
    return spanning


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
    run, info = init(paths, telescope, target, overwrite)

    # ------------------------------------------------------------------------ #
    # Preview
    samples, overview, data_products = preview(run, paths, info, ui, overwrite)

    # ------------------------------------------------------------------------ #
    # Calibrate
    calibration(run, overwrite)

    # ------------------------------------------------------------------------ #
    # Image Registration

    # have to ensure we have single target here
    target = check_single_target(run)

    reg = registration(run, paths, ui, plot, show_cutouts, overwrite)

    # Write data products spreadsheet
    products.write_xlsx(run, paths)

    # ------------------------------------------------------------------------ #
    # Source Tracking
    spanning = track(reg, run)

    #
    tracker.run(hdu.calibrated)
    from IPython import embed
    embed(header="Embedded interpreter at 'src/pyshoc/pipeline/main.py':584")

    # logger.section('Photometry')

    # phot = PhotInterface(run, reg, paths.phot)
    # ts = mv phot.ragged() phot.regions()

    # Launch the GUI
    if plot:
        logger.section('Launching GUI')
        ui.show()
        # app.exec_()
        sys.exit(app.exec_())

    # from IPython import embed
    # embed(header="Embedded interpreter at 'src/pyshoc/pipeline/main.py':676")

# %%
