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
from obstools.phot.tracking import SourceTracker
from recipes import io, op
from recipes.iter import cofilter
from recipes.utils import not_null
from recipes.functionals import negate
from recipes.shell import is_interactive
from recipes.decorators import update_defaults
from recipes.decorators.reporting import trace
from recipes.string import remove_prefix, shared_prefix
from recipes.functionals.partial import PlaceHolder as o

# relative
from .. import CONFIG, shocCampaign
from . import products, lightcurves as lc
from .calibrate import calibrate
from .plotting import PlotFactory
from .logging import logger, config as config_logging


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

CONSOLE_CUTOUTS_TITLE = CONFIG.console.cutouts.pop('title')

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

    stat = CONFIG.samples.params.stat
    min_depth = CONFIG.samples.params.min_depth
    n_intervals = CONFIG.samples.params.n_intervals
    subset = CONFIG.samples.params.subset

    for i, (j, k) in enumerate(get_intervals(hdu, subset, n_intervals)):
        # Source detection. Reporting happens below.
        # NOTE: caching enabled for line below in `setup`
        image = SkyImage.from_hdu(hdu,
                                  stat, min_depth, (j, k),
                                  **{**detection, 'report': False})

        if show_cutouts and i == 0 and image.seg:
            logger.opt(lazy=True).info(
                'Source images:\n{}',
                lambda: image.seg.show.console.format_cutouts(
                    image.data, title=CONSOLE_CUTOUTS_TITLE.format(hdu=hdu),
                    **CONFIG.console.cutouts)
            )

        yield (j, k), image


def get_intervals(hdu, subset, n_intervals):
    n = hdu.nframes
    if subset:
        yield slice(*int2tup(subset)).indices(n)[:2]
        return

    yield from mit.pairwise(range(0, n + 1, n // n_intervals))


# ---------------------------------------------------------------------------- #
# plotting


def plot_sample_images(run, samples, path_template=None, overwrite=True,
                       thumbnails=None, ui=None):

    return tuple(_iplot_sample_images(run, samples, path_template, overwrite,
                                      thumbnails, ui))


def _iplot_sample_images(run, samples, path_template, overwrite,
                         thumbnails, ui):

    if ui:
        logger.info('Adding sample images to ui: {}', ui)

    yield list(
        _plot_sample_images(run, samples, path_template, overwrite, ui)
    )

    # plot thumbnails for sample image from first portion of each data cube
    if not_null(thumbnails, except_=[{}]):
        if thumbnails is True:
            thumbnails = {}

        yield plot_thumbnails(samples, ui, **{'overwrite': overwrite, **thumbnails})


def get_filename_template(ext):
    cfg = CONFIG.samples.params
    if ext := ext.lstrip('.'):
        _j_k = '.{j}-{k}' if (cfg.n_intervals > 1) or cfg.subset else ''
        return f'{{stem}}{_j_k}.{ext}'
    return ''


def _plot_image(image, *args, **kws):
    return image.plot(image, *args, **kws)


def _plot_sample_images(run, samples, path_template, overwrite, ui):

    factory = PlotFactory(ui)   # static args
    task = factory(_plot_image)(fig=o,
                                regions=CONFIG.samples.plots.contours,
                                labels=CONFIG.samples.plots.labels,
                                coords='pixel',
                                use_blit=False)

    for hdu in run.sort_by('t.t0'):
        # grouping
        year, day = str(hdu.t.date_for_filename).split('-', 1)

        for (j, k), image in samples[hdu.file.name].items():
            # get filename
            filename = products.resolve_path(path_template, hdu, j, k)

            # add tab to ui
            key = ('Sample Images', year, day, hdu.file.nr)
            if frames := remove_prefix(hdu.file.stem, filename.stem):
                key.append(frames)

            # plot
            yield factory.add_task(task, key, filename, overwrite, image=image)


# @caching.cached(typed={'hdu': _hdu_hasher}, ignore='save_as')


def plot_thumbnails(samples, ui, filename=None, overwrite=False, **kws):

    # portion = mit.chunked(sample_images, len(run))
    images, = zip(*map(dict.values, samples.values()))

    # filenames, images = zip(*(map(dict.items, samples.values())))
    factory = PlotFactory(ui)
    task = factory(plot_image_grid)(images,
                                    fig=o,
                                    titles=list(samples.keys()),
                                    use_blit=False,
                                    **kws)

    factory.add_task(task, ('Overview', getattr(filename, 'name')),
                     filename, overwrite)


# def plot_image(fig, *indices, image):

#     # image = samples[indices]
#     logger.debug('Plotting image {}', image)

#     display, art = image.plot(fig=fig.figure,
#                               regions=CONFIG.samples.plots.contours,
#                               labels=CONFIG.samples.plots.labels,
#                               coords='pixel',
#                               use_blit=False)

#     return art


def plot_drizzle(fig, *indices, ff):
    # logger.info('POOP drizzle image: {} {}', fig, indices)
    if not indices or indices[-1] == 'Drizzle':
        logger.info('Plotting drizzle image: {} {}', fig, indices)
        ff.show_colorscale(cmap=CONFIG.plotting.cmap)
        fig.tight_layout()


def save_samples_fits(run, samples, wcss, filename_template, overwrite):
    # save samples as fits with wcs

    for (file, subs), wcs in zip(samples.items(), wcss):
        hdu = run[file]
        for (j, k), image in subs.items():
            filename = products.resolve_path(filename_template, hdu, j, k)

            if overwrite or not filename.exists():
                # remove header comment
                # "  FITS (Flexible Image Transport System) format is defined in 'Astronomy"
                # which causes write problems
                del image.meta['COMMENT']
                header = fits.Header(image.meta)
                header.update(wcs.to_header())

                if np.ma.isMA(image.data) and np.ma.is_masked(image.data):
                    raise NotImplementedError

                fits.writeto(filename, np.array(image.data), header,
                             overwrite=overwrite)


# ---------------------------------------------------------------------------- #
# Setup / Load data

def init(paths, telescope, target, overwrite):
    root = paths.folders.root

    run = shocCampaign.load(root, obstype='object')

    # update info if given
    info = check_required_info(run, telescope, target)
    logger.debug('User input from CLI: {}', info)
    if info:
        if info['target'] == '.':
            info['target'] = root.name.replace('_', ' ')
            logger.info('Using target name from root directory name: {!r}',
                        info['target'])

        missing_telescope_info = run[~np.array(run.attrs.telescope, bool)]
        missing_telescope_info.attrs.set(repeat(telescope=info.pop('telescope')))
        run.attrs.set(repeat(info))

    # write script for remote data retrieval
    if ((files := paths.files.remote).get('rsync_script') and
            (overwrite or not all((f.exists() for f in files.values())))):
        write_rsync_script(run, paths)

    return run, info


def write_rsync_script(run, paths, username=CONFIG.remote.username,
                       server=CONFIG.remote.server):

    remotes = list(map(str, run.calls.get_server_path(None)))
    prefix = f'{server}:{shared_prefix(remotes)}'
    filelist = paths.files.remote.rsync_script
    io.write_lines(filelist, [remove_prefix(_, prefix) for _ in remotes])

    outfile = paths.files.remote.rsync_files
    outfile.write_text(
        f'sudo rsync -arvzh --info=progress2 '
        f'--files-from={filelist!s} --no-implied-dirs '
        f'{username}@{prefix} {paths.folders.output!s}'
    )


# ---------------------------------------------------------------------------- #
# Preview

def preview(run, paths, info, ui, plot, overwrite):
    logger.section('Overview')

    # Print summary table
    daily = run.group_by('date')  # 't.date_for_filename'
    logger.info('Observations of {} by date:\n{:s}\n', info['target'],
                daily.pformat(titled=repr))

    # write observation log latex table
    paths.files.info.obslog.write_text(
        run.tabulate.to_latex(
            style='table',
            caption=f'SHOC observations of {info["target"]}.',
            label=f'tbl:obs-log:{info["target"]}'
        )
    )

    # write summary spreadsheet
    path = str(paths.files.info.campaign)
    filename, *sheet = path.split('::')

    run.tabulate.to_xlsx(filename, *sheet, overwrite=True)
    logger.info('The table above is available in spreadsheet format at:\n'
                '{!s:}', path)

    # Sample images prior to calibration and header info
    return compute_preview(run, paths, ui, plot, overwrite)


def compute_preview(run, paths, ui, plot, overwrite, show_cutouts=False):

    # get results from previous run
    overview, data_products = products.get_previous(run, paths)

    # write fits headers to text
    headers_to_txt(run, paths, overwrite)

    # thumbs = ''
    # if overwrite or not products['Images']['Overview']:
    samples = get_sample_images(run, detection=False, show_cutouts=show_cutouts)

    thumbnails = None
    if plot:
        thumbnails = plot_thumbnails(samples, ui,
                                     **{'overwrite': overwrite,
                                        **CONFIG.samples.plots.thumbnails.raw})
    # source regions
    # if not any(products['Images']['Source Regions']):
    #     sample_images = products['Images']['Samples']
    return overview, data_products, samples, thumbnails


def headers_to_txt(run, paths, overwrite):

    showfile = str
    if str(paths.files.info.headers).startswith(str(paths.folders.output)):
        def showfile(h): return h.relative_to(paths.folders.output)

    for hdu in run:
        headfile = products.resolve_path(paths.files.info.headers, hdu)
        if not headfile.exists() or overwrite:
            logger.info('Writing fits header to text at {}.', showfile(headfile))
            hdu.header.totextfile(headfile, overwrite=overwrite)


# ---------------------------------------------------------------------------- #
# Calibrate

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


# ---------------------------------------------------------------------------- #
# Image Registration

def registration(run, paths, ui, plot, show_cutouts, overwrite):
    logger.section('Image Registration (WCS)')

    files = paths.files.registration
    # None in [hdu.wcs for hdu in run]
    if do_reg := (overwrite or not files.registry.exists()):
        # Sample images (after calibration)
        samples_cal = get_sample_images(run, show_cutouts=show_cutouts)
    else:
        logger.info('Loading image registry from file: {}.', files.registry)
        reg = io.deserialize(files.registry)
        reg.params = np.load(files.params)

        # retrieve samples from register
        # TODO: get from fits?
        samples_cal = {
            fn: {('', ''): im}
            for fn, im in zip(run.files.names, list(reg)[1:])
        }
        # samples_cal = get_sample_images(run, show_cutouts=show_cutouts)

    if plot:
        cfg = CONFIG.samples.plots.thumbnails
        plot_sample_images(run, samples_cal, paths.files.samples.filename, overwrite,
                           {**cfg.raw, **cfg.calibrated}, ui)

    # align
    if do_reg:
        reg = register(run, paths, samples_cal, overwrite)

    if plot:
        plot_overview(run, reg, paths, ui, overwrite)

    return reg


def register(run, paths, samples_cal, overwrite):

    # note: source detections were reported above in `get_sample_images`
    reg = run.coalign_survey(
        **CONFIG.registration.params, **{**CONFIG.detection, 'report': False}
    )

    # TODO region files

    # save
    files = paths.files.registration
    logger.info('Saving image registry at: {}.', files.registry)
    reg.params = np.load(files.params)
    io.serialize(files.registry, reg)
    # np.save(paths.reg.params, reg.params)

    # reg.plot_clusters(nrs=True)

    # Build image WCS
    wcss = reg.build_wcs(run)

    # save samples fits
    save_samples_fits(run, samples_cal, wcss,
                      paths.files.samples.filename, overwrite)

    # Drizzle image
    reg.drizzle(files.drizzle, CONFIG.registration.drizzle.pixfrac)

    return reg


def plot_overview(run, reg, paths, ui, overwrite):

    kws = dict(CONFIG.registration.plots.mosaic)
    filename = kws.pop('filename')
    factory = PlotFactory(ui)
    task = factory(reg.mosaic)(names=run.files.stems, **kws)
    factory.add_task(task, ('Overview', 'Mosaic'), filename, overwrite)

    # Create mosaic plot
    # mosaic = reg.mosaic(names=run.files.stems, **kws)

    # txt, arrows = mos.mark_target(
    #     run[0].target,
    #     arrow_head_distance=2.,
    #     arrow_size=10,
    #     arrow_offset=(1.2, -1.2),
    #     text_offset=(4, 5), size=12, fontweight='bold'
    # )

    # drizzle
    filename = paths.files.registration.drizzle
    ff = apl.FITSFigure(str(filename))
    task = factory(plot_drizzle)(o, ff=ff)
    factory.add_task(task, ('Overview', 'Drizzle'),
                     filename.with_suffix('.png'), overwrite, ff._figure)


# ---------------------------------------------------------------------------- #


def track(run, reg, paths, ui, overwrite=False):

    logger.section('Source Tracking')
    spanning = sorted(set.intersection(*map(set, reg.labels_per_image)))
    spanning = np.add(spanning, 1)
    logger.info('Sources: {} span all observations.', spanning)

    filenames = paths.files.tracking
    image_labels = itt.islice(zip(reg, reg.labels_per_image), 1, None)

    for i, (hdu, (img, labels)) in enumerate(zip(run, image_labels)):
        missing_files = _tracker_missing_files(filenames, hdu, spanning)
        if overwrite or missing_files:
            logger.info('Launching tracker for {}.', hdu.file.name)

            # back transform to image coords
            coords = reg._trans_to_image(i).transform(reg.xy[sorted(labels)])
            # path = products.resolve_path(paths.folders.tracking, hdu)
            tracker, ui = _track(hdu, img.seg, spanning, coords, paths, ui)

        # return ui
    logger.info('Source tracking complete.')
    return spanning


def _tracker_missing_files(templates, hdu, sources):
    desired = templates.map(products.resolve_path, hdu)
    position_plots, files = desired.split('position')
    missing_files = {
        key: file for key, path in files.flatten().items()
        if not (file := products.resolve_path(path, hdu)).exists()
    }

    _ppk, position_plots = position_plots.flatten().popitem()
    position_plots = str(position_plots)
    missing_files.update({
        (*_ppk, i): position_plots.replace('$SOURCE', str(i))
        for i in sources
    })
    return missing_files


@update_defaults(CONFIG.tracking.params)
def _track(hdu, seg, labels, coords, paths, ui, overwrite=False, dilate=0, njobs=-1):

    logger.info(motley.stylize('Launching tracker for {:|darkgreen}.'),
                hdu.file.name)

    if (cfg := CONFIG.tracking).params.circularize:
        seg = seg.circularize()

    base = products.resolve_path(paths.folders.tracking.folder, hdu)
    tracker = SourceTracker(coords, seg.dilate(dilate), labels=labels)
    tracker.init_memory(hdu.nframes, base, overwrite=overwrite)
    tracker.run(hdu.calibrated, njobs=njobs)

    # plot
    if cfg.plot:
        tmp, kws = cfg.plots.positions.split('filename')
        tmp = str(products.resolve_path(tmp, hdu))

        factory = PlotFactory(ui)   # static args
        task = factory(tracker.plot.positions)(fig=o, **kws)

        year, day = str(hdu.date_for_filename).split('-', 1)
        tab = (cfg.title, year, day)
        for j in tracker.use_labels:
            # plot source location features
            # factory[(*tab, f'source {j}')] = TaskRunner(task, tmp.format(j), overwrite)
            factory.add_task(task, (*tab, f'source {j}'), tmp.format(j), overwrite)

        # plot positions displacement time series
        (filename, title), kws = (cfg.plots.time_series, ('filename', 'title'))
        task = factory(tracker.plot.displacement_time_series)(ax=o.axes[0], **kws)
        factory.add_task(task, (*tab, title), filename, overwrite)

    return tracker, ui


def lightcurves(run, paths, ui, plot=True, overwrite=False):

    lcs = lc.extract(run, paths, overwrite)

    if not plot:
        return

    for step, db in lcs.items():
        # get path template
        tmp = paths.templates['DATE'].lightcurves[step]
        tmp = getattr(tmp, 'concat', tmp)
        #
        plot_lcs(db, step, ui, tmp, overwrite=False)

    # for hdu in run:
    #     lc.io.load_raw(hdu, products.resolve_path(paths.lightcurves.raw, hdu),
    #                 overwrite)

    # for step in ['flagged', 'diff0', 'decor']:
    #     file = paths.lightcurves[step]
    #     load_or_compute(file, overwrite, LOADERS[step], (hdu, file))


def plot_lcs(lcs, step, ui=None, filename_template=None, overwrite=False, **kws):

    section = CONFIG.lightcurves.title
    factory = PlotFactory(ui)
    task = factory(lc.plot)(o, **kws)

    filenames = {}
    for date, ts in lcs.items():
        filenames[date] = filename\
            = Path(filename_template.substitute(DATE=date)).with_suffix('.png')
        year, day = date.split('-', 1)
        factory.add_task(task,
                         (section, year, day, step),
                         filename, overwrite, None,
                         ts)

    return ui


@trace
def main(paths, target, telescope, top, plot, show_cutouts, overwrite):
    #
    # from obstools.phot import PhotInterface

    # GUI
    ui = None
    if plot:
        if not is_interactive():
            app = QtWidgets.QApplication(sys.argv)
        #
        ui = MplMultiTab(title=CONFIG.plotting.gui.title,
                         pos=CONFIG.plotting.gui.pos)

    # ------------------------------------------------------------------------ #
    # Setup / Load data
    run, info = init(paths, telescope, target, overwrite)

    # ------------------------------------------------------------------------ #
    # Preview
    overview, data_products, samples, thumbnails = preview(
        run, paths, info, ui, plot, overwrite
    )

    # ------------------------------------------------------------------------ #
    # Calibrate
    calibration(run, overwrite)

    # ------------------------------------------------------------------------ #
    # Image Registration

    # have to ensure we have single target here
    target = check_single_target(run)

    reg = registration(run, paths, ui, plot, show_cutouts, overwrite)

    # ------------------------------------------------------------------------ #
    # Source Tracking
    spanning = track(run, reg, paths, ui)

    # ------------------------------------------------------------------------ #
    # Photometry
    logger.section('Photometry')
    lcs = lightcurves(run, paths, ui, plot, overwrite)
    # paths.lightcurves

    # phot = PhotInterface(run, reg, paths.phot)
    # ts = mv phot.ragged() phot.regions()

    # ------------------------------------------------------------------------ #
    # Launch the GUI
    if plot:
        logger.section('Launching GUI')
        # activate tab switching callback (for all tabs)
        ui.add_task()   # needed to check for tab switch callbacks to run
        ui[CONFIG.samples.title].link_focus()
        ui[CONFIG.lightcurves.title].link_focus()
        ui.set_focus('Overview', 'Mosaic')
        ui.show()

        if not is_interactive():
            # sys.exit(app.exec_())
            app.exec_()

        # Run incomplete plotting tasks
        getter = op.AttrVector('plot.func.filename', default=None)
        tabs = list(ui.tabs._leaves())
        filenames, tabs = cofilter(getter.map(tabs), tabs)
        unsaved, tabs = map(list, cofilter(negate(Path.exists), filenames, tabs))
        if n := len(unsaved):
            logger.info('Now running {} incomplete tasks:', n)
            for tab in tabs:
                tab.run_task()

    # ------------------------------------------------------------------------ #
    logger.section('Finalize')

    # get results from this run
    overview, data_products = products.get_previous(run, paths)

    # Write data products spreadsheet
    products.write_xlsx(run, paths, overview)
    # This updates spreadsheet with products computed above
