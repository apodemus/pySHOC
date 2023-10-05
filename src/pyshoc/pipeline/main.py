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
from recipes import io
from recipes.utils import not_null
from recipes.shell import is_interactive
from recipes.decorators import update_defaults
from recipes.decorators.reporting import trace
from recipes.functionals.partial import PlaceHolder
from recipes.string import remove_prefix, shared_prefix

# relative
from .. import CONFIG, shocCampaign
from . import products, lightcurves as lc
from .calibrate import calibrate
from .plotting import PlotTask
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


def _plot_sample_images(run, samples, path_template, overwrite, ui):

    o = PlaceHolder
    for hdu in run:
        # grouping
        year, day = str(hdu.t.date_for_filename).split('-', 1)

        for (j, k), image in samples[hdu.file.name].items():
            # add tab to ui
            key = [year, day, hdu.file.nr]
            if j and k and (j, k) != (0, hdu.nframes):
                key.append('{j}-{k}'.format(j=j, k=k))

            # plot
            filename = products.resolve_path(path_template, hdu, j, k)
            task = PlotTask(ui, ('Samples', *key),
                                  filename, overwrite)
            yield task(plot_image)(o, image=image)


# @caching.cached(typed={'hdu': _hdu_hasher}, ignore='save_as')


def plot_thumbnails(samples, ui, filename=None, overwrite=False, **kws):

    # portion = mit.chunked(sample_images, len(run))
    images, = zip(*map(dict.values, samples.values()))

    # filenames, images = zip(*(map(dict.items, samples.values())))
    task = PlotTask(ui, ('Overview', getattr(filename, 'name')),
                          filename, overwrite)
    return task(plot_image_grid)(images,
                                    use_blit=False,
                                    titles=list(samples.keys()),
                                    **kws)


def plot_image(fig, *indices, image):

    # image = samples[indices]
    logger.debug('Plotting image {}', image)

    display, art = image.plot(fig=fig.figure,
                              regions=CONFIG.samples.plots.contours,
                              labels=CONFIG.samples.plots.labels,
                              coords='pixel',
                              use_blit=False)

    return art


def plot_drizzle(fig, *indices, ff):
    logger.info('Plotting drizzle image.')

    if not indices or indices[-1] == 'Drizzle':
        ff.show_colorscale(cmap=CONFIG.plotting.cmap)
        fig.tight_layout()


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
# Setup / Load data

def init(paths, telescope, target, overwrite):
    files, folders = paths.files.info, paths.folders
    run = shocCampaign.load(folders.root, obstype='object')

    # update info if given
    info = check_required_info(run, telescope, target)
    logger.debug('User input from CLI: {}', info)
    if info:
        if info['target'] == '.':
            info['target'] = folders.root.name.replace('_', ' ')
            logger.info('Using target name from root directory name: {!r}',
                        info['target'])

        missing_telescope_info = run[~np.array(run.attrs.telescope, bool)]
        missing_telescope_info.attrs.set(repeat(telescope=info.pop('telescope')))
        run.attrs.set(repeat(info))

    # write script for remote data retrieval
    if (CONFIG.remote.write_rsync_script and
            (overwrite or not (files.remote.exists() or files.rsync_script.exists()))):
        write_rsync_script(run, paths)

    return run, info


def write_rsync_script(run, paths, username=CONFIG.remote.username,
                       server=CONFIG.remote.server):

    remotes = list(map(str, run.calls.get_server_path(None)))
    prefix = f'{server}:{shared_prefix(remotes)}'
    filelist = paths.folders.info / 'remote-files.txt'
    io.write_lines(filelist, [remove_prefix(_, prefix) for _ in remotes])

    outfile = paths.folders.info / 'rsync-remote.sh'
    outfile.write_text(
        f'sudo rsync -arvzh --info=progress2 '
        f'--files-from={filelist!s} --no-implied-dirs '
        f'{username}@{prefix} {paths.folders.output!s}'
    )


# ---------------------------------------------------------------------------- #
# Preview

def preview(run, paths, info, ui, overwrite):
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
    run.tabulate.to_xlsx(paths.files.info.summary)
    logger.info('The table above is available in spreadsheet format at:\n'
                '{!s:}', paths.files.info.summary)

    # Sample images prior to calibration and header info
    return compute_preview(run, paths, ui, overwrite)


def compute_preview(run, paths, ui, overwrite, show_cutouts=False):

    # get results from previous run
    overview, data_products = products.get_previous(run, paths)

    # write fits headers to text
    headers_to_txt(run, paths, overwrite)

    # thumbs = ''
    # if overwrite or not products['Images']['Overview']:
    samples = get_sample_images(run, detection=False, show_cutouts=show_cutouts)

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
        **CONFIG.registration, **{**CONFIG.detection, 'report': False}
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
    save_samples_fits(samples_cal, wcss, paths.files.samples, overwrite)

    # Drizzle image
    reg.drizzle(files.drizzle, CONFIG.registration.drizzle.pixfrac)

    return reg


def plot_overview(run, reg, paths, ui, overwrite):

    kws = dict(CONFIG.registration.plots.mosaic)
    filename = kws.pop('filename')
    task = PlotTask(ui, ('Overview', 'Mosaic'),
                          filename, overwrite)

    task(reg.mosaic)(names=run.files.stems, **kws)

    # Create mosaic plot
    # mosaic = reg.mosaic(names=run.files.stems, **kws)

    # txt, arrows = mos.mark_target(
    #     run[0].target,
    #     arrow_head_distance=2.,
    #     arrow_size=10,
    #     arrow_offset=(1.2, -1.2),
    #     text_offset=(4, 5), size=12, fontweight='bold'
    # )

    o = PlaceHolder()

    # drizzle
    filename = paths.files.registration.drizzle
    task = PlotTask(ui, ('Overview', 'Drizzle'),
                          filename.with_suffix('.png'),
                          overwrite)

    task(plot_drizzle)(o, ff=apl.FITSFigure(str(filename)))


# ---------------------------------------------------------------------------- #
@update_defaults(CONFIG.tracking.params)
def _track(hdu, seg, labels, coords, path, overwrite=False, dilate=0, njobs=-1):

    logger.info(motley.stylize('Launching tracker for {:|darkgreen}.'),
                hdu.file.name)

    if CONFIG.tracking.params.circularize:
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

        # plot individual features
        fig = plot_positions_individual(tracker)

        # plot positions time series
        fig, ax = plt.subplots(figsize=(12, 5))
        tracker.show.positions_time_series(ax)
        fig.savefig(plotpath / 'positions-ts.png')

    return tracker, ui


def track(run, reg, paths, overwrite=False):

    logger.section('Source Tracking')
    spanning = sorted(set.intersection(*map(set, reg.labels_per_image)))
    logger.info('Sources: {} span all observations.', spanning)

    image_labels = itt.islice(zip(reg, reg.labels_per_image), 1, None)
    folder_pattern = paths.folders.tracking
    filenames = paths.files.tracking
    for i, (hdu, (img, labels)) in enumerate(zip(run, image_labels)):
        missing_files = {
            key: path
            for key, path in filenames.items()
            if not products.resolve_path(folder_pattern, hdu).exists()
        }
        if overwrite or missing_files:
            logger.info('Launching tracker for {}.', hdu.file.name)

            # back transform to image coords
            coords = reg._trans_to_image(i).transform(reg.xy[sorted(labels)])
            tracker, ui = _track(hdu, img.seg, spanning, coords, True)

        # return ui
    logger.info('Source tracking complete.')
    return spanning


def lightcurves(run, paths, ui, plot=True, overwrite=False):

    lcs = lc.extract(run, paths, overwrite)

    if plot:
        for diff, db in lcs.items():
            plot_lcs(db, ui, filename_template=None, overwrite=False, delay=True)
            break

    # for hdu in run:
    #     lc.io.load_raw(hdu, products.resolve_path(paths.lightcurves.raw, hdu),
    #                 overwrite)

    # for step in ['flagged', 'diff0', 'decor']:
    #     file = paths.lightcurves[step]
    #     load_or_compute(file, overwrite, LOADERS[step], (hdu, file))


def plot_lcs(lcs, ui=None, filename_template=None, overwrite=False, **kws):

    # ui = MplMultiTab(title='Light curves', pos='N')

    tab = 'Light curves'

    filenames = {}
    o = PlaceHolder()
    for date, ts in lcs.items():
        filenames[date] = filename = Path(str(filename_template).format(date=date))
        year, day = date.split('-', 1)

        task = PlotTask(ui, (tab, year, day),
                              filename, overwrite)

        task(lc.plot)(o, ts, **kws)

    #     if ui:
    #         fig = ui.add_tab(tab, year, day,
    #                          fig={'figsize': (12, 6)}).figure
    #     # else:
    #     #     fig =

    #     if not delay:
    #         fig = _plot_ts(fig, ts, filename, overwrite, **kws)

    # if ui and delay:  #
    #     ui[tab].add_callback(_plot_ts_ui, ui=ui,
    #                          lcs=lcs, filenames=filenames,
    #                          overwrite=overwrite, **kws)

    return ui


# def _plot_ts_ui(fig, indices, ui, lcs, filenames, overwrite, **kws):
#     date = '-'.join(ui.tabs.tab_text(indices)[-2:])
#     return lc.plot(fig, lcs[date], filenames[date], overwrite, **kws)


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
        run, paths, info, ui, overwrite
    )

    # ------------------------------------------------------------------------ #
    # Calibrate
    calibration(run, overwrite)

    # ------------------------------------------------------------------------ #
    # Image Registration

    # have to ensure we have single target here
    target = check_single_target(run)

    reg = registration(run, paths, ui, plot, show_cutouts, overwrite)

    # Write data products spreadsheet
    products.write_xlsx(run, paths, overview)
    
    raise SystemError()

    # ------------------------------------------------------------------------ #
    # Source Tracking
    spanning = track(run, reg, paths)

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
        ui.show()

        if not is_interactive():
            sys.exit(app.exec_())

    from IPython import embed
    embed(header="Embedded interpreter at 'src/pyshoc/pipeline/main.py':676")

# %%
