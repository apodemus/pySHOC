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
from astropy.io import fits
from matplotlib import rcParams
from mpl_multitab import QtWidgets

# local
import motley
from pyxides.vectorize import repeat
from scrawl.image import plot_image_grid
from obstools.image import SkyImage
from obstools.modelling import int2tup
from obstools.phot.tracking import SourceTracker
from recipes.flow import Catch
from recipes.iter import cofilter
from recipes.functionals import negate
from recipes.shell import is_interactive
from recipes.decorators import update_defaults
from recipes import io, not_null, op, pprint as pp
from recipes.string import remove_prefix, shared_prefix
from recipes.functionals.partial import PlaceHolder as o

# relative
from .. import CONFIG, shocCampaign
from . import products, lightcurves as lc
from .plotting import GUI
from .calibrate import calibrate
from .logging import logger, config as config_logging


# ---------------------------------------------------------------------------- #
# logging config
config_logging()

# plot config
rcParams.update({
    'font.size': CONFIG.plotting.font.size,
    'axes.labelweight': CONFIG.plotting.axes.labelweight,
    'image.cmap': CONFIG.plotting.cmap
})
# rc('text', usetex=False)

#
CONSOLE_CUTOUTS_TITLE = CONFIG.console.cutouts.pop('title')


# ---------------------------------------------------------------------------- #
# TODO group by source

# track
# photomerty
# decorrelate
# spectral analysis

# ---------------------------------------------------------------------------- #


# def identify(run):
#     # identify
#     # is_flat = np.array(run.calls('pointing_zenith'))
#     # run[is_flat].attrs.set(repeat(obstype='flat'))

#     g = run.guess_obstype()


# ---------------------------------------------------------------------------- #
# Utility Helpers

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
# Sample Images

def get_sample_images(run, detection=True, show_cutouts=False):

    # sample = delayed(get_sample_image)
    # with Parallel(n_jobs=1) as parallel:
    # return parallel

    # Get params from config
    if detection:
        logger.section('Source Detection')

        if detection is True:
            detection = dict(CONFIG.detection)
            detection.pop('algorithm')

    #
    samples = defaultdict(dict)
    for hdu in run:
        for interval, image in _get_hdu_samples(hdu, detection, show_cutouts):
            samples[hdu.file.name][interval] = image

    return samples


def _get_hdu_samples(hdu, detection, show_cutouts):
    params = CONFIG.samples.params
    stat, min_depth, n_intervals, subset = \
        op.attrgetter('stat', 'min_depth', 'n_intervals', 'subset')(params)

    for i, (j, k) in enumerate(get_intervals(hdu, subset, n_intervals)):
        # Source detection. Reporting happens below.
        # NOTE: caching enabled for line below in `setup`
        image = SkyImage.from_hdu(hdu,
                                  stat, min_depth, (j, k),
                                  detection, report=False)

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
def get_tab_key(hdu):
    year, day = str(hdu.date_for_filename).split('-', 1)
    return (year, day, hdu.file.nr)


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

        yield plot_thumbnails(samples, ui, **{'overwrite': overwrite,
                                              **thumbnails})


def _plot_image(image, *args, **kws):
    return image.plot(image, *args, **kws)


def _plot_sample_images(run, samples, path_template, overwrite, ui):

    task = ui.task_factory(_plot_image)(fig=o,
                                        regions=CONFIG.samples.plots.contours,
                                        labels=CONFIG.samples.plots.labels,
                                        coords='pixel',
                                        use_blit=False)

    section = CONFIG.samples.tab
    for hdu in run.sort_by('t.t0'):
        # grouping
        tab = get_tab_key(hdu)

        for (j, k), image in samples[hdu.file.name].items():
            # get filename
            filename = products.resolve_path(path_template, hdu, j, k)

            # add tab to ui
            key = (*section, *tab)
            if frames := remove_prefix(hdu.file.stem, filename.stem):
                key = (*key, frames)

            # plot
            yield ui.add_task(task, key, filename, overwrite, image=image)

    if ui:
        ui[section].link_focus()


# @caching.cached(typed={'hdu': _hdu_hasher}, ignore='save_as')
def plot_thumbnails(samples, ui, tab, filename=None, overwrite=False, **kws):

    # portion = mit.chunked(sample_images, len(run))
    images, = zip(*map(dict.values, samples.values()))

    # filenames, images = zip(*(map(dict.items, samples.values())))
    task = ui.task_factory(plot_image_grid)(images,
                                            fig=o,
                                            titles=list(samples.keys()),
                                            use_blit=False,
                                            **kws)
    ui.add_task(task, tab, filename, overwrite)


# def plot_image(fig, *indices, image):

#     # image = samples[indices]
#     logger.debug('Plotting image {}', image)

#     display, art = image.plot(fig=fig.figure,
#                               regions=CONFIG.samples.plots.contours,
#                               labels=CONFIG.samples.plots.labels,
#                               coords='pixel',
#                               use_blit=False)

#     return art

# ---------------------------------------------------------------------------- #
# plotting


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
    logger.bind(indent=' ').info(
        'Observations of {} by date:\n{:s}\n', info['target'],
        daily.pformat(titled=repr)
    )

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

# @flow.log.section('Calibration')
def calibration(run, overwrite):
    logger.section('Calibration')

    # Compute/retrieve master dark/flat. Point science stacks to calibration
    # images.
    gobj, mdark, mflat = calibrate(run, overwrite=overwrite)


# ---------------------------------------------------------------------------- #
# Image Registration

def registration(run, paths, ui, plot, show_cutouts, overwrite):

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
        # -------------------------------------------------------------------- #
        # Plot calibrated thumbnails if calibratuib data available available
        if any(map(any, run.attrs('calibrated.dark', 'calibrated.flat'))):
            cfg = CONFIG.samples.plots.thumbnails
            thumbs = {**cfg.raw, **cfg.calibrated}
        else:
            thumbs = {}
            logger.info("No calibration data available, won't plot calibrated "
                        "thumbnails.")

        # Plot calibrated sample images
        # -------------------------------------------------------------------- #
        plot_sample_images(run, samples_cal,
                           paths.files.samples.filename, overwrite,
                           thumbs, ui)

    # align
    # ------------------------------------------------------------------------ #
    if do_reg:
        logger.section('Image Registration (WCS)')
        reg = register(run, samples_cal, paths, ui, plot, overwrite)

    if plot:
        config = CONFIG.registration

        # DSS mosaic
        # -------------------------------------------------------------------- #
        inner, outer = config.plots.mosaic.split(('show', 'tab', 'filename'))
        if outer.show:
            survey = config.params.survey
            task = ui.task_factory(reg.mosaic)(names=run.files.stems, **inner)
            ui.add_task(task, (*outer.tab, survey.upper()),
                        str(outer.filename).replace('$TEL', survey),
                        outer.get('overwrite', overwrite))

            # TODO mark target

        # Drizzle
        # -------------------------------------------------------------------- #
        if config.drizzle.show:
            filename = paths.files.registration.drizzle
            task = ui.task_factory(plot_drizzle)(fig=o, filename=filename)
            ui.add_task(task, config.drizzle.tab,
                        filename.with_suffix('.png'), overwrite)

    logger.success(' Image Registration complete!')

    return reg


def _registry_plot_tasks(run, paths, overwrite):

    # config
    inner, outer = CONFIG.registration.plots.split(
        ('show', 'tab', 'filename', 'overwrite'))
    input_config = {'alignment': {},
                    'clusters': {},
                    'mosaic': {},
                    **inner}
    input_config['mosaic']['connect'] = False

    # alignment
    templates = paths.templates['HDU'].find('alignment').flatten()
    # get alignment reference hdus. These don't have plots for themselves
    _, indices = run.group_by('telescope', return_index=True)
    indices = np.hstack([idx for idx in indices.values()])
    desired_files = products._resolve_by_file(run[indices], templates)

    section = outer.alignment.tab
    ovr = outer.alignment.get('overwrite', overwrite)
    align_config = input_config['alignment']

    # firsts = run.attrs()
    for stem, (file, ) in desired_files.items():
        key = (*section, *get_tab_key(run[stem]))
        yield 'alignment', key, file, ovr, align_config

    # mosaic / clusters
    templates = paths.templates['TEL'].flatten()
    tel_products = products._get_desired_products(
        sorted({*run.attrs.telescope, 'all'})[::-1], templates, 'TEL')

    for tel, files in tel_products.items():
        for key, file in zip(templates.keys(), files):
            which = key[-1]
            if outer[which].show:
                key = (*CONFIG[key].tab, tel)

            ovr = outer[which].get('overwrite', overwrite)
            yield which, key, file, ovr, input_config[which]


def echo_fig(fig, *args, **kws):
    return fig


def register(run, samples, paths, ui, plot, overwrite):

    config = CONFIG.registration

    plot_config = False
    if plot:
        #
        task = ui.task_factory(echo_fig)(fig=o)

        # pre generate figures and pass to `coalign` via `plots` parameter
        plot_config = defaultdict(dict)
        plot_config['alignment'] = []
        for name, key, file, ovr, kws in _registry_plot_tasks(run, paths, overwrite):

            ui.add_task(task, key, file, ovr)

            kws = {**kws, 'fig': ui[key].figure}
            if name == 'alignment':
                plot_config[name].append(kws)
            else:
                plot_config[name][key[-1]] = kws

        if ui:
            ui[config.plots.alignment.tab].link_focus()
            ui[config.plots.clusters.tab].link_focus()

    # Align with survey image
    # ------------------------------------------------------------------------ #
    # NOTE: source detections were reported above in `get_sample_images`
    reg = run.coalign_survey(**config.params,
                             **{**CONFIG.detection, 'report': False},
                             plots=plot_config)

    # save registry
    # ------------------------------------------------------------------------ #
    files = paths.files.registration
    logger.info('Saving image registry at: {}.', files.registry)
    reg.params = np.load(files.params)
    io.serialize(files.registry, reg)
    # np.save(paths.reg.params, reg.params)
    # TODO region files

    # Build image WCS
    # ------------------------------------------------------------------------ #
    wcss = reg.build_wcs(run)

    # save samples fits
    save_samples_fits(run, samples, wcss,
                      paths.files.samples.filename.with_suffix('.fits'),
                      overwrite)

    # Drizzle image
    # ------------------------------------------------------------------------ #
    reg.drizzle(files.drizzle, config.drizzle.pixfrac)

    return reg


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


def plot_drizzle(fig, *indices, filename):
    # logger.info('POOP drizzle image: {} {}', fig, indices)
    logger.info('Plotting drizzle image: {} {}', fig, indices)

    ff = apl.FITSFigure(str(filename), figure=fig)

    ff.show_colorscale(cmap=CONFIG.plotting.cmap)
    fig.tight_layout()

    # return fig

# def plot_overview(run, reg, paths, ui, overwrite):

#     # ------------------------------------------------------------------------ #

#     # count = itt.count(1)
#     # for name, key, file, ovr, kws in _registry_plot_tasks(run, paths, overwrite):
#     # if name == 'alignment':
#     #     reg.model.gmm.plot(reg[next(count)]

#     # obs = groups.get(key[-1], run)

#     # task = ui.task_factory(func)()
#     # ui.add_task(task, key, file, ovr, **kws)
#     print(name, key, file, kws)

#     # mosaic = reg.mosaic(names=run.files.stems, **kws)

#     # txt, arrows = mos.mark_target(
#     #     run[0].target,
#     #     arrow_head_distance=2.,
#     #     arrow_size=10,
#     #     arrow_offset=(1.2, -1.2),
#     #     text_offset=(4, 5), size=12, fontweight='bold'
#     # )


# def mosaic(fig, *key, reg, **kws):
#     # names=run.files.stems,
#     return reg.mosaic(**kws)


# def plot_clusters(fig, *key, reg, **kws):
#     return reg.plot_clusters(**kws)

# # def plot_gmm_fit(fig, *key, reg, **kws):
# #     reg.model.gmm.plot(**kws, fig=fig)
# #     reg


# ---------------------------------------------------------------------------- #
# Tracking


def track(run, reg, paths, ui, plot=True, overwrite=False, njobs=-1):

    logger.section('Source Tracking')

    # setup detection (needed for recovery when tracking lost)
    spanning = sorted(set.intersection(*map(set, reg.labels_per_image)))
    spanning = np.add(spanning, 1)
    logger.info('Sources: {} span all observations.', spanning)

    filenames = paths.files.tracking
    images_labels = list(itt.islice(zip(reg, reg.labels_per_image), 1, None))
    overwrite = overwrite or CONFIG.tracking.get(overwrite, False)
    for i, (hdu, (img, labels)) in enumerate(zip(run, images_labels), 1):
        # check if we need to run
        if overwrite or _tracker_missing_files(filenames, hdu, spanning):
            # back transform to image coords
            coords = reg._trans_to_image(i).transform(reg.xy[sorted(labels)])
            # path = products.resolve_path(paths.folders.tracking, hdu)
            tracker = _track(reg, hdu, img.seg, spanning, coords,
                             paths, ui, plot, overwrite, njobs=njobs)

    logger.info('Source tracking complete.')
    return spanning


@update_defaults(CONFIG.tracking.params)
def _track(reg, hdu, seg, labels, coords, paths, ui, plot=True, overwrite=False,
           dilate=0, njobs=-1):

    logger.bind(indent=True).opt(lazy=True).info(
        'Launching tracker for {0[0]}.\ncoords = {0[1]}',
        lambda: (motley.darkgreen(hdu.file.name),
                 pp.nrs.matrix(coords, 2).replace('\n', f'\n{"": >9}'))
    )

    # Make circular regions for measuring centroids
    if (cfg := CONFIG.tracking).params.circularize:
        seg = seg.circularize()

    if njobs == -1:
        njobs = CONFIG.tracking.params.njobs

    base = products.resolve_path(paths.folders.tracking.folder, hdu)
    tracker = SourceTracker(coords, seg.dilate(dilate), labels=labels)
    tracker.reg = reg
    # tracker.detection.algorithm = CONFIG.detection
    tracker.init_memory(hdu.nframes, base, overwrite=overwrite)
    tracker.run(hdu.calibrated, njobs=njobs, jobname=hdu.file.stem)

    # plot
    if plot and cfg.plot:
        kws, tmp = cfg.plots.positions.split('filename')
        tmp = str(products.resolve_path(tmp.filename, hdu))

        #                  year, day, nr
        tab = (*cfg.tab, *get_tab_key(hdu))
        for j in tracker.use_labels:
            # plot source location features
            task = ui.task_factory(tracker.plot.positions_source)(o, o[-1], **kws)
            ui.add_task(task, (*tab, f'Source {j}'),
                        tmp.replace('$SOURCE', str(j)), overwrite)

        # plot positions displacement time series
        kws, outer = cfg.plots.time_series.split(('filename', 'tab'))

        task = ui.task_factory(tracker.plot.displacement_time_series)(o.axes[0], **kws)
        ui.add_task(task, (*tab, *outer.tab),
                    products.resolve_path(outer.filename, hdu), overwrite,
                    add_axes=True)

    return tracker


def _tracker_target_files(templates, hdu, sources):
    desired = templates.map(products.resolve_path, hdu)
    files, position_plots = desired.split('positions')
    target_files = {
        key: products.resolve_path(path, hdu)
        for key, path in files.flatten().items()
    }

    # plots
    key, position_plots = position_plots.flatten().popitem()
    position_plots = str(position_plots)
    target_files.update({
        (*key, i): Path(position_plots.replace('$SOURCE', str(i)))
        for i in sources
    })
    return target_files


def _tracker_missing_files(templates, hdu, sources):
    target_files = _tracker_target_files(templates, hdu, sources)
    missing = {key: file
               for key, file in target_files.items()
               if not file.exists()}

    if missing and (missing != target_files):
        logger.bind(indent=4).opt(lazy=True).info(
            'Source Tracker for {0[0]} missing some target files:\n{0[1]}.',
            lambda: (hdu.file.name, pp.pformat(missing, rhs=str))
        )
    else:
        logger.info('First time run for {}.', hdu.file.name)

    return missing


# ---------------------------------------------------------------------------- #
# Light curves

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

    return lcs
    # for hdu in run:
    #     lc.io.load_raw(hdu, products.resolve_path(paths.lightcurves.raw, hdu),
    #                 overwrite)

    # for step in ['flagged', 'diff0', 'decor']:
    #     file = paths.lightcurves[step]
    #     load_or_compute(file, overwrite, LOADERS[step], (hdu, file))


def plot_lcs(lcs, step, ui=None, filename_template=None, overwrite=False, **kws):

    section = CONFIG.lightcurves.tab
    task = ui.task_factory(lc.plot)(o, **kws)

    filenames = {}
    for date, ts in lcs.items():
        filenames[date] = filename \
            = Path(filename_template.substitute(DATE=date)).with_suffix('.png')
        year, day = date.split('-', 1)
        tab = (*section, year, day, step)
        ui.add_task(task, tab, filename, overwrite, None, False, ts)

        if ui:
            ui[tab[:-2]].link_focus()

    return ui


# Main
# ---------------------------------------------------------------------------- #

def main(paths, target, telescope, top, njobs, plot, show_cutouts, overwrite):
    #
    # from obstools.phot import PhotInterface

    # GUI
    if plot:
        if not is_interactive():
            app = QtWidgets.QApplication(sys.argv)

    # GUI
    ui = GUI(title=CONFIG.plotting.gui.title,
             pos=CONFIG.plotting.gui.pos,
             active=plot)

    # ------------------------------------------------------------------------ #
    # Setup / Load data
    run, info = init(paths, telescope, target, overwrite)
    logger.success('Pipeline initialization complete!')

    # ------------------------------------------------------------------------ #
    # Preview
    overview, data_products, samples, thumbnails = preview(
        run, paths, info, ui, plot, overwrite
    )
    logger.success('Preview completed.')

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
    spanning = track(run, reg, paths, ui, overwrite, njobs)

    # ------------------------------------------------------------------------ #
    # Photometry
    logger.section('Photometry')
    lcs = lightcurves(run, paths, ui, plot, overwrite)

    # phot = PhotInterface(run, reg, paths.phot)
    # ts = mv phot.ragged() phot.regions()

    # ------------------------------------------------------------------------ #
    # Launch the GUI
    if plot:
        logger.section('Launching GUI')
        # activate tab switching callback (for all tabs)
        cfg = CONFIG.registration
        survey = cfg.params.survey.upper()
        ui['Overview'].move_tab('Mosaic', 0)
        ui['Overview', 'Mosaic'].move_tab(survey, 0)
        ui.set_focus(*cfg.plots.mosaic.tab, survey)

        ui.show()

        if not is_interactive():
            # sys.exit(app.exec_())
            app.exec_()

        logger.section('UI shutdown')

        # Run incomplete plotting tasks
        trap = Catch(action=logger.warning,
                     message=('Could not complete plotting task '
                              'due the following {err.__class__.__name__}: {err}'))
        getter = op.AttrVector('plot.func.filename', default=None)
        tabs = list(ui.tabs._leaves())
        filenames, tabs = cofilter(getter.map(tabs), tabs)
        unsaved, tabs = map(list, cofilter(negate(Path.exists), filenames, tabs))
        if n := len(unsaved):
            logger.info('Now running {} incomplete tasks:', n)
            with trap:
                for tab in tabs:
                    tab.run_task()

    # ------------------------------------------------------------------------ #
    logger.section('Finalize')

    # get results from this run
    overview, data_products = products.get_previous(run, paths)

    # Write data products spreadsheet
    products.write_xlsx(run, paths, overview)
    # This updates spreadsheet with products computed above

    logger.info('Thanks for using pyshoc! Please report any issues at: '
                'https://github.com/astromancer/pyshoc/issues.')
