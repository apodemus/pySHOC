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
from IPython import embed
from astropy.io import fits
from mpl_multitab import QtWidgets

# local
import motley
from pyxides.vectorize import repeat
from scrawl.image import plot_image_grid
from obstools.image import SkyImage
from obstools.modelling import int2tup
from obstools.phot.tracking import SourceTracker
import obstools.lightcurves as lcs
from recipes.oo.slots import SlotHelper
from recipes.shell import is_interactive
from recipes.decorators import update_defaults
from recipes import io, not_null, op, pprint as pp
from recipes.containers.sets import csv, single_valued
from recipes.string import remove_prefix, shared_prefix
from recipes.functionals.partial import Partial, PlaceHolder as o

# relative
from .. import Campaign, config as cfg
from ..timing import TimeDelta
from ..config import SAVE_KWS, Template, _is_special
from . import products, lightcurves as lc
from .logging import logger
from .calibrate import calibrate
from .plotting import PlotManager


# ---------------------------------------------------------------------------- #
# SHOW_SOURCE_THUMBNAILS_CONSOLE =

# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
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


def last_common_ancestor(paths):
    if len(paths) == 1:
        return paths[0].parent

    common = Path()
    for parts in zip(*map(Path.parts.fget, paths)):
        if len(set(parts)) > 1:
            break

        common /= parts[0]

    return common


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
            detection = dict(cfg.detection)
            detection.pop('algorithm')

    #
    samples = defaultdict(dict)
    for hdu in run:
        for interval, image in _get_hdu_samples(hdu, detection, show_cutouts):
            samples[hdu.file.name][interval] = image

    return samples


def _get_hdu_samples(hdu, detection, show_cutouts):
    params = cfg.samples.params
    stat, min_depth, n_intervals, subset = \
        op.attrgetter('stat', 'min_depth', 'n_intervals', 'subset')(params)

    # cfg.console.cutouts.filter('show')
    for i, (j, k) in enumerate(get_intervals(hdu, subset, n_intervals)):
        # Source detection. Reporting happens below.
        # NOTE: caching enabled for line below in `setup`
        image = SkyImage.from_hdu(hdu, stat, min_depth, (j, k),
                                  detection, report=False)

        if show_cutouts and i == 0 and image.seg:
            title_format = cfg.detection.report.cutouts['title']
            logger.opt(lazy=True).info(
                'Source images:\n{}',
                lambda: image.seg.show.console.format_cutouts(
                    image.data,
                    **{**cfg.detection.report.cutouts,
                        'title': title_format.format(hdu=hdu)}
                )
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
                       thumbnails=None, plotter=None, replace=False):

    return tuple(_iplot_sample_images(run, samples, path_template, overwrite,
                                      thumbnails, plotter, replace))


def _iplot_sample_images(run, samples, path_template, overwrite,
                         thumbnails, plotter, replace):

    if ui := plotter.gui:
        logger.info('Adding sample images to plot GUI: {}', ui)

    yield list(
        _plot_sample_images(run, samples, path_template, overwrite, plotter,
                            replace)
    )

    # plot thumbnails for sample image from first portion of each data cube
    if not_null(thumbnails, except_=[{}]):
        if thumbnails is True:
            thumbnails = {}

        yield plot_thumbnails(samples, plotter, **{'overwrite': overwrite,
                                                   **thumbnails})


def _plot_image(image, *args, **kws):
    return image.plot(*args, **kws)


def _plot_sample_images(run, samples, path_template, overwrite, plotter, replace):

    task = plotter.task_factory(_plot_image)(fig=o,
                                             regions=cfg.samples.plots.contours,
                                             labels=cfg.samples.plots.labels,
                                             coords='pixel',
                                             use_blit=False)

    section = cfg.samples.tab
    for hdu in run.sort_by('t.t0'):
        # grouping
        tab = get_tab_key(hdu)

        for frames, image in samples[hdu.file.name].items():
            # get filename
            filename = path_template.resolve_path(hdu, frames=frames)

            # add tab to ui
            key = (*section, *tab)
            if frames := remove_prefix(hdu.file.stem, filename.stem):
                key = (*key, frames)

            # plot
            if plotter.should_plot([filename], overwrite):
                yield plotter.add_task(task, key, filename, overwrite, image=image,
                                       replace=replace)

    if plotter.gui:
        plotter.gui[section].link_focus()


def get_tab_key(hdu):
    year, day = str(hdu.date_for_filename).split('-', 1)
    *nr, ext = hdu.file.path.suffixes
    nr = ''.join(nr).lstrip('.')
    return (year, day, nr)

# @caching.cached(typed={'hdu': _hdu_hasher}, ignore='save_as')


def plot_thumbnails(samples, plotter, tab, filenames=(), overwrite=False, **kws):

    # portion = mit.chunked(sample_images, len(run))
    images, = zip(*map(dict.values, samples.values()))

    # filenames, images = zip(*(map(dict.items, samples.values())))
    task = plotter.task_factory(plot_image_grid)(images,
                                                 fig=o,
                                                 titles=list(samples.keys()),
                                                 use_blit=False,
                                                 **kws)
    plotter.add_task(task, tab, filenames, overwrite)


# def plot_image(fig, *indices, image):

#     # image = samples[indices]
#     logger.debug('Plotting image {}', image)

#     display, art = image.plot(fig=fig.figure,
#                               regions=cfg.samples.plots.contours,
#                               labels=cfg.samples.plots.labels,
#                               coords='pixel',
#                               use_blit=False)

#     return art


class Pipeline(SlotHelper):

    __slots__ = ('campaign', 'paths', 'plotter', 'overwrite',
                 'info', 'samples', 'registry', 'trackers', 'lightcurves')

    def __init__(self, paths, target, telescope, plot=True, gui=True,
                 overwrite=False):

        self.paths = paths
        self.overwrite = bool(overwrite)

        # load data
        root = paths.folders.root
        run = Campaign.load(root, obstype='object')
        self.campaign = run = run.sort_by('t.t0')

        # update info if given
        self.info = info = check_required_info(run, telescope, target)
        logger.debug('User input from CLI: {}', info)
        if info:
            if info['target'] == '.':
                info['target'] = root.name.replace('_', ' ')
                logger.info('Using target name from root directory name: {!r}',
                            info['target'])

            missing_telescope_info = run[~np.array(run.attrs.telescope, bool)]
            missing_telescope_info.attrs.set(repeat(telescope=info.pop('telescope')))
            run.attrs.set(repeat(info))

        # create output folders for templated paths
        logger.info('Creating folders for templated paths.')
        templated = set(paths.folders.select(values=_is_special).flatten().values())
        for tmp, hdu in itt.product(templated, run):
            path = Template(tmp).resolve_path(hdu)
            path.mkdir(parents=True, exist_ok=True)

        # write script for remote data retrieval
        if ((files := paths.files.remote).get('rsync_script') and
                (overwrite or not all((f.exists() for f in files.values())))):
            write_rsync_script(run, paths)

        # setup plot manager
        self.plotter = PlotManager(plot, gui, **cfg.plotting.gui)

        self.samples = {}
        self.registry = None
        self.trackers = {}
        self.lightcurves = {}

        # done
        logger.success('Initialization complete!')

    # ---------------------------------------------------------------------------- #
    # Preview

    def preview(self, show_cutouts, overwrite=None):
        logger.section('Overview')

        info = self.info
        paths = self.paths
        run = self.campaign

        # show filenames
        logger.info('The following data were loaded:\n{}', run.pformat())

        # write observation log latex table
        paths.files.info.obslog.write_text(
            run.tabulate.to_latex(
                style='table',
                caption=f'SHOC observations of {info["target"]}.',
                label=f'tbl:obs-log:{info["target"]}'
            )
        )

        # Print nightly summary table
        nightly = run.group_by('date_for_filename')
        logger.bind(indent=' ').info(
            'Observations of {} by date:\n{:s}\n', info['target'],
            nightly.pformat(titled=repr)
        )

        # Write nightly summary table to latex
        summarize = {
            'telescope':        single_valued,
            'camera':           single_valued,
            'filters.name':     single_valued,
            'nfiles':           sum,
            'nframes':          sum,
            'timing.t0':        op.itemgetter(0),
            'timing.tn':        op.itemgetter(-1),
            'timing.exp':       csv,
            'timing.duration':  Partial(sum)(o, TimeDelta(0, format='sec'))
        }
        tbl = nightly.tabulate.summarize(summarize)
        tbl.to_latex(paths.files.info.summary)
        logger.info('Night log written to: {!s}.', paths.files.info.summary)

        # write summary spreadsheet
        path = str(paths.files.info.spreadsheets.campaign)
        filename, *sheet = path.split('::')

        run.tabulate.to_xlsx(filename, *sheet, overwrite=True)
        logger.info('The table above is available in spreadsheet format at:\n'
                    '    {!s:}', path)

        # Sample images prior to calibration and header info
        return self.compute_preview_products(show_cutouts, overwrite)

    def compute_preview_products(self, show_cutouts=False, overwrite=None):

        paths = self.paths
        run = self.campaign
        plotter = self.plotter
        overwrite = self.overwrite if overwrite is None else overwrite

        # get results from previous run
        overview, data_products = products.get_previous(run, paths)

        # write fits headers to text
        headers_to_txt(run, paths, overwrite)

        # thumbs = ''
        # if overwrite or not products['Images']['Overview']:
        samples = get_sample_images(run, detection=False, show_cutouts=show_cutouts)
        self.samples = samples

        thumbnails = None
        if plotter.active:
            tmp = Template(paths.files.samples.plots.thumbnails.raw)
            filenames = tmp.resolve_paths()
            if plotter.should_plot(filenames, overwrite):
                thumbnails = plot_thumbnails(samples, plotter,
                                             **{'overwrite': overwrite,
                                                **cfg.samples.plots.thumbnails.raw,
                                                'filenames': filenames})
        # source regions
        # if not any(products['Images']['Source Regions']):
        #     sample_images = products['Images']['Samples']

        logger.success('Preview completed.')
        return overview, data_products, samples, thumbnails

    # ---------------------------------------------------------------------------- #
    # Calibrate

    # @flow.log.section('Calibration')
    def calibration(self, overwrite=None):
        logger.section('Calibration')

        # Compute/retrieve master dark/flat. Point science stacks to calibration
        # images.
        overwrite = self.overwrite if overwrite is None else overwrite
        gobj, mdark, mflat = calibrate(self.campaign, overwrite=overwrite)
        logger.success('Calibration frames set.')
        return gobj, mdark, mflat

    # ---------------------------------------------------------------------------- #
    # Image Registration

    def registration(self, show_cutouts, overwrite=None):

        paths = self.paths
        run = self.campaign
        plotter = self.plotter
        overwrite = self.overwrite if overwrite is None else overwrite

        files = paths.files.registration
        if (use_previous := (files.registry.exists() and not overwrite)):
            # hot start
            logger.info('Loading image registry from file: {}.', files.registry)
            try:
                reg = io.deserialize(files.registry)
            except Exception as err:
                logger.warning(
                    'Could not load image registry from file: {} due to the '
                    'following exception:\n {}', files.registry, str(err)
                )
                use_previous = False
            else:
                reg.params = np.load(files.params)

                # retrieve samples from register
                # TODO: get from fits?
                samples_cal = {
                    fn: {('', ''): im}
                    for fn, im in zip(run.files.names, list(reg)[1:])
                }
        #

        if not use_previous:
            # Sample images (after calibration)
            samples_cal = get_sample_images(run, show_cutouts=show_cutouts)

        #
        if plotter.active:
            # -------------------------------------------------------------------- #
            # Plot calibrated thumbnails if calibration data available available
            if any(map(any, run.attrs('calibrated.dark', 'calibrated.flat'))):
                config = cfg.samples.plots.thumbnails
                thumb_grid_config = {**config.raw, **config.calibrated}
            else:
                thumb_grid_config = {}
                logger.info("No calibration data available, won't plot calibrated "
                            "thumbnail grid.")

            # Plot calibrated sample images
            # -------------------------------------------------------------------- #
            plot_sample_images(run, samples_cal,
                               paths.templates.HDU.samples.filename,
                               overwrite, thumb_grid_config, plotter)

        # Align
        # ------------------------------------------------------------------------ #
        if not use_previous:
            logger.section('Image Registration')  # (World Coordinate System)
            reg = self.register(samples_cal,  overwrite)

        # set
        self.registry = reg

        # Drizzle image
        # ------------------------------------------------------------------------ #
        config = cfg.registration
        if not files.drizzle.filename.exists() or overwrite:
            reg.drizzle(files.drizzle.filename, config.drizzle.pixfrac)

        # plotting
        # ------------------------------------------------------------------------ #
        if plotter.active:
            # DSS mosaic
            # -------------------------------------------------------------------- #
            inner, outer = config.mosaic.split(SAVE_KWS)
            if outer.show:
                survey = config.alignment.survey.survey
                task = plotter.task_factory(reg.mosaic)(names=run.files.stems,
                                                        **inner.plot)
                _, template = paths.templates.find('mosaic').flatten().popitem()
                files = template.resolve_paths(TEL=survey)
                ovr = outer.get('overwrite', overwrite)
                if plotter.should_plot(files, ovr):
                    plotter.add_task(task, (*outer.tab, survey.upper()), paths, ovr)

                # TODO mark target

            # Drizzle
            # -------------------------------------------------------------------- #
            if config.drizzle.show:
                filename = paths.files.registration.drizzle.filename
                tmp = Template(paths.files.registration.drizzle.plot)
                files = tmp.resolve_paths()
                ovr = config.drizzle.get('overwrite', overwrite)
                if plotter.should_plot(files, ovr):
                    task = plotter.task_factory(plot_drizzle)(fig=o, filename=filename)
                    plotter.add_task(task, config.drizzle.tab, files, ovr)

        logger.success('Image Registration complete!')

        return reg

    def register(self, samples, overwrite=None):

        paths = self.paths
        run = self.campaign
        reg = self.registry
        plotter = self.plotter
        overwrite = self.overwrite if overwrite is None else overwrite
        config = cfg.registration

        plot_config = False
        if plotter.active:
            #
            task = plotter.task_factory(echo_fig)(fig=o)
            tasks = {}
            # pre generate figures and pass to `coalign` via `plots` parameter
            plot_config = defaultdict(dict)
            plot_config['alignment'] = []
            for name, key, file, ovr, kws in self._registry_plot_tasks(overwrite):
                # print(name, key, kws)
                key = tuple(key)
                tasks[key] = plotter.add_task(task, key, file, ovr)

                kws = {**kws, 'fig': plotter.figures[key]}
                if name == 'alignment':
                    plot_config[name].append(kws)
                else:
                    plot_config[name][key[-1]] = kws

            if ui := plotter.gui:
                ui[config.alignment.tab].link_focus()
                ui[config.clusters.tab].link_focus()

        # Align with survey image
        # ------------------------------------------------------------------------ #
        # NOTE: source detections were reported above in `get_sample_images`
        reg = run.coalign_survey(
            **{**cfg.detection, 'report': False,
               **config.alignment.self},
            plots=plot_config,
            clustering=config.clusters.filter((*SAVE_KWS, 'plot')),
            source_detection_survey=config.alignment.survey.detection
        )

        if plotter.active:
            for tel in set(run.attrs.telescope):
                # save mosaic /  source ids:
                for x in ('mosaic', 'clusters'):
                    (t := tasks[(*config[x].tab, tel)]).save(t.result)

            # save alignment figs
            for key, task in self.plotter.tasks[config.alignment.tab].flatten().items():
                fig = self.plotter.figures[(*config.alignment.tab, *key)]
                if fig.axes:
                    # save
                    fig.canvas.draw()
                else:
                    # plot model reference in empty figures
                    conf = plot_config['alignment']
                    idx = next(i for i, kws in enumerate(conf) if kws['fig'] == fig)
                    im = reg._reg.model.gmm.plot(**conf[idx])
                    im.ax.set_title('Model Likelihood')
                    # fig = im.figure

                # save figure
                task.save(fig)

        # save registry
        # ------------------------------------------------------------------------ #
        files = paths.files.registration
        logger.info('Saving image registry at: {}.', files.registry)
        # reg.params = np.load(files.params)
        io.serialize(files.registry, reg)
        np.save(paths.files.registration.params, reg.params)

        # Build image WCS
        # ------------------------------------------------------------------------ #
        wcss = reg.build_wcs(run)

        # save samples fits
        self.save_samples_fits(run, wcss,
                               paths.files.samples.filename.with_suffix('.fits'),
                               overwrite)

        # update the source regions in the sample plots
        if plotter.active:
            index = itt.count()
            samples2 = defaultdict(dict)
            for filename, subs in samples.items():
                for interval in subs:
                    samples2[filename][interval] = reg._reg[next(index)]

            tasks = plot_sample_images(run, samples2,
                                       paths.templates.HDU.samples.filename,
                                       overwrite=True, thumbnails=False,
                                       plotter=plotter, replace=True)
            samples = samples2


        return reg

    def _registry_plot_tasks(self, overwrite=None):

        run = self.campaign
        paths = self.paths
        overwrite = self.overwrite if overwrite is None else overwrite

        # config
        inner_config, outer_config = cfg.registration.split(SAVE_KWS)
        plot_config = inner_config.find('plot', True)
        plot_config = {'alignment': {}, 'clusters': {}, 'mosaic': {}, **plot_config}
        plot_config['mosaic']['connect'] = False

        # alignment
        templates = paths.templates['HDU'].find('alignment').flatten()

        # get alignment reference hdus. These don't have plots for themselves
        _, indices = run.group_by('telescope', return_index=True)
        indices = np.hstack([idx for idx in indices.values()])
        desired_files = products.get_desired_products(run[indices], templates, 'file')

        tab = outer_config.alignment.tab
        ovr = outer_config.alignment.get('overwrite', overwrite)
        alignment_config = plot_config['alignment']

        # firsts = run.attrs()
        for stem, files in desired_files.items():
            key = (*tab, *get_tab_key(run[stem]))
            files = list(files['registration']['alignment'].values())
            yield ('alignment', key, files, ovr, alignment_config)

        # ------------------------------------------------------------------------ #
        # mosaic / clusters
        templates = paths.templates['TEL'].flatten()
        telescopes = set(run.attrs.telescope)
        if len(telescopes) > 1:
            telescopes.add('all')
        tel_products = products._get_desired_products(sorted(telescopes)[::-1],
                                                      templates, 'TEL')

        for tel, files in tel_products.items():
            for section, files in files['registration'].items():
                if (config := outer_config[section]).show:
                    key = (*config.tab, tel)
                    ovr = outer_config[section].get('overwrite', overwrite)
                    files = list(files.values())
                    yield (section, key, files, ovr, plot_config[section])

    def save_samples_fits(self, wcss, path_template, overwrite):
        # save samples as fits with wcs

        for (file, subs), wcs in zip(self.samples.items(), wcss):
            hdu = self.campaign[file]
            for frames, image in subs.items():
                filename = Template(path_template).resolve_path(hdu, frames=frames)

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
    # Tracking

    def track(self, top, overwrite=None, njobs=-1):

        logger.section('Source Tracking')

        paths = self.paths
        run = self.campaign
        reg = self.registry
        overwrite = self.overwrite if overwrite is None else overwrite

        # setup detection (needed for recovery when tracking lost)
        spanning = sorted(set.intersection(*map(set, reg.labels_per_image)))
        spanning = np.add(spanning, 1)
        logger.info('Sources: {} span all observations.', spanning)

        templates = paths.templates.HDU.tracking

        reg = reg._reg

        labels = {}
        for i, (hdu, img) in enumerate(zip(run, reg)):
            # select brightest sources
            segment_labels = img.seg.labels[img.counts.argsort()[:-top-1:-1]]

            # check if we need to run
            missing = _tracker_missing_files(
                templates, hdu, segment_labels[:cfg.tracking.plots.top])
            if not (overwrite or missing):
                continue

            # back transform to image coords
            coords = reg._trans_to_image(i).transform(reg.xy[segment_labels - 1])

            # coords = reg._trans_to_image(i).transform(reg.xy[cluster_labels - 1])
            # These are the coordinates for all the sources cross-id in this image
            # These combine info from all images and is more accurate than the
            # centroids from a sampled image from a single cube, but there may be
            # sources with coordinates that do not have corresponding segments in
            # the image due to lower image quality etc. For tracking, we have to
            # remove those coordinate points.

            labels[hdu.file.stem] = segment_labels
            # path = products.resolve_path(paths.folders.tracking, hdu)
            self.trackers[hdu.file.name] = self._track(
                hdu, img.seg, coords, segment_labels, overwrite, njobs=njobs
            )

        logger.info('Source tracking complete.')
        return labels, spanning

    @update_defaults(cfg.tracking.params)
    def _track(self, hdu, seg, coords, labels, overwrite=None, dilate=0,
               njobs=-1, **kws):

        paths = self.paths
        reg = self.registry
        plotter = self.plotter
        overwrite = self.overwrite if overwrite is None else overwrite

        #
        logger.bind(indent=True).opt(lazy=True).info(
            'Launching source tracker for {0[0]}.\ncoords = {0[1]}',
            lambda: (motley.darkgreen(hdu.file.name),
                     pp.nrs.matrix(coords, 2).replace('\n', f'\n{"": >9}'))
        )

        # Make circular regions for measuring centroids
        if (config := cfg.tracking).params.circularize:
            seg = seg.circularize()

        if njobs == -1:
            njobs = config.params.njobs

        seg = seg.dilate(dilate)

        # # check coords
        # region_centres = seg.com(seg.data)
        # delta = cdist(coords, region_centres).min(1)
        # delta -= np.median(delta)
        # no_region = delta > config.params.coord_region_distance_cutoff
        # if no_region.any():
        #     logger.info('Trimming source coordinates that have no corresponding '
        #                 'segment for tracking {}:\n{}', hdu.file.name, coords[no_region])
        #     coords = coords[~no_region]

        # Init tracker
        tracker = SourceTracker(coords, seg, labels, **kws)
        tracker.reg = reg
        # tracker.detection.algorithm = cfg.detection

        # Init memory
        base = Template(paths.folders.tracking.folder).resolve_path(hdu)
        tracker.init_memory(hdu.nframes, base, overwrite=overwrite)
        # run on observation
        tracker.run(hdu.calibrated, njobs=njobs, jobname=hdu.file.stem)
        # save tracker data
        tracker.save(paths.templates.HDU.tracking.init_arrays.resolve_path(hdu))

        # extract lightcurve
        raw = self.lightcurves[('by_file', 'raw', )] = \
            tracker.to_lightcurve(hdu.t.bjd, **lc.get_metadata(hdu))
        # save
        out = self.paths.templates.HDU.lightcurves.by_file.raw.filename.sub(
            HDU=hdu.file.stem, EXT='npz')
        raw.save(out)

        # plot
        SAVE_KWS = ('filename', 'overwrite')
        if plotter and config.plot:
            tmp = paths.templates.HDU.tracking.plots

            kws, save = config.plots.positions.split(SAVE_KWS)
            save.setdefault('overwrite', overwrite)
            save.pop('filename', '')

            #                  year, day, nr
            tab = (*config.tab, *get_tab_key(hdu))
            top = config.plots.top
            logger.success('Tracker plots for top {} sources: {}', top, tab)
            for i, j in enumerate(tracker.use_labels[:top]):
                # plot source location features
                task = plotter.task_factory(tracker.plot.positions_source)(o, i, **kws)
                save['filenames'] = tmp.positions.resolve_paths(hdu, source=j)
                plotter.add_task(task, (*tab, f'Source {j}'), **save)

            # plot positions displacement time series
            kws, save = config.plots.time_series.split(SAVE_KWS)
            save.setdefault('overwrite', overwrite)
            save.pop('filename', '')
            save['filenames'] = tmp.time_series.resolve_paths(hdu)

            tab = (*tab, *kws.pop('tab'))
            task = plotter.task_factory(tracker.plot.displacement_time_series)(o.axes[0], **kws)
            plotter.add_task(task, tab, **save, add_axes=True)

        return tracker

    # ---------------------------------------------------------------------------- #
    # Time Series Analysis

    def time_series_analysis(self, labels, overwrite=None):

        overwrite = self.overwrite if overwrite is None else overwrite
        output_templates = self.paths.templates.find('lightcurves', collapse=True)


        # Init time series pipeline
        # infiles = self.paths.templates.HDU.lightcurves.by_file.raw.filename
        pipeline = lc.Pipeline(self.campaign, cfg.lightcurves,
                               [], output_templates.freeze(),
                               overwrite, self.plotter)

        # Run
        logger.info('Running time series pipeline:\n{}', pipeline)
        lightcurves = pipeline.run()

        return lightcurves

    # Main
    # ---------------------------------------------------------------------------- #

    def main(self, top, njobs, show_cutouts):

        run = self.campaign

        # ------------------------------------------------------------------------ #
        # Preview
        overview, data_products, samples, thumbnails = \
            self.preview(show_cutouts, cfg.samples.get('overwrite'))

        # ------------------------------------------------------------------------ #
        # Calibrate
        self.calibration(cfg.calibration.get('overwrite'))

        # ------------------------------------------------------------------------ #
        # Image Registration

        # have to ensure we have single target here
        target = check_single_target(run)

        # Do alignment and identify target
        reg = self.registration(show_cutouts, cfg.registration.get('overwrite'))

        # ------------------------------------------------------------------------ #
        # Source Tracking
        labels, _ = self.track(top, cfg.tracking.get('overwrite'), njobs)

        # ------------------------------------------------------------------------ #
        # Photometry

        # ------------------------------------------------------------------------ #
        # Time Series Analysis
        logger.section('Photometry')
        self.time_series_analysis(labels, cfg.lightcurves.get('overwrite'))

        # phot = PhotInterface(run, reg, paths.phot)
        # ts = mv phot.ragged() phot.regions()

    def finalize(self):
        logger.section('Finalize')

        # get results from this run
        run = self.campaign
        paths = self.paths
        overview, data_products = products.get_previous(run, paths)

        # Write data products spreadsheet
        products.write_xlsx(run, paths, overview)
        # This updates spreadsheet with products computed above

        logger.success('All tasks done!')

# ---------------------------------------------------------------------------- #
# Setup / Load data


def write_rsync_script(run, paths, username=cfg.remote.username,
                       server=cfg.remote.server):

    remotes = list(map(str, run.calls.get_server_path(None)))
    prefix = f'{server}:{shared_prefix(remotes)}'
    filelist = paths.files.remote.rsync_files
    io.write_lines(filelist, [remove_prefix(_, prefix) for _ in remotes])

    outfile = paths.files.remote.rsync_script
    outfile.write_text(
        f'sudo rsync -arvzh --info=progress2 '
        f'--files-from={filelist!s} --no-implied-dirs '
        f'{username}@{prefix} {paths.folders.output!s}'
    )


# ---------------------------------------------------------------------------- #
# Preview

def headers_to_txt(run, paths, overwrite):

    showfile = str
    if str(paths.files.info.headers).startswith(str(paths.folders.output)):
        def showfile(h): return h.relative_to(paths.folders.output)

    for hdu in run:
        headfile = paths.templates.HDU.info.headers.resolve_path(hdu)
        if not headfile.exists() or overwrite:
            logger.info('Writing fits header to text at {}.', showfile(headfile))
            hdu.header.totextfile(headfile, overwrite=overwrite)


# ---------------------------------------------------------------------------- #
# Calibrate


# ---------------------------------------------------------------------------- #
# Image Registration

def echo_fig(fig, *args, **kws):
    return fig


def plot_drizzle(fig, *indices, filename):
    # logger.info('POOP drizzle image: {} {}', fig, indices)
    logger.info('Plotting drizzle image: {} {}', fig, indices)

    ff = apl.FITSFigure(str(filename), figure=fig)

    ff.show_colorscale(cmap=cfg.plotting.cmap)
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


# ---------------------------------------------------------------------------- #
# Tracking

def _tracker_target_files(templates, hdu, sources):

    target_files = templates.filter('plots').map(Template.resolve_path, hdu)

    # plots
    for key, tmp in templates.plots.items():
        if 'SOURCE' in tmp.get_identifiers():
            target_files.update({
                ('plots', key, i): list(tmp.resolve_paths(hdu, source=i))
                for i in sources
            })
        else:
            target_files['plots', key] = list(tmp.resolve_paths(hdu))

    # plot_file_temps.items()
    return target_files


def _tracker_missing_files(templates, hdu, sources):

    target_files = _tracker_target_files(templates, hdu, sources)

    missing = []
    first_time_run = True
    for file in mit.collapse(target_files.flatten().values()):
        if file.exists():
            first_time_run = False
        else:
            missing.append(file)

    if first_time_run:
        logger.info('First time run for {}.', hdu.file.name)
    elif missing:
        parent = last_common_ancestor(missing)
        missing = (m.relative_to(parent) for m in missing)
        logger.bind(indent=4).opt(lazy=True).info(
            'Source Tracker for {0[0]} missing some target files in the folder '
            '{0[1]}/:\n{0[2]}.',
            lambda: (hdu.file.name, parent,
                     pp.pformat(list(missing), fmt=str, brackets=None, sep='\n'))
        )

    return missing


# ---------------------------------------------------------------------------- #
# Light curves


# Main
# ---------------------------------------------------------------------------- #
def main(paths, target, telescope, top, njobs, plot, gui, show_cutouts, overwrite):

    # create gui app if needed
    app = None
    if plot and not is_interactive():
        app = QtWidgets.QApplication(sys.argv)

    # run pipeline
    pipeline = Pipeline(paths, target, telescope, plot, gui, overwrite)
    pipeline.main(top, njobs, show_cutouts)

    # ------------------------------------------------------------------------ #
    # Launch the GUI
    if gui := pipeline.plotter.gui:
        gui.launch()

        if app:
            app.exec_()

        gui.shutdown()

    # ------------------------------------------------------------------------ #
    pipeline.finalize()

    #
    logger.info('Thanks for using pyshoc! Please report any issues at: '
                'https://github.com/astromancer/pyshoc/issues.')
