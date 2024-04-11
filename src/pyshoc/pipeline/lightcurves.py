
# std
from pathlib import Path
from collections import abc

# third-party
import numpy as np
from mpl_toolkits.axes_grid1.parasite_axes import SubplotHost

# local
import motley
from obstools import lightcurves as lc
from scrawl.ticks import DateTick, _rotate_tick_labels
from tsa import TimeSeries
from tsa.smoothing import tv
from tsa.outliers import MovingWindowDetection
from tsa.ts.plotting import TimeSeriesPlot, make_twin_relative
from recipes.oo import slots
from recipes.config import ConfigNode
from recipes import dicts, io, pprint as ppr
from recipes.decorators import update_defaults
from recipes.functionals.partial import PartialTask, PlaceHolder as o

# relative
from ..timing import Time
from ..core import shocCampaign
from .logging import logger
from .plotting import PlotTask
from .products import resolve_path


# ---------------------------------------------------------------------------- #
# Config
CONFIG = ConfigNode.load_module(__file__)

# ---------------------------------------------------------------------------- #
# Module constants

SPD = 86400

GROUPINGS = dict(
    by_file='file.name',
    by_date='t.date_for_filename',
    # by_cycle=
)
TEMPLATE_KEYS = dict(
    by_file='HDU',
    by_date='DATE',
    # by_cycle=
)

GRAPHICS_EXT = {'png', 'pdf', 'jpg', 'svg', 'eps'}


# ---------------------------------------------------------------------------- #
# Utils

def extract(run, paths, overwrite, plot):
    return Pipeline(run, paths, overwrite, plot).run()


def _get_save_meta(obj, **kws):
    # Campaign
    if isinstance(obj, shocCampaign):
        save = _get_save_meta(obj[0], **kws)
        info = save['meta']['Observing info']
        info.pop('File')
        info['Files'] = ', '.join(obj.files.names)

    # HDU
    kws, _ = dicts.split(kws, 'filename', 'folder',  'tab', 'overwrite')
    return dict(**kws,
                obj_name=obj.target,
                meta={'Observing info':
                      {'T0 [UTC]': obj.t[0].utc,
                       'File':     obj.file.name}}
                )


def _get_plot_config(section, cli_flag):

    if cli_flag is False:
        return False

    kws, specific = CONFIG.plots.split(GROUPINGS)
    specific = specific[section]

    if specific is False:
        return False

    if not isinstance(specific, abc.MutableMapping):
        specific = {}

    return {**kws, **specific}


def plotter(fig, ts, **kws):
    #
    # logger.debug('{}.', pformat(locals()))
    ax = SubplotHost(fig, 1, 1, 1)
    fig.add_subplot(ax)

    #
    jd0 = int(ts.t[0]) - 0.5
    utc0 = Time(jd0, format='jd').utc.iso.split()[0]
    #

    # plot
    axp = make_twin_relative(ax, -(ts.t[0] - jd0) * SPD, 1, 45)
    tsp = ts.plot(ax, t0=[0], tscale=SPD,
                  **{**dict(plims=(-0.1, 99.99), show_masked=True), **kws})
    axp.xaxis.set_minor_formatter(DateTick(utc0))
    _rotate_tick_labels(axp, 45, True)

    cfg = CONFIG.plots
    ax.set(xlabel=cfg.xlabel.bottom, ylabel=cfg.ylabel)
    axp.set_xlabel(cfg.xlabel.top, labelpad=cfg.xlabel.pad)

    # fig.tight_layout()
    fig.subplots_adjust(**cfg.subplotspec)

    # if overwrite or not filename.exists():
    #     save_fig(fig, filename)

    return fig


# alias
plot = plotter


# ---------------------------------------------------------------------------- #
# Raw

def load_memmap(hdu, filename, outfile=None, **kws):

    logger.info(motley.stylize('Loading data for {:|darkgreen}.'), hdu.file.name)

    # CONFIG.pre_subtract
    # since the (gain) calibrated frames are being used below,
    # CCDNoiseModel(hdu.readout.noise)

    # folder = DATAPATH / 'shoc/phot' / hdu.file.stem / 'tracking'
    #
    data = io.load_memmap(filename)
    flux = data['flux']

    return hdu.t.bjd, flux['value'].T, flux['sigma'].T


# ---------------------------------------------------------------------------- #
# Outlier flagging

@update_defaults(CONFIG.by_file.flagged.params)
def flag_outliers(ts, nwindow, noverlap, method, **kws):
    # TODO: move to tsa.ts.outliers

    # flag outliers
    logger.info('Detecting outliers.')

    bjd, flux, _ = ts

    oflag = np.isnan(flux)
    for i, flx in enumerate(flux):
        logger.debug('Source {}.', i)
        mwd = MovingWindowDetection(nwindow, noverlap, method, **kws)
        oflag[i] = mask = mwd(flx)

        logger.info(
            'Flagged {}/{} ({:5.3%}) points as outliers using {} method with {}.',
            (no := mask.sum()), (n := len(bjd)), (no / n), method, kws
        )

    return np.ma.MaskedArray(flux, oflag)


# ---------------------------------------------------------------------------- #
# Differential

def diff0_phot(ts, c=1, meta=None):

    t, flux, sigma = ts  # load_flagged(hdu, paths, overwrite)

    # Zero order differential
    fm = np.ma.median(flux[:, c])
    if meta:
        meta['Flux scale'] = fm

    return (t,
            np.ma.MaskedArray(flux.data / fm, flux.mask),
            sigma / fm + sigma.mean(0) / sigma.shape[1])


def diff_phot(ts, **kws):
    logger.info('Decorrelating light curve for photometry.')
    tss = _diff_smooth_tvr(ts, **kws)
    return tss.t, tss.x.T, tss.u.T


@update_defaults(CONFIG.by_file.diff.params)
def _diff_smooth_tvr(ts, nwindow, noverlap, smoothing):
    # smoothed differential phot
    s = tv.MovingWindowSmoother(ts.t, ts.x, nwindow, noverlap, smoothing)
    s = np.atleast_2d(s).T - 1
    return ts - s


# ---------------------------------------------------------------------------- #
# Decorrelate

def decorrelate(ts, **kws):
    from IPython import embed
    embed(header="Embedded interpreter at 'src/pyshoc/pipeline/lightcurves.py':213")


# ---------------------------------------------------------------------------- #
# Concatenate

def concat_phot(files):
    # stack time series for target run

    # _, step = section
    # compute = COMPUTE[step]

    # data = zip(*(produce(section, compute, paths[section], None, hdu, overwrite)
    #              for hdu in campaign))
    data = (lc.io.txt.read(filename) for filename in files)

    # data
    bjd, rflux, rsigma = map(np.ma.hstack, data)
    return bjd.data, rflux, rsigma.data


# ---------------------------------------------------------------------------- #

def todo(*args, **kws):
    raise NotImplementedError()


COMPUTE = {
    # Time Series
    'raw':          load_memmap,
    'flagged':      flag_outliers,
    'diff0':        diff0_phot,
    'diff':         diff_phot,
    'decor':        decorrelate,

    # Spectral density estimates
    'periodogram':  todo,
    'lombscargle':  todo,
    'welch':        todo,
    'tfr':          todo,
}


# ---------------------------------------------------------------------------- #

class LightCurvePlot(TimeSeriesPlot):

    def setup_figure(self, ax, figsize=(14, 8), twinx='period', **kws):

        fig, ax, hax = super().setup_figure(ax, **kws)

        ts = self.parent
        jd0 = int(ts.t[0]) - 0.5
        utc0 = Time(jd0, format='jd').utc.iso.split()[0]

        # plot
        axp = make_twin_relative(ax, -(ts.t[0] - jd0) * SPD, 1, 45)
        axp.xaxis.set_minor_formatter(DateTick(utc0))
        # _rotate_tick_labels(axp, 45, True)

        cfg = CONFIG.plots
        ax.set(xlabel=cfg.xlabel.bottom, ylabel=cfg.ylabel)
        axp.set_xlabel(cfg.xlabel.top, labelpad=cfg.xlabel.pad)

        # fig.tight_layout()
        fig.subplots_adjust(**cfg.subplotspec)

        return fig, ax, hax

    def __call__(self, *data, **kws):
        kws = {**dict(t0=[0], tscale=SPD, show_masked=True), **kws}
        super().__call__(*data, **kws)


class LightCurve(TimeSeries):

    plot = LightCurvePlot(plims=(-0.1, 99.99))

    @classmethod
    def load(cls, filename, hdu=None):

        filename = Path(filename)
        ext = filename.suffix.strip('.')
        if ext == 'txt':
            return cls(*lc.io.txt.read(filename))

        if ext == 'npy':
            cls(*load_memmap(hdu, filename))

        raise ValueError(f'Unsupported format: {ext!r}')

    def save(self, filename, **kws):
        filename = Path(filename)

        if (suffix := filename.suffix.strip('.')) == 'txt':
            lc.io.write(filename, self.t, self.x, self.u, **kws)
        elif suffix in {'npy', 'dat'}:
            raise NotImplementedError()
        else:
            raise NotImplementedError()

# ---------------------------------------------------------------------------- #

# class ReductionTask(PartialTask):
#     pass


class ReductionStep(PartialTask):

    # __wrapper__ = ReductionTask

    def __init__(self, method, infile, outfiles=(), overwrite=False, save=(),
                 plot=False, *args, **kws):

        super().__init__(method, *args, **kws)
        self.infile = infile
        self.overwrite = bool(overwrite)
        self.plot = plot
        self.save_kws = dict(save)
        self.outfiles, self.plot_files = files = [], []

        for file in outfiles:
            is_img = (file.suffix.lower().strip('.') in GRAPHICS_EXT)
            files[is_img].append(file)

        # logger.info(f'{method = }; {infile = }; {outfile = }')

    def __call__(self, obs, data=None, **kws):

        # Load result from previous run if available  # FIXME: Manage with cache
        if len(obs) > 1:
            from IPython import embed
            embed(header="Embedded interpreter at 'src/pyshoc/pipeline/lightcurves.py':343")
            
        obs = obs[0]
        for path in self.outfiles:
            if (path := resolve_path(path, obs)).exists() and not self.overwrite:
                logger.info('Loading lightcurve for {} from {}.', path)
                return LightCurve.load(path, obs)

        # Load input data
        if data is None:
            infile = resolve_path(self.infile, obs)
            if not infile.exists():
                raise FileNotFoundError(repr(str(infile)))

        # Compute
        path = self.outfiles[0]
        logger.debug('File {!s} {}. Computing: {}.',
                     motley.apply(str(path), 'darkgreen'),
                     (f"will be {('created', 'overwritten')[self.overwrite]}"
                      if path.exists() else 'does not exist'),
                     ppr.caller(self.__wrapped__))  # args, kws
        #
        result = super().__call__(**kws)

        if not isinstance(result, LightCurve):
            result = LightCurve(*result)

        # save text
        if self.save is not False:
            for path in self.outfiles:
                result.save(path, **self.save_kws)

        # plot
        if self.plot is not False:
            plot = self.plot or {}
            kws, init = dicts.split(plot, ('ui', 'keys', 'filename', 'overwrite'))
            init.setdefault('filenames', self.plot_files)
            init.setdefault('overwrite', self.overwrite)
            # load task
            task = PlotTask(**init)
            task(plotter)(o, result, **kws)

        return result


class Pipeline(slots.SlotHelper):

    __slots__ = ('campaign', 'config', 'groupings', 'steps', 'concats', 'results')

    def __init__(self, campaign, config, output_templates=(), overwrite=False,
                 plot=False):

        infile = '$HDU'
        config, groupings = config.split(GROUPINGS)
        config.pop('by_cycle')  # FIXME
        config = {**config, 'overwrite': overwrite, 'plot': plot}

        grouped = {}
        concats = {}
        steps = dicts.DictNode()
        groupings = groupings.filter(('folder', 'filename'))
        for grouping, todo in groupings.items():
            grouped[grouping] = campaign.group_by(GROUPINGS[grouping]).sorted()

            for step in todo.filter('formats'):
                section = (grouping, step)

                # for each step there may be a concat request
                cfg, concat = steps[step].split('concat')

                # load / compute step
                key = (TEMPLATE_KEYS[grouping], *section)
                template = output_templates.get((*key, 'filename'), '')
                template = template or output_templates.get(key, '')
                outfiles = list(template.resolve_paths(('lightcurves', grouping)))

                steps[grouping][step] = \
                    ReductionStep(COMPUTE[step], infile, outfiles,
                                  overwrite, plot=plot, **cfg)
                infile = outfiles[0]

                if concat:
                    # Concatenate
                    concats[section] = \
                        ReductionStep(concat_phot, infile, concat.filename,
                                      **concat)

        # Create attributes
        super().__init__(campaign=campaign,
                         config=config,
                         groupings=grouped,
                         steps=steps,
                         concats=concats,
                         results=dicts.DictNode())

    def run(self):

        logger.info('Extracting lightcurves for {!r}.', self.campaign[0].target)

        # files = self.campaign.files.lightcurves
        for grouping, groups in self.groupings.items():
            steps = self.steps[grouping]
            for step, worker in steps.items():
                # steps
                try:
                    for gid, obs in groups.items():
                        self.results[grouping, gid, step] = worker(obs)
                except Exception as err:
                    import sys, textwrap
                    from IPython import embed
                    from better_exceptions import format_exception
                    embed(header=textwrap.dedent(
                            f"""\
                            Caught the following {type(err).__name__} at 'lightcurves.py':444:
                            %s
                            Exception will be re-raised upon exiting this embedded interpreter.
                            """) % '\n'.join(format_exception(*sys.exc_info()))
                    )
                    raise
                    

                section = (grouping, step)
                if concat := self.concats.get(section):
                    # Concatenate
                    from IPython import embed
                    embed(header="Embedded interpreter at 'src/pyshoc/pipeline/lightcurves.py':225")

                    files = list(groups.values())
                    logger.info('Concatenating {} light curves for {} on {}.',
                                len(files), section)

                    # create TimeSeries
                    self.results[gid, step, 'concat'] = ts = concat(files)
                    #     TimeSeries(bjd, rflux.T, rsigma.T)

        #
        self.results.freeze()
        return self.results

    def plot(self, section, ui=None, filename_template=None, overwrite=False, **kws):

        grouping, step = section
        task = ui.task_factory(lc.plot)(o, **kws)

        filenames = {}
        for date, ts in self.results.items():
            filenames[date] = filename = \
                Path(filename_template.substitute(DATE=date)).with_suffix('.png')
            # add task
            year, day = date.split('-', 1)
            tab = (*self.config.tab, year, day, step)
            ui.add_task(task, tab, filename, overwrite, None, False, ts)

            if ui:
                ui[tab[:-2]].link_focus()

        return ui


# def load_raw(hdu, infile, outfile, overwrite=False, plot=False):

#     logger.info(f'{infile = }; {outfile = }')

#     # _load(('raw', hdu, ))
#     name = 'raw'
#     data = load_or_compute(
#         # load
#         resolve_path(outfile, hdu),
#         CONFIG.by_file[name].get('overwrite', overwrite),
#         # compute
#         delayed(load_memmap)(hdu, resolve_path(infile, hdu)),
#         # save
#         _get_save_meta(hdu, title=CONFIG[name].title),
#         # plot
#         _get_plot_config('by_file', name)
#     )


# def load_flagged(hdu, paths, overwrite=False, plot=False):

#     files = paths.files

#     infile = files.lightcurves[section]
#     outfile = files.lightcurves['flagged']
#     produce(section, infile, outfile, hdu, overwrite, plot)

#     name = 'flagged'
#     return load_or_compute(
#         # load
#         resolve_path(files.lightcurves[name], hdu),
#         CONFIG.by_file[name].get('overwrite', overwrite),
#         # compute
#         delayed(_flag_outliers)(hdu,
#                                 resolve_path(files.tracking.source_info, hdu),
#                                 resolve_path(files.lightcurves.raw, hdu),
#                                 overwrite),
#         # save
#         _get_save_meta(hdu, title=CONFIG[name].title),
#         # plot
#         _get_plot_config(('by_file', name))
#     )


# def _flag_outliers(section, hdu, infile, outfile, overwrite):
#     # load memmap
#     t, flux, sigma = produce(('by_file', 'raw'), infile, outfile, overwrite)
#     # bjd = t.bjd
#     # flux = flag_outliers(t, flux)
#     return t, flag_outliers(t, flux), sigma


# def diff0_phot(hdu, paths, overwrite=False, plot=False):
#     # filename = folder / f'raw/{hdu.file.stem}-phot-raw.txt'
#     # bjd, flux, sigma = load_phot(hdu, filename, overwrite)
#     name = 'diff0'
#     save = _get_save_meta(hdu, title=CONFIG[name].title)
#     # processing metadata added by `_diff0_phot`
#     meta = save['meta']['Processing'] = {}

#     return load_or_compute(
#         # load
#         resolve_path(paths.files.lightcurves.by_file[name].filename, hdu),
#         overwrite,
#         # compute
#         delayed(_diff0_phot)(hdu, paths, meta=meta, overwrite=overwrite),
#         # save
#         save,
#         # plot
#         _get_plot_config(('by_file', name))

#     )


# ---------------------------------------------------------------------------- #
# Decorrelate

# def decor(ts, campaign, paths, overwrite, **kws):

#     kws = {**CONFIG.by_file.decor, **kws}
#     save = _get_save_meta(campaign[0], title=kws.pop('title'))
#     info = save['meta']['Observing info']
#     info.pop('File')
#     info['Files'] = ', '.join(campaign.files.names)

#     meta = save['meta']['Processing'] = kws

#     bjd, rflux, rsigma = load_or_compute(
#         # load
#         resolve_path(paths.files.lightcurves.decor, campaign[0]), overwrite,
#         # compute
#         delayed(_decor)(ts, **kws),
#         # save
#         save,
#         # plot
#         _get_plot_config(('by_date', 'decor'))
#     )

#     return TimeSeries(bjd, rflux.T, rsigma.T)

# # save text
# filename = paths
# lc.io.write_text(
#     filename,
#     tss.t, tss.x.T, tss.u.T,
#     title='Differential (smoothed) ragged-aperture light curve for {}.',
#     obj_name=campaign[0].target,
#     meta={'Observing info':
#           {'T0 [UTC]': Time(tss.t[0], format='jd').utc,
#            # 'Files':    ', '.join(campaign.files.names)
#            }}
# )


# def concat_phot(campaign, paths, section, overwrite=None, plot=False, **meta):

#     # Adjust meta info
#     save = _get_save_meta(campaign[0], **meta)
#     info = save['meta']['Observing info']
#     info.pop('File')
#     info['Files'] = ', '.join(campaign.files.names)

#     # ('by_file', 'decor', 'concat')
#     _, step, _ = section
#     out_group = 'by_date'
#     return load_or_compute(
#         # load
#         resolve_path(paths.files.lightcurves[section], campaign[0]),
#         overwrite,
#         # compute
#         delayed(_concat_phot)(campaign, paths, section, overwrite, plot),
#         # save
#         save,
#         # plot
#         _get_plot_config((out_group, step))
#     )


# def lag_scatter(x, ):
