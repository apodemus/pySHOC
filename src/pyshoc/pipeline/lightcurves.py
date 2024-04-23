
# std
from pathlib import Path

# third-party
import numpy as np
from mpl_toolkits.axes_grid1.parasite_axes import SubplotHost

# local
import motley
from obstools import lightcurves as lcs
from scrawl.ticks import DateTick, _rotate_tick_labels
from tsa.smoothing import tv
from tsa.outliers import MovingWindowDetection
from tsa.ts.plotting import make_twin_relative
from recipes.oo import slots
from recipes.config import ConfigNode
from recipes import dicts, pprint as ppr
from recipes.logging import LoggingMixin
from recipes.decorators import update_defaults
from recipes.functionals.partial import PartialTask, PlaceHolder as o

# relative
from ..timing import Time
from ..config import GROUPING
from ..core import shocCampaign
from .logging import logger
from .plotting import PlotTask
from obstools.lightcurves import io


# ---------------------------------------------------------------------------- #
# Config
CONFIG = ConfigNode.load_module(__file__)

# ---------------------------------------------------------------------------- #
# Module constants

SPD = 86400

# alias
LightCurve = lcs.LightCurve


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
                target=obj.target,
                meta={'Observing info':
                      {'T0 [UTC]': obj.t[0].utc,
                       'File':     obj.file.name}}
                )


# def _get_plot_config(section, cli_flag):

#     if cli_flag is False:
#         return False

#     kws, specific = CONFIG.plots.split(GROUPING)
#     specific = specific[section]

#     if specific is False:
#         return False

#     if not isinstance(specific, abc.MutableMapping):
#         specific = {}

#     return {**kws, **specific}


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
# Load
def load_memmap(hdu, filename, **kws):
    return lcs.io.load_memmap(filename, hdu, **kws)


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
# Decompose Atmospheric / Stellar variability

def median_scale(ts, ref=1, meta=None):

    t, flux, sigma = ts

    # Scale by median flux of reference source
    fm = np.ma.median(flux[:, ref])
    if meta:
        meta['Flux scale'] = fm

    return LightCurve(t,
                      np.ma.MaskedArray(flux.data / fm, flux.mask),
                      sigma / fm + sigma.mean(0) / sigma.shape[1])


def diff_phot(ts, meta=None, **kws):
    logger.info('Decorrelating light curve for photometry.')

    tss = median_scale(ts, meta=meta)
    sm = _diff_smooth_tvr(tss, **kws)

    if meta:
        meta['Differential Photometry'] = kws

    return sm


@update_defaults(CONFIG.by_file.diff.params)
def _diff_smooth_tvr(ts, nwindow, noverlap, smoothing):
    try:
        # smoothed differential phot
        sm = tv.MovingWindowSmoother(nwindow, noverlap)
        s = sm(ts.t, ts.x, smoothing)
        return ts - np.atleast_2d(s) + 1
    except Exception as err:
        import sys
        import textwrap
        from IPython import embed
        from better_exceptions import format_exception
        embed(header=textwrap.dedent(
            f"""\
                Caught the following {type(err).__name__} at 'lightcurves.py':194:
                %s
                Exception will be re-raised upon exiting this embedded interpreter.
                """) % '\n'.join(format_exception(*sys.exc_info()))
        )
        raise


# ---------------------------------------------------------------------------- #
# Decorrelate

def decorrelate(ts, **kws):
    raise NotImplementedError()


# ---------------------------------------------------------------------------- #
# Concatenate

def concatenate(files):
    # stack time series for target run

    # _, step = section
    # compute = COMPUTE[step]

    # data = zip(*(produce(section, compute, paths[section], None, hdu, overwrite)
    #              for hdu in campaign))
    data = (lcs.io.txt.read(filename) for filename in files)

    # data
    bjd, rflux, rsigma = map(np.ma.hstack, data)
    return bjd.data, rflux, rsigma.data


# ---------------------------------------------------------------------------- #

def todo(*args, **kws):
    raise NotImplementedError()


# Time Series Analysis
TSA = {
    'raw':          load_memmap,
    'flagged':      flag_outliers,
    # 'diff0':        median_scale,
    'diff':         diff_phot,
    'decor':        decorrelate,
}

# Spectral density estimates
SDE = {
    'acf':          todo,
    'periodogram':  todo,
    'lombscargle':  todo,
    'welch':        todo,
    'tfr':          todo,
}

KNOWN_STEPS = {*TSA, *SDE}

# ---------------------------------------------------------------------------- #

# class ReductionTask(PartialTask):
#     pass


class ReductionStep(PartialTask):

    # __wrapper__ = ReductionTask

    def __init__(self, method, infile, outfiles=(), /, overwrite=False, save=(),
                 plot=False, id_=(), *args, **kws):

        # init task
        super().__init__(method, o, *args, **kws)

        # NOTE:infile, outfiles, plot_files filename templates
        templates = ConfigNode({'input': infile,
                                'output': [],
                                'plots': []})
        for file in outfiles:
            cat = ('plots', 'output')[io.SupportedFileType.check(str(file))]
            templates[cat].append(file)

        self.id_ = id_
        self.plot = plot
        self.infile = infile
        self.save_kws = dict(save)
        self.templates = templates
        self.overwrite = bool(overwrite)

    def __call__(self, obs, data=None, **kws):

        # Load result from previous run if available  # FIXME: Manage with cache
        if len(obs) > 1:
            from IPython import embed
            embed(header="Embedded interpreter at 'src/pyshoc/pipeline/lightcurves.py':343")

        hdu = obs[0]
        out = [o.resolve_path(hdu) for o in self.templates.output]
        for path in out:
            if path.exists() and not self.overwrite:
                logger.info('Loading lightcurve for {} from {}.', hdu, path)
                return LightCurve.load(path, hdu)

        # Load input data
        if data is None:
            if (infile := self.templates.input.resolve_path(hdu)).exists():
                data = LightCurve.load(infile, hdu)
            else:
                raise FileNotFoundError(repr(str(infile)))

        # Compute
        path = out[0]
        logger.debug('File {!s} {}. Computing: {}.',
                     motley.apply(str(path), 'darkgreen'),
                     (f"will be {('created', 'overwritten')[self.overwrite]}"
                         if path.exists() else 'does not exist'),
                     ppr.caller(self.__wrapped__))  # args, kws
        #
        result = super().__call__(data, **kws)

        if not isinstance(result, LightCurve):
            result = LightCurve(*result)

        # save text
        if self.save_kws is not False:
            for path in out:
                result.save(path, **_get_save_meta(hdu, **self.save_kws))

        # plot
        if self.plot is not False:
            plot = self.plot or {}
            kws, init = dicts.split(plot, ('ui', 'keys', 'filename', 'overwrite'))
            init.setdefault('overwrite', self.overwrite)
            init.setdefault('filenames',  [tmp.resolve_path(hdu)
                                           for tmp in self.templates.plot])

            # load task
            task = PlotTask(**init)
            task(plotter)(o, result, **kws)

        return result


class Pipeline(slots.SlotHelper, LoggingMixin):

    __slots__ = ('campaign', 'config', 'groupings', 'steps', 'concats', 'results')

    def __init__(self, campaign, config, output_templates=(), overwrite=False,
                 plot=True):

        infile = '$HDU'
        config, groupings = config.split(GROUPING)
        config = {**config, 'overwrite': overwrite, 'plot': plot}
        groupings = groupings.filter(('folder', 'filename'))

        grouped = {}
        concats = {}
        steps = dicts.DictNode()

        for grouping, todo in groupings.items():
            template_key, attr = GROUPING[grouping]
            grouped[grouping] = campaign.group_by(attr).sorted()

            for step, cfg in todo.filter('formats').items():
                if step not in KNOWN_STEPS:
                    raise KeyError(
                        f'Unknown reduction step: {step} in section: {grouping}.'
                        f'The following steps are recognised:\n'
                        f'Time Series Analysis: {tuple(TSA)}'
                        f'Spectral Density Esitmators: {tuple(SDE)}'
                    )

                section = (grouping, step)
                tmp_sec = ('lightcurves', grouping)

                # for each step there may be a concat / sde request
                cfg, concat = cfg.split('concat')
                cfg, sde = cfg.split(SDE)
                params = cfg.pop('params', {})
                _plot = cfg.pop('plot', plot)
                plot = _plot if plot else False

                if 'plot' in cfg:
                    from IPython import embed
                    embed(header="Embedded interpreter at 'src/pyshoc/pipeline/lightcurves.py':403")

                # load / compute step
                key = (template_key, *section)
                template = output_templates.get((*key, 'filename'), '')
                template = template or output_templates.get(key, '')
                outfiles = list(template.resolve_paths(section=tmp_sec, partial=True))

                # Add task
                if worker := TSA.get(step, ()) or SDE.get(step, ()):
                    #
                    self.logger.opt(lazy=True).bind(indent=True).info(
                        'Adding reduction step {0[0].__name__!r} for section '
                        '{0[1]}: {0[2]}.',
                        lambda: (worker, section,
                                 f'\n{ppr.pformat(params)}' if params else '')
                    )

                    steps[section] = \
                        ReductionStep(worker, infile, outfiles, overwrite,
                                      save=cfg, plot=plot, **params)

                # Concatenate
                if concat:
                    _, template = (output_templates.find(step).find('concat')
                                   .flatten().popitem())
                    outfiles = list(template.resolve_paths(section=tmp_sec, partial=True))
                    concats[section] = \
                        ReductionStep(concatenate, infile, outfiles, **concat)

                # ??
                infile = outfiles[0]

                # Spectral Density Estimation
                if sde:
                    #
                    # _, template = output_templates.find(step).find('concat').flatten().popitem()

                    from IPython import embed
                    embed(header="Embedded interpreter at 'src/pyshoc/pipeline/lightcurves.py':418")

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
                    logger.exception('Lightcurve pipeline failed at step: {!r}; '
                                     'group: {}', step, gid)
                    raise err

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
        task = ui.task_factory(lcs.plot)(o, **kws)

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


# def lag_scatter(x, ):
