
# std
from IPython import embed
from scrawl.corner import corner
from pathlib import Path

# third-party
import numpy as np

# local
import tsa
from obstools import lightcurves as lcs
from obstools.lightcurves import SupportedFormats, read
from recipes.oo import slots
from recipes.op import MethodCaller
from recipes.functionals import echo
from recipes.config import ConfigNode
from recipes import dicts, pprint as ppr
from recipes.logging import LoggingMixin
from recipes.decorators import update_defaults
from recipes.functionals.partial import Partial, PartialTask, PlaceHolder as o

# relative
from .. import config as cfg
from ..core import Campaign
from ..config import Template
from .logging import logger


# ---------------------------------------------------------------------------- #
# Config
CONFIG = cfg.lightcurves

# alias
LightCurve = lcs.LightCurve


def get_metadata(obj, section='Observing Info', **kws):
    # Campaign
    if isinstance(obj, Campaign):
        save = get_metadata(obj[0], **kws)
        info = save[section]
        info.pop('File')
        info['Files'] = ', '.join(obj.files.names)

    # HDU
    kws, _ = dicts.split(kws, 'filename', 'folder',  'tab', 'overwrite')
    return {**kws,
            'target': obj.target,
            section: {'File':     obj.file.name,
                      'T0 [UTC]': obj.t[0].utc}
            }


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

# ---------------------------------------------------------------------------- #
def from_tracker(tracker, hdu):
    return tracker.to_lightcurve(hdu.t.bjd, **get_metadata(hdu))

# ---------------------------------------------------------------------------- #
# Outlier flagging

# @update_defaults(CONFIG.by_file.flagged.params)
def flag_outliers(ts, nwindow, noverlap, method='gESD', **kws):
    # TODO: move to tsa.ts.outliers

    # flag outliers
    logger.info('Detecting outliers.')

    bjd, flux, std = ts
    flux = flux.T
    oflag = np.isnan(flux)
    for i, flx in enumerate(flux):
        logger.debug('Source {}.', i)

        mwd = tsa.outliers.MovingWindowDetection(nwindow, noverlap, method=method)
        oflag[i] = mask = mwd(bjd, flx, **kws).squeeze()

        logger.info(
            'Flagged {}/{} ({:5.3%}) points as outliers using {} method with {}.',
            (no := mask.sum()), (n := len(bjd)), (no / n), method, kws
        )

    return bjd, np.ma.MaskedArray(flux, oflag).T, std


# ---------------------------------------------------------------------------- #
# Cross calibration
def rescale(ts, ref=1, metadata=None):

    t, flux, sigma = ts

    # Scale by median flux of reference source
    fm = np.ma.median(flux[:, ref])
    if metadata:
        metadata['Flux scale'] = fm

    if sigma is not None:
        sigma = sigma / fm + sigma.mean(0) / sigma.shape[1]

    from IPython import embed
    embed(header="Embedded interpreter at 'src/pyshoc/pipeline/lightcurves.py':115")
    
    return LightCurve(t,
                      np.ma.MaskedArray(flux.data / fm, flux.mask),
                      sigma)

# ---------------------------------------------------------------------------- #
# Decompose Atmospheric / Stellar variability


# def diff_phot(ts, ref=1, meta=None, **kws):
#     logger.info('Applying differential correction.')

#     tsn = ts.normalize(scale_index=False)

#     indices = sorted(set(range(tsn.m)) - {ref})
#     m = LightCurve(tsn.t, np.median(tsn[:, indices].values, 1))

#     # m.plot(ax=plt.gca(), marker='o', color='k', zorder=100)

#     # tss = median_scale(ts, meta=meta)
#     # sm = _diff_smooth_tvr(tss, **kws)

#     if meta:
#         meta['Differential Photometry'] = kws

#     return sm


# @update_defaults(CONFIG.by_file.diff.params)
def _diff_smooth_tvr(ts, nwindow, noverlap, smoothing):
    # smoothed differential phot
    sm = tsa.smooth.tv.MovingWindowSmoother(nwindow, noverlap)
    s = sm(ts.t, ts.x, smoothing)
    return ts - np.atleast_2d(s) + 1


# ---------------------------------------------------------------------------- #
# Decorrelate

def decorrelate(ts, **kws):
    raise NotImplementedError()


# ---------------------------------------------------------------------------- #
# Concatenate

def concatenate(files, **kws):
    # stack time series for target run

    # _, step = section
    # compute = COMPUTE[step]

    # data = zip(*(produce(section, compute, paths[section], None, hdu, overwrite)
    #              for hdu in campaign))

    try:
        data, meta = zip(*(read(filename) for filename in files))
    except Exception as err:
        import sys
        import textwrap
        from IPython import embed
        from better_exceptions import format_exception
        embed(header=textwrap.dedent(
            f"""\
                Caught the following {type(err).__name__} at 'lightcurves.py':201:
                %s
                Exception will be re-raised upon exiting this embedded interpreter.
                """) % '\n'.join(format_exception(*sys.exc_info()))
        )
        raise

    # data
    bjd, rflux, rsigma = map(np.ma.hstack, data)
    return bjd.data, rflux, rsigma.data


# ---------------------------------------------------------------------------- #
# class SpectralDensityEstimate(MethodCaller):
#     def __call__(self, ts, *args, **kws):


def todo(*args, **kws):
    raise NotImplementedError()


# Time Series Analysis
TSA = {
    'raw':          echo,
    'oflag':        flag_outliers,
    'diff':         rescale,
    # 'decor':        decorrelate,
    'concat':       concatenate
}

# Spectral density estimates
SDE = {
    'acf':          MethodCaller('acf'),
    'periodogram':  MethodCaller('periodogram'),
    'lombscargle':  MethodCaller('lombscargle'),
    'welch':        MethodCaller('welch'),
    'tfr':          MethodCaller('tfr')
}

DIAGNOSTICS = {
    'corner':  MethodCaller('corner')
}


KNOWN_STEPS = {**TSA, **SDE, **DIAGNOSTICS}


# ---------------------------------------------------------------------------- #

class Pipeline(slots.SlotHelper, LoggingMixin):

    __slots__ = ('campaign', 'config', 'groupings', 'steps',  'results')

    def __init__(self, campaign, config, infiles=(), output_templates=(),
                 overwrite=False, plotter=True):

        #
        config, groupings = config.split(cfg.GROUPING)

        self.groupings = {}
        self.steps = dicts.DictNode()
        self.output_templates = output_templates
        self.overwrite = bool(overwrite)

        for grouping, todo in groupings.filter(('folder', 'filename')).items():
            # resolve filenames from templated paths
            _, attr = cfg.GROUPING[grouping]
            self.groupings[grouping] = campaign.group_by(attr).sorted()

            for step, config in todo.filter('formats').items():
                section = (grouping, step)

                # init ReductionStep
                tasks = self.add_step(section, infiles, config)

                # file succession
                infiles = tasks[-1].templates.output

        # Create attributes
        super().__init__(campaign=campaign,
                         config=config,
                         results=dicts.DictNode(),
                         plotter=plotter,
                         overwrite=overwrite)

    def __repr__(self):
        return f'{ppr.pformat(self.steps, name=f"<{type(self).__name__}")}>'

    def resolve_output_paths(self, key):
        # resolve filenames from templated paths
        key = grouping, *_ = tuple(key)
        key = (cfg.GROUPING[grouping][0], *key, 'filename')
        tmp = (self.output_templates.get(key, None) or
               self.output_templates.get(key[:-1], None))
        if tmp:
            return tmp.resolve_paths(section=('lightcurves', grouping),
                                     partial=True)
        return []

    # def get_output_template(self, key):
    #     # resolve filenames from templated paths
    #     key = grouping, *_ = tuple(key)
    #     key = (cfg.GROUPING[grouping][0], *key, 'filename')
    #     return (self.output_templates.get(key, None) or
    #             self.output_templates.get(key[:-1], None))

    def add_step(self, section, infiles, config):

        # for each step there may be a concat / sde request
        main_step, sub_steps = config.split(KNOWN_STEPS)
        metadata, kws = main_step.split(('overwrite', 'plot', 'params'))
        kws.update(kws.pop('params', {}))
        main_step = {'overwrite': self.overwrite,
                     'plot': False,
                     'metadata': metadata,
                     **kws}

        # Get worker
        *grouping, step = section
        worker = KNOWN_STEPS.get(step, ())
        if not worker:
            raise KeyError(
                f'Unknown reduction step: {step!r} for grouping: {grouping!r}.'
                f'The following steps are recognised:\n'
                f'Time Series Analysis: {tuple(TSA)}\n'
                f'Spectral Density Esitmators: {tuple(SDE)}\n'
                f'Diagnostics:  {tuple(DIAGNOSTICS)}'
            )

        # log
        self.logger.opt(lazy=True).bind(indent=True).info(
            'Adding reduction step {0[0]!r} for section {0[1]}: {0[2]}.',
            lambda: (ppr.callers.get_name(worker, 1), section,
                     f'\n{ppr.pformat(main_step)}' if main_step else '')
        )
        # partially resolve output file patterns
        outfiles = self.resolve_output_paths(section)
        # Initialize & add step
        kls = tsa.spectral.Periodogram if step in SDE else LightCurve
        task = ReductionStep(worker, infiles, outfiles, kls, section, **main_step)
        task.pipeline = self  # add reference to pipeline

        # ensure unique names
        if sub_steps:
            section = (*section, 'main')
        self.steps[section] = task

        tasks = [task]
        for sub, conf in sub_steps.items():
            # file succession
            infiles = task.templates.output
            tasks.extend(self.add_step((*section[:-1], sub), infiles, conf))

        return tasks

        # # Concatenate
        # if not concat:
        #     continue

        # # run concat at the next grouping
        # section = (*section[:-1], 'concat')
        # next_group = (g := list(cfg.GROUPING))[g.index(grouping) + 1]
        # tmp_key, _ = cfg.GROUPING[next_group]
        # outfiles = self.resolve_output_paths((tmp_key, *section))

        # # add concat step
        # task = self.add_step(section, infiles, outfiles,
        #                      save=save, plot=plot, **concat)

    def run(self, datamap):
        logger.info('Extracting lightcurves for {!r}.', self.campaign[0].target)

        previous = None
        for (grouping, *step), worker in self.steps.flatten().items():
            try:
                for gid, obs in self.groupings[grouping].items():
                    section = (grouping, gid, *step)
                    # Compute
                    if previous:
                        (p := list(previous)).insert(1, gid)
                        data = self.results[tuple(p)]
                    else:
                        # first step
                        data = datamap.get(section)

                    self.results[section] = worker(obs, data)

            except Exception as err:
                logger.exception('Lightcurve pipeline failed at step: {!r}; '
                                 'group: {}', step, gid)
                raise err

            previous = (grouping, *step)

        #
        self.results.freeze()
        return self.results

    def plot(self, section, filename_template=None, overwrite=False, **kws):

        _, step = section
        manager = self.plotter
        task = manager.task_factory(lcs.plot)(o, **kws)

        filenames = {}

        for date, ts in self.results.items():
            filenames[date] = filename = \
                Path(filename_template.substitute(DATE=date)).with_suffix('.png')

            # add task
            year, day = date.split('-', 1)
            tab = (*self.config.tab, year, day, step)
            manager.add_task(task, tab, filename, overwrite, None, False, ts)

            if ui := manager.gui:
                ui[tab[:-2]].link_focus()

        return ui


def get_tab_key(hdu):
    year, day = str(hdu.date_for_filename).split('-', 1)
    return (year, day, hdu.file.nr)


def sort_load_order(path, preference):
    # out = sorted(paths,
    suf = path.suffix.lstrip('.')
    return preference.index(suf) if suf in preference else 100


class ReductionStep(PartialTask):

    # __wrapper__ = ReductionTask
    _load_order_prefered = ('npz', 'txt')

    def __init__(self, func, infile=None, outfiles=(), output_class=LightCurve, /,
                 section=(), overwrite=False, metadata=(), plot=None,  *args, **kws):

        # init task
        # Placeholder here for time series
        super().__init__(func, o, *args, **kws)

        # NOTE: infile, outfiles, plot_files are path templates
        templates = ConfigNode({'input': infile,
                                'output': [],
                                'plots': []}).freeze()
        for file in outfiles:
            cat = ('plots', 'output')[SupportedFormats.check(str(file))]
            templates[cat].append(file)

        self.plot = plot
        self.section = section
        self.metadata = dict(metadata)
        self.templates = templates
        self.overwrite = bool(overwrite)
        self.output_class = output_class

    def resolve_paths(self, tmp, hdu, **kws):
        if tmp:
            section = ('lightcurves', *self.section)
            return sorted(tmp.resolve_paths(hdu, section, **kws),
                          key=Partial(sort_load_order)(o, self._load_order_prefered))
        return []

    def load_or_compute(self, obs, data=None, load_kws=(), *args, **kws):

        hdu = obs[0]
        datafiles = [o.resolve_path(hdu) for o in self.templates.output]

        load_kws = load_kws or {}
        for path in datafiles:
            if path.exists() and not self.overwrite:
                logger.info('Loading lightcurve for {} from {}.', hdu, path)
                return self.output_class.load(path, **load_kws)

        # Load input data
        if data is None:
            infiles = self.resolve_paths(self.templates.input, hdu)
            if infiles and (infile := infiles[0]).exists():
                data = self.output_class.load(infile, **load_kws)
            else:
                raise FileNotFoundError(repr(str(infile)))

        # Compute
        logger.opt(lazy=True).debug('Computing: {0}.',
                                    lambda: ppr.caller(self.__wrapped__))

        for path in datafiles:
            logger.opt(lazy=True).debug(
                'File {0[0]!s} {0[1]}.',
                lambda: (path, (f"will be {('created', 'overwritten')[self.overwrite]}"
                                if path.exists() else 'does not exist'))
            )

        # Do compute for step
        return super().__call__(data, *args, **kws)
        

    def __call__(self, obs, data=None, load_kws=(), *args, **kws):

        # Load result from previous run if available  # TODO: Manage with cache
        if len(obs) > 1:
            from IPython import embed
            embed(header="Embedded interpreter at 'src/pyshoc/pipeline/lightcurves.py':343")

        # if '20130212.010' in obs[0].file.name:
        #     from IPython import embed
        #     embed(header="Embedded interpreter at 'src/pyshoc/pipeline/lightcurves.py':480")
        
        hdu = obs[0]
        self.templates.output
        out = [o.resolve_path(hdu) for o in self.templates.output]

        #
        result = self.load_or_compute(obs, data, load_kws, *args, **kws)

        # ensure LightCurve
        if not isinstance(result, self.output_class):
            result = self.output_class(*result)

        # save text
        if self.metadata is not False:
            for path in out:
                if self.overwrite or not path.exists():
                    # if path.suffix == '.npz':
                    # from IPython import embed
                    # embed(header="Embedded interpreter at 'src/pyshoc/pipeline/lightcurves.py':485")
                    result.save(path, **get_metadata(hdu, **self.metadata))

        # plot
        if self.plot is not False:
            plot = {} if self.plot is True else dict(self.plot)
            params, kws = dicts.split(plot, ('keys', 'filename', 'overwrite'))
            kws = {**kws,
                   'overwrite': self.overwrite,
                   'filenames':  [tmp.resolve_path(hdu)
                                  for tmp in self.templates.plots]}

            # load task
            if plotter := self.pipeline.plotter:
                try:
                    grouping, step = self.section
                    tab = (*cfg.lightcurves.tab, grouping, *get_tab_key(hdu), step)
                    task = plotter.task_factory(result.plot)(*result, **params)
                    plotter.add_task(task, tab, **kws)
                except Exception as err:
                    import sys
                    import textwrap
                    from IPython import embed
                    from better_exceptions import format_exception
                    embed(header=textwrap.dedent(
                        f"""\
                            Caught the following {type(err).__name__} at 'lightcurves.py':500:
                            %s
                            Exception will be re-raised upon exiting this embedded interpreter.
                            """) % '\n'.join(format_exception(*sys.exc_info()))
                    )
                    raise

        return result


# def lag_scatter(x, ):
