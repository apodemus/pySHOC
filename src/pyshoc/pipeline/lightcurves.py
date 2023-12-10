
# std
from collections import abc

# third-party
import numpy as np
from tqdm import tqdm
from mpl_toolkits.axes_grid1.parasite_axes import SubplotHost

# local
import motley
from obstools import lc
from scrawl.ticks import DateTick, _rotate_tick_labels
from tsa import TimeSeries
from tsa.smoothing import tv
from tsa.ts.plotting import make_twin_relative
from tsa.outliers import WindowOutlierDetection
from recipes.array import fold
from recipes.config import ConfigNode
from recipes import dicts, io, pprint as ppr
from recipes.decorators import delayed, update_defaults
from recipes.functionals.partial import PlaceHolder as o

# relative
from ..timing import Time
from .logging import logger
from .plotting import PlotTask
from .products import resolve_path


# ---------------------------------------------------------------------------- #
SPD = 86400

# ---------------------------------------------------------------------------- #
CONFIG = ConfigNode.load_module(__file__)

# ---------------------------------------------------------------------------- #


def load_or_compute(path, overwrite, worker, save, plot):

    if path.exists() and not overwrite:
        logger.info('Loading lightcurve from {}.', path)
        return lc.io.read_text(path)

    # Compute
    logger.debug('File {!s} does not exist. Computing: {}.',
                 motley.apply(str(path), 'darkgreen'),
                 ppr.caller(worker))  # args, kws
    #
    data = worker()

    # save text
    if save is not False:
        lc.io.write_text(path, *data, **(save or {}))

    # plot
    if plot is not False:
        plot = plot or {}
        kws, init = dicts.split(plot, ('ui', 'keys', 'filename', 'overwrite'))
        init.setdefault('filename', path.with_suffix('png'))
        init.setdefault('overwrite', overwrite)

        if not isinstance(data, TimeSeries):
            data = TimeSeries(*data)

        task = PlotTask(**init)
        task(plotter)(o, data, **kws)

    return data


def _get_save_meta(hdu, **kws):
    return dict(
        **kws,
        obj_name=hdu.target,
        meta={'Observing info':
              {'T0 [UTC]': hdu.t[0].utc,
               'File':     hdu.file.name}}
    )


def _get_save_meta_stack(campaign):
    save = _get_save_meta(campaign[0], title=CONFIG['diff0'].title)
    info = save['meta']['Observing info']
    info.pop('File')
    info['Files'] = ', '.join(campaign.files.names)


# ---------------------------------------------------------------------------- #

def _get_plot_config(grouping, step):

    kws, specific = CONFIG.plots.split(('by_file', 'by_date'))
    specific = specific[grouping][step]

    if specific is False:
        return False

    if not isinstance(specific, abc.MutableMapping):
        specific = {}

    return {**kws, **specific}


# def _plot_helper(ts, filename, overwrite, ui=None, key=None, delay=True, **kws):
#     if not isinstance(ts, TimeSeries):
#         ts = TimeSeries(*ts)

#     if not isinstance(kws, abc.MutableMapping):
#         kws = {}

#     #
#     fig = get_figure(ui, key, **figkws)

#     if ui and delay and not fig.plot:
#         ui[key].add_task(plotter, (fig, ts, filename, overwrite), **kws)
#     else:
#         plotter(fig, ts, filename, overwrite, **kws)


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
# def _load(id_, infile, outfile, overwrite, plot):

#     # outfile = paths.lightcurves[step]
#     step, key = id_
#     data = load_or_compute(
#         # load
#         resolve_path(outfile, key), overwrite,
#         # compute
#         delayed(COMPUTE[step])(key, resolve_path(infile, key)),
#         # save
#         _get_save_meta(key, title=CONFIG[step].title),
#         # plot
#         _get_plot_config('by_file', 'raw')
#     )

#     return

# ---------------------------------------------------------------------------- #


def load_raw(hdu, infile, outfile, overwrite=False, plot=False):

    logger.info(f'{infile = }; {outfile = }')

    # _load(('raw', hdu, ))

    data = load_or_compute(
        # load
        resolve_path(outfile, hdu),
        CONFIG.raw.get('overwrite', overwrite),
        # compute
        delayed(load_memmap)(hdu, resolve_path(infile, hdu)),
        # save
        _get_save_meta(hdu, title=CONFIG['raw'].title),
        # plot
        _get_plot_config('by_file', 'raw')
    )


def load_memmap(hdu, filename):

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


def load_flagged(hdu, paths, overwrite=False, plot=False):
    files = paths.files
    return load_or_compute(
        # load
        resolve_path(files.lightcurves.flagged, hdu),
        CONFIG.flagged.get('overwrite', overwrite),
        # compute
        delayed(_flag_outliers)(hdu,
                                resolve_path(files.tracking.source_info, hdu),
                                resolve_path(files.lightcurves.raw, hdu),
                                overwrite),
        # save
        _get_save_meta(hdu, title=CONFIG['flagged'].title),
        # plot
        _get_plot_config('by_file', 'flagged')

    )


def _flag_outliers(hdu, infile, outfile, overwrite):
    # load memmap
    # _load('raw', )
    t, flux, sigma = load_raw(hdu, infile, outfile, overwrite)
    # bjd = t.bjd
    # flux = flag_outliers(t, flux)
    return t, flag_outliers(t, flux), sigma


@update_defaults(CONFIG.flagged.params)
def flag_outliers(bjd, flux, nwindow, noverlap, kmax):

    # flag outliers
    logger.info('Detecting outliers.')

    oflag = np.isnan(flux)
    for i, flx in enumerate(flux):
        logger.debug('Source {}.', i)
        oidx = WindowOutlierDetection(flx, nwindow, noverlap, kmax=kmax)
        logger.info('Flagged {}/{} ({:5.3%}) points as outliers.',
                    (no := len(oidx)), (n := len(bjd)), (no / n))
        oflag[i, oidx] = True

    return np.ma.MaskedArray(flux, oflag)

# ---------------------------------------------------------------------------- #


COMPUTE = {
    'raw':      load_memmap,
    'flagged':  _flag_outliers,
    # 'diff0':    diff0_phot,
    # 'decor':    load_decor
}


def diff0_phot(hdu, paths, overwrite=False, plot=False):
    # filename = folder / f'raw/{hdu.file.stem}-phot-raw.txt'
    # bjd, flux, sigma = load_phot(hdu, filename, overwrite)

    save = _get_save_meta(hdu, title=CONFIG['diff0'].title)
    # processing metadata added by `_diff0_phot`
    meta = save['meta']['Processing'] = {}

    return load_or_compute(
        # load
        resolve_path(paths.files.lightcurves.diff0.filename, hdu), overwrite,
        # compute
        delayed(_diff0_phot)(hdu, paths, meta=meta, overwrite=overwrite),
        # save
        save,
        # plot
        _get_plot_config('by_file', 'diff0')

    )


def _diff0_phot(hdu, paths, c=1, meta=None, overwrite=False):

    t, flux, sigma = load_flagged(hdu, paths, overwrite)

    # Zero order differential
    fm = np.ma.median(flux[:, c])
    if meta:
        meta['flux scale'] = fm

    return (t,
            np.ma.MaskedArray(flux.data / fm, flux.mask),
            sigma / fm + sigma.mean(0) / sigma.shape[1])


# ---------------------------------------------------------------------------- #

def concat_phot(campaign, paths, overwrite=None, plot=False):
    #
    cfg = CONFIG.find('concat')
    title, = cfg.find('title').flatten().values()
    save = _get_save_meta(campaign[0], title=title)

    info = save['meta']['Observing info']
    info.pop('File')
    info['Files'] = ', '.join(campaign.files.names)

    return load_or_compute(
        # load
        resolve_path(paths.files.lightcurves.diff0.concat, campaign[0]),
        cfg.get('overwrite', overwrite),
        # compute
        delayed(_concat_phot)(campaign, paths, overwrite, plot),
        # save
        save,
        # plot
        _get_plot_config('by_date', 'diff0')
    )


def _concat_phot(campaign, paths, overwrite, plot=False):
    # stack time series for target run
    logger.info('Concatenating {} light curves for run on {}.',
                len(campaign), campaign[0].date_for_filename)

    # data
    bjd, rflux, rsigma = map(np.ma.hstack,
                             zip(*(diff0_phot(hdu, paths, overwrite, plot)
                                   for hdu in campaign)))
    return bjd.data, rflux, rsigma.data

# ---------------------------------------------------------------------------- #


def extract(run, paths, overwrite=False, plot=False):

    logger.info('Extracting lightcurves for {!r}.', run[0].target)

    lightcurves = dicts.DictNode()
    for date, obs in run.group_by('t.date_for_filename').sorted().items():
        date = str(date)
        # year, day = date.split('-', 1)

        bjd, rflux, rsigma = concat_phot(obs, paths, overwrite, plot)
        lightcurves['diff0'][date] = ts = TimeSeries(bjd, rflux.T, rsigma.T)

        # decorrellate
        lightcurves['decor'][date] = decor(ts, obs, paths, overwrite)

    lightcurves.freeze()
    return lightcurves


def decor(ts, campaign, paths, overwrite, **kws):

    kws = {**CONFIG.decor, **kws}
    save = _get_save_meta(campaign[0], title=kws.pop('title'))
    info = save['meta']['Observing info']
    info.pop('File')
    info['Files'] = ', '.join(campaign.files.names)

    meta = save['meta']['Processing'] = kws

    bjd, rflux, rsigma = load_or_compute(
        # load
        resolve_path(paths.files.lightcurves.decor, campaign[0]), overwrite,
        # compute
        delayed(_decor)(ts, **kws),
        # save
        save,
        # plot
        _get_plot_config('by_date', 'decor')
    )

    return TimeSeries(bjd, rflux.T, rsigma.T)

    # # save text
    # filename = paths
    # lc.io.write_text(
    #     filename,
    #     tss.t, tss.x.T, tss.u.T,
    #     title='Differential (smoothed) ragged-aperture light curve for {}.',
    #     obj_name=run[0].target,
    #     meta={'Observing info':
    #           {'T0 [UTC]': Time(tss.t[0], format='jd').utc,
    #            # 'Files':    ', '.join(campaign.files.names)
    #            }}
    # )


def _decor(ts, **kws):
    logger.info('Decorrelating light curve for photometry.')
    tss = _diff_smooth_phot(ts, **kws)
    return tss.t, tss.x.T, tss.u.T


def _diff_smooth_phot(ts, nwindow, noverlap, smoothing):
    # smoothed differential phot
    s = tv_window_smooth(ts.t, ts.x, nwindow, noverlap, smoothing)
    s = np.atleast_2d(s).T - 1
    # tss = ts - s
    return ts - s


@update_defaults(CONFIG.decor.params)
def tv_window_smooth(t, x, nwindow, noverlap, smoothing):

    n = len(t)
    nwindow = fold.resolve_size(nwindow, n)
    noverlap = fold.resolve_size(noverlap, nwindow)
    half_overlap = noverlap // 2

    tf = fold.fold(t, nwindow, noverlap)
    xf = fold.fold(x[:, 1], nwindow, noverlap)
    nsegs = tf.shape[0]

    a = 0
    r = []
    for i, (tt, xx) in tqdm(enumerate(zip(tf, xf)),
                            total=nsegs,
                            **{**CONFIG.parent.console.progress,
                               'unit': ' segments'}):
        s = tv.smooth(tt, xx, smoothing)
        r.extend(s[a:-half_overlap])
        a = half_overlap

    if len(r) < n:
        r.extend(s[-half_overlap:-half_overlap + n - len(r)])
    elif len(r) > n:
        r = r[:n]

    return np.ma.array(r)
