
# third-party
import numpy as np
from tqdm import tqdm
from mpl_multitab import MplMultiTab
from mpl_toolkits.axes_grid1.parasite_axes import SubplotHost

# local
import motley
from obstools import lc
from scrawl.ticks import DateTick, _rotate_tick_labels
from recipes.array import fold
from recipes.dicts import DictNode
from recipes import io, pprint as ppr
from tsa import TimeSeries
from tsa.smoothing import tv
from tsa.ts.plotting import make_twin_relative
from tsa.outliers import WindowOutlierDetection

# relative
from ..timing import Time
from ..config import CONFIG
from .logging import logger
from .utils import save_fig
from .products import resolve_path


# ---------------------------------------------------------------------------- #
SPD = 86400

# ---------------------------------------------------------------------------- #


def load_or_compute(path, overwrite, compute, args=(), kws=None, save=False):

    if path.exists() and not overwrite:
        logger.info('Loading lightcurve from {}.', path)
        return lc.io.read_text(path)

    #
    logger.debug('File {!s} does not exist. Computing: {}',
                 motley.apply(str(path), 'darkgreen'),
                 ppr.caller(compute))  # args, kws

    if not isinstance(args, tuple):
        args = args,

    data = compute(*args, **(kws or {}))

    if save:
        lc.io.write_text(path, *data, **(save or {}))

    return data


def load_raw(hdu, infile, outfile, overwrite=False):

    logger.info(f'{infile = }; {outfile = }')

    return load_or_compute(
        # load
        resolve_path(outfile, hdu), overwrite,
        # compute
        load_memmap, (hdu, resolve_path(infile, hdu)),
        # save
        save=_get_save_meta(
            hdu,
            title='Raw ragged-aperture light curve for {}.',
        )
    )


def _get_save_meta(hdu, **kws):
    return dict(
        **kws,
        obj_name=hdu.target,
        meta={'Observing info':
              {'T0 [UTC]': hdu.t[0].utc,
               'File':     hdu.file.name}}
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


def load_flagged(hdu, paths, overwrite=False):

    data = load_or_compute(
        # load
        resolve_path(paths.lightcurves.flagged, hdu), overwrite,
        # compute
        _flag_outliers,
        (hdu,
         resolve_path(paths.tracking.source_info, hdu),
         resolve_path(paths.lightcurves.raw, hdu),
         overwrite),
        # save
        save=_get_save_meta(
            hdu,
            title='Flagged ragged-aperture light curve for {}.',
        )
    )

    # from IPython import embed
    # embed(header="Embedded interpreter at 'src/pyshoc/pipeline/lightcurve.py':111")
    return data


def _flag_outliers(hdu, infile, outfile, overwrite):
    # load memmap
    t, flux, sigma = load_raw(hdu, infile, outfile, overwrite)
    #bjd = t.bjd
    # flux = flag_outliers(t, flux)
    return t, flag_outliers(t, flux), sigma


def flag_outliers(bjd, flux, nwindow=1000, noverlap='50%', kmax='0.1%'):

    # flag outliers
    logger.info('Detecting outliers.')

    oflag = np.isnan(flux)
    for i, flx in enumerate(flux):
        logger.debug('Source {}', i)
        oidx = WindowOutlierDetection(flx, nwindow, noverlap, kmax=kmax)
        logger.info('Flagged {}/{} ({:5.3%}) points as outliers.',
                    (no := len(oidx)), (n := len(bjd)), (no / n))
        oflag[i, oidx] = True

    return np.ma.MaskedArray(flux, oflag)

# ---------------------------------------------------------------------------- #


def diff0_phot(hdu, paths, overwrite=False):
    # filename = folder / f'raw/{hdu.file.stem}-phot-raw.txt'
    # bjd, flux, sigma = load_phot(hdu, filename, overwrite)

    save = _get_save_meta(
        hdu,
        title='Differential (0th order) ragged-aperture light curve for {}.',
    )
    meta = save['meta']['Processing'] = {}

    return load_or_compute(
        # load
        resolve_path(paths.lightcurves.diff0, hdu), overwrite,
        # compute
        _diff0_phot, (hdu, paths), dict(meta=meta, overwrite=overwrite),
        # save
        save=save
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


def concat_phot(obs, paths, overwrite):
    #
    save = _get_save_meta(
        obs[0],
        title='Differential (0th order) ragged-aperture light curve for {}.'
    )
    info = save['meta']['Observing info']
    info.pop('File')
    info['Files'] = ', '.join(obs.files.names)

    return load_or_compute(
        # load
        resolve_path(paths.lightcurves.nightly.diff0, obs[0]), overwrite,
        # compute
        _concat_phot, (obs, paths, overwrite),
        # save
        save=save
    )


def _concat_phot(obs, paths, overwrite):
    # stack time series for target run
    logger.info('Concatenating {} light curves for run on {}',
                len(obs), obs[0].date_for_filename)

    data = [diff0_phot(hdu, paths, overwrite) for hdu in obs]

    bjd, rflux, rsigma = map(np.ma.hstack, zip(*data))
    return bjd.data, rflux, rsigma.data

# ---------------------------------------------------------------------------- #


def extract(run, paths, overwrite=False):

    logger.info('Extracting lightcurves for {!r}', run[0].target)

    lightcurves = DictNode()
    nightly = run.group_by('t.date_for_filename').sorted()
    for date, obs in nightly.items():
        date = str(date)
        # year, day = date.split('-', 1)
        bjd, rflux, rsigma = concat_phot(obs, paths, overwrite)
        lightcurves['diff0'][date] = ts = TimeSeries(bjd, rflux.T, rsigma.T)

        # decorrellate
        lightcurves['decor'][date] = decor(ts, obs, paths, overwrite)

    lightcurves.freeze()
    return lightcurves


def decor(ts, obs, paths, overwrite, **kws):

    save = _get_save_meta(
        obs[0],
        title='Differential (smoothed) ragged-aperture light curve for {}.',
    )
    info = save['meta']['Observing info']
    info.pop('File')
    info['Files'] = ', '.join(obs.files.names)

    meta = save['meta']['Processing'] = kws

    bjd, rflux, rsigma = load_or_compute(
        # load
        resolve_path(paths.lightcurves.nightly.decor, obs[0]), overwrite,
        # compute
        _decor, ts, kws,
        # save
        save=save
    )
    tss = TimeSeries(bjd, rflux.T, rsigma.T)

    # # save text
    # filename = paths
    # lc.io.write_text(
    #     filename,
    #     tss.t, tss.x.T, tss.u.T,
    #     title='Differential (smoothed) ragged-aperture light curve for {}.',
    #     obj_name=run[0].target,
    #     meta={'Observing info':
    #           {'T0 [UTC]': Time(tss.t[0], format='jd').utc,
    #            # 'Files':    ', '.join(obs.files.names)
    #            }}
    # )


def _decor(ts, **kws):
    logger.info('Decorrelating light curve for photometry.')
    tss = _diff_smooth_phot(ts, **kws)
    return tss.t, tss.x.T, tss.u.T


def _diff_smooth_phot(ts, nwindow=1000, noverlap='10%', smoothing=0.1):
    # smoothed differential phot
    s = tv_window_smooth(ts.t, ts.x, nwindow, noverlap, smoothing)
    s = np.atleast_2d(s).T - 1
    # tss = ts - s
    return ts - s


def tv_window_smooth(t, x, nwindow=1000, noverlap='10%', smoothing=0.1):

    n = len(t)
    nwindow = fold.resolve_size(nwindow, n)
    noverlap = fold.resolve_size(noverlap, nwindow)
    half_overlap = noverlap // 2

    tf = fold.fold(t, nwindow, noverlap)
    xf = fold.fold(x[:, 1], nwindow, noverlap)
    nsegs = tf.shape[0]

    
    
    r = []
    a = 0
    for i, (tt, xx) in tqdm(enumerate(zip(tf, xf)),
                            total=nsegs, 
                            **{**CONFIG.tracking.progress,
                               'unit': 'segments'}):
        s = tv.smooth(tt, xx, smoothing)
        r.extend(s[a:-half_overlap])
        a = half_overlap
    
    # if len(r) < n:
    #     s = tv.smooth(tf[-1], xf[-1], smoothing)
    #     r.extend(s[half_overlap:half_overlap + n - len(r)])
    # else:
    #     r = r[:n]
    
    if len(r) != n:
        from IPython import embed
        embed(header="Embedded interpreter at 'src/pyshoc/pipeline/lightcurve.py':309")
    
    return np.ma.array(r)


# LOADERS = {
#     'raw':      load_raw,
#     'flagged':  load_flagged,
#     # 'diff0':    load_diff0,
#     # 'decor':    load_decor
# }
# ---------------------------------------------------------------------------- #


def _plot_lc_ui(fig, indices, ui, lcs, filenames, overwrite, **kws):
    date = '-'.join(ui.tabs.tab_text(indices))
    return _plot_lc(fig, lcs[date], filenames[date], overwrite, **kws)


def _plot_lc(fig, ts, filename, overwrite, **kws):
    #
    # logger.debug('{}', pformat(locals()))
    ax = SubplotHost(fig, 1, 1, 1)
    fig.add_subplot(ax)

    #
    jd0 = int(ts.t[0]) - 0.5
    utc0 = Time(jd0, format='jd').utc.iso.split()[0]
    #
    axp = make_twin_relative(ax, -(ts.t[0] - jd0) * SPD, 1, 45, utc0)

    # plot
    tsp = ts.plot(ax, t0=[0], tscale=SPD,
                  **{**dict(plims=(-0.1, 99.99), show_masked=True), **kws})

    axp.xaxis.set_minor_formatter(DateTick(utc0))
    _rotate_tick_labels(axp, 45, True)

    ax.set(xlabel='Δt (s)', ylabel='Relative Flux')
    axp.set_xlabel('Time (UTC)', labelpad=-17.5)

    # fig.tight_layout()
    fig.subplots_adjust(
        top=0.81,
        bottom=0.1,
        left=0.065,
        right=0.94
    )

    if overwrite or not filename.exists():
        save_fig(fig, filename)

    return fig

# CONFIG.plotting.delay:


def plot(lcs, filename_template=None, overwrite=False, delay=True, **kws):

    ui = MplMultiTab(title='Light curves', pos='N')

    filenames = {}
    for date, ts in lcs.items():
        filenames[date] = filename = Path(str(filename_template).format(date=date))
        year, day = date.split('-', 1)
        fig = ui.add_tab(year, day, fig={'figsize': (12, 6)}).figure

        if not delay:
            fig = _plot_lc(fig, ts, filename, overwrite, **kws)

    if delay:  #
        ui.add_callback(_plot_lc_ui, ui=ui,
                        lcs=lcs, filenames=filenames,
                        overwrite=overwrite, **kws)

    return ui
