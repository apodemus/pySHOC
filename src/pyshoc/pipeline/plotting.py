
# std
import sys
from pathlib import Path

# third-party
from loguru import logger
from matplotlib.figure import Figure

# local
from recipes import dicts
from recipes.pprint import callers
from recipes.logging import LoggingMixin
from recipes.oo.slots import SlotHelper, _sanitize_locals
from recipes.functionals.partial import Partial, PartialAt

# relative
from ..config import CONFIG
from .logging import logger


# ---------------------------------------------------------------------------- #
FIG_KWS = ('figsize',
           'dpi',
           'facecolor',
           'edgecolor',
           'linewidth',
           'frameon',
           'subplotpars',
           'tight_layout',
           'constrained_layout',
           'layout')

# ---------------------------------------------------------------------------- #


def get_figure(ui=None, *tab, figure=None, figsize=None, add_axes=False, **kws):

    figure = _get_figure(ui, tab, figure, figsize=figsize, **kws)

    if add_axes:
        if figure.axes:
            logger.warning('Ignoring `add_axes=True` since figure not empty.')
        else:
            figure.add_subplot()

    # resize if requested (normally handled by ui)
    if not ui:
        if figsize:
            self.logger.info('Resizing figure: {}', figsize)
            figure.set_size_inches(figsize)

    return figure


def _get_figure(ui, tab, fig, **kws):

    # Create figure if needed, add to ui tab if active
    if ui:
        if tab:
            logger.debug('UI active, adding tab {}.', tab)
            tab = ui.add_tab(*tab, fig=fig, **kws)
            return tab.figure

        logger.debug('UI active, but no tab. Figure will not be embedded.')

    if fig:
        assert isinstance(fig, Figure)
        return fig

    if plt := sys.modules.get('matplotlib.pyplot'):
        logger.debug('pyplot active, launching figure {}.', tab)
        return plt.figure(**kws)

    logger.debug('No UI, creating figure. {}.', tab)
    return Figure(**kws)


def save_figure(fig, filename, overwrite=False, **kws):

    if not filename:
        logger.debug('Not saving {} since filename is {!r}.', fig, filename)
        return False

    filename = Path(filename)
    if filename.exists():
        if not overwrite:
            logger.info('Not overwriting: {!s}', filename)
            return False

        logger.info('Overwriting image: {!s}.', filename)
        fig.savefig(filename, **kws)
        return True

    logger.info('Saving image: {!s}.', filename)
    fig.savefig(filename, **kws)
    return True


# alias
save_fig = save_figure


# ---------------------------------------------------------------------------- #

class PlotTask(LoggingMixin, PartialAt):

    def __init__(self, ui, args, kws):
        self.ui = ui
        super().__init__(args, kws)

    def __wrapper__(self, func, figure, tab, *args, **kws):

        self.logger.opt(lazy=True).info(
            'Plotting tab {0[0]} with {0[1]} at figure: {0[2]}.',
            lambda: (self.ui.tabs.tab_text(tab), callers.describe(func), figure))

        # Fill dynamic parameter values
        nfree = self.nfree
        if nfree:
            args = (*self._get_args((figure, *([tab] * (nfree == 2)))), *args)
            kws = self._get_kws(kws)

        if self._keywords:
            args = (*self.args, *args)
            kws = {**self._get_kws({list(self._keywords).pop(): figure}), **kws}

        self.logger.opt(lazy=True).info(
            'Invoking call for plot task:\n>>> {}.',
            lambda: callers.pformat(func, args, kws)
        )

        # plot
        art = func(*args, **kws)

        if not (self.nfree or self._keywords):
            # We will have generated a figure to fill the tab in the ui, we have
            # to replace it after the task executes with the actual figure we
            # want in our tab.
            figure = art.fig
            mgr = self.ui[tuple(tab)]._parent()
            mgr.replace_tab(tab[-1], figure, focus=False)

        return figure, art


class FigureSaver(SlotHelper):

    __slots__ = ('task', 'figure', 'filename', 'overwrite', 'save_kws')

    def __init__(self, task, figure, filename=None, overwrite=False, **save_kws):
        if filename:
            filename = Path(filename)

        super().__init__(**_sanitize_locals(locals()))

    def __call__(self, figure, tab, *args, **kws):

        # run
        figure, art = self.task(figure, tab, *args, **kws)

        # save
        save_fig(figure, self.filename, self.overwrite, **self.save_kws)

        return art

# ---------------------------------------------------------------------------- #


class PlotFactory(LoggingMixin, Partial):  # PlotInterface
    """Plotting task factory"""

    Task = PlotTask

    def __init__(self, ui=None, delay=CONFIG.plotting.gui.delay):
        self.ui = ui
        self.delay = (ui is not None) and delay

    def __wrapper__(self, func, *args, **kws):
        # split keywords for figure init
        kws, self.fig_init = dicts.split(kws, FIG_KWS)
        return self.Task(self.ui, args, kws)(func)

    def add_task(self, task, tab, filename, overwrite, figure=None, add_axes=False,
                 *args, **kws):

        # Task requires Figure
        # func_creates_figure = not (task.nfree or task._keywords)
        # # next line will generate figure to fill the tab, we have to replace it
        # # after the task executes with the actual fgure we want in our tab

        figure = get_figure(self.ui, *tab, figure=figure, add_axes=add_axes,
                            **self.fig_init)

        self.logger.debug('Task will save figure {} at {}, {}.',
                          figure, filename, f'{overwrite = }')
        _task = FigureSaver(task, figure, filename, overwrite)

        if self.delay:
            # Future task
            self.logger.info('Plotting delayed: Adding plot callback for {}: {}.',
                             tab, _task)

            self.ui[tab].add_task(_task, *args, **kws)

            return _task

        # execute task
        self.logger.debug('Plotting immediate: {}.', tab)
        result = _task(figure, tab, *args, **kws)
        _task._results_cache.append(result)
        return _task
