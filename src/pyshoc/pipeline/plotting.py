
# std
import sys
from pathlib import Path

# third-party
from loguru import logger
from matplotlib.figure import Figure

# local
from recipes.pprint import callers
from recipes.logging import LoggingMixin
from recipes.functionals.partial import Partial, PartialAt

# relative
from ..config import CONFIG
from .logging import logger


# ---------------------------------------------------------------------------- #
def get_figure(ui=None, *key, fig=None, **kws):
    if ui:
        logger.debug('UI active, adding tab {}', key)
        tab = ui.add_tab(*key, fig=fig, **kws)
        return tab.figure

    if fig:
        assert isinstance(fig, Figure)
        return fig

    if plt := sys.modules.get('matplotlib.pyplot'):
        logger.debug('pyplot active, launching figure {}', key)
        return plt.figure(**kws)

    logger.debug('No UI, creating figure. {}', key)
    return Figure(**kws)


def save_figure(fig, filename, overwrite=False):
    if filename:
        filename = Path(filename)
        if not filename.exists() or overwrite:
            logger.info('Saving image: {}', filename)
            fig.savefig(filename)


# alias
save_fig = save_figure


# ---------------------------------------------------------------------------- #

class PlotTask(LoggingMixin, PartialAt):

    def __init__(self, ui, args, kws):
        self.ui = ui
        super().__init__(args, kws)

    def __wrapper__(self, func, figure, key, *args, **kws):

        self.logger.opt(lazy=True).info(
            'Plotting tab {0[0]} with {0[1]} at figure: {0[2]}.',
            lambda: (self.ui.tabs.tab_text(key), callers.describe(func), figure))

        # Fill dynamic parameter values
        if self.nfree:
            args = (*self._get_args((figure, )), *args)
            kws = self._get_kws(kws)
        if self._keywords:
            args = (*self.args, *args)
            kws = {**self._get_kws({list(self._keywords).pop(): figure}), **kws}

        self.logger.opt(lazy=True).info(
            'Invoking call for plot task: {}',
            lambda: callers.pformat(func, args, kws)
        )

        # plot
        art = func(*args, **kws)

        if not (self.nfree or self._keywords):
            # We will have generated a figure to fill the tab in the ui, we have
            # to replace it after the task executes with the actual figure we
            # want in our tab.
            figure = art.fig
            mgr = self.ui[tuple(key)]._parent()
            mgr.replace_tab(key[-1], figure, focus=True)

        return figure, art


class _SaveTask:
    def __init__(self, task, filename=None, overwrite=False):
        self.task = task
        self.filename = filename
        self.overwrite = overwrite

    def __call__(self, figure, key, *args, **kws):
    
        # run
        figure, art = self.task(figure, key, *args, **kws)

        # save
        save_fig(figure, self.filename, self.overwrite)

        return art


class PlotFactory(LoggingMixin, Partial):
    """Plotting task factory"""

    task = PlotTask

    def __init__(self, ui=None, delay=CONFIG.plotting.gui.delay):
        self.ui = ui
        self.delay = (ui is not None) and delay

    def __wrapper__(self, func, *args, **kws):
        return self.task(self.ui, args, kws)(func)

    def add_task(self, task, key, filename, overwrite, figure=None, *args, **kws):

        # Task requires Figure
        # func_creates_figure = not (task.nfree or task._keywords)
        # # next line will generate figure to fill the tab, we have to replace it
        # # after the task executes with the actual fgure we want in our tab
        figure = get_figure(self.ui, *key, fig=figure)

        _task = _SaveTask(task, filename, overwrite)

        if self.delay:
            # Future task
            self.logger.info('Plotting delayed: Adding plot callback for {}: {}.',
                             key, task)

            tab = self.ui[key]
            tab.add_task(_task, *args, **kws)

            return _task

        # execute task
        self.logger.debug('Plotting immediate: {}.', self.key)
        return _task(figure, key, *args, **kws)


# ---------------------------------------------------------------------------- #

class PlotInterface:
    pass
