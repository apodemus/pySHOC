
# std
import sys
from pathlib import Path
from IPython import embed

# third-party
from loguru import logger
from matplotlib.figure import Figure

# local
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

    def __wrapper__(self, func, figure, key, filename, overwrite, *args, **kws):

        self.logger.info('Plotting tab {} with {!r} at figure {}.',
                         key, func.__name__, figure)

        # Fill dynamic parameter values
        if self.nfree:
            args = self._get_args((figure, *args))
            kws = self._get_kws(kws)
        if self._keywords:
            args = self._get_args(args)
            kws = self._get_kws({list(self._keywords).pop(): figure})

        func_creates_figure = not (self.nfree or self._keywords)
        # We will have generated a figure to fill the tab in the ui, we have to
        # replace it after the task executes with the actual figure we want in
        # our tab.

        # plot
        art = func(*args, **kws)

        if func_creates_figure:
            figure = art.fig
            mgr = self.ui[tuple(key)]._parent()
            mgr.replace_tab(key[-1], figure, focus=True)

        # save
        save_fig(figure, filename, overwrite)

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

        if self.delay:
            # Future task
            self.logger.info('Plotting delayed: Adding plot callback for {}: {}.',
                             key, task)

            # self.ui[key]._parent().add_callback(
            #   task, filename, overwrite, *args, **kws)
            tab = self.ui[key]
            tab.add_callback(task, filename, overwrite, *args, **kws)
            parent = tab._parent()
            parent._cid = parent.tabs.currentChanged.connect(parent._on_change)

            return task

        # execute task
        self.logger.debug('Plotting immediate: {}.', self.key)
        return task(figure, key, filename, overwrite, *args, **kws)


# ---------------------------------------------------------------------------- #

class PlotInterface:
    pass
