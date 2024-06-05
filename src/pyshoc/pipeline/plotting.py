
# std
import sys
from pathlib import Path

# third-party
from matplotlib.figure import Figure
from mpl_multitab import MplMultiTab, QtGui

# local
from scrawl.utils import save_figure
from recipes import dicts
from recipes.oo import slots
from recipes.string import indent
from recipes.pprint import callers
from recipes.containers import ensure
from recipes.logging import LoggingMixin
from recipes.pprint.callers import Callable
from recipes.functionals.partial import Partial, PartialTask

# relative
from .. import config as cfg
from .logging import logger


# ---------------------------------------------------------------------------- #
FIG_KWS = Figure.__init__.__code__.co_varnames[1:-2]
SAVE_KWS = ('filename', 'overwrite', 'dpi')

# ---------------------------------------------------------------------------- #


def get_figure(ui, tab, fig, **kws):

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


# ---------------------------------------------------------------------------- #

class _TaskBase(PartialTask, LoggingMixin):
    """Base class for Tasks."""


class TaskFactory(Partial, LoggingMixin):

    def __init__(self, ui):
        self.ui = ui

    def __wrapper__(self, func, *args, **kws):

        # resolve placholders and static params
        task = PlotTask(func, self.ui, *args, **kws)

        # create TaskRunner, which will run PlotTask when called
        return TaskRunner(task, **task.fig_kws)


class TaskRunner(_TaskBase):
    # Run task when called

    def __init__(self, task, **fig_kws):
        self.ui = None  # set in `GUI.add_task`
        self.task = task  # PlotTask
        self.fig_kws = fig_kws

    def __call__(self, figure, tab, *args, **kws):
        # Call signature for tasks requires: figure, tab
        return self.task(figure, tab, *args, **kws)

    def __repr__(self):
        name = type(self).__name__
        # inner = callers.pformat(self.task, (), self.fig_kws, hang=False)
        inner = indent(repr(self.task), 4)
        return f'{name}({inner})'


class PlotTask(_TaskBase):
    """Plot Task"""

    def __init__(self, func, ui, *args, **kws):

        # split keywords for figure init
        kws, self.fig_kws = dicts.split(kws, FIG_KWS)
        kws, self.save_kws = dicts.split(kws, SAVE_KWS)

        if 'filename' in Callable(func).sig.parameters:
            kws['filename'] = self.save_kws['filename']

        #
        self.ui = ui
        
        # resolve placholders and static params
        super().__init__(func, *args, **kws)

    def __call__(self, figure, tab, *args, **kws):
        # Call signature for tasks requires: figure, tab

        # Fill dynamic parameter values
        nreq = self.nreq
        if nreq:
            args = (*self._get_args((figure, *([tab] * (nreq == 2)))), *args)
            kws = self._get_kws(kws)

        if self._keywords:
            args = (*self.args, *args)
            kws = {**self._get_kws({list(self._keywords).pop(): figure}), **kws}

        # Now run the task
        func = self.__wrapped__
        self.logger.opt(lazy=True).info(
            'Invoking call for plot task:\n>>> {}.',
            lambda: indent(callers.pformat(func, args, kws, ppl=1))
        )

        # plot
        art = func(*args, **kws)

        if not (self.nreq or self._keywords):
            # We will have generated a figure to fill the tab in the ui, we have
            # to replace it after the task executes with the actual figure we
            # want in our tab.
            figure = art.fig
            mgr = self.ui[tuple(tab)]._parent()
            mgr.replace_tab(tab[-1], figure, focus=False)

        return figure, art


class TabTask(slots.SlotHelper, LoggingMixin):

    __slots__ = (
        'ui', 'task', 'tab', 'filenames', 'result', 'overwrite', 'save_kws'
    )

    def __init__(self, ui, task, tab, filenames=(), overwrite=False, **save_kws):
        # `task` is TaskRunner instance
        self.logger.debug('Creating {0.__name__} for {1}.', type(self), task)

        if filenames:
            filenames = ensure.tuple(filenames, Path)
            self.logger.debug('Figure for task {.task.__wrapped__.__name__!r} '
                              'will be saved at {}. {}.',
                              task, filenames, f'{overwrite = }')
        # init namespace
        super().__init__(**slots.sanitize(locals()), result=())

    def __call__(self, figure, tab, *args, **kws):
        # Execute task
        self.logger.opt(lazy=True).info(
            'Plotting tab {0[0]} with {0[1]} at figure: {0[2]}.',
            lambda: (self.ui.tabs.tab_text(tab), callers.describe(self.task), figure)
        )

        # run
        figure, self.result = self.task(figure, tab, *args, **kws)

        # save
        self.save(figure)

        # art
        return self.result

    def get_figure(self, figure=None, figsize=None, add_axes=False, **kws):

        #
        figure = get_figure(self.ui, self.tab, figure, figsize=figsize, **kws)

        if add_axes:
            if figure.axes:
                self.logger.warning('Ignoring `add_axes=True` since figure not empty.')
            else:
                figure.add_subplot()

        # resize if requested (normally handled by ui)
        if not self.ui:
            if figsize:
                self.logger.info('Resizing figure: {}', figsize)
                figure.set_size_inches(figsize)

        return figure
    
    def save(self, figure):
        save_figure(figure, self.filenames, self.overwrite, **self.save_kws)


class GUI(MplMultiTab):

    def __init__(self, title, pos,
                 active=cfg.plotting.gui.active,
                 delay=cfg.plotting.gui.delay):
        #
        super().__init__((), title, pos)

        # self.setWindowIcon(QtGui.QIcon('/home/hannes/Pictures/mCV.jpg'))

        self.active = bool(active)
        self.delay = active and delay

        #
        self.task_factory = TaskFactory(self)
        # creates TaskRunner when called

    def __bool__(self):
        return self.active

    def add_task(self, task, tab,
                 filenames=(), overwrite=False,
                 figure=None, add_axes=False,
                 *args, **kws):

        self.logger.info('Adding task for tab {}:\n{}', tab, task)

        # set ui
        task.ui = self

        # Task requires Figure
        # next line will generate figure to fill the tab, we have to replace it
        # after the task executes with the actual figure we want in our tab
        if filename := kws.pop('filename', ()):
            filenames = (filename, *ensure.tuple(filenames, Path))

        tab_task = TabTask(self, task, tab, filenames, overwrite)
        figure = tab_task.get_figure(figure, add_axes=add_axes, **task.fig_kws)

        if self.delay:
            # Future task
            self.logger.info('Plotting delayed: Adding plot callback for {}: {}.',
                             tab, tab_task)

            self[tab].add_task(tab_task, *args, **kws)
            return tab_task

        # execute task
        self.logger.debug('Plotting immediate: {}.', tab)
        tab_task(figure, tab, *args, **kws)
        return tab_task

    def show(self):
        super().add_task()   # needed to check for tab switch callbacks to run
        return super().show()
