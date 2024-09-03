
# std
import sys
from pathlib import Path

# third-party
from matplotlib.figure import Figure
from mpl_multitab import MplMultiTab

# local
from scrawl.utils import save_figure
from recipes.oo import slots
from recipes import dicts, op
from recipes.flow import Catch
from recipes.iter import cofilter
from recipes.string import indent
from recipes.pprint import callers
from recipes.containers import ensure
from recipes.functionals import negate
from recipes.logging import LoggingMixin
from recipes.pprint.callers import Callable
from recipes.functionals.partial import Partial, PartialTask

# relative
from .. import config as cfg
from .logging import logger


# ---------------------------------------------------------------------------- #
cfg = cfg.plotting
FIG_KWS = Figure.__init__.__code__.co_varnames[1:-2]
SAVE_KWS = ('filename', 'overwrite', 'dpi')

# ---------------------------------------------------------------------------- #


class PlotManager(LoggingMixin):

    def __init__(self, plot=True, gui=cfg.gui.active, **kws):

        self.active = bool(plot)
        self.tasks = dicts.DictNode()
        self.figures = dicts.DictNode()

        # GUI
        self.gui = GUI(**kws, active=gui) if gui else None

        #
        self.task_factory = TaskFactory(self)

    def should_plot(self, paths, overwrite):
        if not self.active:
            return False

        return overwrite or any(not p.exists() for p in paths) or self.gui

    def get_figure(self, tab, fig, replace=False, **kws):

        # Create figure if needed, add to ui tab if active
        if self.gui:
            if tab:
                do = ('add', 'replace')[replace]
                caller = getattr(self.gui, f'{do}_tab')
                tab = caller(tab, fig=fig, **kws)
                return tab.figure

            logger.debug('GUI active, but no tab. Figure will not be embedded.')

        if fig:
            assert isinstance(fig, Figure)
            return fig

        if plt := sys.modules.get('matplotlib.pyplot'):
            logger.debug('pyplot active, launching figure {}.', tab)
            return plt.figure(**kws)

        logger.debug('No GUI, creating figure. {}.', tab)
        return Figure(**kws)

    def add_task(self, task, tab,
                 filenames=(), overwrite=False,
                 figure=None, add_axes=False, replace=False,
                 *args, **kws):

        # check if plotting is active
        if not self.active:
            self.logger.debug('Plotting deactivated, ignoring task: {}', tab)
            return

        # resolve filenames
        filenames = ensure.tuple(filenames, Path)
        if filename := kws.pop('filename', ()):
            filenames = (Path(filename), *filenames)

        # check if plot needed
        if not self.should_plot(filenames, overwrite):
            self.logger.debug('Not plotting {} since GUI inactive, files exist,'
                              ' and not overwriting.', tab)
            return

        #
        self.logger.info('Adding plotting task for {} {}:\n{}',
                         ('section', 'tab')[bool(self.gui)], tab, task)

        # create the task
        fig_kws = task.fig_kws
        task = PlotTask(self, task, tab, filenames, overwrite)
        self.tasks[tab] = task

        # next line will generate figure to fill the tab, we have to replace it
        # after the task executes with the actual figure we want in our tab
        figure = task.get_figure(figure, add_axes=add_axes, replace=replace,
                                 **fig_kws)
        self.figures[tab] = figure

        if self.gui and self.gui.delay:
            # Future task
            self.logger.info('Plotting delayed: Adding plot callback for {}: {}.',
                             tab, task)

            self.gui[tab].add_task(task, *args, **kws)
            return task

        # execute task
        self.logger.debug('Plotting immediate: {}.', tab)
        task(figure, tab, *args, **kws)
        return task

    def save(self, key, filename, overwrite=False, **kws):
        save_figure(self.figures[key], filename, overwrite, **kws)


class GUI(MplMultiTab):

    def __init__(self, title, pos, active=cfg.gui.active, delay=cfg.gui.delay):
        #
        super().__init__((), title, pos)

        # self.setWindowIcon(QtGui.QIcon('/home/hannes/Pictures/mCV.jpg'))
        self.active = bool(active)
        self.delay = active and delay

    def __bool__(self):
        return self.active

    def show(self):
        super().add_task()   # needed to check for tab switch callbacks to run
        return super().show()

    def launch(self):

        logger.section('Launching GUI')

        # activate tab switching callback (for all tabs)
        config = cfg.registration
        if config.mosaic.show:
            survey = config.alignment.survey.survey.upper()
            self['Overview'].move_tab('Mosaic', 0)
            self['Overview', 'Mosaic'].move_tab(survey, 0)
            self.set_focus(*config.mosaic.tab, survey)

        self.show()

    def shutdown(self):

        logger.section('GUI shutdown')

        # Run incomplete plotting tasks
        trap = Catch(action=logger.warning,
                     message=('Could not complete plotting task '
                              'due the following {err.__class__.__name__}: {err}'))
        getter = op.AttrVector('plot.func.filename', default=None)
        tabs = list(self.tabs._leaves())
        filenames, tabs = cofilter(getter.map(tabs), tabs)
        unsaved, tabs = map(list, cofilter(negate(Path.exists), filenames, tabs))
        if n := len(unsaved):
            logger.info('Now running {} incomplete tasks:', n)
            with trap:
                for tab in tabs:
                    tab.run_task()


class TaskFactory(Partial, LoggingMixin):

    def __init__(self, mgr):
        self.manager = mgr

    def __wrapper__(self, func, *args, **kws):

        # resolve placholders and static params
        task = PlotTaskWrapper(self.manager.gui, func, *args, **kws)

        # create TaskRunner, which will run PlotTask when called
        return TaskRunner(task, **task.fig_kws)


class _TaskBase(PartialTask, LoggingMixin):
    """Base class for Tasks."""


class TaskRunner(_TaskBase):
    # Run task when called

    def __init__(self, task, **fig_kws):
        # self.ui = None  # set in `GUI.add_task`
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


# ---------------------------------------------------------------------------- #


class PlotTaskWrapper(_TaskBase):
    """
    This class wraps the plotting task. It handles the figure initialization and
    will ensure the correct keywords get passed to the figure creation and
    plotting methods. Also handles the case for plot methods that create their 
    own figures.
    """

    def __init__(self, gui, func, *args, **kws):
        #
        self.gui = gui
        # split keywords for figure init
        kws, self.fig_kws = dicts.split(kws, FIG_KWS)
        kws, self.save_kws = dicts.split(kws, SAVE_KWS)

        if 'filename' in Callable(func).sig.parameters:
            kws['filename'] = self.save_kws['filename']

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

        # This part is needed for GUI embedded tasks that generate their own
        # figures
        if not (self.nreq or self._keywords):
            # We will have generated a figure to fill the tab in the ui, we have
            # to replace it after the task executes with the actual figure we
            # want in our tab.
            figure = art.fig
            if gui := self.gui:
                mgr = gui[tuple(tab)]._parent()
                mgr.replace_tab(tab[-1], figure, focus=False)

        return figure, art


class PlotTask(slots.SlotHelper, LoggingMixin):

    __slots__ = (
        'manager', 'task', 'tab', 'filenames', 'result', 'overwrite', 'save_kws'
    )

    def __init__(self, manager, task, tab, filenames=(), overwrite=False, **save_kws):
        #
        self.logger.debug('Creating {0.__name__} for {1}.', type(self), task)

        if filenames:
            filenames = ensure.tuple(filenames, Path)

            self.logger.debug(
                'Figure for task {!r} will be saved at {}. overwrite = {}.',
                callers.get_name(task.task.__wrapped__, 1), filenames, overwrite
            )

        # init namespace
        super().__init__(**slots.sanitize(locals()), result=())

    def __call__(self, figure, tab, *args, **kws):
        # Execute task
        self.logger.opt(lazy=True).info(
            'Plotting tab {0[0]} with callable {0[1]} at figure: {0[2]}.',
            lambda: (tab, callers.describe(self.task), figure)
        )

        # run
        figure, self.result = self.task(figure, tab, *args, **kws)

        # save
        self.save(figure)

        # art
        return self.result

    def get_figure(self, figure=None, figsize=None, add_axes=False,
                   replace=False, **kws):

        #
        figure = self.manager.get_figure(self.tab, figure, replace, figsize=figsize, **kws)

        if add_axes:
            if figure.axes:
                self.logger.warning('Ignoring `add_axes=True` since figure not empty.')
            else:
                figure.add_subplot()

        # resize if requested (normally handled by ui)
        if not self.manager.gui and figsize:
            self.logger.info('Resizing figure: {}', figsize)
            figure.set_size_inches(figsize)

        return figure

    def save(self, figure):
        save_figure(figure, self.filenames, self.overwrite, **self.save_kws)


# class TabTask(PlotTask):
#     pass

    # def __call__(self, figure, tab, *args, **kws):
    #     # Execute task
    #     self.logger.opt(lazy=True).info(
    #         'Plotting tab {0[0]} with callable {0[1]} at figure: {0[2]}.',
    #         lambda: (self.ui.tabs.tab_text(tab),
    #                  callers.describe(self.task),
    #                  figure)
    #     )

    #     # run
    #     figure, self.result = self.task(figure, tab, *args, **kws)

    #     # save
    #     self.save(figure)

    #     # art
    #     return self.result
