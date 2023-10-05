
# third-party
from loguru import logger
from matplotlib.figure import Figure

# local
from recipes.functionals.partial import Partial, PartialAt

# relative
from .. import CONFIG
from .logging import logger
from .utils import get_figure, save_fig


class _PlotTask(PartialAt):

    def __wrapper__(self, figure, keys, filename=None, overwrite=False):
        # data[keys]

        logger.info('Plotting tab {} with {}. filename = {}, overwrite = {}:',
                    keys, self.func.__name__, filename, overwrite)

        # plot
        art = self.func(figure, *self.args, **self.kws)

        # save
        save_fig(figure, filename, overwrite)

        return art


class PlotTask(Partial):  # PlotWrapper
    
    factory = _PlotTask

    def __init__(self, ui=None, keys=(), delay=CONFIG.plotting.gui.delay,
                 filename=None, overwrite=False):

        self.ui = ui
        self.keys = keys
        self.delay = (ui is not None) and delay
        self.overwrite = overwrite
        self.filename = filename

    def __wrapper__(self, func, *args, **kws):
        
        callback = super().__wrapper__(func, *args, filename=self.filename,
                                       overwrite=self.overwrite, **kws)

        if not (fig := next((_ for _ in args if isinstance(_, Figure)), None)):
            fig = get_figure(self.ui, *self.keys)

        if self.delay:
            logger.info('Plotting delayed: Adding plot callback for {}: {}.',
                        self.keys, callback)

            self.ui[self.keys].add_callback(callback)
            return callback

        logger.debug('Plotting immediate: {}.', self.keys)
        return callback(fig, *self.keys)
