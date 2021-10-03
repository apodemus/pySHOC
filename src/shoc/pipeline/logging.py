"""
Logging config for pyshoc pipeline
"""

# std
import sys
import atexit
import warnings

# third-party
from loguru import logger

# local
import motley
from recipes.pprint.nrs import hms


class FilterRepeatsHelper:
    """Helper class for filtering repeat log messages."""

    pass_next = True

    def __call__(self, record):
        return self.pass_next


class RepeatMessageFilter:
    """
    Patch that filters repeat log messages.
    """

    # previous record
    previous = None
    n_repeats = 1

    # filter func
    filter = FilterRepeatsHelper()

    def __init__(self, handler_id=0):
        self.handler_id = handler_id
        atexit.register(self.close)

    def __call__(self, record):
        previous = self.previous
        repeats = bool(previous) and (record['message'] == previous['message'])
        self.filter.pass_next = not repeats
        self.n_repeats += repeats

        if not repeats and self.n_repeats > 1:
            self.flush()

        self.previous = record
    
    @property
    def handler(self):
        return logger._core.handlers[self.handler_id]
    
    def flush(self):
        previous = self.previous
        previous['message'] = '[Previous message repeats ×{}]\n'.format(
            self.n_repeats)
        self.filter.pass_next = True
        self.handler.emit(previous, previous['level'], False, True, None)
        # reset
        self.n_repeats = 1

    def close(self):
        if self.previous and self.n_repeats > 1:
            self.flush()


log_level_styles = {lvl: obj.color.strip('<>').replace('><', ',')
                    for lvl, obj in logger._core.levels.items()}


def formatter(record):
    # {time:YYYY-MM-DD HH:mm:ss zz}
    format_string = (
        '{elapsed:s|Bb}|'
        '{{{name}.{function}:|green}:{line:d|orange}: <52}|'
        '{{level}: {message}:|{style}}'
    )
    if record['exception']:
        format_string += '{exception}'
    format_string += '\n'

    return motley.format(
        format_string,
        **{**record, **dict(
            style=log_level_styles[record['level'].name],
            elapsed=hms(record['elapsed'].total_seconds(), 1, unicode=True)
        )}
    )

# ---------------------------------------------------------------------------- #
# Capture warnings
# _showwarning = warnings.showwarning


def _showwarning(message, *_, **__Z):
    logger.opt(depth=1).warning(message)
    # _showwarning(message, *args, **kwargs)


warnings.showwarning = _showwarning

# ---------------------------------------------------------------------------- #


def config(path):
    # logger config
    # logger.level('DEBUG', color='<black><bold>')

    logger.configure(
        handlers=[
            # console logger
            dict(sink=sys.stdout,
                 level='DEBUG',
                 catch=True,
                 colorize=True,
                 format=formatter,
                 filter=RepeatMessageFilter.filter,
                 ),

            # File logger
            dict(sink=path,
                 # serialize= True
                 level='DEBUG',
                 format=formatter
                 ),
        ],
        # "extra": {"user": "someone"}
        patcher=RepeatMessageFilter()
    )

    return logger
