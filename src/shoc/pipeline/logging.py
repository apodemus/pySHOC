"""
Logging config for pyshoc pipeline.
"""


# std
import sys
import atexit
import warnings

# third-party
import better_exceptions as bx
from loguru import logger

# local
import motley
from recipes.pprint.nrs import hms


def markup_to_list(tags):
    """convert html tags eg "<red><bold>" to comma separated list "red,bold"."""
    return tags.strip('<>').replace('><', ',')


fmt = ('{elapsed:s|Bb}|{{{name}.{function}:|green}:{line:d|orange}: <52}|'
       '{{level.name}: {message}:|{style}}'
       # '{exception:?}',
       )
level_formats = {level.name: motley.stylize(
    fmt, level=level, style=markup_to_list(level.color)
) for level in logger._core.levels.values()
}


class RepeatMessageHandler:
    """
    A loguru sink that filters repeat log messages and instead emits a 
    configurable summary message.
    """

    def __init__(self, target=sys.stderr, template=motley.stylize(
            '{[Previous message repeats ×{repeats}]:|kB}\n')):
        self._target = target
        self._previous_args = None
        self._repeats = 0
        self._template = str(template)
        atexit.register(self._write_repeats)

    def write(self, message):
        args = (message.record['message'], message.record['level'].no)
        if self._previous_args == args:
            self._repeats += 1
            return

        self._write_repeats()

        self._target.write(message)
        self._repeats = 0
        self._previous_args = args

    def _write_repeats(self):
        if self._repeats > 0:
            self._target.write(self._template.format(repeats=self._repeats))


class TimeDeltaFormatter:
    """
    Helper for printing elapsed time in hms format eg: 00ʰ02ᵐ33.2ˢ
    """

    def __init__(self, timedelta, **kws):
        self.timedelta = timedelta
        self._kws = {**kws,
                     # defaults
                     **dict(precision=1,
                            unicode=True)}

    def __format__(self, spec):
        return hms(self.timedelta.total_seconds(), **self._kws)


def patch(record):
    set_elapsed_time_hms(record)
    escape_module(record)


def set_elapsed_time_hms(record):
    record['elapsed'] = TimeDeltaFormatter(record['elapsed'])


# if is_interactive():
def escape_module(record):
    """This prevents loguru from trying to parse <module> as an html tag."""
    if record['function'] == '<module>':
        record['function'] = r'\<module>'
# else:
#     escape_module = noop

# @ftl.lru_cache()
# def stylize(format_string, level):
#     return motley.format_partial(format_string,
#                                  level=level,
#                                  style=log_level_styles[level.name])


def formatter(record):
    # {time:YYYY-MM-DD HH:mm:ss zz}
    format_string = level_formats[record['level'].name]
    if record['exception']:
        format_string += '\n{exception}'
        # record['exception'] = format_exception(record['exception'])

    format_string += '\n'

    # If we format the message here, loguru will try format a second time, which
    # is usually fine, except when the message contains braces (eg dict as str),
    # in which case it fails.
    # record['message'] = '{message}'
    # motley.format_partial(record['message']) # 
    return motley.format(format_string, **{**record, 'message':'{message}'}) 


def format_exception(exc_info=None):
    return '\n'.join(bx.format_exception(*(exc_info or sys.exc_info())))

# ---------------------------------------------------------------------------- #
# Capture warnings
# _showwarning = warnings.showwarning


def _showwarning(message, *_, **__):
    logger.opt(depth=2).warning(message)
    # _showwarning(message, *args, **kwargs)


warnings.showwarning = _showwarning

# ---------------------------------------------------------------------------- #


def config():
    # logger config
    # logger.level('DEBUG', color='<black><bold>')

    # formatter = motley.stylize(
    #     '{elapsed:s|Bb}|'
    #     '{{{name}.{function}:|green}:{line:d|orange}: <52}|'
    #     '<level>{level}: {message}</>'
    # )

    logger.configure(
        handlers=[
            # console logger
            dict(sink=RepeatMessageHandler(),
                 level='DEBUG',
                 catch=False,
                 colorize=False,
                 format=formatter,
                 ),

            # File logger
            # dict(sink=path,
            #      # serialize= True
            #      level='DEBUG',
            #      format=formatter,
            #      colorize=False,
            #      ),
        ],
        
        # "extra": {"user": "someone"}
        patcher=patch,
        
        # disable logging for motley.formatter, since it is being used here to
        # format the log messages and will thus recurse infinitely
        activation=[('obstools', True),
                    ('recipes', True),
                    ('motley.formatter', False)]
    )
    
    return logger
