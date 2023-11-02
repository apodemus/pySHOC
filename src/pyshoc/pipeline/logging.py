"""
Logging config for pyshoc pipeline.
"""


# std
import sys
import warnings
from functools import partialmethod

# third-party
import better_exceptions as bx
from loguru import logger

# local
import motley
from recipes.misc import get_terminal_size
from recipes.logging import RepeatMessageHandler, TimeDeltaFormatter

# relative
from .. import CONFIG


# ---------------------------------------------------------------------------- #
# Capture warnings
# _showwarning = warnings.showwarning


def _showwarning(message, *_, **__):
    logger.opt(depth=2).warning(message)
    # _showwarning(message, *args, **kwargs)


warnings.showwarning = _showwarning


# ---------------------------------------------------------------------------- #
# Log levels

def markup_to_list(tags):
    """convert html tags eg "<red><bold>" to comma separated list "red,bold"."""
    return tags.strip('<>').replace('><', ',')


level_formats = {
    level.name: motley.stylize(CONFIG.logging.format,
                               level=level,
                               style=markup_to_list(level.color))
    for level in logger._core.levels.values()
}

# custom level for sectioning
level_formats['SECTION'] = motley.stylize(CONFIG.logging.section, '',
                                          width=get_terminal_size()[0])
logger.level('SECTION', no=15)
Logger = type(logger)
Logger.section = partialmethod(Logger.log, 'SECTION')


# ---------------------------------------------------------------------------- #


def patch(record):
    # dynamic formatting tweaks
    set_elapsed_time_hms(record)
    escape_module(record)


def set_elapsed_time_hms(record):
    # format elapsed time
    record['elapsed'] = TimeDeltaFormatter(record['elapsed'])


def escape_module(record):
    """This prevents loguru from trying to parse <module> as an html tag."""
    if record['function'] == '<module>':
        # '\N{SINGLE LEFT-POINTING ANGLE QUOTATION MARK}'
        # '\N{SINGLE RIGHT-POINTING ANGLE QUOTATION MARK}'
        record['function'] = '‹module›'


def formatter(record):
    # {time:YYYY-MM-DD HH:mm:ss zz}
    format_string = level_formats[record['level'].name]
    if record['exception']:
        format_string += '\n{exception}'
        # record['exception'] = format_exception(record['exception'])

    format_string += '\n'

    # If we format the `message` here, loguru will try format a second time,
    # which is usually fine, except when the message contains braces (eg dict as
    # str), in which case it fails.
    # FIXME: just escape the braces. Use color=False
    # record['message'] = '{message}'
    # motley.format_partial(record['message']) #

    return motley.format(format_string, **{**record, 'message': '{message}'})


def format_exception(exc_info=None):
    return '\n'.join(bx.format_exception(*(exc_info or sys.exc_info())))

# ---------------------------------------------------------------------------- #
# Configure log sinks


def config():
    # logger config
    # logger.level('DEBUG', color='<black><bold>')

    # formatter = motley.stylize(
    #     '{elapsed:s|Bb}|'
    #     '{{{name}.{function}:|green}:{line:d|orange}: <52}|'
    #     '<level>{level}: {message}</>'
    # )

    console_sink = RepeatMessageHandler()
    console_sink._template = motley.apply(console_sink._template, *'kB')

    logger.configure(
        handlers=[
            # console handler
            dict(sink=console_sink,
                 level='DEBUG',
                 catch=False,
                 colorize=False,
                 format=formatter,
                 ),

            # File handler
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
        # format the log messages and will thus recurse infinitely.
        activation=[('pyshoc', True),
                    ('obstools', True),
                    ('recipes', True),
                    ('motley.formatter', False)]
    )

    return logger

# ---------------------------------------------------------------------------- #

# @dataclass
# class ConditionalString:
#     s: str = ''

#     def __or__(self, n):
#         return 's' if n > 1 else ''
