"""
Logging config for pyshoc pipeline.
"""


# std
import warnings
from pathlib import Path

# third-party
from loguru import logger

# local
import motley
from recipes.shell import terminal
from recipes.logging import (ParagraphWrapper, RepeatMessageHandler,
                             TimeDeltaFormatter)

# relative
from .. import config as cfg


# ---------------------------------------------------------------------------- #
# Capture warnings

def _showwarning(message, *_, **__):
    logger.opt(depth=2).warning(message)


warnings.showwarning = _showwarning


# ---------------------------------------------------------------------------- #
# Log levels

def markup_to_list(tags):
    """convert html tags eg "<red><bold>" to comma separated list "red,bold"."""
    return tags.strip('<>').replace('><', ',')


# custom level for sectioning
def section(message):
    # bind(indent=False)
    logger.log('SECTION', message, depth=1)


# Update levels
logger.level('SECTION', no=20)
logger.__class__.section = staticmethod(section)

console = cfg.logging.console
for level, icon in console.icons.items():
    logger.level(level.upper(), icon=icon)

LEVEL_FORMATS = {
    # generic level format
    **{level.name: motley.stylize(console.format,
                                  level=level,
                                  style=markup_to_list(level.color))
       for level in logger._core.levels.values()},
    # custom level for sectioning
    'SECTION':      motley.stylize(console.section, '',
                                   width=terminal.get_size()[0])
}


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
    # If we format the `message` here, loguru will try format a second time,
    # which is usually fine, except when the message contains braces (eg dict as
    # str), in which case it fails.

    return motley.format(f'{LEVEL_FORMATS[record["level"].name]}\n',
                         **{**record, 'message': '{message}'})


def cleanup(logfile):
    logfile = Path(logfile)
    logfile.write_text(motley.ansi.strip(logfile.read_text()))


# ---------------------------------------------------------------------------- #
# Configure log sinks

def config():
    # logger config
    config = cfg.logging
    return logger.configure(
        # disable logging for motley.formatter, since it is being used here to
        # format the log messages and will thus recurse infinitely.
        activation=list(config.activation.items()),

        # File sink is added by pipeline.cli.setup once output path is known
        handlers=[{
            # console handler
            'sink':
                RepeatMessageHandler(
                    ParagraphWrapper(indent=f'{"⎢ ": >66}'),
                    template=config.console.repeats
                ),
            'level':    config.console.level,
            'catch':    config.console.catch,
            'colorize': False,
            'format':   formatter,
            # 'filter':   filter_console
            # One can also pass a dict mapping module names to minimum
            # required level. In such case, each log record will search for
            # it’s closest parent in the dict and use the associated level
            # as the filter. The dict values can be int severity, str level
            # name or True and False to respectively authorize and discard
            # all module logs unconditionally. In order to set a default
            # level, the "" module name should be used as it is the parent
            # of all modules (it does not suppress global level threshold,
            # though).
        }],

        # "extra": {"user": "someone"}
        patcher=patch,
    )


# ---------------------------------------------------------------------------- #

# @dataclass
# class ConditionalString:
#     s: str = ''

#     def __or__(self, n):
#         return 's' if n > 1 else ''
