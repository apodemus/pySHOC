"""
Logging config for pyshoc pipeline.
"""


# std
import sys
import time
import atexit
import warnings
from dataclasses import dataclass

# third-party
import better_exceptions as bx
from loguru import logger

# local
import motley
from recipes.pprint.nrs import hms


def markup_to_list(tags):
    """convert html tags eg "<red><bold>" to comma separated list "red,bold"."""
    return tags.strip('<>').replace('><', ',')


fmt = ('{elapsed:s|Bb}|'
       '{{{name}.{function}:s|green}:{line:d|orange}: <52}|'
       '{{level.name}: {message}:|{style}}')
level_formats = {
    level.name: motley.stylize(fmt,
                               level=level,
                               style=markup_to_list(level.color))
    for level in logger._core.levels.values()
}


# @dataclass
# class ConditionalString:
#     s: str = ''

#     def __or__(self, n):
#         return 's' if n > 1 else ''


class RepeatMessageHandler:
    """
    A loguru sink that filters repeat log messages and instead emits a 
    custom summary message.
    """

    _keys = (
        # 'file',
        'function', 'line',
        'message',
        'exception', 'extra'
    )

    def __init__(self,
                 target=sys.stderr,
                 template=motley.stylize(
                     '{ ⤷ [Previous {n_messages} {n_repeats} in {t}]:|kB}\n'),
                 x='×',
                 xn=' {x}{n:d}',
                 buffer_size=12):

        self._target = target
        self._repeats = 0
        self._repeating = None
        self._template = str(template)
        self._x = str(x)
        self._xn = str(xn)
        self._memory = []
        self.buffer_size = int(buffer_size)
        self._timestamp = None

        atexit.register(self._write_repeats)

    def write(self, message):
        #
        args = (message.record['level'].no, message.record['file'].path,
                *(message.record[k] for k in self._keys))
        if args in self._memory:  # if self._previous_args == args:
            if self._repeats:  # multiple consecutive messages repeat
                idx = self._memory.index(args)
                if idx == 0:
                    self._repeats += 1
                    self._repeating = 0
                elif idx == (self._repeating + 1):
                    self._repeating = idx
                else:
                    # out of sequence, flush
                    self._flush()
            else:
                # drop all previous unique messages
                self._memory = self._memory[self._memory.index(args):]
                self._repeating = 0
                self._repeats += 1
                self._timestamp = time.time()

            return

        # add to buffered memory
        if self._repeats:
            # done repeating, write summary of repeats, flush memory
            self._flush()

        self._memory.append(args)
        if len(self._memory) > self.buffer_size:
            self._memory.pop(0)

        self._target.write(message)

    def _flush(self):
        self._write_repeats()

        self._memory = []
        self._repeats = 0
        self._repeating = None

    def _write_repeats(self):
        if self._repeats == 0:
            return

        # xn = #('' if self._repeats == 1 else
        xn = self._xn.format(x=self._x, n=self._repeats + 1)

        # {i} message{s|i} repeat{s|~i}{xn}
        i = len(self._memory) - 1
        n_messages = f'{f"{i + 1} " if (many := i > 1) else ""}message{"s" * many}'
        n_repeats = f'repeat{"s" * (not many)}{xn}'
        t = hms(time.time() - self._timestamp, precision=3, short=True, unicode=True)
        self._target.write(motley.format(self._template, **locals()))


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
            # console handler
            dict(sink=RepeatMessageHandler(),
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
