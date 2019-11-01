"""
Tools for generating output file names for SHOC observations
"""

import re
import os

import more_itertools as mit

from recipes.containers.set_ import OrderedSet


def adapt_path(base, path, extension, output_type):
    return output_type(os.path.join(path, base + extension))


def unique_filenames(run, path='', extension='.fits', pattern=None,
                     output_type=str):
    if pattern is None:
        pattern = run.nameFormat
    #
    naming = NamingConvention(pattern)
    path = str(path)
    if not extension.startswith('.'):
        extension = '.%s' % extension

    return naming.unique(run, path, extension, output_type)


class NamingConvention(object):
    """
    Implements a very basic mini language to name multiple files in a sequence
    using the shortest possible unique names. Uniqueness is determined by a
    set of keys that map to attributes of the shocRun
    """

    # re pattern matchers
    optPattern = '<[^\>]+>'
    optMatcher = re.compile(optPattern)
    # matches the optional keys sections (including <> brackets) in the
    # format specifier string from the args.names namespace
    keyPattern = '\{(\w+)\}'
    keyMatcher = re.compile(keyPattern)

    # matches the key (including curly brackets) and key (excluding curly
    # brackets) for each section of the format string

    def __init__(self, pattern):
        self.pattern = pattern
        self.keys = self.keyMatcher.findall(pattern)
        self.keySet = OrderedSet(self.keys) - {'sep'}

        self.optMatches = self.optMatcher.findall(self.pattern)
        optKeys = mit.flatten(map(self.keyMatcher.findall, self.optMatches))
        self.optKeySet = OrderedSet(optKeys) - {'sep'}
        self.optParts = dict(zip(self.optKeySet, self.optMatches))
        self.reqKeySet = self.keySet - self.optKeySet

        self.has_opt = bool(len(self.optKeySet))
        if self.has_opt:
            self.base = pattern[:pattern.index('<')]
        else:
            self.base = self.pattern

    # def format(self, **kws):
    #     self.pattern.format(kws)

    def unique(self, run, path=None, extension='.fits', output_type=str):
        """
        Generates a unique sequence of file names for observation cubes
        in a run based on the provided pattern and the unique attributes of
        those data cubes.
        """

        if not extension.startswith('.'):
            extension = '.%s' % extension

        # if only a single observation in the run, can simply use required keys
        # to create file name
        if len(run) == 1:
            cube = run[0]
            if path is None:
                # output path will be same as input path per file
                path, _ = os.path.split(cube.filename())

            filename = self.base.format(**cube.get_attr_repr())
            return [adapt_path(filename, path, extension, output_type)]

        filenames = []
        for keys, group in run.group_by(*self.reqKeySet).items():
            use_opt = []
            nr_flag = False
            if len(group) > 1:
                # check which optional keys to add for uniqueness
                for opt, partFmt in self.optParts.items():
                    if group.varies_by(opt):
                        use_opt.append(partFmt)
                        break

                # none of the optional keys help in uniquely identifying the
                # files. use numbers instead.
                nr_flag = len(use_opt) == 0

            for i, cube in enumerate(group):
                fmt = self.base + ''.join(use_opt)
                fmt = fmt.replace('<', '').replace('>', '')  # remove brackets
                if nr_flag:
                    fmt += '{sep}%i' % i

                if path is None:
                    # output path will be same as input path per file
                    path, _ = os.path.split(cube.filename())

                filename = fmt.format(**cube.get_attr_repr())
                filename = adapt_path(filename, path, extension, output_type)
                filenames.append(filename)

        return filenames


class FilenameGenerator(object):
    """
    Used to generate sequential file names with given pattern
    """

    def __init__(self, basename, path='', padwidth=0, start=1, sep='.',
                 extension='.fits', output_type=str):
        self._count = 0
        self.start = int(start)
        self.basename = basename
        self.path = path
        self.padwidth = padwidth
        self.sep = sep
        self.extension = extension
        self.output_type = output_type

    def __call__(self, n=None, **kws):
        """
        Generator of file names of unpacked cube.

        Parameters
        ----------
        n: int or None
            The number of file names that will be generated.  If omitted (None)
            (the default) or 0, a single filename without a number extension
            will be returned.
            If `n` is a positive integer, a generator that will yield `n`
            file names in sequence is returned.

        kws

        Yields
        ------
        filename : str
            The next filename in the sequence. e.g. darkstar.0013.fits

        Examples
        --------
        >>> g = FilenameGenerator('darkstar')
        >>> g()
        'darkstar.fits'
        >>> list(g(2, start=0))
        ['darkstar.0.fits', 'darkstar.1.fits']
        >>> next(g(1))
        'darkstar.1.fits'
        >>> list(g(3, padwidth=4))
        ['darkstar.0001.fits', 'darkstar.0002.fits', 'darkstar.0003.fits']
        """

        path = kws.get('path', self.path)
        extension = kws.get('extension', self.extension)
        base = os.path.join(path, self.basename)

        if (n is None) or (n == 0):
            return self.output_type(base + extension)

        # If we got here, create generator
        assert int(n) >= 0, '`n` should be a non-negative integer'
        return self._iter(n, **kws)

    def _iter(self, n, **kws):

        path = kws.get('path', self.path)
        base = os.path.join(path, self.basename)
        extension = kws.get('extension', self.extension)
        sep = kws.get('extension', self.sep)
        padwidth = kws.get('padwidth') or 0
        start = kws.get('start', self.start)

        n = int(n)
        padwidth = max(padwidth, len(str(n)))

        for i in range(n):
            current = start + i
            # image number string. eg: '0013'
            nr = '{1:0>{0}}'.format(padwidth, current)
            # name string eg. 'darkstar.0013.fits'
            name = ''.join((base, sep, nr, extension))
            yield self.output_type(name)


if __name__ == '__main__':
    g = FilenameGenerator('darkstar')
    print(
            list(g(2, start=0))
    )
    print(
            list(g(1))
    )
    print(
            list(g(3, padwidth=4))
    )

    namer = NamingConvention(
            '{kind}{date}{sep}{binning}<{sep}m{mode}><{sep}t{kct}>'
    )
