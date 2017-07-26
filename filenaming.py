"""
A very basic mini language to name multiple files in a sequence using the
shortest possible unique names.  Uniqueness is determined by a set of keys

"""

import re
import os

class FilenameGenerator(object):
    """
    Used to generate sequential filenames with given pattern
    """
    def __init__(self, basename, path='', padwidth=0, start=1, sep='.',
                 extension='.fits'):
        self._count = 0
        self.start = int(start)
        self.basename = basename
        self.path = path
        self.padwidth = padwidth
        self.sep = sep
        self.extension = extension


    def __call__(self, n=None, **kws):
        """
        Generator of filenames of unpacked cube.

        Parameters
        ----------
        n : int or None
            The number of filenames that will be generated.  If (None)
            a single filename without a number extention will be yielded.
            If a positive integer, that number of

        kw

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

        if (n is None) or (n < 1):
            return ''.join((base, extension))
        return self._iter(n, **kws)

        # elif isinstance(n, int):
        #     return self._iter(n, **kws)
        # else:
        #     raise TypeError('Invalid number %s' %n)


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
            imnr = '{1:0>{0}}'.format(padwidth, current)  # image number string. eg: '0013'
            outname = ''.join((base, sep, imnr, extension))  # name string eg. 'darkstar.0013.fits'
            yield outname



class NamingConvention():

    # re pattern matchers
    optPattern = '<[^\>]+>'
    optMatcher = re.compile(optPattern)
    # matches the optional keys sections (including <> brackets) in the
    # format specifier string from the args.names namespace
    keyPattern = '\{(\w+)\}'
    keyMatcher = re.compile(keyPattern)
    # matches the key (including curly brackets) and key (excluding curly
    # brackets) for each section of the format string

    # nameFormat = '{basename}'       # Naming convention defaults

    # TODO: maybe have get_binning, get_mode etc methods. alternatively,
    # a Mode, Binning(NamedTuple) class with __str__

    # def format(self, **kws):
    #     self.pattern.format(kws)

    # def get_mode(self):

    def __init__(self, pattern):
        self.pattern = pattern
        self.keys = self.keyMatcher.findall(pattern)
        self.keyset = set(self.keys) - {'sep'}

    def unique(self, run):
        """
        Generates a unique sequence of filenames based on the name_dict.
        """
        from recipes.iter import partition, roundrobin
        from recipes.list import count_repeats

        # check which keys help in generating unique set of filenames - these won't be used
        use_keys, remove_keys = partition(run.varies_by, self.keyset)
        remove_keys = set(remove_keys)
        fmt = self.pattern[:]

        if len(remove_keys):
            for i, mo in enumerate(self.optMatcher.finditer(self.pattern)):
                opt = self.pattern[slice(*mo.span())]
                if len(remove_keys & set(self.keyMatcher.findall(opt))):
                    fmt = fmt.replace(mo.group(), '')

        # fmt str now has only keys that should be used for naming
        fmt = fmt.replace('<', '').replace('>', '') # remove brackets

        # create the filenames
        filenames = [fmt.format(**cube.get_name_dict()) for cube in run]

        # last resort append numbers to the non-unique filenames
        counts = count_repeats(filenames)
        kws = dict(sep='', extension='')
        generators = (FilenameGenerator(fn, **kws)(count if count > 1 else None)
                        for fn, count in counts.items())
        yield from roundrobin(*generators)



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

    n = NamingConvention(
        '{label[0]}{date}{sep}{binning}<{sep}m{mode}><{sep}t{kct}>'
    )
