"""
Universal build script for python project git repos.
"""

# std
import os
import re
import sys
import glob
import site
import math
import fnmatch
import subprocess as sub
from pathlib import Path
from collections import abc
from distutils import debug

# third-party
from setuptools.command.build_py import build_py
from setuptools import Command, find_packages, setup


# ---------------------------------------------------------------------------- #
debug.DEBUG = True

# allow editable user installs
# see: https://github.com/pypa/pip/issues/7953
site.ENABLE_USER_SITE = ('--user' in sys.argv[1:])


# Git ignore
# ---------------------------------------------------------------------------- #
# Source: https://github.com/astromancer/recipes/blob/main/src/recipes/io/gitignore.py


def _git_status(raises=False):
    # check if we are in a repo
    status = sub.getoutput('git status --porcelain')
    if raises and status.startswith('fatal: not a git repository'):
        raise RuntimeError(status)

    return status


UNTRACKED = re.findall(r'\?\? (.+)', _git_status())
IGNORE_IMPLICIT = ('.git', )


# ---------------------------------------------------------------------------- #

def read(path):
    # read glob patterns from file
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'No such file: {path!s}.')

    return list(_read(path))


def _read(path):
    return filter(None, (line.strip(' ')
                         for line in path.read_text().splitlines()
                         if not line.startswith('#')))


class GlobPatternList:
    """
    Class to filter files matching any in a list of glob patterns.
    """

    __slots__ = ('root', 'names', 'patterns')

    @classmethod
    def from_file(cls, path):
        path = Path(path)
        return cls(path.parent, read(path))

    fromfile = from_file

    def __init__(self, root, patterns):
        self.root = Path(root)
        self.names = list(IGNORE_IMPLICIT)
        self.patterns = []
        self.add(patterns)

    def add(self, items):
        if isinstance(items, str):
            self._add(items)
        elif isinstance(items, abc.Iterable):
            list(map(self.add, items))
        else:
            raise TypeError(f'Invalid object type {type(items).__name__}: {items}.')

    def _add(self, pattern):
        items = (self.names, self.patterns)
        if pattern:
            items[glob.has_magic(pattern)].append(pattern.rstrip('/'))

    def match(self, filename):
        path = Path(filename).relative_to(self.root)
        filename = str(path)
        for pattern in self.patterns:
            if fnmatch.fnmatchcase(filename, pattern):
                return True

        return filename.endswith(tuple(self.names))

    def iterdir(self, folder=None, depth=any, _level=0):
        depth = math.inf if depth is any else depth
        folder = folder or self.root

        _level += 1
        if _level > depth:
            return

        for path in folder.iterdir():
            if self.match(path):
                continue

            if path.is_dir():
                yield from self.iterdir(path, depth, _level)
                continue

            yield path

    # alias
    iter = iterdir


class GitIgnore(GlobPatternList):
    """
    Class to read `.gitignore` files and filter source trees.
    """

    def __init__(self, filename='.gitignore'):
        path = Path(filename)
        return super().__init__(path.parent, read(path))

# Setuptools
# ---------------------------------------------------------------------------- #


class Builder(build_py):
    # need this to exclude ignored files from the build archive

    def find_package_modules(self, package, package_dir):
        # filter folders
        if gitignore.match(package_dir) or gitignore.match(Path(package_dir).name):
            self.debug_print(f'(git)ignoring {package_dir}')
            return

        # package, module, files
        info = super().find_package_modules(package, package_dir)

        for package, module, path in info:
            # filter files
            if path in UNTRACKED:
                self.debug_print(f'Ignoring untracked: {path}.')
                continue

            if gitignore.match(path):
                self.debug_print(f'(git)ignoring: {path}.')
                continue

            self.debug_print(f'Found: {package = }: {module = } {path = }')
            yield package, module, path


class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""

    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        os.system('rm -vrf ./build ./dist ./*.pyc ./*.tgz ./src/*.egg-info')


# Main
# ---------------------------------------------------------------------------- #
gitignore = GitIgnore()

setup(
    packages=find_packages(exclude=['tests', "tests.*"]),
    use_scm_version=True,
    include_package_data=True,
    exclude_package_data={'': [*gitignore.patterns, *gitignore.names]},
    cmdclass={'build_py': Builder,
              'clean': CleanCommand}
    # extras_require = dict(reST = ["docutils> = 0.3", "reSTedit"])
    # test_suite = 'pytest',
)
