"""
Photometry pipeline for the Sutherland High-Speed Optical Cameras.
"""


# std
from pathlib import Path
from collections import abc

# local
from obstools.phot import tracking
from recipes import dicts
from recipes.string import sub
from recipes.config import ConfigNode
from recipes.functionals import negate
from recipes.oo import AttributeAutoComplete

# relative
from .. import CONFIG
from .logging import logger
from .banner import make_banner


# ---------------------------------------------------------------------------- #
WELCOME_BANNER = ''
if CONFIG.console.banner.pop('show', True):
    WELCOME_BANNER = make_banner(**CONFIG.console.banner)

# # overwrite tracking default config
# tracking.CONFIG = CONFIG.tracking
# tracking.CONFIG['filenames'] = CONFIG.tracking.filenames


SUPPORTED_APERTURES = [
    'square',
    'ragged',
    'round',
    'ellipse',
    'optimal',
    # 'psf',
    # 'cog',
]
APPERTURE_SYNONYMS = {
    'circle':     'round',
    'circular':   'round',
    'elliptical': 'ellipse'
}

# ---------------------------------------------------------------------------- #


def prefix_paths(mapping, output, substitutions):
    relative_paths = _sub_interal_refs(mapping, substitutions)
    return dict(_prefix_paths(relative_paths, output), output=output)


def _sub_interal_refs(folders, substitutions):
    return _recurse(sub, folders, substitutions)


def _prefix_paths(mapping, prefix='./'):
    return _recurse(_prefix_relative_path, mapping, prefix)


def _prefix_relative_path(path, prefix):
    return o if (o := Path(path)).is_absolute() else (prefix / path).resolve()


def _recurse(func, mapping, arg):
    for key, item in mapping.items():
        if isinstance(item, abc.MutableMapping):
            _recurse(func, item, arg)
        else:
            mapping[key] = func(item, arg)
    return mapping


def _is_special(path):
    return ('$HDU' in (s := str(path))) or ('$DATE' in s)


def get_parents(pathdict):
    for item in pathdict.values():
        if isinstance(item, abc.MutableMapping):
            yield from get_parents(item)
        elif not _is_special(item.parent):
            yield item.parent


class FolderTree(AttributeAutoComplete):
    """
    Filesystem tree helper. Attributes point to the full system folders and
    files for pipeline data products.
    """

    def __init__(self, root, output=None, folders=CONFIG.folders, **output_files):
        #
        self.root = Path(root).resolve()

        #
        folders = dict(folders)
        output_root_default = folders.pop('output_root')
        output = output or (self.root / output_root_default)

        substitutions = {f'${name.upper()}': loc
                         for name, loc in folders.items()}
        substitutions = _sub_interal_refs(substitutions.copy(), substitutions)
        for key, folder in prefix_paths(folders, output, substitutions).items():
            setattr(self, key, folder)

        # folders to create
        self.folders = dicts.AttrReadItem(vars(self))

        # files
        self.files = prefix_paths(output_files, output, substitutions)
        for alias, filename in self.files.items():
            setattr(self, alias, filename)

    def __getitem__(self, key):
        return self.__dict__[key]

    def __repr__(self):
        return dicts.pformat(vars(self), rhs=self._relative_to_root,
                             ignore=('folders', 'files'))

    def _set_path_attrs(self, mapping, prefix):
        for key, item in mapping.items():
            if isinstance(item, abc.MutableMapping):
                return self._set_path_attrs(item)

    def _relative_to_root(self, path):
        return str(path.relative_to(self.root)
                   if self.root in path.parents
                   else '$ROOT' / path)

    def create(self):
        logger.info('Creating output folder structure.')
        required = list(set(get_parents(self.files)))
        for _, path in self.folders.items():
            if not _is_special(path):
                required.append(path)

        for path in filter(negate(Path.exists), required):
            logger.debug('Creating folder: {}', path)
            path.mkdir()
