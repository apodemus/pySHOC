"""
Photometry pipeline for the Sutherland High-Speed Optical Cameras.
"""


# std
from pathlib import Path

# local
from recipes import dicts
from recipes.oo import AttributeAutoComplete

# relative
from .. import CONFIG
from .banner import make_banner


# ---------------------------------------------------------------------------- #
WELCOME_BANNER = ''
if CONFIG.console.banner.pop('show'):
    WELCOME_BANNER = make_banner(**CONFIG.console.banner)


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


class FolderTree(AttributeAutoComplete):
    """Filesystem tree helper"""

    def __init__(self, root, output=None, folders=CONFIG.folders, **output_files):
        #
        self.root = Path(root).resolve()

        #
        folders = dict(folders)
        output_root_default = folders.pop('output_root')
        if output is None:
            output = self.root / output_root_default
        self.output = output

        for key, folder in folders.items():
            setattr(self, key, output / folder)

        # folders to create
        self.folders = dict(vars(self))

        # files
        for alias, filename in output_files.items():
            setattr(self, alias,  self.output / filename)

    def __repr__(self):
        return dicts.pformat(vars(self), rhs=self._relative_to_root)

    def _relative_to_root(self, path):
        return str(path.relative_to(self.root)
                   if self.root in path.parents
                   else path)

    def create(self):
        for _, path in self.folders.items():
            path.mkdir(exist_ok=True)

    # def folders(self):
    #     return list(filter(Path.is_dir, self.__dict__.values()))
