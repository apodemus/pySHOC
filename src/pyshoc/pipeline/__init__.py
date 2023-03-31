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
    'round', 'circle',
    'ellipse',
    'optimal',
    # 'psf',
    # 'cog',
]
APPERTURE_SYNONYMS = {'round': 'circle'}

# ---------------------------------------------------------------------------- #


class FolderTree(AttributeAutoComplete):
    """Filesystem tree helper"""

    def __init__(self, root, output=None, folders=CONFIG.folders, **output_files):
        #
        self.root = Path(root).resolve()

        if output is None:
            output = self.root / CONFIG.folders.output_root
        self.output = output

        for key, folder in folders.items():
            path = output / folder
            setattr(self, path.name.lstrip('.'), path)

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
