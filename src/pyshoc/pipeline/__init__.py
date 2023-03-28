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

# Folder structure for results
# cfg = CONFIG.files.copy()
# cfg.pop('output_root')
# cfg.freeze()
# _files = {}
# for key, rpath in cfg.items():
#     _files[rpath.parent].append(rpath.name)

_file_struct = {
    'plots': ('thumbs', 'thumbs_cal', 'mosaic'),
    'info': ('summary', 'products', 'obslog'),
}


_folders = (
    'info',
    'info/headers',
    'info/logs',
    'plots',
    'plots/sample_images',
    'phot',
    '.cache'
)

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

    def __init__(self, root, output=None, folders=_folders, **output_files):
        #
        self.root = Path(root).resolve()

        if output is None:
            output = self.root / CONFIG.files.output_root
        self.output = output

        for folder in folders:
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
