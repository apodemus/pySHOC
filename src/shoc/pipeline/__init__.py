"""
Photometry pipeline for the Sutherland High-Speed Optical Cameras.
"""


# std
from pathlib import Path

# local
from recipes import dicts
from recipes.oo import AttributeAutoComplete

# relative
from .. import make_banner


BANNER_WIDTH = None
WELCOME_BANNER = make_banner(
    'Photometry Pipeline', BANNER_WIDTH,
    fg=('Bold', 'blue'),
    linestyle=('-', 'B'),
    linecolor=['teal']
)

# Folder structure for results
OUTPUT_ROOT = '.pyshoc'
SUMMARY_FILENAME = 'campaign-files.xlsx'
PRODUCTS_FILENAME = 'data-products.xlsx'
OBSLOG_FILENAME = 'observing-log.tex'
_folders = (
    'headers',
    'logs',
    'plots',
    'plots/sample_images',
    'phot',
    'phot/source_regions',
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


class FolderTree(AttributeAutoComplete):
    """Filesystem tree helper"""

    def __init__(self,
                 root, output=None, folders=_folders,
                 obslog=OBSLOG_FILENAME,
                 summary=SUMMARY_FILENAME,
                 products=PRODUCTS_FILENAME):
        #
        self.root = Path(root).resolve()

        if output is None:
            output = self.root / OUTPUT_ROOT
        self.output = output

        for folder in folders:
            path = output / folder
            setattr(self, path.name.strip('.'), path)

        # folders to create
        self.folders = dict(vars(self))

        # files
        self.obslog = self.output / obslog
        self.summary = self.output / summary
        self.products = self.output / products

    def __repr__(self):
        return dicts.pformat(vars(self), rhs=lambda x: x.relative_to(self.root))

    def _relative_to_root(self, path):
        if path is self.root:
            return str(path)
        return f'$root/{path.relative_to(self.root)}'

    def create(self):
        for _, path in self.folders.items():
            path.mkdir(exist_ok=True)

    # def folders(self):
    #     return list(filter(Path.is_dir, self.__dict__.values()))
