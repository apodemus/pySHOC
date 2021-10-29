"""
Photometry pipeline for the Sutherland High-Speed Optical Cameras.
"""


# std
from pathlib import Path


# relative
from .. import make_banner


BANNER_WIDTH = 120
WELCOME_BANNER = make_banner('Photometry Pipeline', BANNER_WIDTH)

# Folder structure for results
OUTPUT_ROOT = '.pyshoc'
SUMMARY_FILENAME = 'campaign.xlsx'
PRODUCTS_FILENAME = 'data_products.xlsx'
_folders = (
    'headers',
    'logs',
    'plots',
    'plots/image_samples',
    'plots/source_regions',
    'phot'
)


class PartialAttributeLookup:
    """
    Attribute lookup that returns if the lookup key matches the start of the
    attribute name and the match is one-to-one. Raises AttributeError otherwise.
    """

    def __getattr__(self, key):
        try:
            return super().__getattribute__(key)
        except AttributeError as err:
            maybe = [_ for _ in self.__dict__ if _.startswith(key)]
            real, *others = maybe or (None, ())
            if others or not real:
                raise err from None
            return super().__getattribute__(real)


class FolderTree(PartialAttributeLookup):
    """Filesystem tree helper"""

    def __init__(self, root, output=None, folders=_folders,
                 summary=SUMMARY_FILENAME, products=PRODUCTS_FILENAME):
        self.root = Path(root)
        if output is None:
            output = self.root / OUTPUT_ROOT
        self.output = output

        for folder in folders:
            path = output / folder
            setattr(self, path.name, path)

        # folders to create
        self.folders = dict(vars(self))

        # files
        self.summary = self.output / summary
        self.products = self.output / products

    def __repr__(self):
        return str(vars(self))

    def create(self):
        for _, path in self.folders.items():
            path.mkdir(exist_ok=True)

    # def folders(self):
    #     return list(filter(Path.is_dir, self.__dict__.values()))
