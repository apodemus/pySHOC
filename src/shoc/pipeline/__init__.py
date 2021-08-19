"""
Photometry pipeline for the Sutherland High-Speed Optical Cameras
"""


# std libs
from pathlib import Path


# relative libs
from .. import make_banner


BANNER_WIDTH = 120
WELCOME_BANNER = make_banner('Photometry Pipeline', BANNER_WIDTH)


_folders = (
    'logs',
    'plots',
    # 'detection',
    # 'samples',
    'photometry'
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

    def __init__(self, input_, output=None, folders=_folders):
        self.input = Path(input_)
        if output is None:
            output = self.input / '.pyshoc'
        self.output = output

        for folder in folders:
            path = output / folder
            setattr(self, folder, path)

    def create(self):
        for _, path in self.__dict__.items():
            path.mkdir(exist_ok=True)

    def folders(self):
        return list(filter(Path.is_dir, self.__dict__.values()))
