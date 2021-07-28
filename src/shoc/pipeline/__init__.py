"""
Photometry pipeline for the Sutherland High-Speed Optical Cameras
"""


# std libs
from pathlib import Path


# relative libs
from .. import make_banner


BANNER_WIDTH = 100
WELCOME_BANNER = make_banner('Photometry Pipeline', BANNER_WIDTH)


_folders = (
    'logs',
    'plots',
    'detection',
    'sample',
    'photometry'
)


class FolderTree:
    """Filesystem tree helper"""

    def __init__(self, input, output=None, folders=_folders):
        self.input = Path(input)
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
