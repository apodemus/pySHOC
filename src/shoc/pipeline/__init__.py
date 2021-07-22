"""
Photometry pipeline for the Sutherland High-Speed Optical Cameras
"""


# std libs
from pathlib import Path

# local libs


# relative libs
from .. import make_banner
from .main import *


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
    def __init__(self, input, output=None, folders=_folders):
        input = Path(input)
        if output is None:
            output = input / '.pyshoc'

        for folder in folders:
            path = output / folder
            setattr(self, folder, path)

    def create(self):
        for _, path in self.__dict__.items():
            path.mkdir(exist_ok=True)

    def folders(self):
        return list(filter(Path.is_dir, self.__dict__.values()))
