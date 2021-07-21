"""
Photometry pipeline for the Sutherland High-Speed Optical Cameras
"""



# std libs
from pathlib import Path

# local libs
from motley import banner

# relative libs
from .. import __version__


LOGO = r"""
                 _____ __  ______  ______
    ____  __  __/ ___// / / / __ \/ ____/
   / __ \/ / / /\__ \/ /_/ / / / / /     
  / /_/ / /_/ /___/ / __  / /_/ / /___   
 / .___/\__, //____/_/ /_/\____/\____/   
/_/    /____/                            
"""
SUBTITLE = f"""\
Photometry Pipeline
v{__version__}\
"""
WELCOME_BANNER = '\n'.join(banner(_, bar='', side='', align=al)
                   for _, al in zip([LOGO, SUBTITLE], '^>'))


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
