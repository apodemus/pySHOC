"""
Photometry pipeline for the Sutherland High-Speed Optical Cameras.
"""


from .. import config as cfg
from .logging import logger
from .banner import make_banner


# ---------------------------------------------------------------------------- #
WELCOME_BANNER = ''
if cfg.console.banner.pop('show', True):
    WELCOME_BANNER = make_banner(**cfg.console.banner)


# # overwrite tracking default config
# tracking.cfg = cfg.tracking
# tracking.cfg['filenames'] = cfg.tracking.filenames

# TODO: Enum
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
