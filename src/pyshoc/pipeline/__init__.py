"""
Photometry pipeline for the Sutherland High-Speed Optical Cameras.
"""


# relative
from .. import CONFIG
from .logging import logger
from .banner import make_banner


# ---------------------------------------------------------------------------- #
WELCOME_BANNER = ''
if CONFIG.console.banner.pop('show', True):
    WELCOME_BANNER = make_banner(**CONFIG.console.banner)


# # overwrite tracking default config
# tracking.CONFIG = CONFIG.tracking
# tracking.CONFIG['filenames'] = CONFIG.tracking.filenames


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
