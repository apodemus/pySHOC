
# std
import os
import pwd
from pathlib import Path

# third-party
import yaml

# local
import motley
from recipes.dicts import AttrReadItem, DictNode


# ---------------------------------------------------------------------------- #

def get_username():
    return pwd.getpwuid(os.getuid())[0]

# ---------------------------------------------------------------------------- #


class ConfigNode(DictNode, AttrReadItem):
    pass


CONFIG = ConfigNode(
    **yaml.load((Path(__file__).parent / 'config.yaml').read_text(),
                Loader=yaml.FullLoader)
)
# CONFIG.freeze()

# load cmasher if needed
plt = CONFIG.plotting
for cmap in (plt.cmap, plt.segments.contours.cmap, plt.mosaic.cmap):
    if cmap.startswith('cmr.'):
        import cmasher
        break


# set remote username default
if CONFIG.remote.username is None:
    CONFIG.remote['username'] = get_username()

# uppercase logging level
for cfg in CONFIG.logging.values():
    cfg['level'] = cfg.level.upper()
del cfg

CONFIG.logging.console['repeats'] = motley.stylize(CONFIG.logging.console.repeats)
