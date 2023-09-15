
# std
from recipes import op
import os
import pwd

# local
import motley
from recipes.config import ConfigNode


# ---------------------------------------------------------------------------- #

def get_username():
    return pwd.getpwuid(os.getuid())[0]

# ---------------------------------------------------------------------------- #


CONFIG = ConfigNode.load_module(__file__)


# load cmasher if needed
plt = CONFIG.plotting
for cmap in CONFIG.filtered(op.contained('cmap').within).values():
    if cmap.startswith('cmr.'):
        import cmasher
        break


# set remote username default
if CONFIG.remote.username is None:
    CONFIG.remote['username'] = get_username()


# uppercase logging level
for sink, cfg in CONFIG.logging.items():
    if sink == 'folder':
        continue
    cfg['level'] = cfg.level.upper()
del cfg



# stylize log repeat handler
CONFIG.logging.console['repeats'] = motley.stylize(CONFIG.logging.console.repeats)

# stylize progressbar
prg = CONFIG.console.progress
prg['bar_format'] = motley.stylize(prg.bar_format)
del prg

# make config read-only
CONFIG.freeze()
