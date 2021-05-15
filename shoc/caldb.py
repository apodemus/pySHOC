from . import shocCampaign, MATCH
from pathlib import Path
from recipes.io import serialize


CALDB = Path('/media/Oceanus/work/Observing/data/SHOC/calibration/')
RAW = {'dark': CALDB / 'darks/raw',
       'flat': CALDB / 'flats/raw'}
MASTER = {'dark': CALDB / 'darks/master',
          'flat': CALDB / 'flats/master'}


def make_calbd(kind):

    path = RAW[kind]
    run = shocCampaign.load(path, recurse=True)
    grp = run.group_by(*MATCH[kind][0])
    filenames = {str(k): list(map(str, run.files.paths))
                 for k, run in grp.items()}
    serialize(path.parent / 'raw.json', filenames)
