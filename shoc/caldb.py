
import numpy as np
from pathlib import Path
from recipes import io  # serialize, deserialize
from recipes.dicts import AutoVivify, AttrDict
from . import shocCampaign, shocObsGroups, MATCH
from .timing import Date

from recipes.logging import logging, get_module_logger
from motley.table import Table
from collections import defaultdict


# module level logger
logger = get_module_logger()
logging.basicConfig()
logger.setLevel(logging.INFO)

MATCH_CLOSE_CLASSES = {'dark': int,
                       'flat': Date}


def strings(items):
    return map(str, items)


def join(items):
    return ' '.join(map(str, items)) + '\n'


class CalDB(AttrDict, AutoVivify):
    def __init__(self, path=None):
        super().__init__()
        if path is None:
            return

        self.path = Path(path)
        for which in ('raw', 'master'):
            for kind in ('dark', 'flat'):
                self[kind] = self.path / f'{kind}s'
                self[which][kind] = self[kind] / which

        # The actual db
        self.db = CalDB()

    def make(self, kind, master=True):
        which = 'master' if master else 'raw'
        path = self[which][kind]
        run = shocCampaign.load(path, recurse=True)
        tbl = run.attrs('file.path', *sum(MATCH[kind], ()))
        # close all
        run.calls('_file.close')

        # filenames = {str(k): list(map(str, run.files.paths))
        #              for k, run in grp.items()}

        # write text
        io.safe_write(path.with_suffix('.db'), map(join, tbl))

    def load(self, kind, master=True):
        """Lazy load the json database"""
        which = 'master' if master else 'raw'
        fn = self[kind] / f'{which}.db'
        ex, cl = defaultdict(list), {}
        if fn.exists():
            attex, attcl = MATCH[kind]
            nex = len(attex)
            nsplit = nex + len(attcl)
            for line in io.read_lines(fn):
                filename, *data = line.split(' ', nsplit)
                ex[tuple(data[:nex])].append(filename)
                cl[filename] = data[nex:]

            self.db[which][kind] = ex, cl
        return ex, cl

    def get(self, run, kind, master=True):
        att, _ = MATCH[kind]
        which = 'master' if master else 'raw'
        if kind in self.db[which]:
            db, dbc = self.db[which][kind]
        else:
            db, dbc = self.load(kind, master)

        filenames = []
        not_found = []
        kls = MATCH_CLOSE_CLASSES[kind]
        grp = run.groupby(*att)
        
        for key in set(run.attrs(*att)):
            key = str(key)
            x = np.array([kls(dbc[fn]) if fn else np.ma.masked
                          for fn in db.get(key)])

            not_found.append(key)

        if not_found:
            logger.info(
                'No files available in calibration database for %s %s with '
                'observational setup(s):\n%s',
                which, kind, Table(not_found, col_headers=att, nrs=True)
            )
        if filenames:
            return shocCampaign.load(filenames, obstype=kind).group_by(*att)
        return shocObsGroups()
