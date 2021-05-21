
from pathlib import Path
from recipes import io
from recipes.containers import ArrayLike1D, AttrGrouper, OfType
from recipes.dicts import AutoVivify, AttrDict
from . import shocCampaign, MATCH
from .core import split_dist
from .timing import Date

from recipes.logging import logging, get_module_logger
from motley.table import Table
from collections import defaultdict


# module level logger
logger = get_module_logger()
logging.basicConfig()
logger.setLevel(logging.INFO)


class DB(AttrDict, AutoVivify):
    def __init__(self, mapping=(), **kws):
        super().__init__()
        kws.update(mapping)
        for a, v in kws.items():
            self[a] = v

    def __setitem__(self, key, val):
        if '.' in key:
            key, tail = key.split('.', 1)
            self[key][tail] = val
        else:
            super().__setitem__(key, val)


class MockHDU(DB):
    pass


class MockRun(ArrayLike1D, AttrGrouper, OfType(MockHDU)):
    pass


class CalDB(DB):

    format = 'pkl'
    suffix = f'.{format}'

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
        self.db = DB()

    def make(self, kind, master=True):
        which = 'master' if master else 'raw'
        path = self[which][kind]
        run = shocCampaign.load(path, recurse=True)

        # set the telescope from the folder path
        run.set_attrs(telescope=run.attrs('file.path.parent.parent.name'),
                      each=True)
        # get appropriate attributes for dark / flat
        files, *data = zip(*run.attrs('file.path', *sum(MATCH[kind], ())))

        # close all files
        run.calls('_file.close')

        # write text
        io.serialize(path.with_suffix(self.suffix),
                     dict(zip(files, zip(*data))))

    def load(self, kind, master=True):
        """Lazy load the json database"""
        which = 'master' if master else 'raw'
        fn = (self[kind] / which).with_suffix(self.suffix)

        db = MockRun()
        if fn.exists():
            att = sum(MATCH[kind], ())
            for filename, data in io.deserialize(fn).items():
                db.append(MockHDU(zip(att, data), filename=filename))

            self.db[which][kind] = db
        return db

    def get(self, run, kind, master=True):

        which = 'master' if master else 'raw'
        if kind in self.db[which]:
            db = self.db[which][kind]
        else:
            db = self.load(kind, master)

        not_found = []
        grp = run.new_groups()
        gobj, gcal = run.match(db, *MATCH[kind])
        for key, mock in gcal.items():
            if mock is None:
                not_found.append(key)
                continue

            grp[key] = shocCampaign.load(mock.attrs('filename'), obstype=kind)
            if kind == 'flat':
                grp[key].set_attrs(
                    telescope=mock.attrs('telescope'), each=True)

        if not_found:
            logger.info(
                'No files available in calibration database for %s %s with '
                'observational setup(s):\n%s',
                which, kind, Table(not_found, col_headers=sum(MATCH[kind], ()),
                                   nrs=True)
            )

        return grp
