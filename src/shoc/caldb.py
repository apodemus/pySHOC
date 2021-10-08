"""
Calibration files database
"""


# std
import shutil
import itertools as itt
from pathlib import Path
from datetime import datetime

# third-party
from loguru import logger

# local
import motley
from motley.table import Table
from pyxides.containers import ArrayLike1D, AttrGrouper, OfType
from recipes import io
from recipes.logging import LoggingMixin
from recipes.string import numbered, plural
from recipes.dicts import AutoVivify, AttrDict

# relative
from . import shocCampaign, MATCH


def move_to_backup(file, ext='bak'):
    """
    Rename and move the `file` to a backup storage location.

    Parameters
    ----------
    file : Path
        File to move.
    ext : str
        Filename extension for backup. An integer will be appended if necessary
        for uniqueness.
    """
    if not file.exists():
        return

    path = file.parent
    date = datetime.now().strftime('%Y%m%d')
    new_file = path / f'{file.name}{date}.{ext}'
    i = 1
    while new_file.exists():
        new_file = path / f'{file.stem}{date}.{ext}{i}'
        i += 1

    shutil.move(file, new_file)


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
    """An HDU mocking helper"""


class MockRun(ArrayLike1D, AttrGrouper, OfType(MockHDU)):
    """A lightweight Campaign emulator"""


class CalDB(DB, LoggingMixin):
    """Calibration file database"""

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

    def make(self, new=False):
        for kind, master in itt.product(('dark', 'flat'), (0, 1)):
            if new:
                # move to backup
                path = self.get_path(kind, master)
                filename = path.with_suffix(self.suffix)
                move_to_backup(filename)

            self.update(kind=kind, master=master)

    def get_path(self, kind, master=True):
        which = 'master' if master else 'raw'
        return self[which][kind]

    @staticmethod
    def get_dict(run, kind, master):
        # get filenames -> *attributes mapping
        # set the telescope from the folder path
        if kind == 'flat' and not master:
            run.attrs.set(filters=run.attrs('file.path.parent.name'),
                          telescope=run.attrs('file.path.parent.parent.name'))

        # get appropriate attributes for dark / flat
        files, *data = zip(*run.attrs('file.path', *sum(MATCH[kind], ())))
        return dict(zip(files, zip(*data)))

    def get_new_files(self, kind, master):
        path = self.get_path(kind, master)
        filelist = set(path.rglob('*.fits'))
        db = self.load_mock(kind, master)
        return filelist - set(db.attrs('filename'))

    def update(self, new=(), kind=None, master=True):
        if new:
            kind = kind or next(iter(new.attrs('obstype')))
            close = False
        else:
            if not kind:
                raise ValueError('Require `kind` to be given if no `new` '
                                 'observations (shocCampaign) is provided.')

            # look for new files
            new = self.get_new_files(kind, master)
            if new:
                shocCampaign.logger = self.logger
                new = shocCampaign.load(new)
                shocCampaign.logger = shocCampaign.Logger()
                # logger.info('Loaded {:d} {:s}.', i, plural('file', new))

            close = True

        if not new:
            return

        # update
        db = self.load(kind, master)
        db.update(self.get_dict(new, kind, master))

        # (re)write
        path = self.get_path(kind, master)
        io.serialize(path.with_suffix(self.suffix), db)

        if close:
            new.close()

    def load(self, kind, master=True):
        """Lazy load the json database"""
        filename = self.get_path(kind, master).with_suffix(self.suffix)
        if filename.exists():
            return io.deserialize(filename)
        return {}

    def load_mock(self, kind, master=True):
        """Load data as a MockRun object"""
        which = 'master' if master else 'raw'
        db = self.db[which][kind] = MockRun()
        att = sum(MATCH[kind], ())
        for filename, data in self.load(kind, master).items():
            db.append(MockHDU(zip(att, data), filename=filename))
        return db

    def get(self, run, kind, master=True):
        """
        Get calibration files matching observation campaign `run`.

        Parameters
        ----------
        run : shocCampaign
            Observations to get calibration files for.
        kind : {'dark', 'flat'}
            The kind of calibration files required.
        master : bool, optional
            Should master files or unprocessed raw stacks be returned, by
            default True.

        Returns
        -------
        shocObsGroups
            The matched and grouped `shocCampaign`s.
        """
        from .import CONFIG
        
        which = 'master' if master else 'raw'
        name = motley.apply(f'{which} {kind}', CONFIG['colors'][kind])
        logger.info("Searching for {:s} files in database: '{!s}'",
                    name, self[which][kind])

        if kind in self.db[which]:
            db = self.db[which][kind]
        else:
            db = self.load_mock(kind, master)

        not_found = []
        attx, attc = MATCH[kind]
        attrs = (*attx, *attc)
        grp = run.new_groups()
        grp.group_id = attrs, {}
        _, gcal = run.match(db, attx, attc)

        i = 0
        for key, mock in gcal.items():
            if mock is None:
                not_found.append(key)
                continue

            # load
            logger.disable('obstools.campaign')
            # catch repetative log message. Will emit below for loop
            cal = grp[key] = shocCampaign.load(mock.attrs('filename'),
                                               obstype=kind)
            i += 1
            logger.enable('obstools.campaign')

            # set telescope from db folder path for flats
            if kind == 'flat':
                cal.attrs.set(telescope=mock.attrs('telescope'))
                assert None not in cal.attrs.telescope

        if not_found:
            logger.info(
                'No {:s} available in database for {:s} with '
                'observational {:s}:\n{:s}',
                name, numbered(not_found, 'file'), plural('setup', not_found),
                Table(not_found, col_headers=attrs, nrs=True)
            )
        else:
            logger.info('Found {:d} {:s} files.', sum(map(len, grp.values())),
                        name)

        return grp
