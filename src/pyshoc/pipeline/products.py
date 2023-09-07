

# std
import functools as ftl
import itertools as itt
from pathlib import Path
from collections import defaultdict

# third-party
import more_itertools as mit
from loguru import logger

# local
import motley
from motley.table import Table, hstack
from recipes import cosort, string
from recipes.tree import FileSystemNode
from recipes.dicts import DictNode, vdict

# relative
from .. import CONFIG
from .utils import get_file_age, human_time


# ---------------------------------------------------------------------------- #

class Node(DictNode, vdict):
    pass


class ProductNode(FileSystemNode):

    @staticmethod
    def get_label(path):
        return f'{path.name}{"/" * path.is_dir()}'

    def __getitem__(self, key):
        # sourcery skip: assign-if-exp, reintroduce-else
        if self.is_leaf:
            raise IndexError('Leaf node')

        if isinstance(key, Path):
            key = key.relative_to(self.root.as_path())
            item = self
            for key in key.parts:
                item = item[key]
            return item

        try:
            return super().__getitem__(key)
        except KeyError as err:
            return super().__getitem__(f'{key}/')

    def get_ages(self):
        for leaf in self.leaves:
            leaf.age = get_file_age(leaf.as_path())
        return self

    def as_dict(self, attr='name', leaf_attr='age'):
        if self.is_leaf:
            return getattr(self, leaf_attr)

        return {getattr(child, attr): child.as_dict(attr, leaf_attr)
                for child in self.children}


def resolve_path(path, hdu):
    return Path(string.sub(str(path),
                           {'$HDU':  hdu.file.stem,
                            '$DATE': str(hdu.t.date_for_filename)}))


def get_previous(run, paths):
    # get data products
    products = ProductNode.from_path(paths.output, ignore=('.cache', '_old', 'logs'))
    products.get_ages()

    # get hdu products          # ignore='**/_old/*'
    phot_products = products[paths.folders.phot]  # .as_dict()

    # get overview products
    overview = {
        part: products[paths.folders[part]].as_dict()
        for part in ('info', 'plots', 'registry')
    }

    _TODO = overview['info'].pop(f'{paths.headers.parent.name}/')
    sample_plots = overview['plots'].pop(f'{paths.sample_images.name}/')

    logger.opt(lazy=True).debug(
        'Found previous data products: \n{}\n{}\n{}',

        # overview
        lambda: _overview_table_vstack(overview, paths),

        # hdu products
        lambda: _product_table(phot_products, sample_plots,  paths),

        # light curves
        lambda: _lc_nightly_table(paths, products)
    )

    return overview, phot_products


def _overview_table_hstack(overview, paths):
    subtitles = {key: f'{key.title()}: /{paths.folders[key].relative_to(paths.output)}/*'
                 for key in overview}
    tables = {
        key: Table.from_dict(
            dict(zip(*cosort(*zip(*items.items())))),
            title=('Data Products: Overview' if (first := (key == 'info')) else ''),
            col_groups=[subtitles[key]] * 2,  # if first else None,
            col_groups_align='<',
            col_headers=['file', 'Age'],  # if first else None,
            formatter=human_time,
            order='c',
            **(CONFIG.console.products if first else
                {'title_align': '<', 'title_style': ('B', '_')}),
        )
        for key, items in overview.items()
    }

    # return tables.values()
    return motley.utils.hstack(tables.values(), spacing=0)
    # return motley.utils.vstack.from_dict(tables, vspace=1)


def _overview_table_vstack(overview, paths):
    subtitles = {
        key: f'{key.title()}: /{paths.folders[key].relative_to(paths.output)}/*'
        for key in overview
    }
    tables = {
        key: Table.from_dict(
            dict(zip(*cosort(*zip(*items.items())))),
            title=('Data Products: Overview' if (first := (key == 'info'))
                   else f'\n{subtitles[key]}'),
            col_groups=[subtitles[key]] * 2 if first else None,
            col_groups_align='<',
            col_headers=['file', 'Age'] if first else None,
            formatter=human_time,
            order='c',
            **(CONFIG.console.products if first else
               {'title_align': '<', 'title_style': ('B', '_')}),
        )
        for key, items in overview.items()
    }
    # return tables.values()
    return motley.utils.vstack.from_dict(tables, vspace=0)


# print(_overview_table_vstack(overview, paths))

# return Table.from_dict(
#     dict(zip(*cosort(*zip(*items.items())))),
#     # convert_keys='/'.join,
#     title='Data Products: Overview',
#     col_headers=['file', 'Age'],
#     formatter=human_time,
#     order='c',
#     align=('<', '>'),
#     **CONFIG.console.products,
# )


def _product_table(phot_products, sample_plots, paths):

    hdu_products = phot_products['tracking/'].as_dict()
    tracking_files = tuple(set(mit.collapse(p.keys() for p in hdu_products.values())))
    tracking_rpath = paths.folders.tracking.relative_to(paths.output)

    sample_plots = dict(zip(*cosort(*zip(*sample_plots.items()),
                                    key=(lambda x: x[-12:]))))

    return hstack(

        [  # sample images
            Table.from_dict(
                sample_plots,
                row_level=0,
                col_groups=['Sample Images'],
                convert_keys=(lambda s: f'$HDU.{s[0].split(".")[-1]}'),
                # sort=(lambda x: x[-15:]),
                formatter=human_time,
                order='c'
            ),

            # tracking files
            Table.from_rows(
                hdu_products,
                ignore_keys='plots/',
                col_groups=([
                    motley.format('{:|B}: {:|darkgreen}',
                                  'Tracking data', f'/{tracking_rpath}/*.dat')
                ] * len(tracking_files)),
                convert_keys=ftl.partial(string.remove_suffix, suffix='.dat'),
                sort=(lambda x: x[-12:]),
                #
                formatter=human_time,
                align='<',
            ),

        ],
        title='HDU Data Products',
        # title_align='<',
        subtitle=motley.format('{Output root: {:|darkgreen}/:|B}', paths.output),
        subtitle_align='<',
        subtitle_style=('_'),
        col_groups_align='<',
        **CONFIG.console.products
    )


def _lc_nightly_table(paths, products):

    groups = {}
    lcs = defaultdict(dict)
    folder = paths.folders.lightcurves
    for name, filename_pattern in paths.lightcurves.nightly.items():
        sub = filename_pattern.parent
        for file, age in products[sub].as_dict().items():
            date, tail = file[:10], file[10:]
            lcs[date][tail] = age
            groups[tail] = f'/{sub.relative_to(folder)}/'

    tbl = Table.from_rows(
        lcs,
        title='Data Products: Light curves',
        formatter=human_time,
        align='<',
        sort=True,
        subtitle=motley.format('{Output root: {:|darkgreen}/:|B}', folder),
        subtitle_align='<',
        subtitle_style=('_'),
        col_groups=list(groups.values()),
        col_groups_align='<',
        convert_keys='*{}'.format,
        **CONFIG.console.products
    )

    tbl.pre_table[0, 0] = 'Date'
    return tbl


# ---------------------------------------------------------------------------- #


def match(run, filenames, reference=None, empty=''):
    if isinstance(filenames, Path) and filenames.is_dir():
        filenames = filenames.iterdir()

    return list(_match(run, filenames, reference, empty))


def _match(run, filenames, reference, empty):

    incoming = {stem: list(vals) for stem, vals in
                itt.groupby(sorted(mit.collapse(filenames)), Path.stem.fget)}

    if reference is None:
        _, reference = cosort(run.attrs('t.t0'), run.files.stems)

    for base in reference:
        yield incoming.get(base, empty)


# ---------------------------------------------------------------------------- #


def hyperlink_ext(path):
    return f'=HYPERLINK("{path}", "{path.suffix[1:]}")'


def hyperlink_path(path):
    return f'=HYPERLINK("{path}", "{path.name}")'


def write_xlsx(run, paths, filename=None):

    overview, _products = get_previous(run, paths)

    #
    out = DictNode()
    out['FITS']['files'] = run.files.paths
    out['FITS']['headers'] = match(run, paths.headers)

    # Images
    # duplicate Overview images so that they get merged below
    out['Images']['Overview'] = \
        [[(paths.plots / _) for _ in overview['plots']]] * len(run)

    out['Images']['Samples'] = match(run, paths.sample_images)
    # out['Images']['Source Regions'] = list(match(run, paths.source_regions.iterdir()))

    # Light curves
    # cmb_txt = io.iter_files(paths.phot, 'txt')
    # individual = io.iter_files(paths.phot, '*.*.{txt,npy}', recurse=True)
    # combined = io.iter_files(paths.phot, '*.{txt,npy}')
    # out['Light Curves']['Raw'] = [
    #     (*indiv, *cmb) for indiv, cmb in
    #     zip(match(run, individual, empty=['']),
    #         match(run, combined, bases))
    # ]
    # TODO
    # ['Spectral Estimates', ]
    # 'Periodogram','Spectrogram'

    # create table
    tbl = Table.from_dict(out,
                          title='Data Products',
                          convert={Path: hyperlink_ext,
                                   'files': hyperlink_path,
                                   'Overview': hyperlink_path},
                          split_nested_types={tuple, list})

    # write
    return tbl.to_xlsx(filename or paths.products,
                       formats=';;;[Blue]@',
                       widths={'files': 23,
                               'headers': 10,
                               'Overview': 4,
                               'Samples': 10,
                               # 'Source Regions': 10,
                               ...: 7},
                       align={'files': '<',
                              'Overview': dict(horizontal='center',
                                               vertical='center',
                                               text_rotation=90),
                              ...: dict(horizontal='center',
                                        vertical='center')},
                       merge_unduplicate=('data', 'headers'))
