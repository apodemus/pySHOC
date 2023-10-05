

# std
import re
import itertools as itt
from pathlib import Path
from string import Template
from collections import defaultdict

# third-party
import more_itertools as mit
from loguru import logger

# local
import motley
from motley.table import Table
from recipes import cosort, string
from recipes.dicts.node import DictNode
from recipes.tree import FileSystemNode

# relative
from .. import CONFIG
from .utils import get_file_age, human_time


class ProductNode(FileSystemNode):

    @staticmethod
    def get_label(path):
        return path.name
        # return f'{path.name}{"/" * path.is_dir()}'

    def __getitem__(self, key):
        # sourcery skip: assign-if-exp, reintroduce-else
        if self.is_leaf:
            raise IndexError('Leaf node')

        if isinstance(key, Path):
            if key.is_absolute():
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


def resolve_path(path, hdu, *frames):
    if isinstance(path, DictNode):
        raise TypeError

    path = str(path)
    subs = {'$HDU':  hdu.file.stem,
            '$DATE': str(hdu.t.date_for_filename)}

    if frames and '$FRAMES' in path:
        j, k = frames
        if j and k and (j, k) != (0, hdu.nframes):
            subs['$FRAMES'] = '{j}-{k}'.format(j=j, k=k)
        else:
            subs['$FRAMES'] = ''

    return Path(string.sub(path, subs))


def get_previous(run, paths):
    # get data products
    output = paths.folders.output
    products = ProductNode.from_path(output,
                                     ignore=('.cache', '_old', 'logs'))
    products.get_ages()

    # get overview products
    overview = {
        section: products[path].as_dict()
        for section, path in
        paths.folders.filtered(('info', 'plotting', 'registration')).items()
    }

    # _TODO = overview['info'].pop(f'{paths.files.info.headers.parent.name}/')
    _sample_plots = overview['plotting'].pop(f'{paths.folders.samples.name}/')

    #
    logger.opt(lazy=True).debug(
        'Found previous data products: \n{}\n{}\n{}',

        # overview
        lambda: _overview_table_vstack(overview, paths),

        # hdu products
        lambda: _hdu_products_table(run, paths.patterns['HDU'], output),

        # nightly data products
        lambda: _nightly_products_table(run, paths.patterns['DATE'], output)
    )

    return overview, products


def _overview_table_hstack(overview, paths):
    subtitles = {key: f'{key.title()}: /{paths.folders[key].relative_to(paths.folders.output)}/*'
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
        key: f'{key.title()}: /{paths.folders[key].relative_to(paths.folders.output)}/*'
        for key in overview
    }

    tables = {
        key: Table.from_dict(
            dict(zip(*cosort(*zip(*items.items())))),
            title=('Data Products: Overview' if (first := (key == 'info'))
                   else f'\n{subtitles[key]}'),
            ignore_keys=('headers/'),
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

    tables = dict(zip(*cosort(tables.keys(), tables.values())))

    # return tables.values()
    return motley.utils.vstack.from_dict(tables, vspace=0)


# class XNode(Node):
#     _join_names = '/'.join

#     @staticmethod
#     def _get_name(path):
#         return f'{path.name}{"/" * path.is_dir()}'


def _sort_cols(keys):
    for i, x in enumerate(('info', 'samples', 'tracking', 'lightcurves')):
        if x in keys:
            return i
    return 99


def _resolve_desired_hdu_products(run, filename_patterns):
    # sort rows
    stems, _ = cosort(run.files.stems, run.attrs('t.date'))
    patterns = filename_patterns.flatten()
    rows = defaultdict(list)
    for pattern in patterns.values():
        tmp = Template(pattern)
        for stem in stems:
            rows[stem].append(Path(tmp.substitute(HDU=stem, FRAMES='')))

    # sections, rows
    return list(patterns.keys()), rows


def _hdu_products_table(run, filename_patterns, output):

    section_titles = dict(samples='Sample Images',
                          lightcurves='Light Curves',
                          tracking='Tracking Data')

    sections, desired_files = _resolve_desired_hdu_products(run, filename_patterns)

    # table = defaultdict(list)
    patterns = filename_patterns.flatten()
    headers = []
    for section in sorted(sections, key=_sort_cols):
        path = Path(patterns[section])
        rpath = path.parent.relative_to(output)
        header = (section_titles.get(section[0], ''),
                  f'{rpath.parent}/', f'{rpath.name}/', path.name)
        # table[header] = desired_files[section]
        headers.append(header)

    # Sort columns
    return Table(list(desired_files.values()),
                 title='HDU Data Products',
                 row_headers=desired_files.keys(),
                 col_groups=headers,
                 formatter=lambda f: human_time(get_file_age(f)),
                 align='<',
                 col_groups_align='<',
                 subtitle=motley.format('{Output folder: {:|darkgreen}/:|B}', output),
                 subtitle_align='<',
                 subtitle_style=('_'),
                 **CONFIG.console.products
                 )


def _nightly_products_table(run, filename_patterns, output):

    dates = sorted(set(run.attrs('date_for_filename')))
    section_titles = {}

    cols = defaultdict(list)
    for keys, pattern in filename_patterns.flatten().items():
        path = Path(pattern)
        rpath = path.parent.relative_to(output)

        header = (section_titles.get(keys[0], ''), f'{rpath.parent}/', f'{rpath.name}/', path.name)
        for date in dates:
            filename = Path(Template(pattern).substitute(DATE=date))
            cols[header].append(get_file_age(filename))

    return Table(cols,
                 title='Nightly Data Products',
                 row_headers=dates,
                 col_groups=cols.keys(),
                 formatter=human_time,
                 align='<',
                 col_groups_align='<',
                 **CONFIG.console.products
                 )


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
    if path.exists():
        return (f'=HYPERLINK("{path}", "{path.suffix[1:]}")')
    return '--'


def hyperlink_name(path):
    if path.exists():
        return (f'=HYPERLINK("{path}", "{path.name}")')
    return '--'


def write_xlsx(run, paths, overview, filename=None):

    sections, desired_files = _resolve_desired_hdu_products(run, paths.patterns['HDU'])
    run = run[list(desired_files.keys())]

    out = DictNode()
    files = 'files'
    out['FITS', 'base'] = run.files.stems
    out['FITS', files] = run.files.paths

    # duplicate Overview images so that they get merged below
    out['Images', 'Overview'] = \
        [[(paths.folders.plotting / _) for _ in overview['plotting']]] * len(run)

    #
    sections = [(section.title(), name) for section, name in sections]
    sections[sections.index(('Info', 'headers'))] = ('FITS', 'headers')
    sections[sections.index(('Samples', 'filename'))] = ('Images', 'samples')

    sort = ('FITS', 'Images', 'Tracking', 'Lightcurves').index
    sections, headers, *data = cosort(*zip(*sections), *desired_files.values(), 
                                    key=sort)
    
    sections = zip(sections, headers)
    out.update(dict(zip(sections, zip(*data))))

    out['Light Curves']  = out.pop('Lightcurves')
    
    # Images
    # out['Images']['Source Regions'] = list(match(run, paths.source_regions.iterdir()))

    # TODO
    # ['Spectral Estimates', ]
    # 'Periodogram','Spectrogram'

    # create table
    tbl = Table.from_dict(out,
                          title='Data Products',
                          convert={Path: hyperlink_ext,
                                   'files': hyperlink_ext,
                                   'Overview': hyperlink_name
                                   },
                          split_nested_types={tuple, list},
                          )

    # write
    # header_formatter=str.title
    tbl.to_xlsx(filename or paths.files.info.products,
                formats=';;;[Blue]@',
                widths={'base': 14,
                        files: 5,
                        'headers': 7,
                        'Overview': 4,
                        'samples': 7,
                        # 'Source Regions': 10,
                        ...: 7},
                align={'base': '<',
                       'Overview': dict(horizontal='center',
                                        vertical='center',
                                        text_rotation=90),
                       ...: dict(horizontal='center',
                                 vertical='center')},
                merge_unduplicate=('data', 'headers'))
