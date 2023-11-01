

# std
from pathlib import Path
from collections import defaultdict

# third-party
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

    #
    _sample_plots = overview['plotting'].pop(f'{paths.folders.samples.name}/')

    #
    logger.opt(lazy=True).debug(
        'Found previous data products: \n{}\n{}\n{}',

        # overview
        lambda: _overview_table_vstack(overview, paths),

        # hdu products
        lambda: _hdu_products_table(run, paths),

        # nightly data products
        lambda: _nightly_products_table(run, paths)
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


def _resolve_by_file(run, templates):
    # sort rows
    stems = run.sort_by('t.t0').files.stems
    return _resolve_by(stems, templates, 'HDU', FRAMES='')


def _resolve_by_date(run, templates):
    # sort rows
    dates = sorted(set(run.attrs('date_for_filename')))
    return _resolve_by(dates, templates, 'DATE')


def _resolve_by(items, templates, key, **kws):

    rows = defaultdict(list)
    for tmp in templates.values():
        for row in items:
            rows[row].append(
                Path(tmp.substitute(**{key: row, **kws}))
            )
    return rows


def _get_column_header(base, keys, paths):
    section, *_ = keys
    rpath = paths.folders[section].relative_to(paths.folders.output)
    return (CONFIG[section].get('title', ''),
            f'{rpath.parent}/',
            f'{rpath.name}/',
            Path(paths.templates[(base, *keys)].template).name)


def _hdu_products_table(run, paths):

    # resolve required data products (paths) from campaign and folder config
    templates = paths.templates['HDU'].flatten()
    desired_files = _resolve_by_file(run, templates)
    headers = [_get_column_header('HDU', s, paths) for s in templates.keys()]

    # Sort columns
    return Table(list(desired_files.values()),
                 title='HDU Data Products',
                 row_headers=desired_files.keys(),
                 col_groups=headers,
                 formatter=lambda f: human_time(get_file_age(f)),
                 align='<',
                 col_groups_align='<',
                 subtitle=motley.format('{Output folder: {:|turquoise}/:|B}', paths.folders.output),
                 subtitle_align='<',
                 subtitle_style=('_'),
                 **CONFIG.console.products
                 )


def _nightly_products_table(run, paths):

    templates = paths.templates['DATE'].flatten()
    desired_files = _resolve_by_date(run, templates)
    headers = [_get_column_header('DATE', s, paths) for s in templates.keys()]

    return Table(list(desired_files.values()),
                 title='Nightly Data Products',
                 row_headers=desired_files.keys(),
                 col_groups=headers,
                 formatter=lambda f: human_time(get_file_age(f)),
                 align='<',
                 col_groups_align='<',
                 **CONFIG.console.products
                 )


# ---------------------------------------------------------------------------- #


def _hyperlink(path, target):
    if path.exists():
        return (f'=HYPERLINK("{path}", "{target}")')
    return '--'


def hyperlink_name(path):
    return _hyperlink(path, path.name)


def hyperlink_ext(path):
    return hyperlink_exts(path, 1)


def hyperlink_exts(path, tail=0, strip=''):
    ext = ''.join(path.suffixes[-tail:]).lstrip(strip) if path.exists() else ''
    return _hyperlink(path, ext)


def write_xlsx(run, paths, overview):

    # HDU
    filename, *sheet = str(paths.files.info.products.by_file).split('::')
    _write_hdu_products_xlsx(run, paths, overview, filename, *sheet)

    # DATE
    filename, *sheet = str(paths.files.info.products.by_date).split('::')
    return _write_nightly_products_xlsx(run, paths, filename, *sheet)


def _write_hdu_products_xlsx(run, paths, overview, filename=None, sheet=None,
                             overwrite=True):

    templates = paths.templates['HDU'].flatten()
    desired_files = _resolve_by_file(run, templates)

    # sort
    # run = run[list(desired_files.keys())]

    out = DictNode()
    files = 'files'
    out['FITS', 'HDU'] = run.files.stems
    out['FITS', files] = run.files.paths

    # duplicate Overview images so that they get merged below
    out['Images', 'Overview'] = [[(paths.folders.plotting / _)
                                  for _ in overview['plotting']]] * len(run)

    # CONFIG[section].get('title', section.title())
    sections = [(section.title(), name) for section, name, *_ in templates]
    sections[sections.index(('Info', 'headers'))] = ('FITS', 'headers')
    sections[sections.index(('Samples', 'filename'))] = ('Images', 'samples')

    order = ('FITS', 'Images', 'Tracking', 'Lightcurves')
    sections, headers, *data = cosort(*zip(*sections), *desired_files.values(),
                                      key=order.index)
    sections = zip(sections, headers)
    d = dict(zip(sections, zip(*data)))
    out.update(d)

    out['Light Curves'] = out.pop('Lightcurves')

    # Images
    # out['Images']['Source Regions'] = list(match(run, paths.source_regions.iterdir()))

    # TODO
    # ['Spectral Estimates', ]
    # 'Periodogram','Spectrogram'

    # create table
    tbl = Table.from_dict(out,
                          title='HDU Data Products',
                          convert={Path: hyperlink_ext,
                                   'files': hyperlink_ext,
                                   'Overview': hyperlink_name
                                   },
                          split_nested_types={tuple, list},
                          )

    # write
    # header_formatter=str.title

    # sheet = sheet[0] if sheet else None

    return tbl.to_xlsx(
        filename, sheet, overwrite=overwrite,
        formats={'HDU': str,
                 ...: ';;;[Blue]@'},
        widths={'HDU': 14,
                files: 5,
                'headers': 7,
                'Overview': 4,
                'samples': 7,
                # 'Source Regions': 10,
                ...: 7},
        align={'HDU': '<',
               'Overview': dict(horizontal='center',
                                vertical='center',
                                text_rotation=90),
               ...: dict(horizontal='center',
                         vertical='center')},
        merge_unduplicate=('data', 'headers')
    )


def _write_nightly_products_xlsx(run, paths, filename, sheet=None,
                                 overwrite=True):

    templates = paths.templates['DATE'].flatten()
    desired_files = _resolve_by_date(run, templates)

    # col_titles = dict(diff0='Differential',
    #                   decor='Decorrelated')
    headers = []
    for s in templates.keys():
        head, *_pathinfo, last = _get_column_header('DATE', s, paths)
        # ''.join((*_pathinfo, last))
        headers.append((head, s[1]))

    *groups, headers = zip((*[''] * (len(headers[0]) - 1), 'DATE'), *headers)

    tbl = Table(
        [(b, *f) for b, f in desired_files.items()],
        title='Nightly Data Products',
        col_groups=list(zip(*groups)),
        col_headers=headers,
        formatters={'DATE': str,
                    ...: hyperlink_ext},
        too_wide=False
        #  align='<',
        #  col_groups_align='<',
        #  **CONFIG.console.products
    )

    return tbl.to_xlsx(
        filename, sheet, overwrite=overwrite,
        formats={'DATE': str,
                 ...: ';;;[Blue]@'},
        widths={'DATE': 10,
                ...: 12},
        align={'DATE': '<',
               ...: dict(horizontal='center',
                         vertical='center')}
    )
