

# std
import itertools as itt
from pathlib import Path
from collections import defaultdict

# third-party
from loguru import logger

# local
import motley
from motley.table import Table
from recipes import cosort, op, string
from recipes.dicts.node import DictNode
from recipes.tree import FileSystemNode

# relative
from .. import CONFIG
from ..config import Template
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


def sanitize_filename(name):
    return name.lower().replace(" ", "")


def resolve_path(path, hdu, *frames, **kws):
    if isinstance(path, DictNode):
        raise TypeError(f'{type(path)}')

    path = str(path)
    subs = {'$HDU':  hdu.file.stem,
            '$DATE': str(hdu.t.date_for_filename)}

    if frames and '$FRAMES' in path:
        j, k = frames
        if j and k and (j, k) != (0, hdu.nframes):
            subs['$FRAMES'] = '{j}-{k}'.format(j=j, k=k)
        else:
            subs['$FRAMES'] = ''

    # if '$SOURCE' in path and kws.:
    return Path(string.sub(path, {**subs, **kws}))


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
        paths.folders.select(('info', 'plotting', 'registration')).items()
    }

    #
    samples = paths.folders.samples
    _sample_plots = overview[samples.parent.name].pop(f'{samples.name}/')

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
    return _get_desired_products(stems, templates, 'HDU', FRAMES='')


def _resolve_by_date(run, templates):
    # sort rows
    dates = sorted(set(run.attrs('t.date_for_filename')))
    return _get_desired_products(dates, templates, 'DATE')


def _get_desired_products(items, templates, key, **kws):
    rows = defaultdict(list)
    for val, tmp in itt.product(items, templates.values()):
        rows[val].append(
            Path(tmp.substitute(**{key: val, **kws}))
        )
    return rows


def _get_column_header(base, keys, paths):
    section, *_ = keys
    rpath = paths.get_folder(keys).relative_to(paths.folders.output)

    return (CONFIG[section].get('title', ''),
            f'{rpath.parent}/',
            f'{rpath.name}/',
            Path(paths.templates[(base, *keys)].template).name)


def _hdu_products_table(run, paths):

    # resolve required data products (paths) from campaign and folder config
    templates = paths.templates['HDU'].filter('plots').flatten()
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


def _get_templates(paths, key, add_ext=CONFIG.lightcurves.plots.format):

    def _append(k, ext):
        return (*k[:-1],
                *([k[-1]] if k[-1] != 'filename' else ()),
                ext)

    def _prepend(k, pre):
        return (pre, *k)

    def _png_expected(k):
        attr = {'HDU': 'by_file',
                'DATE': 'by_date'}[key]
        return CONFIG.lightcurves.plots[attr].get(k[0])

    #
    tmp = paths.templates[key].copy()
    txt = tmp.pop('lightcurves').map(op.AttrGetter('template'))

    png = txt.map(str.replace, '.txt', f'.{add_ext}').transform(_append, add_ext)
    png = png.select(_png_expected)
    png = png.transform(_prepend, 'lightcurves')
    txt = txt.transform(_prepend, 'lightcurves').transform(_append, 'txt')

    return DictNode({**tmp.flatten(),
                     **txt.map(Template).flatten(),
                     **png.map(Template).flatten()}).flatten()


def _write_hdu_products_xlsx(run, paths, overview, filename=None, sheet=None,
                             overwrite=True):

    templates = _get_templates(paths, 'HDU')
    desired_files = _resolve_by_file(run, templates)

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
    # order = ('FITS', 'Images', 'Tracking', 'Lightcurves')
    # sections, headers, *data = cosort(*zip(*sections), *desired_files.values(),
    #                                   key=order.index)
    out.update(dict(zip(sections, zip(*desired_files.values()))))

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

    return tbl.to_xlsx(
        filename, sheet, overwrite=overwrite,
        formats={'HDU': str,
                 ...:   ';;;[Blue]@'},
        widths={'HDU':      14,
                files:      5,
                'headers':  7,
                'Overview': 4,
                'samples':  7,
                ...:        7},
        align={'HDU': '<',
               'Overview': dict(horizontal='center',
                                vertical='center',
                                text_rotation=90),
               ...: dict(horizontal='center',
                         vertical='center')},
        merge_unduplicate=('Overview', 'headers')
    )


def _write_nightly_products_xlsx(run, paths, filename, sheet=None,
                                 overwrite=True):

    templates = _get_templates(paths, 'DATE')
    desired_files = _resolve_by_date(run, templates)

    # col_titles = dict(diff0='Differential',
    #                   decor='Decorrelated')
    headers = [(CONFIG[section].get('title', ''), step)
               for section, step, *_ in templates.keys()]

    *groups, headers = zip((*[''] * (len(headers[0]) - 1), 'DATE'), *headers)

    tbl = Table(
        [(b, *f) for b, f in desired_files.items()],
        title='Nightly Data Products',
        col_groups=list(zip(*groups)),
        col_headers=headers,
        formatters={'DATE': str,
                    ...: hyperlink_ext},
        too_wide=False
    )

    return tbl.to_xlsx(
        filename, sheet, overwrite=overwrite,
        formats={'DATE': str,
                 ...: ';;;[Blue]@'},
        widths={'DATE': 10,
                ...: 6},
        align={'DATE': '<',
               ...: dict(horizontal='center',
                         vertical='center')}
    )
