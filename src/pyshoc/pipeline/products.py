

# std
import itertools as itt
from pathlib import Path

# third-party
from loguru import logger

# local
import motley
from motley.table import Table
from motley.table.xlsx import hyperlink_ext
from recipes import cosort, op, string
from recipes.dicts.node import DictNode
from recipes.tree import FileSystemNode
from recipes.logging import LoggingMixin
from recipes.utils import ensure_list, ensure_tuple
from recipes.lists import remove, remove_all, replace
from recipes.functionals.partial import Partial, PlaceHolder as o

# relative
from .. import CONFIG
from ..config import Template
from .utils import get_file_age, human_time


# ---------------------------------------------------------------------------- #

def tpop(obj, *to_remove):
    return tuple(remove_all(list(obj), to_remove))


def sanitize_filename(name):
    return name.lower().replace(' ', '')


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
        paths.folders.select(('info', 'registration')).items()
    }

    # targets = DataProducts(run, paths)
    #
    # samples = paths.folders.samples
    # _sample_plots = overview[samples.parent.name].pop(f'{samples.name}/')

    #
    debug = logger.bind(indent=' ').opt(lazy=True).info
    pprint = (
        # overview
        lambda: ('overview',  _overview_table_vstack(overview, paths)),

        # hdu products
        lambda: ('hdu', _hdu_products_table(run, paths)),

        # nightly data products
        lambda: ('nightly', _nightly_products_table(run, paths))
    )
    for f in pprint:
        debug('Found previous {0[0]} data products: \n{0[1]}', f)

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


# ---------------------------------------------------------------------------- #
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


# ---------------------------------------------------------------------------- #

# GROUPINGS = dict(
#     by_file='file.stems',
#     by_date='t.date_for_filename',
#     # by_cycle=
# )
GROUPING_TO_TEMPLATE_KEY = {'file':  'HDU',
                            'date':  'DATE',
                            'cycle': 'E'}


class DataProducts(LoggingMixin):

    GROUPING_TO_TEMPLATE_KEY = {'file':  'HDU',
                                'date':  'DATE',
                                'cycle': 'E'}

    GROUPING_SORT_ATTR = {'file':  't.t0',
                          'date':  't.date_for_filename', }
    # 'cycle': 'E'}

    _remove_keys = ('by_file', 'by_date', 'concat')

    def __init__(self, run, paths, groupings=('file', 'date')):
        self.campaign = run
        self.paths = paths
        for g in groupings:
            names, targets = self.resolve(g)
            self.targets = {g: targets}

    def get_templates(self, key):

        # Get template tree
        tmp = self.paths.templates[key].copy()
        tmp = tmp.filter(('tracking', 'plots'))  # FIXME
        tmp, lcs = tmp.split('lightcurves')
        lcs = lcs.map(op.AttrGetter('template'))

        return DictNode({**tmp.flatten(),
                        **lcs.map(Template).flatten()}).flatten()

    def resolve(self, by):
        by = by.lower()
        if by == 'cycle':
            raise ValueError('TODO')

        key = GROUPING_TO_TEMPLATE_KEY[by]
        if by == 'file':
            # sort rows
            stems = self.campaign.sort_by('t.t0').files.stems
            templates = self.get_templates('HDU')
            files = _get_desired_products(stems, templates, key, FRAMES='')

        if by == 'date':
            # sort rows
            dates = sorted(set(self.campaign.attrs('t.date_for_filename')))
            templates = self.get_templates('DATE')
            files = _get_desired_products(dates, templates, key)

        files = files.transform(Partial(tpop)(o, self.to_remove))
        return files.stack(level=0)

    # def get_desired_products(self, items, templates,  key, **kws):

    #     rows = DictNode()
    #     for val, (section, template) in itt.product(items, templates.items()):
    #         files = template.resolve_paths(section, **{key: val, **kws})
    #         rows[(val, *section)] = {file.suffix.strip('.'): file for file in files}

    #     return rows

    def write_xlsx(self, overview):

        paths = self.paths
        run = self.campaign

        # HDU
        filename, *sheet = str(paths.files.info.products.by_file).split('::')
        _write_hdu_products_xlsx(run, paths, overview, filename, *sheet)

        # DATE
        filename, *sheet = str(paths.files.info.products.by_date).split('::')
        return _write_nightly_products_xlsx(run, paths, filename, *sheet)


# ---------------------------------------------------------------------------- #

SAMPLE = ['sample']
SERIES = ['raw', 'flagged', 'diff0', 'diff', 'decor']
SPECTRAL = ['periodogram', 'lombscargle', 'welch', 'tfr', 'acf']
SECTIONS = dict((*zip(SAMPLE, itt.repeat('Sample Images')),
                 *zip(SERIES, itt.repeat('Light Curves')),
                 *zip(SPECTRAL, itt.repeat('Spectral Density Estimates'))))
# dict(*(zip(thing, itt.repeat(name))
#        for thing, name in [(SAMPLE, 'Sample Images'),
#                            (SERIES, 'Light Curves'),
#                            (SPECTRAL, 'Spectral Density Estimates')])
#      )


def _get_templates(paths, key):

    # Get template tree
    tmp = paths.templates[key].copy()
    tmp = tmp.filter(('tracking', 'plots'))  # FIXME
    tmp, lcs = tmp.split('lightcurves')
    lcs = lcs.map(op.AttrGetter('template'))

    return DictNode({**tmp.flatten(),
                     **lcs.map(Template).flatten()}).flatten()


def get_desired_products(run, templates, by):

    by = by.lower()
    key = GROUPING_TO_TEMPLATE_KEY[by]
    if by == 'file':
        # sort rows
        stems = run.sort_by('t.t0').files.stems
        return _get_desired_products(stems, templates, key, FRAMES='')

    if by == 'date':
        # sort rows
        dates = sorted(set(run.attrs('t.date_for_filename')))
        return _get_desired_products(dates, templates, key)

    if by == 'cycle':
        raise ValueError('TODO')


def _get_desired_products(items, templates, key, **kws):

    rows = DictNode()
    for val, (section, template) in itt.product(items, templates.items()):
        files = template.resolve_paths(section, **{key: val, **kws})
        rows[(val, *section)] = {file.suffix.strip('.'): file for file in files}

    return rows


def _get_hdu_products(run, paths, overview=None, **kws):

    #
    templates = _get_templates(paths, 'HDU')
    to_remove = ('by_file', 'by_date', 'concat')
    desired_files = get_desired_products(run, templates, by='file')
    desired_files = desired_files.transform(Partial(tpop)(o, *to_remove))
    names, desired_files = desired_files.stack(level=0)

    # input for table
    tree = DictNode()
    base, files = 'base', 'fits'
    tree['input', base] = names
    tree['input', files] = run.files.paths

    # add files
    tree.update(desired_files)

    if overview:
        # re-section
        tree['registration', 'samples'] = tree.pop('samples')

        # duplicate Overview images so that they get merged below
        rplot_paths = paths.files.registration.plots
        path = rplot_paths.alignment.parent
        tree['registration', 'alignment', 'png'] = \
            [path / _ for _ in overview['registration'][f'{path.name}/']]

    tree = tree.transform(Partial(get_header)(o, tree.depth()))
    tree = tree.sorted(['input', 'info', 'registration', 'lightcurves'])

    return tree


def _hdu_products_table(run, paths):

    # resolve required data products (paths) from campaign and folder config
    # templates = paths.templates['HDU'].filter('plots').flatten()
    # desired_files = get_desired_products(run, templates, by='file')
    # headers = [_get_column_header('HDU', s, paths) for s in templates.keys()]

    tree = _get_hdu_products(run, paths)
    # tree = tree.transform(_sub_titles)

    fmt = motley.stylize('{!s:|turquoise}')
    tree = _add_path_info(tree, paths, fmt)

    return Table.from_dict(
        tree,
        title='HDU Data Products',
        converters={Path: lambda f: human_time(get_file_age(f))},
        align='<',
        col_groups_align='<',
        subtitle=motley.bold(
            f'Output folder: {fmt.format(f"{paths.folders.output}/")}'),
        subtitle_align='<',
        subtitle_style=('_'),
        **CONFIG.console.products
    )


def _sub_titles(section, at='tab'):
    section, *key = section
    header = ensure_list(CONFIG.get((section, at), section.title()))
    return (*header, *key)


def _add_path_info(tree, paths, fmt):

    out = DictNode()
    for section, items in tree.flatten().items():
        section, *key = section
        header = ensure_list(CONFIG.get((section, 'tab'), section.title()))

        if parent := CONFIG.get((section, 'folder')):
            parent = parent.relative_to(paths.folders.output)
            header[-1] = f'{header[-1]}: {fmt.format(f"/{parent}/")}'

            if p2 := CONFIG.get((section, key[0], 'folder')):
                p2 = p2.relative_to(paths.folders.output).relative_to(parent)
                key[0] = f'{key[0]}: {fmt.format(f"/{p2!s}/")}'

        key = [*header, *key]
        out[tuple(key)] = items

    return out


def _write_hdu_products_xlsx(run, paths, overview, filename=None, sheet=None,
                             overwrite=True):

    base, files = 'base', 'fits'
    tree = _get_hdu_products(run, paths, overview)
    tree = tree.transform(_sub_titles)

    tbl = Table.from_dict(tree,
                          title='HDU Data Products',
                          convert={Path: hyperlink_ext,
                                   files: hyperlink_ext,
                                   #    'Overview': hyperlink_name
                                   })
    # create table
    # col_sort = op.index(

    # section, tmp = paths.templates['TEL'].find('mosaic').flatten().popitem()
    # tmp.substitute(TEL=CONFIG.registration.params.survey)
    # path = rplot_paths.mosaic.parent
    # out['Images', 'Overview'] = [[(paths.folders.plotting / _)
    #                               for _ in overview['plotting']]] * len(run)

    # order = ('FITS', 'Images', 'Tracking', 'Lightcurves')
    # sections, headers, *data = cosort(*zip(*sections), *desired_files.values(),
    #                                   key=order.index)

    # write
    # header_formatter=str.title

    return tbl.to_xlsx(
        filename, sheet, overwrite=overwrite,
        formats={base: str,
                 ...:  ';;;[Blue]@'},
        widths={base:       14,
                files:      5,
                'headers':  7,
                # 'Overview': 4,
                'samples':  7,
                ...:        7},
        align={base: '<',
               #    'Overview': dict(horizontal='center',
               #                     vertical='center',
               #                     text_rotation=90),
               ...: dict(horizontal='center',
                         vertical='center')},
        merge_unduplicate=('headers', )  # 'Overview',
    )


# ---------------------------------------------------------------------------- #
def _nightly_products_table(run, paths):

    # templates = paths.templates['DATE'].flatten()
    # desired_files = get_desired_products(run, templates, by='date')
    # headers = [_get_column_header('DATE', s, paths) for s in templates.keys()]

    tree = _get_nightly_products(run, paths)
    fmt = motley.stylize('{!s:|turquoise}')
    out = _add_path_info(tree, paths, fmt)

    return Table.from_dict(
        out,
        title='Nightly Data Products',
        convert={Path: lambda f: human_time(get_file_age(f))},
        align='<',
        col_groups_align='<',
        **CONFIG.console.products
    )


def get_header(key, depth):
    key = list(key)

    # replace 'filename' header
    fixup = {'lightcurves': 'ts'}
    repl = fixup.get(key[0], '')
    replace(key, 'filename', repl)

    # balance depth of the branches for table
    while len(key) < depth:
        key.insert(-1, repl)

    return tuple(key)


def _get_nightly_products(run, paths):

    #
    date = 'date'
    templates = _get_templates(paths, date.upper())
    to_remove = ('by_file', 'by_date', 'concat')
    desired_files = get_desired_products(run, templates, by=date)
    desired_files = desired_files.transform(Partial(tpop)(o, *to_remove))
    names, stacked = desired_files.stack(level=0)

    #
    tree = DictNode()
    tree['input', date] = names
    tree.update(stacked)
    return tree.transform(Partial(get_header)(o, tree.depth()))


def _write_nightly_products_xlsx(run, paths, filename, sheet=None,
                                 overwrite=True):

    tree = _get_nightly_products(run, paths)
    tree = tree.transform(_sub_titles)

    # fmt = motley.stylize('{!s:|turquoise}')
    # tree = _add_path_info(tree, paths, fmt)

    # section, subsection, *key = key
    # section = CONFIG.get((section, 'tab'), section.title())
    # if subsection in SECTIONS:
    #     section = SECTIONS[subsection]

    # key = [*ensure_tuple(section), subsection, *key]

    tbl = Table.from_dict(
        tree,
        title='Nightly Data Products',
        convert={Path: hyperlink_ext},
        too_wide=False
    )

    date = 'date'
    return tbl.to_xlsx(
        filename, sheet, overwrite=overwrite,
        formats={date: str,
                 ...: ';;;[Blue]@'},
        widths={date: 10,
                ...: 6},
        align={date: '<',
               ...: dict(horizontal='center',
                         vertical='center')}
    )


# def _get_column_header(base, keys, paths):
#     section, *_ = keys
#     rpath = paths.get_folder(keys).relative_to(paths.folders.output)

#     return (CONFIG[section].get('title', ''),
#             f'{rpath.parent}/',
#             f'{rpath.name}/',
#             Path(paths.templates[(base, *keys)].template).name)

def write_xlsx(run, paths, overview):

    # HDU
    filename, *sheet = str(paths.files.info.products.by_file).split('::')
    _write_hdu_products_xlsx(run, paths, overview, filename, *sheet)

    # DATE
    filename, *sheet = str(paths.files.info.products.by_date).split('::')
    return _write_nightly_products_xlsx(run, paths, filename, *sheet)
