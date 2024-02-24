

# std
from recipes.functionals import echo
import itertools as itt
from pathlib import Path

# third-party
from loguru import logger

# local
import motley
from motley.table import Table
from motley.table.xlsx import hyperlink_ext
from recipes.functionals import echo
from recipes.containers import ensure
from recipes import cosort, op, string
from recipes.tree import FileSystemNode
from recipes.logging import LoggingMixin
from recipes.containers.lists import remove
from recipes.containers.dicts import DictNode
from recipes.functionals.partial import Map, Partial, over, PlaceHolder as o

# relative
from .. import CONFIG
from ..config import Template
from .utils import get_file_age, human_time


# ---------------------------------------------------------------------------- #


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

        #
        files = files.reshape(Partial(remove)(o, self.to_remove))
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

# SAMPLE = ['sample']
# SERIES = ['raw', 'flagged', 'diff0', 'diff', 'decor']
SPECTRAL = dict(zip(
    ['periodogram', 'lombscargle', 'welch', 'tfr', 'acf'],
    itt.repeat('sde')))

# SECTIONS = dict((*zip(SAMPLE, itt.repeat('Sample Images')),
#                  *zip(SERIES, itt.repeat('Light Curves')),
#                  *zip(SPECTRAL, itt.repeat('Spectral Density Estimates'))))
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


# ---------------------------------------------------------------------------- #

def replace_from(lookup, keys, level, at=None, fallback=None):

    if level < len(keys):
        keys = list(keys)
        at = int(level if at is None else at)

        if new := lookup.get(keys[level]):
            keys[at] = new
        elif fallback:
            keys[at] = fallback(keys[at])

    return tuple(keys)


def insert_from(key, lookup, level, insert=0):

    if new := lookup.get(key[level]):
        return (*key[:insert], *ensure.list(new), *key[insert:])

    return key


def balance_depth(key, depth, insert=''):
    # balance depth of the branches for table
    key = list(key)
    while len(key) < depth:
        key.insert(-1, insert)

    return tuple(key)


# ---------------------------------------------------------------------------- #

def add_path_info(section, info, fmt='{}: /{}/'):
    return tuple(_add_path_info(section, info, fmt))


def _add_path_info(section, info, fmt):
    for key, rpath in itt.zip_longest(section, info):
        yield fmt.format(key, rpath) if rpath else key


def get_path_info(sections, paths, templates):
    return DictNode(_get_path_info(sections, paths, templates))


def _get_path_info(sections, paths, templates):
    for section in sections:
        section = section[:-1]
        rpaths = tuple(_rpaths(section, paths, fill=False))
        if tmp := templates.get(section):
            yield section, (*rpaths, Path(tmp.template).name)


def get_relative_paths(sections, paths, depth=-1):
    return {s: tuple(_rpaths(s, paths, depth)) for s in sections}


def _rpaths(section, paths, depth=-1, fill=False, fill_value=''):
    parent = paths.folders.output
    depth = depth % (n := len(section))
    for i in range(1, n + 1):
        if ((i <= depth) and (folder := paths.get_folder(section[:i]))
                and (folder != parent)):
            yield str(folder.relative_to(parent))
            parent = folder
        elif fill:
            yield fill_value


def replace_section_title(key, at='tab', level=0, fallback=str.title):
    section = key[level]
    new = CONFIG.get((section, at), fallback(section))
    return (*key[:level], *ensure.list(new),  *key[(level + 1):])


def get_titles(section, lookup_at='tab', fallbacks={0: str.title}):
    return tuple(_get_titles(section, lookup_at, fallbacks))


def _get_titles(section, lookup_at, fallbacks):
    for i in range(1, len(section) + 1):
        key = section[:i]
        if not (title := CONFIG.get((*key, lookup_at))):
            fallback = fallbacks.get(i, echo)
            title = fallback(key[-1])

        yield from ensure.list(title)

# ---------------------------------------------------------------------------- #


def _get_hdu_products(run, paths, overview=None, **kws):

    #
    templates = _get_templates(paths, 'HDU')
    desired_files = get_desired_products(run, templates, by='file')

    # input for table
    out = DictNode()
    base, files = 'base', 'fits'
    out['input', base] = list(desired_files.keys())
    out['input', files] = run.files.paths

    # add files
    out.update(desired_files.stack(level=0))

    if overview:
        # re-section
        out['registration', 'samples'] = out.pop('samples')

        # duplicate Overview images so that they get merged below
        rplot_paths = paths.files.registration.plots
        path = rplot_paths.alignment.parent
        out['registration', 'alignment', 'png'] = \
            [path / _ for _ in overview['registration'][f'{path.name}/']]

    out = out.sorted(['input', 'info', 'registration', 'lightcurves'])

    return out, DictNode(templates)


def _hdu_products_table(run, paths):
    #
    tree, templates = _get_hdu_products(run, paths)

    # get relative paths
    sections = list(tree.flatten().keys())
    path_info = get_path_info(sections, paths, templates)

    # replace 'filename' header
    fixup = {'filename': 'ts'}
    headers = Map(replace_from)(fixup, Over(sections), -2)

    # get section title
    headers = Map(replace_from)(SPECTRAL, Over(headers), -2, 0)

    # remove verbose keys
    to_remove = ('by_file', 'by_date', 'concat')
    headers = Map(remove)(Over(headers), *to_remove)

    # balance depth of branches
    depth = max(map(len, headers))
    headers = Map(balance_depth)(Over(headers), depth)

    from IPython import embed
    embed(header="Embedded interpreter at 'src/pyshoc/pipeline/products.py':472")

    # get section titles
    fmt = motley.stylize(R'{{}: {{!s}:|turquoise}:|bold}')
    new = []
    for section, header in zip(sections, headers):
        titles = get_titles(header)
        infos = path_info.get(section[:-1], ())
        new.append(add_path_info(titles, infos, fmt))

    # finally combine headers and data
    out = DictNode(zip(new, tree.flatten().values()))

    return Table.from_dict(
        out,
        title='HDU Data Products',
        converters={Path: Partial(get_file_age)(o, human=True)},
        align='<',
        col_groups_align='<',
        subtitle=fmt.format('Output folder', paths.folders.output),
        subtitle_align='<',
        subtitle_style=('_'),
        **CONFIG.console.products
    )


def _write_hdu_products_xlsx(run, paths, overview, filename=None, sheet=None,
                             overwrite=True):

    base, files = 'base', 'fits'
    tree = _get_hdu_products(run, paths, overview)
    tree = tree.reshape(replace_section_title)

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
# def _get_column_header(base, keys, paths):
#     section, *_ = keys
#     rpath = paths.get_folder(keys).relative_to(paths.folders.output)

#     return (CONFIG[section].get('title', ''),
#             f'{rpath.parent}/',
#             f'{rpath.name}/',
#             Path(paths.templates[(base, *keys)].template).name)


def _get_nightly_products(run, paths):

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

    #
    from IPython import embed
    embed(header="Embedded interpreter at 'src/pyshoc/pipeline/products.py':561")

    date = 'date'
    templates = _get_templates(paths, date.upper())
    to_remove = ('by_file', 'by_date', 'concat')
    desired_files = get_desired_products(run, templates, by=date)
    desired_files = desired_files.reshape(Partial(remove)(o, *to_remove))

    #
    tree = DictNode()
    tree['input', date] = list(desired_files.keys())
    tree.update(desired_files.stack(level=0))
    return tree.reshape(Partial(get_header)(o, tree.depth()))


def _write_nightly_products_xlsx(run, paths, filename, sheet=None,
                                 overwrite=True):

    tree = _get_nightly_products(run, paths)
    tree = tree.reshape(replace_section_title)
    # tree = tree.reshape(_replace_key_from(o, SECTIONS, 2, True))

    # tree = _add_path_info(tree, paths, '{!s}')

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


def _nightly_products_table(run, paths):

    # templates = paths.templates['DATE'].flatten()
    # desired_files = get_desired_products(run, templates, by='date')
    # headers = [_get_column_header('DATE', s, paths) for s in templates.keys()]

    tree = _get_nightly_products(run, paths)
    fmt = motley.stylize(R'{{}: {/{!s}/:|turquoise}:|bold}')
    out = _add_path_info(tree, paths, fmt)

    return Table.from_dict(
        out,
        title='Nightly Data Products',
        convert={Path: Partial(get_file_age)(o, human=True)},
        align='<',
        col_groups_align='<',
        **CONFIG.console.products
    )


def write_xlsx(run, paths, overview):

    # HDU
    filename, *sheet = str(paths.files.info.products.by_file).split('::')
    _write_hdu_products_xlsx(run, paths, overview, filename, *sheet)

    # DATE
    filename, *sheet = str(paths.files.info.products.by_date).split('::')
    return _write_nightly_products_xlsx(run, paths, filename, *sheet)
