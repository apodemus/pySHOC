

# std
import itertools as itt
from pathlib import Path

# third-party
from loguru import logger

# local
import motley
from motley.table import Table
from motley.table.xlsx import hyperlink_ext
from recipes import op
from recipes.functionals import echo
from recipes.tree import FileSystemNode
from recipes.containers import ensure, remove
from recipes.containers.dicts import DictNode
from recipes.containers.dicts.node import balance_depth
from recipes.functionals.partial import Partial, PlaceHolder as o

# relative
from .. import CONFIG
from ..config import GROUPING, Template
from .utils import get_file_age


# ---------------------------------------------------------------------------- #
class ProductNode(FileSystemNode):

    @staticmethod
    def _get_name(path):
        return path.name
        # return f'{path.name}{"/" * path.is_dir()}'

    def __getitem__(self, key):
        # sourcery skip: assign-if-exp, reintroduce-else
        if self.is_leaf:
            raise IndexError('Leaf node')

        if isinstance(key, Path):
            if key.is_absolute():
                key = key.relative_to(self.root.as_path)
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
            leaf.age = get_file_age(leaf.as_path)
        return self

    def as_dict(self, attr='name', leaf_attr='age'):
        if self.is_leaf:
            return getattr(self, leaf_attr)

        return {getattr(child, attr): child.as_dict(attr, leaf_attr)
                for child in self.children}


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


def insert_from(lookup, key, level, insert=0):

    if new := lookup.get(key[level]):
        return (*key[:insert], *ensure.list(new), *key[insert:])

    return key


# ---------------------------------------------------------------------------- #

def _prepare_headers(products, path_info, to_remove, fixups, fmt, path_stop,
                     title_lookup='tab'):

    sections = list(products.flatten().keys())
    headers = DictNode(zip(sections, sections))

    # any ad hoc changes
    for fixup in fixups:
        headers = headers.map(fixup)

    # get section title
    headers = headers.map(get_titles, title_lookup)

    # remove verbose keys
    if to_remove:
        headers = headers.map(Partial(remove)(o, *to_remove))

    # add paths
    headers = add_path_infos(headers, path_info, fmt, stop=path_stop)

    # balance branch depths
    # headers = headers.balance(insert='') # FIXME
    headers = headers.map(Partial(balance_depth)(o, headers.depth()))

    # filter empty row header lines
    return remove_empty_rows(headers)


def remove_empty_rows(headers):
    # filter empty row header lines
    keys, vals = zip(*headers.flatten().items())
    return DictNode(zip(keys, zip(*filter(any, zip(*vals)))))


def get_titles(section, lookup_at='tab', fallbacks={0: str.title}):
    return tuple(_get_titles(section, lookup_at, fallbacks))[::-1]


def _get_titles(section, lookup_at, fallbacks):
    i = len(section)
    while i:
        key = section[:i]
        if not (title := CONFIG.get((*key, lookup_at))):
            fallback = fallbacks.get(i - 1, echo)
            title = fallback(key[-1])

        yield from (titles := ensure.tuple(title))[::-1]
        i -= len(titles)


def add_path_infos(headers, path_info, fmt, stop=None):
    out = DictNode()
    for section, header in headers.flatten().items():
        info = path_info.get(section[:stop], ())
        # print(f'{section = }\n{header = }\n{info = }')
        out[section] = add_path_info(header, info, fmt)
    return out


def add_path_info(section, info, fmt='{}: {}'):
    return tuple(_add_path_info(section, info, fmt))


def _add_path_info(section, info, fmt):
    for key, rpath in itt.zip_longest(section, info):
        yield fmt.format(key, rpath) if rpath else key


def get_path_info(sections, paths, templates):
    return DictNode(_get_path_info(sections, paths, templates))


def _get_path_info(sections, paths, templates):
    for section in sections:
        section = section[:-1]
        rpaths = tuple(_get_relative_paths(section, paths))
        # print(rpaths)
        if tmp := templates.get(section):
            yield section, (*rpaths, Path(tmp.template).name)


def get_relative_paths(sections, paths, depth=-1, order=1, slash=None):
    return {s: tuple(_get_relative_paths(s, paths, depth, order, slash))
            for s in sections}


def _get_relative_paths(section, paths, depth=-1, order=1, slash=None,
                        parent=None, fill=False, fill_value=''):

    top = parent = (parent or paths.folders.output)
    depth = (n if (depth == -1) else depth) if (n := len(section)) else 0

    slash = True
    slashed = [*[('/' * (slash or slash is None))] * (n - 1), '/' * bool(slash)]
    itr = zip(range(1, n + 1), slashed)
    if order == -1:
        itr = list(itr)[::-1]

    ancestors = []
    for i, slash in itr:
        if ((i <= depth) and (section[0] != 'input')
                and (folder := paths.get_folder(section[:i])) and (folder != parent)
                and isinstance(folder, Path)
            ):
            #
            if folder.is_relative_to(parent):
                yield f'/{folder.relative_to(parent)!s}{slash}'
                ancestors.append(parent)
                parent = folder
            elif folder.is_relative_to(top):
                # yield f'/{folder.relative_to(top)!s}{slash}'
                for trial in ancestors[::-1]:
                    if folder.is_relative_to(trial):
                        yield f'/{folder.relative_to(trial)!s}{slash}'
                        parent = trial
                        break

        elif fill:
            yield fill_value


def replace_section_title(key, at='tab', level=0, fallback=str.title):
    section = key[level]
    new = CONFIG.get((section, at), fallback(section))
    return (*key[:level], *ensure.list(new),  *key[(level + 1):])


# ---------------------------------------------------------------------------- #

def sanitize_filename(name):
    return name.lower().replace(' ', '')


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
    key, attr = GROUPING[f'by_{by}']

    run = run.sort_by('t.t0')
    if by == 'file':
        # sort rows
        stems = run.attrs(attr)
        return _get_desired_products(stems, templates, key, FRAMES='')

    if by == 'date':
        # sort rows
        dates = sorted(set(run.attrs(attr)))
        return _get_desired_products(dates, templates, key)

    if by == 'cycle':
        orbits = list(itt.collapse(run.attrs(attr)))
        return _get_desired_products(orbits, templates, key)


def _get_desired_products(items, templates, key, **kws):

    rows = DictNode()
    for val, (section, template) in itt.product(items, templates.items()):
        files = template.resolve_paths(section=section[0], **{key: val, **kws})
        rows[(val, *section)] = {file.suffix.strip('.'): file for file in files}

    return rows

# ---------------------------------------------------------------------------- #


def get_previous(run, paths):
    # get data products
    output = paths.folders.output
    products = ProductNode.from_path(output,
                                     ignore=('.cache', '_old', 'logs'))
    products.get_ages()

    # get overview products
    overview = _get_overview_products(paths)

    # targets = DataProducts(run, paths)

    #
    info = logger.bind(indent=' ').opt(lazy=True).info
    pprint = (
        # overview
        lambda: ('overview',  _overview_products_table(overview, paths)),

        # hdu products
        lambda: ('hdu', _hdu_products_table(run, paths)),

        # nightly data products
        lambda: ('nightly', _nightly_products_table(run, paths))
    )
    for f in pprint:
        info('Found previous {0[0]} data products: \n{0[1]}', f)

    return overview, products


def _get_overview_products(paths):
    return paths.files.filter(values=lambda s: '$' in str(s))


def _overview_products_table(overview, paths):
    return _overview_products_hstack(overview, paths)


def _overview_products_hstack(overview, paths):

    # get overview products
    overview = overview.filter('logging')

    # header edits
    to_remove = ('Overview', )
    fixup = Partial(replace_from)({'Overview': 'Registration'}, o, 0, 0)

    # path info format
    fmt = R'{{:<{width:}}: {{!s}:|turquoise}:|bold}'
    fmt_subtitle = motley.stylize(fmt, width=13)
    fmt_header = motley.stylize(fmt, width='')

    # get relative paths
    sections = list(overview.flatten().keys())
    path_info = get_relative_paths(sections, paths, slash=False)

    # Map to path name and age
    out = overview.map(lambda path: (path.name, path))

    # get headers
    headers = _prepare_headers(out, path_info, to_remove, [fixup],
                               fmt_header, None)

    return Table.from_dict(
        out.reshape(headers.get),
        # too_wide=False,
        title='Data Products: Overview',
        row_headers=['File', 'Age'],
        converters={Path: Partial(get_file_age)(o, human=True)},
        col_header_align='<',
        align='<',
        subtitle='\n'.join(
            fmt_subtitle.format(f'{io.title()}put folder',
                                getattr(paths.folders, f'{io}put'))
            for io in ('in', 'out')),
        subtitle_align='<',
        subtitle_style='_',
        **CONFIG.console.products
    )


def _write_overview_products_table(overview, paths, filename=None, sheet=None,
                                   overwrite=True):
    #
    out = overview.filter(('spreadsheets', 'logging')).sorted(['info'])
    headers = _prepare_headers(out, {}, ('plots', ), {}, '{}', 0, title_lookup='header')
    tbl = Table.from_dict(out.reshape(headers.get),
                          title='Overview Data Products ',
                          convert={Path: hyperlink_ext})

    return tbl.to_xlsx(
        filename, sheet, overwrite=overwrite,
        formats={...:  ';;;[Blue]@'},
        widths={...:        10},
        align={  # base: '<',
            #    'Overview': dict(horizontal='center',
            #                     vertical='center',
            #                     text_rotation=90),
            ...: dict(horizontal='center',
                      vertical='center')},
        merge_unduplicate=('headers', )  # 'Overview',
    )


# ---------------------------------------------------------------------------- #

def _get_hdu_products(run, paths, **kws):

    #
    templates = _get_templates(paths, 'HDU')
    desired_files = get_desired_products(run, templates, by='file')

    # input for table
    products = DictNode()
    base, files = 'base', 'fits'
    products['input', base] = list(desired_files.keys())
    products['input', files] = run.files.paths

    # add files
    products.update(desired_files.stack(level=0))

    # sort columns
    products = products.sorted(['input', 'info', 'samples', 'lightcurves'])

    return products, DictNode(templates)


def _hdu_products_table(run, paths):
    #
    products, templates = _get_hdu_products(run, paths)

    # remove verbose keys
    to_remove = ('filename', 'concat', 'by_file', 'by_date')

    fixups = (
        # get section title
        Partial(replace_from)(SPECTRAL, o, -2, 0),
        # add level header for time series
        Partial(insert_from)({'lightcurves': 'ts'}, o, 0, -1)
    )

    # get relative paths
    sections = list(products.flatten().keys())
    path_info = get_path_info(sections, paths, templates)

    # path info format
    fmt = motley.stylize(R'{{}: {{!s}:|turquoise}:|bold}')

    # get headers
    headers = _prepare_headers(products, path_info, to_remove, fixups, fmt,
                               path_stop=-1)
    row_headers = products.pop(('input', 'base'))

    tbl = Table.from_dict(
        products.reshape(headers.get),
        title='HDU Data Products',
        converters={Path: Partial(get_file_age)(o, human=True)},
        row_headers=row_headers,
        align='<',
        col_header_align='<',
        subtitle='\n'.join(fmt.format(f'{io.title()}put folder',
                                      getattr(paths.folders, f'{io}put'))
                           for io in ('in', 'out')),
        subtitle_align='<',
        subtitle_style='_',
        **CONFIG.console.products
    )

    tbl.headers_header_block[:, 0] = ('Input', '', '', 'base')
    return tbl


def _write_hdu_products_xlsx(run, paths, filename=None, sheet=None, overwrite=True):

    base, files = 'base', 'fits'
    out, templates = _get_hdu_products(run, paths)

    # remove verbose keys
    to_remove = ('filename', 'concat', 'by_file', 'by_date')

    fixups = (
        # get section title
        Partial(replace_from)(SPECTRAL, o, -2, 0),
        # add level header for time series
        Partial(insert_from)({'lightcurves': 'ts'}, o, 0, -1)
    )

    # get headers
    headers = _prepare_headers(out, {}, to_remove, fixups, '{}', -1, 'header')

    tbl = Table.from_dict(out.reshape(headers.get),
                          title='HDU Data Products',
                          convert={Path: hyperlink_ext,
                                   files: hyperlink_ext})

    # write
    widths = {base:       20,
              files:      5,
              'headers':  7,
              'samples':  7}
    widths = {headers.find(key, True, default=key).flatten().popitem()[1]: val
              for key, val in widths.items()}

    return tbl.to_xlsx(
        filename, sheet, overwrite=overwrite,
        formats={base:  None,
                 ...:   ';;;[Blue]@'},
        widths={**widths,
                ...: 7},
        align={base: '<',
               #    'Overview': dict(horizontal='center',
               #                     vertical='center',
               #                     text_rotation=90),
               ...: dict(horizontal='center',
                         vertical='center')},
        merge_unduplicate=('headers', )  # 'Overview',
    )


# ---------------------------------------------------------------------------- #

def _get_nightly_products(run, paths):

    date = 'date'
    templates = _get_templates(paths, date.upper())
    desired_files = get_desired_products(run, templates, by=date)

    #
    products = DictNode()
    products['input', date] = list(desired_files.keys())
    products.update(desired_files.stack(level=0))
    return products, templates


def _nightly_products_table(run, paths):

    #
    products, templates = _get_nightly_products(run, paths)

    # remove verbose keys
    to_remove = ()
    fixups = (
        # get section title
        Partial(replace_from)(SPECTRAL, o, 2, 0),
    )

    # get relative paths
    sections = list(products.flatten().keys())
    path_info = get_path_info(sections, paths, templates)

    # path info format
    fmt = motley.stylize(R'{{}: {{!s}:|turquoise}:|bold}')

    # get headers
    headers = _prepare_headers(products, path_info, to_remove, fixups,
                               fmt, path_stop=-1)

    return Table.from_dict(
        products.reshape(headers.get),
        title='Nightly Data Products',
        convert={Path: Partial(get_file_age)(o, human=True)},
        align='<',
        col_header_align='<',
        subtitle='\n'.join(fmt.format(f'{io.title()}put folder',
                                      getattr(paths.folders, f'{io}put'))
                           for io in ('in', 'out')),
        subtitle_align='<',
        subtitle_style='_',
        **CONFIG.console.products
    )


def _write_nightly_products_xlsx(run, paths, filename, sheet=None,
                                 overwrite=True):

    #
    products, templates = _get_nightly_products(run, paths)
    # sections = list(products.flatten().keys())

    # remove verbose keys
    to_remove = ('filename', 'concat', 'by_file', 'by_date')

    fixups = (
        # get section title
        Partial(replace_from)(SPECTRAL, o, -2, 0),
        # add level header for time series
        Partial(insert_from)({'lightcurves': 'ts'}, o, 0, -1)
    )

    # get headers
    headers = _prepare_headers(products, {}, to_remove, fixups, '{}', -1, 'header')

    date = 'date'
    tbl = Table.from_dict(
        products.reshape(headers.get),
        title='Nightly Data Products',
        convert={Path: hyperlink_ext,
                 date: str},
        too_wide=False
    )

    return tbl.to_xlsx(
        filename, sheet, overwrite=overwrite,
        formats={date: None,
                 ...: ';;;[Blue]@'},
        widths={date: 10,
                ...: 6},
        align={date: '<',
               ...: dict(horizontal='center',
                         vertical='center')}
    )


def write_xlsx(run, paths, overview):
    sheets = paths.files.info.spreadsheets

    # Overview
    filename, *sheet = str(sheets.overview).split('::')
    _write_overview_products_table(overview, paths, filename, *sheet)

    # HDU
    filename, *sheet = str(sheets.by_file).split('::')
    _write_hdu_products_xlsx(run, paths, filename, *sheet)

    # DATE
    filename, *sheet = str(sheets.by_date).split('::')
    return _write_nightly_products_xlsx(run, paths, filename, *sheet)
