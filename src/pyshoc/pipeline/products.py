

# std
import itertools as itt
from pathlib import Path

# third-party
import more_itertools as mit
from loguru import logger

# local
from motley.table import Table
from recipes import cosort, io
from recipes.dicts import DictNode, vdict

# relative
from .. import CONFIG
from . import _file_struct
from .utils import get_file_age, human_time


# ---------------------------------------------------------------------------- #
TABLE_STYLE = dict(title_props={'bg': 'darkgreen',
                                'fg': ('lightgrey', 'B', '_')},
                   col_head_props={'bg': 'lightgreen',
                                   'fg': ('darkslategrey', 'B')})


# ---------------------------------------------------------------------------- #
def _match(run, filenames, reference, empty):

    incoming = {stem: list(vals) for stem, vals in
                itt.groupby(sorted(mit.collapse(filenames)), Path.stem.fget)}

    if reference is None:
        _, reference = cosort(run.attrs('t.t0'), run.files.stems)

    for base in reference:
        yield incoming.get(base, empty)


def match(run, filenames, reference=None, empty=''):
    if isinstance(filenames, Path) and filenames.is_dir():
        filenames = filenames.iterdir()

    return list(_match(run, filenames, reference, empty))


# ---------------------------------------------------------------------------- #
class Node(DictNode, vdict):
    pass


def get_previous(run, paths):

    cfg = CONFIG.files
    overview = Node()
    for key, items in _file_struct.items():
        for item in items:
            path = paths.output / cfg[item]
            overview[key][path.name] = get_file_age(path)

    overview.freeze()

    hdu_products = Node()
    for stem in run.files.stems:
        # get_info = ftl.partial(_get_info, stem)
        for path in io.iter_files(paths.output/f'**/{stem}*', recurse=True):
            mid = str(path.parent.relative_to(paths.output))  # .replace(stem, '*')
            end = path.name  # .replace(stem, '*')
            hdu_products[stem][mid][end] = get_file_age(path)

    hdu_products.freeze()

    #

    def get_key(key):
        base, mid, end = key
        return f'/{mid.replace(base, "*")}/', f'{end.replace(base, "*")}'

    # op.index()

    logger.opt(lazy=True).debug(
        'Found previous data products: \n{}\n{}\n',
        lambda: Table.from_dict(
            overview,
            convert_keys='/'.join,
            title='Data Products: Overview',
            col_headers=['file', 'Age'],
            formatter=human_time,
            order='c',
            align={'Age': '>'},
            **TABLE_STYLE,
        ),
        lambda: Table.from_dict(
            hdu_products,
            convert_keys=get_key,
            title='Data Products: HDU (Age)',
            row_headers=['*.fits', *hdu_products.keys()],
            col_head_align='<',
            ignore_keys=('origin.dat', 'centroids-mean.dat', 'coords-delta.dat'),
            col_sort=('*.txt', 'flux.dat', 'flux-std.dat', 'snr.dat',
                      'centroids.dat', 'coords-std.dat', '*.png').index,
            formatter=human_time,
            **TABLE_STYLE,
        )
    )

    return overview, hdu_products

# ---------------------------------------------------------------------------- #

def hyperlink_ext(path):
    return f'=HYPERLINK("{path}", "{path.suffix[1:]}")'


def hyperlink_path(path):
    return f'=HYPERLINK("{path}", "{path.name}")'


def write_xlsx(run, paths, filename=None):

    overview, products = get_previous(run, paths)

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

