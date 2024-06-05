
# std
import os
import pwd
import itertools as itt
from pathlib import Path
from importlib.metadata import version

# third-party
from loguru import logger
from matplotlib import rcParams
from platformdirs import user_data_path

# local
import motley
from recipes import op
from recipes.shell import bash
from recipes.functionals import always
from recipes.string import Template, sub
from recipes.containers.dicts import DictNode
from recipes.containers import ensure, replace
from recipes.config import ConfigNode, create_user_config
from recipes.functionals.partial import Partial, placeholder as o


# ---------------------------------------------------------------------------- #
# user_config_path = user_config_path('pyshoc')

# Create user config file if needed
user_config_path = create_user_config('config.yaml', __file__,
                                      version_stamp=version('pyshoc'))

# ---------------------------------------------------------------------------- #


def get_username():
    return pwd.getpwuid(os.getuid())[0]


def load(file=None):

    if file is None:
        file = Path(__file__).with_suffix('.yaml')

    # Load config file
    config = ConfigNode.load(file)

    # load cmasher if needed
    # ---------------------------------------------------------------------------- #
    for _ in config.select('cmap').filtered(values=None).flatten().values():
        if _.startswith('cmr.'):
            # load the cmasher colormaps into the matplotlib registry
            import cmasher
            break

    # coerce list to tuple so we can cache safely
    if r := config.detection.roundness:
        config.detection['roundness'] = tuple(r)

    # user details
    # ---------------------------------------------------------------------------- #
    # calibration database default
    if not config.calibration.get('folder'):
        config.calibration['folder'] = user_data_path('pyshoc') / 'caldb'

    # set remote username default
    if not config.remote.get('username'):
        config.remote['username'] = get_username()

    # logging
    # ---------------------------------------------------------------------------- #
    # uppercase logging level
    for sink, cfg in config.logging.select(('file', 'console')).items():
        config.logging[sink, 'level'] = cfg.level.upper()

    # stylize log repeat handler
    config.logging.console['repeats'] = motley.stylize(config.logging.console.repeats)
    config.console.cutouts['title'] = motley.stylize(config.console.cutouts.pop('title'))

    # stylize progressbar
    prg = config.console.progress
    prg['bar_format'] = motley.stylize(prg.bar_format)

    # GUI
    # ---------------------------------------------------------------------------- #
    # Convert tab specifiers to tuple
    config.update(config.select('tab').map(ensure.tuple))

    # make config read-only
    config.freeze()

    # plot config
    rcParams.update({
        'font.size':        config.plotting.font.size,
        'axes.labelweight': config.plotting.axes.labelweight,
        'image.cmap':       config.plotting.cmap
    })
    #  rc('text', usetex=False)

    # set in module namespace
    global CONFIG
    CONFIG = config

    return config


# Configure
# ---------------------------------------------------------------------------- #
#  Load
CONFIG = load()

_Null = object()


# Module attribute lookup goes through CONFIG
def __getattr__(section):
    if section in CONFIG:
        return CONFIG[section]

    raise AttributeError(section)


def get(section, *default):
    return CONFIG.get(section, *default)


#
GROUPING = {
    'by_file':  ('HDU',  'file.stem'),
    'by_date':  ('DATE', 't.date_for_filename'),
    'by_orbit': ('E',    't.orbits_partial')
}

SAVE_KWS = ('show', 'tab', 'filename', 'overwrite')


# ---------------------------------------------------------------------------- #
# Get file / folder tree for config
# PATHS = get_paths(CONFIG)

_section_aliases = dict(
    plotting='plots',
    lightcurves='lc',
    calibration='cal',
    registration='registry'
)


# ---------------------------------------------------------------------------- #
# Path template resolution

def resolve_internal_path_refs(folders, **aliases):
    return _resolve_internal_path_refs(get_internal_path_refs(folders, **aliases))


def _resolve_internal_path_refs(subs):
    return dict(zip(subs, (sub(v, subs) for v in subs.values())))


def get_internal_path_refs(folders, **aliases):
    return {f'${name.upper()}': str(loc).rstrip('/')
            for name, loc in _get_internal_path_refs(folders, **aliases)}


def _get_internal_path_refs(folders, **aliases):
    for name, loc in folders.items():
        if isinstance(loc, DictNode):
            if (loc := loc.get('folder')):
                yield name, loc
            continue

        yield name, loc

        if name := aliases.get(name):
            yield name, loc


def _prefix_paths(node, prefix):

    prefix = Path(prefix)

    if isinstance(node, (str, Path)):
        return prefix / node

    done = set()
    paths, parents = node.split('folder')
    needs_prefix = paths.map(Path).filter(values=Path.is_absolute)
    for key, parent in parents.flatten().items():
        # set folder again on paths for completeness
        paths[key] = current = prefix / parent

        # last key is 'folder', ignore
        section = key[:-1]
        # only prefix relative paths
        for key, path in needs_prefix[section].flatten().items():
            done.add(key := (*section, *key))
            paths[key] = current / path

    # prefix paths without explicit folder
    for key in (set(paths.flatten().keys()) - done):
        paths[key] = prefix / paths[key]

    return paths


def _is_special(path):
    return ('$HDU' in (s := str(path))) or ('$DATE' in s)


def _ignore_any(ignore):

    if isinstance(ignore, str):
        ignore = [ignore]

    if not (ignore := list(ignore)):
        return always(False)

    def wrapper(keys):
        return any(key in ignore for key in keys)

    return wrapper


# ---------------------------------------------------------------------------- #

class Template(Template):

    def resolve_path(self, hdu=None, frames=(), partial=False, **kws):
        subs = self.get_subs(hdu, frames, partial, **kws)
        return self.sub(partial, **subs, **kws)

    def get_subs(self, hdu=None, frames=(), partial=False, **kws):
        identifiers = set(self.get_identifiers())

        # convert kws to uppercase
        subs = {key.upper(): str(val) for key, val in kws.items()}

        missing = identifiers - set(subs)
        if not partial and (missing & ({'HDU', 'DATE', 'E'})):
            assert hdu

        for grouping, (key, attr) in GROUPING.items():
            if hdu and key in missing:
                subs[key] = op.AttrGetter(attr)(hdu)

        if 'FRAMES' in missing:
            j, k = frames or kws.pop('FRAMES')
            if j and k and (j, k) != (0, hdu.nframes):
                subs['FRAMES'] = '{j}-{k}'.format(j=j, k=k)
            else:
                subs['FRAMES'] = ''

        return subs

    def sub(self, partial=False, **kws):
        obj = super().sub(partial, **kws)
        if partial:
            return obj
        return Path(obj)

    def resolve_paths(self, hdu=None, section=(), partial=False, **kws):
        return list(self._resolve_paths(hdu, section, partial, **kws))

    def _resolve_paths(self, hdu, section, partial, **kws):

        kws = self.get_subs(hdu, partial=partial, **kws)

        if '$EXT' in self.template:
            section = ensure.tuple(section)
            # search upwards in config for `formats`
            for i in range(1, len(section) + 1)[::-1]:
                if formats := CONFIG[section[:i]].find('formats').get('formats'):
                    for ext in formats:
                        yield self.sub(partial, **{**kws, 'EXT': ext})
                    return

            raise ValueError(
                f'No formats specified in config section {section} for '
                f'template: {self}.'
            )

        # expand braced expressions
        for tmp in bash.brace_expand(self.template):
            yield Template(tmp).sub(partial, **kws)


# ---------------------------------------------------------------------------- #

class PathConfig(ConfigNode):  # AttributeAutoComplete
    """
    Filesystem trse helper. Attributes `files` and `folders` point to the full
    system paths for pipeline data products.
    """

    @classmethod
    def from_config(cls, root, output, config):

        # input / output root paths
        root = Path(root).resolve()
        output = root / output

        # split folder / filenames from config
        # create root node
        node = cls()
        attrs = [('files', 'filename'), ('folders', 'folder')]
        remapped_keys = DictNode()
        # catch both singular and plural form keywords
        for (key, term), s in itt.product(attrs, ('', 's')):
            found = config.find(term + s, True, remapped_keys[key])
            for keys, val in found.flatten().items():
                node[(key, *keys)] = val

        # read in paths, fill templates
        node.resolve_paths(output)

        # add root
        node.folders['root'] = node.folders['input'] = root

        # update config!
        # for (kind, *orignal), new in remapped_keys.flatten().items():
        #     config[tuple(orignal)] = node[(kind, *new)]

        # isolate the file template patterns
        templates = node.files.select(values=lambda v: '$' in str(v))
        # sort sections
        section_order = ('info', 'samples', 'tracking', 'lightcurves')
        templates = templates.sorted(section_order).map(str).map(Template)
        for section, tmp in templates.flatten().items():
            node[('templates', tmp.get_identifiers()[0], *section)] = tmp

        # make readonly
        node.freeze()

        return node

    # def __repr__(self):
    #     return dicts.pformat(self, rhs=self._relative_to_output)

    # def _relative_to_output(self, path):
    #     print('ROOT', self._root(), '-' * 88, sep='\n')
    #     out = self._root().folders.output
    #     return f'/{path.relative_to(out)}' if out in path.parents else path

    # @cached.
    # @property
    # def _section_folders(self):
    #     return DictNode(self.folders.reshape(Partial(remove)(o, 'folder')))

        # return [key[:(-1 if key[-1] == 'folder' else None)]
        # for key in self.folders.flatten().keys()]

    def create(self, ignore=()):
        logger.debug('Checking for missing folders in output tree.')

        required = set(self.folders
                       .filter(('root', 'input'))
                       .filter(_ignore_any(ignore))
                       .filter(values=_is_special)
                       .filter(values=Path.exists)
                       .flatten().values())

        if not required:
            logger.debug('All folders in output tree already exist.')
            return

        logger.info('The following folders will be created:\n    {}.',
                    '\n    '.join(map(str, required)))

        for path in required:
            logger.debug('Creating folder: {}.', path)
            path.mkdir(parents=True)

    def resolve_paths(self, output):
        # resolve files / folders
        self.resolve_folders(output)
        self.resolve_files()
        # All paths resolved as far as possbile (barring $HDU $DATE templates)

        # ensure we have unique path to each folder node
        folders = DictNode()
        for section, folder in self.folders.flatten().items():
            if section in self.files:
                section = (*section, 'folder')
            folders[section] = folder

        # add file parent folders to node.folders
        parents = DictNode(self.files.map(Path.parent.fget))
        parents = parents.reshape(Partial(replace)(o, 'filename', 'folder'))
        folders.update(parents)
        self['folders'] = folders

    def resolve_folders(self, output):
        # resolve internal folder references $HDU etc. Prefix paths when needed.

        # path $ substitutions
        folders = self.folders
        folders['output'] = ''
        substitutions = resolve_internal_path_refs(folders, **_section_aliases)

        # Convert folders to absolute paths
        folders = folders.map(sub, substitutions)
        self['folders'] = _prefix_paths(folders, output)

    def resolve_files(self):
        # convert filenames to absolute paths where necessary

        # find config sections where 'filename' given as relative path, and
        # there is also a 'folder' given in the same group. Prefix the filename
        # with the folder path.

        # sub internal path refs
        substitutions = get_internal_path_refs(self.folders, **_section_aliases)

        files = self.files.map(sub, substitutions).map(Path)
        needs_prefix = files.filter(values=Path.is_absolute)

        for section, path in needs_prefix.flatten().items():
            if prefix := self.get_folder(section):
                files[section] = prefix / path

        # make sure everything converted to full path
        assert len(files.filter(values=Path.is_absolute)) == 0

        self['files'] = files

    def get_folder(self, section):
        if folder := self.folders.get((*section, 'folder')):
            return folder

        for end in range(1, len(section) + 1)[::-1]:
            if (folder_key := section[:end]) in self.folders:
                folder = self.folders[folder_key]
                return getattr(folder, 'folder', folder)
