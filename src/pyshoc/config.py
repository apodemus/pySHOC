
# std
import os
import pwd
import itertools as itt
from pathlib import Path
from string import Template

# third-party
from loguru import logger
from platformdirs import user_config_path, user_data_path

# local
import motley
from recipes.shell import bash
from recipes.string import sub
from recipes.caching import cached
from recipes.dicts import DictNode
from recipes.config import ConfigNode
from recipes.utils import ensure_tuple
from recipes.functionals import always, negate


# ---------------------------------------------------------------------------- #

def get_username():
    return pwd.getpwuid(os.getuid())[0]


# ---------------------------------------------------------------------------- #
# Load package config
CONFIG = ConfigNode.load_module(__file__)


# load cmasher if needed
# ---------------------------------------------------------------------------- #
plt = CONFIG.plotting

for _ in CONFIG.select('cmap').filtered(values=None).flatten().values():
    if _.startswith('cmr.'):
        # load the cmasher colormaps into the matplotlib registry
        import cmasher
        break


# user details
# ---------------------------------------------------------------------------- #
user_config_path = user_config_path('pyshoc')

# calibration database default
if not CONFIG.calibration.get('folder'):
    CONFIG.calibration['folder'] = user_data_path('pyshoc') / 'caldb'

# set remote username default
if not CONFIG.remote.get('username'):
    CONFIG.remote['username'] = get_username()


# logging
# ---------------------------------------------------------------------------- #
# uppercase logging level
for sink, cfg in CONFIG.logging.select(('file', 'console')).items():
    CONFIG.logging[sink, 'level'] = cfg.level.upper()
del sink, cfg


# stylize log repeat handler
CONFIG.logging.console['repeats'] = motley.stylize(CONFIG.logging.console.repeats)
CONFIG.console.cutouts['title'] = motley.stylize(CONFIG.console.cutouts.pop('title'))


# stylize progressbar
prg = CONFIG.console.progress
prg['bar_format'] = motley.stylize(prg.bar_format)
del prg


# GUI
# ---------------------------------------------------------------------------- #
# Convert tab specifiers to tuple
CONFIG.update(CONFIG.select('tab').map(ensure_tuple))


# make config read-only
CONFIG.freeze()

# ---------------------------------------------------------------------------- #
# Get file / folder tree for config
# PATHS = get_paths(CONFIG)

_section_aliases = dict(registration='registry',
                        plotting='plots')


# ---------------------------------------------------------------------------- #


def resolve_internal_path_refs(folders, **aliases):
    return _resolve_internal_path_refs(_get_internal_path_refs(folders, **aliases))


def _resolve_internal_path_refs(subs):
    return dict(zip(subs, (sub(v, subs) for v in subs.values())))


def _get_internal_path_refs(folders, **aliases):
    return {f'${aliases.get(name, name).upper()}': str(loc).rstrip('/')
            for name, loc in folders.items()}


def _prefix_paths(node, prefix):
    # print(f'{node = }    {prefix = }')

    if isinstance(node, PathConfig):
        result = PathConfig()
        if 'folder' in node:
            node, folder = node.split('folder')
            prefix = Path(prefix) / folder.folder
            result['folder'] = prefix

        result.update({
            key: _prefix_paths(child, prefix)
            for key, child in node.items()
        })
        return result

    if prefix:
        return Path(prefix) / node

    return node


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

    def get_identifiers(self):
        # NOTE: python 3.11 has Template.get_identifiers
        _, keys, *_ = zip(*self.pattern.findall(self.template))
        return keys

    def __repr__(self):
        return f'{type(self).__name__}({self.template})'

    def resolve_paths(self, section, **kws):
        if '$EXT' in self.template:
            if formats := CONFIG[section[0]].find('formats').get('formats'):
                for ext in formats:
                    yield Path(self.substitute(**{'EXT': ext, **kws}))
                return

            raise ValueError(
                f'No formats specified in config section {section} for '
                f'template: {self.template} '
            )

        # expand braced expressions
        yield from map(Path, bash.brace_expand(self.substitute(**kws)))

# ---------------------------------------------------------------------------- #

class PathConfig(ConfigNode):  # AttributeAutoComplete
    """
    Filesystem tree helper. Attributes point to the full system folders and
    files for pipeline data products.
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
            found = config.find(term + s,  True, remapped_keys[key])
            for keys, val in found.flatten().items():
                node[(key, *keys)] = val

        # resolve files / folders
        node.resolve_folders(output)
        node.resolve_files()
        # All paths are resolved

        # update config!
        for (kind, *orignal), new in remapped_keys.flatten().items():
            config[tuple(orignal)] = node[(kind, *new)]

        # add root
        node.folders['root'] = root

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

    @cached.property
    def _folder_sections(self):
        return [key[:(-1 if key[-1] == 'folder' else None)]
                for key in self.folders.flatten().keys()]

    def get_folder(self, section):
        for end in range(1, len(section) + 1)[::-1]:
            if section[:end] in self._folder_sections:
                folder_key = section[:end]
                break

        folder = self.folders[folder_key]

        return getattr(folder, 'folder', folder)

    def create(self, ignore=()):
        logger.debug('Checking for missing folders in output tree.')

        node = self.filtered(_ignore_any(ignore))
        required = {*node.folders.flatten().values(),
                    *map(Path.parent.fget, node.files.flatten().values())}
        required = set(filter(negate(_is_special), required))
        required = set(filter(negate(Path.exists), required))

        if not required:
            logger.debug('All folders in output tree already exist.')
            return

        logger.info('The following folders will be created:\n    {}.',
                    '\n    '.join(map(str, required)))

        for path in required:
            logger.debug('Creating folder: {}.', path)
            path.mkdir(parents=True)

    def resolve_folders(self, output):
        # resolve internal folder references $EXAMPLE. Prefix paths where needed

        # path $ substitutions
        folders = self.folders
        folders['output'] = ''
        substitutions = resolve_internal_path_refs(folders, **_section_aliases)

        # Convert folders to absolute paths
        self['folders'] = _prefix_paths(folders.map(sub, substitutions), output)

    def resolve_files(self):
        # convert filenames to absolute paths where necessary

        # find config sections where 'filename' given as relative path, and
        # there is also a 'folder' given in the same group. Prefix the filename
        # with the folder path.

        # sub internal path refs
        substitutions = _get_internal_path_refs(self.folders, **_section_aliases)

        files = self.files.map(sub, substitutions).map(Path)
        needs_prefix = files.filter(values=Path.is_absolute)

        for section, path in needs_prefix.flatten().items():
            prefix = self.get_folder(section)
            files[section] = prefix / path

        # make sure everything converted to full path
        assert len(files.filter(values=Path.is_absolute)) == 0

        self['files'] = files
