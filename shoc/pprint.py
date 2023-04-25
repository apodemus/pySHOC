import more_itertools as mit
import re
# from anytree.node.nodemixin import NodeMixin
from recipes.dicts import AVDict
# from anytree import RenderTree
# from collections import defaultdict
from recipes.containers import PrettyPrinter
# import itertools as itt
from recipes import bash  # Node, brace_contract, render_tree

from recipes.logging import logging, LoggingMixin


# # module level logger
# logger = get_module_logger()
# logging.basicConfig()
# logger.setLevel(logging.INFO)

#                            prefix,    year,  month, date,    nr
RGX_FILENAME = re.compile(r'(SH[ADH]_|)(\d{4})(\d{2})(\d{2})\.(.+)')


# def name_grouper(name):
#     if not name:
#         return 0, name

#     mo = RGX_FILENAME.match(name)
#     if mo:
#         return 1, mo

#     return 2, name


# def group_names(run):
#     names = defaultdict(list)
#     for name in run.files.stems:
#         g, mo = name_grouper(name)
#         names[g].append(mo)
#     return names


def morph(dic, parent):
    for name, branch in dic.items():
        morph(branch, bash.Node(name, parent))


def get_tree_ymd(names, depth=-1):
    tree = AVDict()
    for file in names:
        mo = RGX_FILENAME.match(file)
        if mo is None:
            continue
        parts = mo.groups()
        node = tree
        for part in parts:
            node = node[part]

    root = bash.Node('')
    morph(tree, root)
    root.collapse(depth)
    return root


class TreeRepr(PrettyPrinter, LoggingMixin):

    brackets: str = ('', '')
    depth = -1

    def get_tree(self, run):
        try:
            # Filenames partitioned by year, month day
            return get_tree_ymd(run.files.names, self.depth)
        except ValueError as err:
            self.logger.debug('Failed to get filename tree with YMD pattern.\n$=%s\n'
                              'Building tree letter by letter', err)

        # more general partitioning
        return bash.get_tree(run.files.names, self.depth)

    def joined(self, run, indent=None):
        return self.get_tree(run).render()


class BraceContract(TreeRepr):
    """
    Make compact representation of files that follow a numerical sequence.  
    Eg:  'SHA_20200822.00{25,26}.fits' representing
         ['SHA_20200822.0025.fits', 'SHA_20200822.0026.fits']
    """

    per_line = 1
    depth = 1

    def __call__(self, run):

        if len(run) <= 1:
            return super().__call__(run)

        # contracted
        return super().__call__(run)

    # @ftl.rlu_cache()
    def joined(self, run):
        return PrettyPrinter.joined(self, self.get_tree(run).to_list())
