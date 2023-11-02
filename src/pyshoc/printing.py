"""
Pretty printing list of filenames as tree structures.
"""

# std
import re

# local
from pyxides.pprint import PrettyPrinter
from recipes.shell import bash
from recipes.logging import LoggingMixin


#                            prefix,    year,  month, date,  nr
RGX_FILENAME = re.compile(r'(SH[ADH]_|)(\d{4})(\d{2})(\d{2})(.+)')

# ---------------------------------------------------------------------------- #
# class Node(bash.BraceExpressionNode):
#     def __init__(self, name, parent=None, children=None, **kwargs):
#         super().__init__(name, parent=parent, children=children, **kwargs)

#         # tree from parts of filename  / from letters
#         itr = iter(mo.groups() if (mo := RGX_FILENAME.match(name)) else name)
#         self.get_prefix = lambda _: next(itr)

# class Node(bash.BraceExpressionNode):
#     def make_branch(self, words):
#         for base, words in itt.groupby(filter(None, words), self.get_prefix):
#             child = self.__class__(base, parent=self)
#             child.make_branch((remove_prefix(w, base)
#                                for w in filter(None, words)))


def _split_filenames(names):
    for file in names:
        if (mo := RGX_FILENAME.match(file)):
            yield mo.groups()
            continue

        raise ValueError('Filename does not have YYYYMMDD.nnn pattern')


def get_tree_ymd(names, depth=-1):
    tree = bash.BraceExpressionNode.from_list(_split_filenames(names))
    tree.collapse(depth)
    return tree

# ---------------------------------------------------------------------------- #


class TreeRepr(PrettyPrinter, LoggingMixin):

    brackets: str = ('', '')
    depth = 1

    def get_tree(self, run, depth=None):
        if depth is None:
            depth = self.depth
        try:
            # Filenames partitioned by year, month day
            return get_tree_ymd(run.files.names, depth)
        except ValueError as err:
            self.logger.debug(
                'Failed to get filename tree with YYYYMMDD.nnn pattern. '
                'Building tree letter by letter'
            )

        # fully general partitioning of filenames
        return bash.get_tree(run.files.names, depth)

    def joined(self, run):
        return self.get_tree(run).render()


class BraceContract(TreeRepr):
    """
    Make compact representation of files that follow a numerical sequence.  
    Eg:  'SHA_20200822.00{25,26}.fits' representing
         ['SHA_20200822.0025.fits', 'SHA_20200822.0026.fits']
    """

    per_line = 1
    depth = 1

    # def __call__(self, run):
    #     if len(run) <= 1:
    #         return super().__call__(run)

    #     # contracted
    #     return super().__call__(run)

    # @ftl.rlu_cache()
    def joined(self, run):
        return PrettyPrinter.joined(self, self.get_tree(run).to_list())
