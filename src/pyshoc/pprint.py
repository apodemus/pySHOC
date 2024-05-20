"""
Pretty printing list of filenames as tree structures.
"""

# std
import re
import operator as op
import textwrap as txw
from string import Template

# third-party
import numpy as np

# local
from pyxides.pprint import PrettyPrinter
from motley.table import Table
from motley.utils import ALIGNMENT_MAP_INV
from motley.table.attrs import AttrTable, AttrColumn as Column
from recipes import pprint
from recipes.shell import bash
from recipes.oo.temp import temporary
from recipes.logging import LoggingMixin
from recipes.string import indent as indented

# relative
from .timing import Trigger


# ---------------------------------------------------------------------------- #
#                            prefix,    year,  month, date,  nr
RGX_FILENAME = re.compile(r'(SH[ADH]_|)(\d{4})(\d{2})(\d{2})(.+)')

# ---------------------------------------------------------------------------- #
# class Node(bash.BraceExpressionNode):
#     def __init__(self, name, parent=None, children=None, **kws):
#         super().__init__(name, parent=parent, children=children, **kws)

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


# ---------------------------------------------------------------------------- #
# Tables

def hms(t):
    """sexagesimal formatter"""
    return pprint.hms(t.to('s').value, unicode=True, precision=1)


def hms_latex(x, precision=0):
    return R'\hms{{{:02.0f}}}{{{:02.0f}}}{{{:04.1f}}}'.format(
        *pprint.nrs.sexagesimal(x.value, precision=precision)
    )


class TableHelper(AttrTable):

    # def get_table(self, run, attrs=None, **kws):
    #     #
    #     self._foot_fmt = ' {flag}  {info}'
    #     table = super().get_table(run, attrs, **kws)

    #     # HACK: compacted `filter` displays 'A = ∅' which is not very clear. Go more
    #     # verbose again for clarity
    #     if table.summary.items and table.summary.loc != -1:
    #         replace = {'A': 'filter.A',
    #                    'B': 'filter.B'}
    #         table.summary.items = {replace.get(name, name): item
    #                                for name, item in table.summary.items.items()}
    #         table.inset = table.summary()
    #     return table

    def to_latex(self, style='table', indent=2, **kws):
        fname = f'_to_{style}'
        if not hasattr(LatexWriter, fname):
            raise NotImplementedError(fname)

        writer = LatexWriter(self.parent)
        worker = getattr(writer, fname)
        table = worker(**kws).replace('\n    ', f'\n{" " * indent}')
        return '\n'.join(map(str.rstrip,  table.splitlines()))

    def to_xlsx(self, path, sheet=None, overwrite=False):
        tabulate = AttrTable.from_columns({
            'file.name':          Column('filename',
                                         align='<'),
            'timing.t0.datetime': Column('Time', '[UTC]',
                                         fmt='YYYY-MM-DD HH:MM:SS',
                                         convert=str,
                                         align='<'),
            'timing.exp':         Column('Exposure', '[s]',
                                         fmt='0.?????',
                                         align='<'),
            'timing.duration':    Column(convert=lambda t: t.value / 86400,
                                         fmt='[HH]"ʰ"MM"ᵐ"SS"ˢ"', unit='[hms]',
                                         total=True),
            'telescope':          ...,
            'filters.name':       Column('Filter'),
            'camera':             ...,
            'readout.mode':       Column(convert=str, align='<'),
            # 'nframes':            Column('n', total=True),
            # 'binning':            Column('bin', unit='y, x', header_level=1),
            'binning.y':          ...,
            'binning.x':          ...,

        },
            header_levels={'binning': 1},
            show_groups=False,
            title='Observational Setup'
        )

        tabulate.parent = self.parent
        return tabulate.to_xlsx(path, sheet, overwrite=overwrite,
                                align={...: '^'}, header_formatter=str.title)
        # widths={'binning': 5})

        # tabulate = AttrTable.from_spec({
        #     'Filename':        'file.name',
        #     'Time [UTC]':      'timing.t0.datetime:YYYY-MM-DD HH:MM:SS',
        #     'Exp [s]':         'timing.exp',
        #     'Duration [s]':    'timing.duration',
        #     'camera':          '',
        #     'telescope':       '',
        #     'Filter':           'filters.name',
        #     'binning.y':       '',
        #     'binning.x':       '',
        #     'Mode':            'readout.mode!s'
        # },
        #     converters={'timing.duration': lambda t: t.value / 86400},
        #     header_levels={'binning', 1},
        #     show_groups=False,
        #     totals='timing.duration'
        # )


class LatexWriter:
    def __init__(self, container):
        self.parent = container

    def _tabular_body(self,
                      booktabs=True, unicodemath=False,
                      flag_fmt=R'$\:^{{{flag}}}$',
                      foot_fmt=R'\hspace{{1eM}}$^{{{flag}}}$\ {info}\\',
                      summary_fmt=R'\hspace{{1eM}}{{{key} = {val}<? [{unit}]?>}}\\',
                      timing_flags=None,
                      **kws):

        if timing_flags is None:
            # †  ‡  §
            if unicodemath:
                timing_flags = {-1: '!',  # ⚠ not commonly available in all fonts
                                0:  '↓',
                                1:  '⟳'}
            else:
                timing_flags = {-1: '!',
                                0:  '*',
                                1:  R'\dagger'}  # '

        # change flags symbols temporarily
        with temporary(Trigger, FLAG_SYMBOLS=timing_flags):
            # controls which attributes will be printed
            tabulate = TableHelper.from_columns({
                # FIXME: these columns just change the titles from tabulate
                'timing.t0':       Column('$t_0$', fmt='{.iso}'.format, unit='UTC',
                                          flags=op.attrgetter('t.trigger.t0_flag')),
                'telescope':       Column('Telescope'),
                'camera':          Column('Camera'),
                'filters.name':    Column('Filter'),
                'nframes':         Column('n', total=True),
                'readout.mode':    Column('Readout Mode'),
                'binning':         Column('Binning', unit='y, x', align='^',
                                          fmt=R'{0.y}\times{0.x}'),
                'timing.exp':      Column(R'$t_{\mathrm{exp}}$', fmt=str, unit='s',
                                          flags=op.attrgetter('t.trigger.texp_flag')),
                'timing.duration': Column('Duration', fmt=hms_latex, unit='hh:mm:ss',
                                          total=True)},
                row_nrs=1,
                frame=False,
                hlines=False,
                col_head_style=None,
                borders={...: '& ', -1: R'\\'},
                summary=dict(footer=True, n_cols=1, bullets='', align='<',
                             pillars=['t0'], fmt=summary_fmt),
                too_wide=False,
                insert=({
                    -2:                         R'\toprule',
                    0:                          R'\midrule',
                    (n := len(self.parent)):    R'\midrule',
                    n + 1:                      R'\bottomrule'
                } if booktabs else {}),
                flag_fmt=flag_fmt,
                footnotes=(footnotes := Trigger.get_flags()),
                foot_fmt=foot_fmt
            )
            tabulate.parent = self.parent
            with temporary(Table, _nrs_header=R'\#'):  # HACK for latex symbol
                tbl = tabulate(title=False, col_groups=None, **kws)

        # HAck out the footnotes for formatting downstream
        footnotes = tbl.footnotes[:]
        tbl.footnotes = []
        return tbl, footnotes

    def _tabular_colspec(self, tbl, indent=4):
        col_spec = list(map(ALIGNMENT_MAP_INV.get, tbl.align[tbl._idx_shown]))
        # letters = list(map(chr, range(65, 65 + len(col_spec))))
        letters = [f'{i:c}' for i in range(65, 65 + len(col_spec))]
        letters[0] = f'% {letters[0]}'
        col_spec = Table([letters, col_spec],
                         col_borders=[*['& '] * (len(letters) - 1), ''],
                         frame=False, too_wide=False)
        # col_spec._idx_shown = tbl._idx_shown
        widths = tbl.col_widths[tbl._idx_shown]
        col_spec.max_width = 1000  # HACK since split is happening here... # FIXME
        col_spec.col_widths = np.array([(l, w)[l < w]
                                        for l, w in zip(map(len, letters), widths)])
        col_spec.truncate_cells(widths)
        letters, spec = indented(str(col_spec).rstrip(), indent).splitlines()

        return '\n'.join((letters, spec.replace('&', ' ')))

    def _to_table(self, star='*', pos='ht!',
                  options=R'\centering',
                  caption=None, cap=None,
                  label=None, env='tabular',
                  booktabs=True, unicodemath=False):
        indent = 4
        # options
        star = '*' if star else ''
        cap = f'[{{{cap!s}}}]' if cap else ''
        caption = fR'\caption{cap}{{{caption}}}' if caption else ''
        label = fR'\\label{{{label}}}' if label else ''

        #
        tbl, footnotes = self._tabular_body(booktabs, unicodemath)

        # get column spec
        return Template(txw.dedent(R'''
            \begin{table$star}[$pos]
                $options
                $caption
                $label
                %
                \begin{$env}{%
                $colspec
                }
                $body
                \end{$env}

                \footnotesize
                \raggedright
                $footnotes

            \end{table$star}
            ''')).substitute(
            locals(),
            colspec=self._tabular_colspec(tbl),
            body=indented(str(tbl), indent),
            footnotes=f'\n{" " * indent}'.join(np.char.strip(footnotes))
        )

    def _to_ctable(self, options='star, nosuper', pos='ht!',
                   caption=None, cap=None, label=None,
                   hspace_body='-1.5cm', hspace_footer='-1.2cm',
                   booktabs=True, unicodemath=True):
        indent = 4
        options = f',\n{" " * indent}'.join(
            filter(None, (f'{options}',
                          f'{pos     = !s}',
                          f'caption = {{{caption!s}}}' if caption else '',
                          f'cap     = {{{cap!s}}}' if cap else '',
                          f'{label   = !s}' if label else ''))
        )

        tbl, footnotes = self._tabular_body(
            booktabs, unicodemath,
            flag_fmt=R'\tmark[$\:{{{flag}}}$]',
            foot_fmt=R'\tnote[$^{{{flag}}}\:$]{{{info}}}',
            summary_fmt=R'\tnote[{{}}]{{{key} = {val}<? {unit}?>}}'
        )

        hack = '% ' if (len(tbl._idx_shown) < 7) else ''
        return Template(txw.dedent(R'''
            \ctable[
                $options,
                % hack to move table into left margin
                ${hack}doinside= {\hspace*{$hspace_body}}
            ]{
                % column spec
                $colspec
            }{
                % footnotes
                % also move footnotes to keep alignment consistent
                $hack\hspace*{$hspace_footer}
                $footnotes
            }{
                % tabular body
                $body
            }
            ''')).substitute(
            locals(),
            body=indented(tbl, indent),
            colspec=self._tabular_colspec(tbl),
            footnotes=f'\n{" " * indent}'.join(footnotes)
        )

    def _to_tabularray(self, options='',  # pos='ht!',
                       caption=None, cap=None, label=None,
                       hspace_body='-1.5cm', hspace_footer='-1.2cm',
                       booktabs=True, unicodemath=True):

        indent = 4  # since Template is hard coded indent 4
        options = f',\n{" " * indent}'.join(
            filter(None, (f'{options}',
                          #   f'{pos     = !s}',
                          f'caption = {{{caption!s}}}' if caption else '',
                          f'entry   = {{{cap!s}}}' if cap else '',
                          f'{label   = !s},' if label else ''))
        )

        tbl, footnotes = self._tabular_body(
            booktabs, unicodemath,
            flag_fmt=R'\TblrNote{{{flag}}}',
            foot_fmt=R'note{{${{{flag}}}$}} = {{{info}}},',
            summary_fmt=R'remark{{{key}}} = {{{val}{unit}}},'
        )

        return Template(txw.dedent(R'''
            \begin{tblr}[
                % outer spec
                $options
                % footnotes
                % \hspace*{$hspace_footer}
                $footnotes
            ]{% inner spec
                % Column headers (to appear on every page for page-split tables)
                row{1-2} = {c, m, font=\bfseries},
                rowhead = 2,
                % rowfoot = 1,
                colspec={
                $colspec
                },
            }
            $body
            \end{tblr}
            ''')).substitute(
            locals(),
            colspec=self._tabular_colspec(tbl, indent),
            body=indented(tbl, indent),
            footnotes=f'\n{" " * indent}'.join(footnotes)
        )


# ------------------------------------- ~ ------------------------------------ #
