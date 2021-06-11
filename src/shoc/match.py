
# std libs
from recipes.logging import LoggingMixin
import textwrap as txw
import functools as ftl
from collections import defaultdict

# local libs
import motley
from motley.utils import Filler, GroupTitle, ConditionalFormatter
from recipes.sets import OrderedSet

# third-party libs
import numpy as np

# relative libs
from .utils import str2tup


def proximity_groups(self, other, keys):
    # Do the matching - map observation to those in `other` with attribute
    # values matching most closely
    if not (self and other and keys):
        #vals = self.attrs(*keys) if self else (None, )
        yield (None, ), self, other, None
        return

    # split sub groups for closest match
    for vals, l0, l1, deltas in proximity_split(self.attrs(*keys),
                                                other.attrs(*keys)):
        # deltas shape=(l0.sum(), l1.sum(), len(keys))
        yield vals, self[l0], other[l1], deltas


def proximity_split(v0, v1):
    # split sub groups for closest match. closeness of runs is measured by
    # the sum of the relative distance between attribute values.
    # todo: use cutoff

    v0 = np.c_[v0][:, None]
    v1 = np.c_[v1][None]
    delta_mtx = np.abs(v0 - v1)

    # with wrn.catch_warnings():
    #     wrn.filterwarnings('ignore', 'divide by zero', RuntimeWarning)
    # scale = delta_mtx.max(1, keepdims=True))
    dist = delta_mtx.sum(-1)
    selection = (dist == dist.min(1, keepdims=True))
    for l1 in selection:
        # there may be multiple HDUs that are equidistant from the selected set
        # group these together
        l0 = (l1 == selection).all(1)
        # values are the same for this group (selected by l0), so we can just
        # take the first row of attribute values
        # vals array to tuple for hashing
        # deltas shape=(l0.sum(), l1.sum(), len(keys))
        yield tuple(v0[l0][0, 0]), l0, l1, delta_mtx[l0][:, l1]


class MatchedObservations(LoggingMixin):
    """
    Match observational data sets with each other according to their attributes. 
    """

    def __init__(self, a, b):
        """
        Initialize the pairing

        Parameters
        ----------
        a, b: shocCampaign
            `shocCampaign` instances from which observations will be picked to
            match those in the other list
        """
        self.a = a
        self.b = b
        self.matches = {}
        self.deltas = {}
        #  dict of distance matrices between 'closest' attributes

    def __call__(self, exact, closest=(), cutoffs=(), keep_nulls=False):
        """
        Match these observations with those in `other` according to their
        attribute values. Matches exactly the attributes given in `exact`, and
        as closely as possible to those in `closest`. Group both campaigns by
        the values of those attributes.

        Parameters
        ----------
        exact: str or tuple of str
            single or multiple attribute names to check for equality between
            the two runs. For null matches, None is returned.
        closest: str or tuple of str, optional, default=()
            single or multiple keywords to match as closely as possible between
            the two runs. The attributes which are pointed to by these should
            support item subtraction since closeness is taken to mean the
            absolute difference between the two attribute values.
        keep_nulls: bool, optional, default=False
            Whether to keep the empty matches. ie. if there are observations in
            `other` that have no match in this observation set, keep those
            observations in the grouping and substitute `None` as the value for
            the corresponding key in the resulting dict. This parameter affects
            only matches in the grouping of the `other` shocCampaign.
            Observations without matches in `self` (this run) are always kept so
            that full set of observations are always accounted for in the
            resultant grouping. A consequence of setting this to False (the
            default) is therefore that the two groupings returned by this
            function will have different keys, which may or may not be desired
            for further analysis.

        Returns
        -------
        out0, out1: shocObsGroups
            a dict-like object keyed on the attribute values of `keys` and
            mapping to unique `shocCampaign` instances
        """

        self.logger.info(
            'Matching %i files to %i files by:\n\tExact  : %r;\n\tClosest: %r',
            len(self.a), len(self.b), exact, closest
        )

        # create the GroupedRun for science frame and calibration frames
        self.exact = exact = str2tup(exact)
        self.closest = closest = str2tup(closest)
        self.attrs = OrderedSet(filter(None, exact + closest))

        if not self.attrs:
            raise ValueError('Need at least one `key` (attribute name) by which'
                             ' to match')
        # assert len(other), 'Need at least one other observation to match'

        g0 = self.a.group_by(*exact)
        g1 = self.b.group_by(*exact)

        # iterate through all group keys. There may be unmatched groups in both
        self.deltas = {}
        keys = set(g0.keys())
        if keep_nulls:
            keys |= set(g1.keys())

        for key in keys:
            obs0 = g0.get(key)
            obs1 = g1.get(key)
            for id_, sub0, sub1, delta in proximity_groups(obs0, obs1, closest):
                gid = (*key, *id_)
                # group
                self.matches[gid] = sub0, sub1
                # delta matrix
                self.deltas[gid] = delta

        return self

    def __str__(self):
        return self.pformat()

    def __iter__(self):
        yield from (self.left, self.right)

    def _make(self, i):
        run = (self.a, self.b)[i]
        split = list(zip(*self.matches.values()))[i]
        groups = run.new_groups(zip(self.matches.keys(), split))
        groups.group_id = self.attrs, {}
        return groups

    @property
    def left(self):
        return self._make(0)

    @property
    def right(self):
        return self._make(1)

    def delta_matrix(self, keys):
        """get delta matrix.  keys are attributes of the HDUs"""
        v0 = self.a.attrs(*keys)
        v1 = self.b.attrs(*keys)
        return np.abs(v0[:, None] - v1)

    def pformat(self, 
                title='Matched Observations', title_props=('g', 'bold'),
                group_header_style=('g', 'bold'), g1_style='c',
                no_match_style='r',
                **kws):
        """
        Format the resulting matches in a table

        Parameters
        ----------
        title : str, optional
            [description], by default 'Matched Observations'
        title_props : tuple, optional
            [description], by default ('g', 'bold')
        group_header_style : str, optional
            [description], by default 'bold'
        g1_style : str, optional
            [description], by default 'c'
        no_match_style : str, optional
            [description], by default 'r'

        Returns
        -------
        [type]
            [description]
        """

        # create temporary shocCampaign instance so we can use the builtin
        # pprint machinery
        g0, g1 = self.left, self.right
        tmp = g0.default_factory()

        # remove group-by keys that are same for all
        varies = [(g0.varies_by(key) | g1.varies_by(key)) for key in self.attrs]
        unvarying, = np.where(~np.array(varies))
        # use_key, = np.where(varies)

        # remove keys that runs are grouped into
        attrs = OrderedSet(tmp.table.attrs) - self.attrs
        insert = defaultdict(list)
        highlight = {}
        # hlines = []
        n = 0
        for i, key in enumerate(self.matches.keys()):
            obs = g0[key]
            other = g1[key]
            use = varies[:len(key)]
            display_keys = np.array(key, 'O')[use]
            # headers = tmp.table.get_headers(np.array(group_id)[varies])
            # info = dict(zip(headers, display_keys))

            # insert group headers
            group_header = GroupTitle(i, display_keys, group_header_style)
            insert[n].append((group_header, '<', 'underline'))

            for j, (run, c) in enumerate([(other, no_match_style), (obs, '')]):
                if run is None:
                    insert[n].append(Filler(c))
                else:
                    tmp.extend(run or ())
                    end = n + len(run)
                    # highlight other
                    if j == 0:
                        for m in range(n, end):
                            highlight[m] = g1_style
                    n = end

            # separate groups by horizontal lines
            # hlines.append(n - 1)

        # get title
        colour = ftl.partial(motley.codes.apply, txt=title_props)
        title = txw.dedent(f'''\
            {colour(title)}
            {colour("exact  :")} {self.exact}
            {colour("closest:")} {self.closest}\
            ''')

        # get attribute table
        tbl = tmp.table.get_table(tmp, attrs,
                                  title=title, title_align='<',
                                  insert=insert,  # hlines=hlines,
                                  row_nrs=False, totals=False,
                                  title_props='underline')

        # filler lines
        Filler.make(tbl)
        GroupTitle.width = tbl.get_width() - 1

        # fix for final run null match
        if run is None:
            # tbl.hlines.pop(-1)
            tbl.insert[n][-1] = (tbl.insert[n][-1], '', 'underline')

        # highlight `other`
        tbl.highlight = highlight

        # hack compact repr
        tbl.compact_items = dict(zip(np.take(list(self.attrs), unvarying),
                                     np.take(key, unvarying)))

        # create delta table
        # if False:
        #     dtbl = _delta_table(tbl, deltas, tmp.table.get_headers(closest),
        #                         threshold_warn)
        #     print(hstack((tbl, dtbl)))
        # else:

        # print()
        return tbl

    def pprint(self, title='Matched Observations', title_props=('g', 'bold'),
               group_header_style='bold', g1_style='c', no_match_style='r',
               **kws):
        """
        Pretty print the resulting matches in a table
        """
        print(self.pformat(title, title_props,
                           group_header_style, g1_style, no_match_style,
                           **kws))

    def _delta_table(self, tbl, deltas, headers, threshold_warn):
        #        threshold_warn: int, optional, default=None
        # If the difference in attribute values for attributes in `closest`
        #     are greater than `threshold_warn`, a warning is emitted

        if threshold_warn is not None:
            threshold_warn = np.atleast_1d(threshold_warn)
            assert threshold_warn.size == len(self.closest)

        # size = sum(sum(map(len, filter(None, g.values()))) for g in (g0, g1))
        # depth = np.product(
        #     np.array(list(map(np.shape, deltas.values()))).max(0)[[0, -1]])

        # dtmp = np.ma.empty((size, depth), 'O')  # len(closest)
        # dtmp[:] = np.ma.masked

        # for key, other in g1.items()
        #     # populate delta table
        #     s0 = n + np.size(other)
        #     delta_mtx = np.ma.hstack(deltas.get(key, [np.ma.masked]))
        #     dtmp[s0:s0 + np.size(obs), :delta_mtx.shape[-1]] = delta_mtx

        headers = list(map('Δ({})'.format, headers))
        formatters = []
        fmt_db = {'date': lambda d: d.days}
        deltas0 = next(iter(deltas.values())).squeeze()
        for d, w in zip(deltas0, threshold_warn):
            fmt = ConditionalFormatter('yellow', op.gt,
                                       type(d)(w.item()), fmt_db.get(kw))
            formatters.append(fmt)
        #
        insert = {ln: [('\n', '>', 'underline')] + ([''] * (len(v) - 2))
                  for ln, v in tbl.insert.items()}
        formatters = formatters or None
        headers = headers * (depth // len(closest))
        return Table(dtmp, col_headers=headers, formatters=formatters,
                     insert=insert, hlines=hlines)
