

# std
import textwrap as txw
import functools as ftl
from collections import abc, defaultdict

# third-party
import numpy as np

# local
import motley
from motley.utils import Filler, GroupTitle
from motley.formatters import ConditionalFormatter
from recipes import op
from recipes.sets import OrderedSet
from recipes.logging import LoggingMixin

# relative
from .utils import str2tup


def proximity_groups(self, other, keys, null=None):
    """
    Match observations with each other for calibration - map observations in
    `self` to those in `other` with attribute values matching most closely.

    Parameters
    ----------
    self, other : shocCampaign
        Observation sets to be matched.
    keys : tuple of str
        Attributes according to which the grouping and matching will be done.
    null : object, optional
        Sentinel value for a null match, by default None.

    Yields
    -------
    gid : tuple
        Group identifier: The values of attributes in `keys`, identical for all 
        observations in the sub groups. If no matching observations exist in one 
        of the observations sets for a particular subgroup, (null, ) is used as 
        the group identifier.
    sub0, sub1 : shocCampaign
        The sub groups of matched observations.
    """

    if not (self and other and keys):
        #vals = self.attrs(*keys) if self else (None, )
        yield (null, ), self, other, None
        return

    # split sub groups for closest match
    for vals, l0, l1, deltas in proximity_split(self.attrs(*keys),
                                                other.attrs(*keys)):
        # deltas shape=(l0.sum(), l1.sum(), len(keys))
        yield vals, self[l0], other[l1], deltas


def proximity_split(v0, v1):
    """
    Split into sub-groups for closest match. Closeness of runs is measured by
    the sum of the relative distance between attribute values.
    """

    # TODO: use cutoff

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


class Any:
    """
    Sentinel for proximity matching. If no exact match is found, use this object
    to represent the value of the `closest` matching attribute values which is
    allowed to take any value.
    """

    def __str__(self):
        return 'any'

    __repr__ = __str__


ANY = Any()


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
            Observation runs from which matching files will be chosen to match
            those in the other list.
        """
        data = dict(a=a, b=b)
        for k, v in data.items():
            if isinstance(v, dict):  # shocObsGroups
                data[k] = v.to_list()
            elif not isinstance(v, abc.Container):  # (shocCampaign, MockRun)
                raise ValueError(
                    f'Cannot match objects of type {type(a)} and {type(b)}. '
                    f'Please ensure you pass `shocCampaign` or `shocObsGroups`'
                    f' instances to this class.'
                )

        self.a, self.b = data.values()
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

        self.logger.info(txw.dedent('''
            Matching {:d} files to {:d} files by attributes:
               Exact  : {!r:};
               Closest: {!r:}
            '''), len(self.a), len(self.b), exact, closest
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
            for id_, sub0, sub1, delta in proximity_groups(
                    obs0, obs1, closest, null=ANY):
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
        run0 = (self.a, self.b)[i]
        split = list(zip(*self.matches.values()))[i]
        groups = run0.new_groups(zip(self.matches.keys(), split))
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

    def tabulate(self,
                 title='Matched Observations',
                 title_props=('g', 'bold'),
                 group_header_style=('g', 'bold'),
                 g1_style='c',
                 no_match_style='r',
                 **kws):

        #  style=dict(title='gB',
        #              group_headers='gB',
        #              g1='gB',
        #              no_match='rB'),

        #  formatters=dict(title='{:s|gB}',
        #                  group_headers='{:s|gB}',
        #                  g1='{:s|cB}',
        #                  no_match='{:s|rB}'),
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

        # set missing formatter
        tmp.tabulate.formatters['timing.t0'] = op.AttrGetter('iso')

        # remove group-by keys that are same for all
        varies = [(g0.varies_by(key) | g1.varies_by(key)) for key in self.attrs]
        unvarying, = np.where(~np.array(varies))
        # use_key, = np.where(varies)

        # remove keys that runs are grouped into
        n = 0
        highlight = {}
        insert = defaultdict(list)
        head_keys = np.array([*g0.group_id[0]])
        attrs = OrderedSet(tmp.tabulate.attrs) - self.attrs
        for i, gid in enumerate(self.matches.keys()):
            obs = g0[gid]
            other = g1[gid]
            use = varies[:len(gid)]
            
            # insert group header
            head = [f'{key}={tmp.tabulate.formatters.get(key, str)(val)}'
                    for key, val in zip(head_keys, np.array(gid, 'O')[use])]
            head = GroupTitle(i, head, group_header_style, '^')
            insert[n].append((head, '<', 'underline'))

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
        colour = ftl.partial(motley.apply, txt=title_props)

        title = txw.dedent(f'''\
            {colour(title)}
            {colour("exact  :")} {self.exact}
            {colour("closest:")} {self.closest}\
            ''')

        # get attribute table
        tbl = tmp.tabulate(attrs,
                           title=title, title_align='<',
                           insert=insert,  # hlines=hlines,
                           row_nrs=False, totals=False,
                           title_props='underline')

        # filler lines
        Filler.make(tbl)
        GroupTitle.width = tbl.get_width() - 2

        # fix for final run null match
        if run is None:
            # tbl.hlines.pop(-1)
            tbl.insert[n][-1] = (tbl.insert[n][-1], '', 'underline')

        # highlight `other`
        tbl.highlight = highlight

        # hack summary repr
        tbl.summary.items = dict(zip(np.take(list(self.attrs), unvarying),
                                     np.take(gid, unvarying)))

        # create delta table
        # if False:
        #     dtbl = _delta_table(tbl, deltas, tmp.table.get_headers(closest),
        #                         threshold_warn)
        #     print(hstack((tbl, dtbl)))
        # else:

        # print()
        return tbl

    def pformat(self, title='Matched Observations', title_props=('g', 'bold'),
                group_header_style='bold', g1_style='c', no_match_style='r',
                **kws):
        """
        Pretty format the resulting matches in a table.
        """
        return str(self.tabulate(title, title_props,
                                 group_header_style, g1_style, no_match_style,
                                 **kws))

    def pprint(self, title='Matched Observations', title_props=('g', 'bold'),
               group_header_style='bold', g1_style='c', no_match_style='r',
               **kws):
        """
        Pretty print the resulting matches in a table.
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
        headers *= (depth // len(closest))
        return Table(dtmp, col_headers=headers, formatters=formatters,
                     insert=insert, hlines=hlines)
