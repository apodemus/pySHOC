# std
import re
import itertools as itt
from collections import defaultdict

# third-party
import numpy as np
from loguru import logger
from astropy.io.fits import Header

# local
from motley.table import Table
from recipes.sets import OrderedSet

# relative
from .convert import KEYMAP


# ---------------------------------------------------------------------------- #

def table(run, keys=None, ignore=('COMMENT', 'HISTORY')):
    agg = defaultdict(list)
    if keys is None:
        keys = set(itt.chain(*run.calls('header.keys')))

    for key in keys:
        if key in ignore:
            continue
        for header in run.attrs('header'):
            agg[key].append(header.get(key, '--'))

    return Table(agg)
    # return Table(agg, order='r', minimalist=True,
    # width=[5] * 35, too_wide=False)


def intersection(run, merge_histories=False):
    """
    For the headers of the observation set, keep only the keywords that have
    the same value across all headers.

    Parameters
    ----------
    run

    Returns
    -------

    """
    size = len(run)
    assert size > 0

    headers = h0, *hrest = run.attrs('header')
    # if single stack we are done
    if not hrest:
        return h0

    all_keys = OrderedSet(h0.keys())
    for h in hrest:
        all_keys &= OrderedSet(h.keys())

    all_keys -= {'COMMENT', 'HISTORY', ''}
    out = Header()
    for key in all_keys:
        vals = {h[key] for h in headers}
        if len(vals) == 1:
            # all values for this key are identical -- keep
            out[KEYMAP.get(key, key)] = vals.pop()
        else:
            logger.debug('Header key {} nonunique values: {}', key, list(vals))

    # merge comments / histories
    for key in ('COMMENT', *(['HISTORY'] * merge_histories)):
        # each of these are list-like and thus not hashable.  Wrap in
        # tuple to make them hashable then merge.
        agg = OrderedSet()
        for h in headers:
            if key in h:
                agg |= OrderedSet(tuple(h[key]))

        for msg in agg:
            getattr(out, f'add_{key.lower()}')(msg)
        continue

    return out


def match_term(kw, header_keys):
    """Match terminal input with header key"""
    matcher = re.compile(kw, re.IGNORECASE)
    # the matcher may match multiple keywords (eg: 'RA' matches 'OBJRA' and
    # 'FILTERA'). Tiebreak on witch match contains the greatest fraction of
    # the matched key
    f = [np.diff(m.span())[0] / len(k) if m else m
         for k in header_keys
         for m in (matcher.search(k),)]
    f = np.array(f, float)
    if np.isnan(f).all():
        # print(kw, 'no match')
        return

    i = np.nanargmax(f)
    # print(kw, hk[i])
    return header_keys[i]
