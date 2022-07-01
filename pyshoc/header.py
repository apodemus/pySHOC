"""

"""


# import logging
import re
import itertools as itt
import logging
from collections import defaultdict

import numpy as np
from astropy.io.fits import Header

from .convert_keywords import KEYWORDS as KWS_OLD_NEW
# from .io import (ValidityTests as validity,
#                  Conversion as convert,
#                  InputCallbackLoop
#                  )
from recipes.introspection.utils import get_module_name

# from recipes.io import warn

# module level logger
logger = logging.getLogger(get_module_name(__file__))

HEADER_KEYS_MISSING_OLD = \
    [
        'OBJECT',
        'OBJEPOCH',
        # 'OBJEQUIN',  # don't need both 'OBJEPOCH' and 'OBJEQUIN'
        'OBJRA',
        'OBJDEC',
        'OBSERVER',
        'OBSTYPE',
        'DATE-OBS',

        # 'TELESCOP',
        # 'TELFOCUS',
        # 'TELRA',
        # 'TELDEC',
        # 'INSTRUME',
        # 'INSTANGL',
        #
        # 'WHEELA',
        # 'WHEELB',
        # 'DATE-OBS',
        # 'GPS-INT',
        # 'GPSSTART',
        #
        # 'HA',
        # 'AIRMASS',
        # 'ZD',

        # 'DOMEPOS',  # don't know post facto

        # # Spectrograph stuff
        # 'ESHTMODE',
        # 'INSTSWV',
        # 'NSETHSLD',
        # 'RAYWAVE',
        # 'CALBWVNM',

    ]

KWS_REMAP = {old.replace('HIERARCH ', ''): new for old, new in KWS_OLD_NEW}


def get_new_key(old):
    return KWS_REMAP.get(old, old)


def headers_table(run, keys=None, ignore=('COMMENT', 'HISTORY')):
    agg = defaultdict(list)
    if keys is None:
        keys = set(itt.chain(*run.calls('header.keys')))

    for key in keys:
        if key in ignore:
            continue
        for header in run.attrs('header'):
            agg[key].append(header.get(key, '--'))

    return agg
    # return Table(agg, order='r', minimalist=True,
    # width=[5] * 35, too_wide=False)


def headers_intersect(obs, merge_histories=False):
    """
    For the headers of the observation set, keep only the keywords that have
    the same value across all headers.

    Parameters
    ----------
    obs

    Returns
    -------

    """

    assert len(obs)

    from recipes.containers.sets import OrderedSet
    from astropy.io.fits.verify import VerifyWarning
    from astropy.io.fits.header import Header
    import warnings

    headers = h0, *hrest = obs.attrs('header')
    all_keys = OrderedSet(h0.keys())
    for h in hrest:
        all_keys &= OrderedSet(h.keys())

    all_keys -= {'COMMENT', 'HISTORY', ''}
    out = Header()
    for key in all_keys:
        vals = {h[key] for h in headers}
        if len(vals) == 1:
            # all values for this key are identical -- keep
            with warnings.catch_warnings():
                # filter stupid HIERARCH warnings of which there seem to be
                # millions
                warnings.filterwarnings('ignore', category=VerifyWarning)
                out[get_new_key(key)] = vals.pop()

    # merge comments / histories
    merge_keys = ('COMMENT', )
    if merge_histories:
        merge_keys += ('HISTORY', )

    for key in merge_keys:
        # each of these are list-like and thus not hashable.  Wrap in
        # tuple to make them hashable then merge.
        agg = OrderedSet()
        for h in headers:
            if key in h:
                agg |= OrderedSet(tuple(h[key]))

        for x in agg:
            getattr(out, f'add_{key.lower()}')(x)
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


def get_header_info(do_update, from_terminal, header_for_defaults,
                    strict=False):
    """
    Create a Header object containing the key-value pairs that will be used to
    update the headers of the run.  i.e. These keywords are the same across
    multiple cubes.

    Ensure we have values of the following header keywords:
        OBJECT, OBJRA, OBJDEC, EPOCH, OBSERVAT, TELESCOP, FILTERA, FILTERB,
        OBSERVER

    if *do_update* is True (i.e. "update-headers" supplied at terminal):
        if the required keywords are missing from the headers, and not supplied
         at terminal, they will be asked for interactively and checked for
         validity in a input callback loop
    if *do_update* is False
        we don't care what is in the header.
        NOTE that timing data will not be barycentre corrected if object
        coordinates are not contained in the header


    Parameters
    ----------
    do_update:
        Whether the "update-headers" argument was supplied at terminal
    from_terminal:
        argparse.Namespace of terminal arguments for updating headers.
    header_for_defaults:
        The header object that will be used to populate the default
    strict:
        if True, always ask for required keyword values. else ignore them

    Returns
    -------

    """

    # TODO: if interactive
    # set the defaults for object info according to those (if available) in the
    # header of the first cube
    # update_head = shocHeader()
    # update_head.set_defaults(header_for_defaults)

    egRA = "'03:14:15' or '03 14 15'"
    egDEC = "'+27:18:28.1' or '27 18 28.1'"
    # key, comment, example, assumed_if_not_given, check_function, conversion_function, ask_for_if_not_given
    table = [
        ('OBJECT', 'IAU name of observed object', '', None, validity.trivial,
         convert.trivial),
        ('OBJRA', 'Right Ascension', egRA, None, validity.RA, convert.RA),
        ('OBJDEC', 'Declination', egDEC, None, validity.DEC, convert.DEC),
        ('EPOCH', 'Coordinate epoch', '2000', 2000, validity.epoch,
         convert.trivial),
        # ('OBSERVAT', 'Observatory', '', validity.trivial, convert.trivial, 0),
        ('TELESCOP', 'The telescope name', '', '1.9m', validity.trivial,
         convert.trivial),
        ('FILTERA', 'The active filter in wheel A', 'Empty', 'Empty',
         validity.trivial, convert.trivial,),
        # ('FILTERB', 'The active filter in wheel B', 'Empty', validity.trivial,  convert.trivial, 0),
        ('OBSERVER', 'Observer who acquired the data', '', None,
         validity.trivial, convert.trivial,)
        # ('RON', 'CCD Readout Noise',  '', validity.trivial, convert.trivial),
    ]

    # pull out the keywords from the list above
    keywords = next(zip(*table))

    # setup
    infoDict = {}
    said = False
    if do_update:  # args.update_headers
        supplied_keys = from_terminal.__dict__.keys()
        msg = ('\nPlease enter the following information about the observations '
               'to populate the image header. If you enter nothing that item '
               'will not be updated.')
        for term_key in supplied_keys:
            # match the terminal (argparse) input arguments with the keywords in
            # table above
            header_key = match_term(term_key, keywords)
            # match is now non-empty str if the supplied key matches a keyword
            # in the table
            # get terminal input value (or default) for this keyword
            if not header_key:
                # some of the terminal supplied info is not relevant here and
                # thus won't match
                continue

            # get the default value if available in header
            default = header_for_defaults.get(header_key, None)
            info = getattr(from_terminal, term_key) or default
            # print(header_key, info)
            ask = not bool(info) and strict
            # if no info supplied for this header key and its value could not be
            # determined from the default header it will be asked for
            # interactively (only if strict=True) the keywords in the table
            # that do have corresponding terminal input will not be asked
            # for unless strict is True
            _, comment, eg, assumed, check, converter = \
                table[keywords.index(header_key)]
            if ask:
                if not said:
                    print(msg)
                    said = True
                # get defaults from header
                info = InputCallbackLoop.str(comment, default, example=eg,
                                             check=check, verify=False,
                                             what=comment, convert=converter)
            elif assumed and not default:
                # assume likely values and warn user
                logger.warning('Assuming %s is %r' % (header_key, assumed))
                info = assumed

            if info:
                infoDict[header_key] = info

    # finally set the keys we don't have to ask for
    infoDict['OBSERVAT'] = 'SAAO'
    return infoDict


class shocHeader(Header):
    """Extend the pyfits.Header class for interactive user input"""

    def __init__(self, cards=[], copy=False):
        # Convert to new style keywords

        super().__init__(cards, copy)

    def has_old_keys(self):
        old, new = zip(*KWS_OLD_NEW)
        return any((kw in self for kw in old))

    def convert_old_new(self, forward=True):
        """Convert old HIERARCH keywords to new short equivalents"""
        success = True
        if self.has_old_keys():
            logger.info(('The following header keywords will be renamed:' +
                         '\n'.join(itt.starmap('{:35}--> {}'.format,
                                               KWS_OLD_NEW))))

        for old, new in KWS_OLD_NEW:
            try:
                if forward:
                    self.rename_keyword(old, new)
                else:
                    self.rename_keyword(new, old)
            except ValueError as e:
                logger.warning('Could not rename keyword %s due to the '
                               'following exception \n%s' % (old, e))
                success = False

        return success

    # def get_readnoise(self):
    #     """
    #     Readout noise, sensitivity, saturation as taken from ReadNoiseTable
    #     """
    #     from pyshoc import readNoiseTable
    #     return readNoiseTable.get_readnoise(self)
    #
    # def get_readnoise_dict(self, with_comments=False):
    #     """
    #     Readout noise, sensitivity, saturation as taken from ReadNoiseTable
    #     """
    #     data = self.get_readnoise()
    #     keywords = 'RON', 'SENSITIV', 'SATURATE'
    #     if with_comments:
    #         comments = ('CCD Readout Noise', 'CCD Sensitivity',
    #                     'CCD saturation counts')
    #         data = zip(data, comments)
    #     return dict(zip(keywords, data))
    #
    # def set_readnoise(self):
    #     """set Readout noise, sensitivity, observation date in header."""
    #     # Readout noise and Sensitivity as taken from ReadNoiseTable
    #     ron, sensitivity, saturation = self.readNoiseTable.get_readnoise(self)
    #
    #     self['RON'] = (ron, 'CCD Readout Noise')
    #     self['SENSITIV'] = sensitivity, 'CCD Sensitivity'
    #     # self['OBS-DATE'] = header['DATE'].split('T')[0], 'Observation date'
    #     # self['SATURATION']??
    #     # Images taken at SAAO observatory
    #
    #     return ron, sensitivity, saturation

    def needs_update(self, info, verbose=False):
        """check which keys actually need to be updated"""
        to_update = {}
        for key, val in info.items():
            if self.get(key, None) != val:
                to_update[key] = val
            elif verbose:
                print(f"{key} will not be updated")
        return to_update
