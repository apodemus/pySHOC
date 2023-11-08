
import more_itertools as mit
from loguru import logger


# ---------------------------------------------------------------------------- #
# A mapping of old to new keywords.
KEYWORDS = {
    # old                                   new
    'HIERARCH EMREALGAIN':                  'EMREALGN',
    'HIERARCH COUNTCONVERTMODE':            'CNTCVTMD',
    'HIERARCH COUNTCONVERT':                'CNTCVT',
    'HIERARCH DETECTIONWAVELENGTH':         'DTNWLGTH',
    'HIERARCH SENSITIVITY':                 'SNTVTY',
    'HIERARCH SPURIOUSNOISEFILTER':         'SPSNFLTR',
    'HIERARCH THRESHOLD':                   'THRSHLD',
    'HIERARCH PHOTONCOUNTINGENABLED':       'PCNTENLD',
    'HIERARCH NOTHRESHOLDS':                'NSETHSLD',
    'HIERARCH PHOTONCOUNTINGTHRESHOLD1':    'PTNTHLD1',
    'HIERARCH PHOTONCOUNTINGTHRESHOLD2':    'PTNTHLD2',
    'HIERARCH PHOTONCOUNTINGTHRESHOLD3':    'PTNTHLD3',
    'HIERARCH PHOTONCOUNTINGTHRESHOLD4':    'PTNTHLD4',
    'HIERARCH AVERAGINGFILTERMODE':         'AVGFTRMD',
    'HIERARCH AVERAGINGFACTOR':             'AVGFCTR',
    'HIERARCH FRAMECOUNT':                  'FRMCNT'
}


KWS_REMAP = {
    old.replace('HIERARCH ', ''): new
    for old, new in KEYWORDS.items()
}

# ---------------------------------------------------------------------------- #


def get_old_keys(header):
    return {kw for kw in KEYWORDS if kw in header}


def convert(header, forward=True):
    """Convert old HIERARCH keywords to new short equivalents"""
    success = True
    if to_rename := get_old_keys(header):
        logger.debug(
            'The following header keywords will be renamed:' +
            ('\n{: <35}--> {:}' * len(to_rename)),
            *mit.interleave(to_rename, map(KEYWORDS.get, to_rename))
        )

    for old in to_rename:
        try:
            header.rename_keyword(*(old, KEYWORDS[old])[::(-1, 1)[forward]])
        except ValueError as error:
            logger.warning('Could not rename keyword {:s} due to the '
                           'following exception \n{}', old, error)
            success = False

    return success
