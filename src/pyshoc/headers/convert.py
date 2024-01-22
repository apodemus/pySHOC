
import more_itertools as mit
from loguru import logger


# ---------------------------------------------------------------------------- #
# A mapping of old to new keywords.
KEYMAP = {
    # old                                   new
    'EMREALGAIN':                  'EMREALGN',
    'COUNTCONVERTMODE':            'CNTCVTMD',
    'COUNTCONVERT':                'CNTCVT',
    'DETECTIONWAVELENGTH':         'DTNWLGTH',
    'SENSITIVITY':                 'SNTVTY',
    'SPURIOUSNOISEFILTER':         'SPSNFLTR',
    'THRESHOLD':                   'THRSHLD',
    'PHOTONCOUNTINGENABLED':       'PCNTENLD',
    'NOTHRESHOLDS':                'NSETHSLD',
    'PHOTONCOUNTINGTHRESHOLD1':    'PTNTHLD1',
    'PHOTONCOUNTINGTHRESHOLD2':    'PTNTHLD2',
    'PHOTONCOUNTINGTHRESHOLD3':    'PTNTHLD3',
    'PHOTONCOUNTINGTHRESHOLD4':    'PTNTHLD4',
    'AVERAGINGFILTERMODE':         'AVGFTRMD',
    'AVERAGINGFACTOR':             'AVGFCTR',
    'FRAMECOUNT':                  'FRMCNT'
}

KEYWORDS = {
    f'HIERARCH {old}': new
    for old, new in KEYMAP.items()
}


REMOVE = {
    'PREAMPGAINTEXT',
    'SPECTROGRAPHSERIAL',
    'SHAMROCKISACTIVE',
    'SPECTROGRAPHNAME',
    'SPECTROGRAPHISACTIVE',
    'IRIGDATAAVAILABLE'
}

# ---------------------------------------------------------------------------- #


def get_old_keys(header):
    return {kw for kw in KEYWORDS if kw in header}


def convert(header, forward=True):
    """Convert old HIERARCH keywords to new short equivalents"""
    success = True
    if to_rename := get_old_keys(header):
        logger.bind(indent=True).debug(
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
    
    # 
    for key in REMOVE:
        header.remove(key, ignore_missing=True)
    
    return success
