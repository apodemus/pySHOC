#!/usr/bin/env python

"""
A utility for users to convert between different versions of header keywords.
"""

__version__ = '1.0'

import os
import argparse

from astropy.io import fits

# A mapping of old to new keywords.
KEYWORDS = [
    # Example: ('oldkeyword', 'newkeyword'),
    ('HIERARCH EMREALGAIN', 'EMREALGN'),
    ('HIERARCH COUNTCONVERTMODE', 'CNTCVTMD'),
    ('HIERARCH COUNTCONVERT', 'CNTCVT'),
    ('HIERARCH DETECTIONWAVELENGTH', 'DTNWLGTH'),
    ('HIERARCH SENSITIVITY', 'SNTVTY'),
    ('HIERARCH SPURIOUSNOISEFILTER', 'SPSNFLTR'),
    ('HIERARCH THRESHOLD', 'THRSHLD'),
    ('HIERARCH PHOTONCOUNTINGENABLED', 'PCNTENLD'),
    ('HIERARCH NOTHRESHOLDS', 'NSETHSLD'),
    ('HIERARCH PHOTONCOUNTINGTHRESHOLD1', 'PTNTHLD1'),
    ('HIERARCH PHOTONCOUNTINGTHRESHOLD2', 'PTNTHLD2'),
    ('HIERARCH PHOTONCOUNTINGTHRESHOLD3', 'PTNTHLD3'),
    ('HIERARCH PHOTONCOUNTINGTHRESHOLD4', 'PTNTHLD4'),
    ('HIERARCH AVERAGINGFILTERMODE', 'AVGFTRMD'),
    ('HIERARCH AVERAGINGFACTOR', 'AVGFCTR'),
    ('HIERARCH FRAMECOUNT', 'FRMCNT'),
]

try:
    input = raw_input
except NameError:
    pass


def get_input(prompt, validator=lambda r: r):
    """Tries to get user input until the validator function succeeds."""
    while True:
        response = input(prompt)
        if validator(response):
            break
    return response


def convert(cube, forward=True):
    """Convert between different versions of FITS keywords."""
    basename, extension = os.path.splitext(cube.filename())
    hdu = cube[0]

    print("Converting '{0}'.".format(cube.filename()))

    for old, new in KEYWORDS:
        try:
            if forward:
                hdu.header.rename_keyword(old, new)
            else:
                hdu.header.rename_keyword(new, old)
        except ValueError as e:
            print(e)

    filename = '{0}_converted.fits'.format(basename)
    while os.path.exists(filename):
        overwrite = get_input(
                "The file '{0}' already exists. Would you like to overwrite it?"
                " [y/n] ".format(filename),
                validator=lambda r: r.lower() in ('y', 'n')).lower()

        if overwrite == 'y':
            overwrite = get_input(
                    "Are you sure that you want to overwrite '{0}'? "
                    "[y/n] ".format(filename),
                    validator=lambda r: r.lower() in ('y', 'n')).lower()

        if overwrite == 'n':
            filename = get_input('Please enter a new filename: ')
        else:
            break
    hdu.writeto(filename, clobber=True)

    print("Keywords successfully updated and saved to '{0}'.".format(filename))
    print('')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=convert.__doc__)
    parser.add_argument('files', nargs='+', help='Files to convert.')
    parser.add_argument('-r', '--revert', action='store_false',
                        help='Revert the headers of an already converted file.')

    arguments = parser.parse_args()

    for path in arguments.files:
        try:
            cube = fits.open(path, do_not_scale_image_data=True)
        except IOError as e:
            print(e)
        else:
            convert(cube, forward=arguments.revert)
            cube.close()
