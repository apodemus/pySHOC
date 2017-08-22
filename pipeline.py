#!/usr/bin/python3

# TODO: """Integrate various components into fully automated pipeline"""
#  i.e photometry

# TODO: IDENTIFICATION OF CALIBRATION FRAMES WITH HEADER KEYWORDS - will
# allow passing directories and discriminating previously reduced calib files

# FIXME: write calibration frames to input directory of calibration frames....
# FIXME: all references to bias here should be DARK since bias traditionally
#  refers to amplifier offset value
# FIXME: printing too many match tables

# TODO: logging
# TODO:  + indicate different compute sections (timing / flats / etc) by ANSI marker at the beginning of terminal lines?

# TODO: LOCALIZE IMPORTS FOR PERFORMANCE GAIN / import in thread and set global??

# TODO: use fastfits for performance gain for file reads ??

# TODO: time / profile sections

from decor.profiler.timers import Chrono, timer, timer_extra
chrono = Chrono()
# NOTE do this first so we can profile import times


import os
# import warnings
# import traceback
import itertools as itt
import multiprocessing as mp
from collections import OrderedDict

# from datetime import datetime

# from warnings import warn

# WARNING: THIS IMPORT IS MEGA SLOW!! ~10s      # Localize to mitigate?
import numpy as np
import matplotlib.pyplot as plt
from astropy.io.fits.header import Header
from astropy.coordinates import SkyCoord


from pySHOC.core import shocSciRun, shocBiasRun, shocFlatFieldRun #, StructuredRun,
from pySHOC.header import get_header_info
from pySHOC.io import (ValidityTests as validity,
                       Conversion as convert,
                       InputCallbackLoop)

from recipes.io import iocheck, warn  # , note
from recipes.io import parse
# from recipes.list import flatten
from recipes.iter import grouper, partition  # , flatiter
from recipes.misc import getTerminalSize
from ansi.table import Table as sTable

# from ansi.progress import ProgressBar

from decor.profiler import profiler
# profiler = profile()
# @profiler.histogram

from IPython import embed

chrono.mark('imports')
chrono.report()


# setup warnings to print full traceback
# TODO: enable next section if mode is debug
# from recipes.io.tracewarn import *
# warning_traceback_on()
# logging.captureWarnings(True)



################################################################################
# Misc Function definitions
################################################################################
def section_header(msg, swoosh='=', _print=True):
    width = getTerminalSize()[0]
    # swoosh = swoosh * width
    # barfmt = '{1:{1}<{2}}'
    # msgfmt = '{0:^{2}}'
    info = '{1:{1}<{2}}\n{0:^{2}}\n{1:{1}<{2}}\n'.format(msg, swoosh, width)
    # info = '\n'.join(['\n', swoosh, msg.center(width), swoosh, '\n'])
    if _print:
        print(info)
    return info


def imaccess(filename):
    return True  # i.e. No validity test performed!
    # try:
    #     pyfits.open( filename )
    #     return True
    # except BaseException as err:
    #     print( 'Cannot access the file {}...'.format(repr(filename)) )
    #     print( err )
    #     return False


def median_scaled_median(data, axis):
    frame_med = np.median(data, (1,2))[:, None, None]
    scaled = data / frame_med
    return np.median(scaled, axis)

def plot(**kws):
    pname = mp.current_process().name
    print(pname, 'starting plot')
    im = self.plot(**kws)
    plt.show()      # child will stop here until the graph is closed
    print(pname, 'Done')


################################################################################
# MAIN
################################################################################
# TODO: PRINT SOME INTRODUCTORY STUFFS

def parse_input():
    """Parse sys.argv arguments from terminal input"""
    # FIXME: merge -d & -c option in favour of positional argument ??

    # exit clause for script parser.exit(status=0, message='')
    from sys import argv
    import argparse

    # Main parser
    main_parser = argparse.ArgumentParser(
        description='Data reduction pipeline for SHOC.')

    # group = parser.add_mutually_exclusive_group()
    # main_parser.add_argument('-v', '--verbose', action='store_true')
    # main_parser.add_argument('-s', '--silent', action='store_true')

    inputOK = ['1) a directory (in which case all fits files within will be read)',
               '2) name(s) of master {0} file(s)',
               '3) name(s) of unprocessed {0} cube(s)',
               '4) name of txt list containing (2) or (3) as one entry per line',
               '5) a glob expression resolving to (2) or (3)']
    inputOKfmt = '\n\t'.join([''] + inputOK)

    main_parser.add_argument(
        '-i',
        '--interactive',
        action='store_true',
        default=False,
        dest='interactive',
        help='Run the script in interactive mode.  You will be prompted for input when necessary')

    # main_parser.add_argument(
    #     '-d',
    #     '--dir',
    #     default=None,
    #     help='The data directory. Defaults to current working directory.')

    main_parser.add_argument(
        'files_or_directory', nargs='*',  # metavar='N',
        help='Science data cubes to be processed.  Requires at least one argument. ')

    main_parser.add_argument(
        '-o',
        '--outdir',
        help='The data directory where the reduced data is to be placed. Defaults to input '
             'directory')
    main_parser.add_argument(
        '-w',
        '--write-to-file',
        nargs='?',
        const=True,
        default=True,
        dest='w2f',
        help='Controls whether the script creates txt list of the files created. Requires -c '
             'option. Optionally takes filename base string for txt lists.')
    main_parser.add_argument(
        '-s',
        '--science',
        dest='sci',
        nargs='+',
        type=str,
        help='Science data cubes to be processed.  Requires at least one argument. Argument can '
             'be explicit list of files, a glob expression, a txt list, or a directory.')
    main_parser.add_argument(
        '-b',
        '--bias',
        nargs='+',
        default=False,
        help='Files to use for bias correction. Requires -c option. Optional argument(s) '
             'can be one of the following:' + inputOKfmt.format('bias'))
    main_parser.add_argument(
        '-f',
        '--flats',
        nargs='+',
        default=False,
        help='Files to use for flat field correction.  Requires -c option.  Optional argument(s) '
             'can be one of the following:' + inputOKfmt.format('flat field'))
    main_parser.add_argument(
        '-x',
        '--split',
        nargs='?',
        const=True,
        default=False,
        help='Split (burst) the data cubes into a sequence single-frame fits files. This is utterly'
             ' unnecessary, inefficient, and unproductive, but you can still do it if you are stuck in'
             ' your oldschool ways and want to clutter the output folder with cruft. Obviously requires'
             ' -c option.')
    main_parser.add_argument(
        '-t',
        '--timing',
        nargs='?',
        const=True,
        default=True,
        help='Calculate the timestamps for data cubes. Note that time-stamping is done by default '
             'The timing data will be written to a text files with the cube basename and extention '
             '".time"')     #TODO: eliminate this argument
    main_parser.add_argument(
        '-g',
        '--gps',
        nargs='+',
        default=None,
        help='GPS triggering times. Explicitly or listed in txt file')  # NOTE: only applies to old data
    main_parser.add_argument(
        '-k',
        '--kct',
        default=[],
        nargs='+',
        help='Kinetic Cycle Time for External GPS triggering.')
    main_parser.add_argument(
        '-c',
        '--combine',
        nargs='+',
        default=['daily', 'median'],
        help='Specifies how the bias/flats will be combined. Options are daily/weekly mean/median.')

    main_parser.add_argument('--plot', action='store_true', default=True, help='Do plots')
    main_parser.add_argument('--no-plots', dest='plot', action='store_false', help="Don't do plots")

    args = argparse.Namespace()

    # mx = main_parser.add_mutually_exclusive_group
    # NOTE: NEED PYTHON3.3 AND MULTIGROUP PATCH FOR THIS...  OR YOUR OWN ERROR ANALYSIS???

    # Header update parser
    # TODO: -coo  6 38 19.71 -48 59 15.6 ??? will this work??
    # FIXME: why not in main parser????
    # main_parser.add_argument('-u', '--update-headers',  help = 'Update fits file headers.')

    head_parser = argparse.ArgumentParser()
    head_parser.add_argument('update-headers',  # TODO: dest=someBetterName
                             nargs='?',
                             help='Update fits file headers.')  # action=store_const?
    head_parser.add_argument('-obj', '--object',
                             nargs='*', help='',
                             default=[''])
    head_parser.add_argument('-ra', '--right-ascension',
                             nargs='*',
                             default=[''],
                             dest='ra',
                             help='')  # TODO
    head_parser.add_argument('-dec', '--declination',
                             nargs='*',
                             default=[''],
                             dest='dec',
                             help='')  # TODO
    head_parser.add_argument('-epoch',
                             '--epoch',
                             default=None,
                             help='')
    head_parser.add_argument('-date', '--date',
                             nargs='*',
                             default=[''],
                             help='')
    head_parser.add_argument('-filter', '--filter',
                             default=None,
                             help='')
    head_parser.add_argument('-tel', '--telescope',
                             dest='tel',
                             default=None,
                             type=str,
                             help='')  # TODO
    head_parser.add_argument('-obs', '--observatory',
                             dest='obs', default=None,
                             help='')
    # head_parser.add_argument('-em', '--em-gain', dest='em', default=None, help='Electron Multiplying gain level')
    head_info = argparse.Namespace()

    # Name convension parser
    name_parser = argparse.ArgumentParser()
    name_parser.add_argument(
        'name',
        nargs='?',
        help=(
            "template for naming convension of output files.  "
            "eg. 'foo{sep}{basename}{sep}{filter}[{sep}b{binning}][{sep}sub{sub}]' where"
            "the options are: "
            "basename - the original filename base string (no extention); "
            "name - object designation (if given or in header); "
            "sep - separator character(s); filter - filter band; "
            "binning - image binning. eg. 4x4; "
            "sub - the subframed region eg. 84x60. "
            "[] brackets indicate optional parameters."))  # action=store_const?
    name_parser.add_argument(
        '-fl',
        '--flats',
        nargs=1,
        default='f[{date}{sep}]{binning}[{sep}sub{sub}][{sep}filt{filter}]')
    name_parser.add_argument(
        '-bi',
        '--bias',
        default='b{date}{sep}{binning}[{sep}m{mode}][{sep}t{kct}]',
        nargs=1)
    name_parser.add_argument('-sc',
                             '--science-frames',
                             nargs=1,
                             dest='sci',
                             default='{basename}')
    names = argparse.Namespace()

    parsers = [main_parser, head_parser, name_parser]
    namespaces = [args, head_info, names]

    valid_commands = ['update-headers', 'names']

    # TODO: get  parser.add_subparsers vibes working to simplify this stuff below.....
    # NOTE: argparse supports textlist input through @list.txt syntax!
    # ===========================================================================
    def groupargs(arg, currentarg=[None]):
        """Groups the arguments in sys.argv for parsing."""
        if arg in valid_commands:
            currentarg[0] = arg
        return currentarg[0]

    # Groups the arguments in sys.argv for parsing
    commandlines = [list(argz) for cmd, argz in itt.groupby(argv, groupargs)]
    for vc in valid_commands:
        setattr(args, vc.replace('-', '_'), vc in argv)
        if not vc in argv:
            commandlines.append([''])  # DAMN HACK!

    for cmds, parser, namespace in zip(commandlines, parsers, namespaces):
        parser.parse_args(cmds[1:], namespace=namespace)

    return namespaces, main_parser


################################################################################################################
def sanity_checks(args, main_parser):
    # Sanity checks for mutually exclusive keywords
    args_dict = args.__dict__
    prix = main_parser.prefix_chars
    prix2 = prix * 2

    # no other keywords allowed in interactive mode # FIXME: thiswill be annoying. find a way to mesh
    disallowedkw = {'interactive': set(args_dict.keys()) - set(['interactive'])}
    if args.interactive:
        for key in disallowedkw['interactive']:
            if args_dict[key]:
                # TODO: ignore these with warning
                raise KeyError('%s (%s) option not allowed in interactive mode'
                               % (prix + key[0], prix2 + key))

    # Sanity checks for non-interactive mode
    # any one of these need to be specified for an actionable outcome
    rqd = ['files_or_directory', 'sci', 'bias', 'flats']
    if not any([args_dict[key] for key in rqd]):
        main_parser.print_help()
        raise ValueError('\nNo files specified! Please specify files to be processed')
        # ''.format('/'.join(rqd)))


        ##, 'split', 'timing', 'update_headers', 'names']
        # raise ValueError('No action specified!\n'
        # 'Please specify one or more commands:'
        # '-s, -t, update-headers, name, or -i for interactive mode')
    return

    # FIXME: needs work since default arguments apply

    # Sanity checks for mutually inclusive keywords. Any one (or more) of the
    # listed keywords are required => or
    required = (('sci', ['timing', 'flats', 'bias', 'update-headers']),
                # ('bias',             ['sci']),       #if no science cubes specified, simply compute masters
                # ('flats',            ['sci']),
                ('update-headers', ['sci']),
                ('split', ['sci']),
                ('timing', ['sci', 'gps']),
                ('ra', ['update-headers']),
                ('dec', ['update-headers']),
                ('obj', ['update-headers']),
                ('epoch', ['update-headers', 'ra', 'dec']),
                ('date', ['update-headers']),
                ('filter', ['update-headers']),
                ('kct', ['gps']),)
    #    ('combine',          ['bias', 'flats'])       )
    requiredkw = OrderedDict(required)
    for key in args_dict:
        if args_dict[key] and (key in requiredkw):  # if this option has required options
            if not any(args_dict.get(rqk) for rqk in
                       requiredkw[key]):  # if none of the required options for this option are given
                ks = prix2 + key  # long string for option which requires option(s)
                ks_desc = '%s (%s)' % (ks[1:3], ks)
                rqks = [prix2 + rqk for rqk in requiredkw[key]]  # list of long string for required option(s)
                rqks_desc = ' / '.join(['%s (%s)' % (rqk[1:3], rqk)
                                        for rqk in rqks])
                raise KeyError('One or more of the following option(s) required'
                               ' with option {}: {}'.format(ks_desc, rqks_desc))


################################################################################################################
def process_args(namespaces):
    from pathlib import Path

    # TODO: make so that we can run without science frames
    args, head_info, names = namespaces

    # Positional argument and -c argument mean the same thing, we keep both for convenience
    if args.files_or_directory and not args.sci:
        args.sci = args.files_or_directory

    # embed()

    if args.outdir:
        # output directory given explicitly
        args.outdir = iocheck(args.outdir, os.path.exists, 1)
    # else:
        # infer output directory from images provided

    _infer_indir = not Path(args.sci[0]).is_dir()       # FIXME: NO sci ?
    _infer_outdir = not bool(args.outdir)
    workdir = ''
    for name in ('sci', 'flats', 'bias'): #args.dark   # 'sci',
        images = getattr(args, name)
        if images:
            # Resolve the input images
            images = parse.to_list(images,
                                   os.path.exists,
                                   include='*.fits',
                                   path=workdir,
                                   abspaths=True,
                                   raise_error=1)
            setattr(args, name, images)
            if _infer_indir:
                workdir = Path(images[0]).parent
                _infer_indir = False

            if _infer_outdir:
                args.outdir = os.path.split(images[0])[0]
                _infer_outdir = False
    # All inputs should now be resolved to lists of filenames

    if args.sci:
        # Initialize Run
        args.sci = shocSciRun(filenames=args.sci, label='science')

        # for cube in args.sci:  # DO YOU NEED TO DO THIS IN A LOOP?
        #     cube._needs_flip = not cube.cross_check(args.sci[0], 'flip_state')
            # self-consistency check for flip state of science cubes
            # #NOTE: THIS MAY BE INEFICIENT IF THE FIRST CUBE IS THE ONLY ONE WITH A DIFFERENT FLIP STATE...

    # ===========================================================================
    if args.gps:
        args.timing = True  # Do timing if gps info given

        if len(args.gps) == 1:
            # triggers give either as single trigger time string or filename of trigger list
            valid_gps = iocheck(args.gps[0], validity.RA,
                                raise_error=-1)  # if valid single time this will return that same str else None
            if not valid_gps:
                args.gps = parse.to_list(args.gps, validity.RA,
                                         path=workdir,
                                         abspath=0,
                                         sort=0,
                                         raise_error=1)

        # at ths point args.gps is list of explicit time strings.
        # Check if they are valid representations of time
        args.gps = [iocheck(g, validity.RA, raise_error=1, convert=convert.RA)
                    for g in args.gps]

        # Convert and set as cube attribute
        args.sci.that_need_triggers().set_gps_triggers(args.gps)

        # if any cubes are GPS triggered on each individual frame
        grun = args.sci.that_need_kct()
        if len(args.kct) == 1 and len(grun) != 1:
            warn('A single GPS KCT provided for multiple externally triggered runs. '
                 'Assuming this applies for all these files: %s' % grun)
            args.kct *= len(grun)   # expand by repeating

        elif len(grun) != len(args.kct):
            l = str(len(args.kct)) or 'No'
            s = ': %s' % str(args.kct) if len(args.kct) else ''
            raise ValueError('%s GPS KCT values provided%s for %i file(s): %s'
                             '' % (l, s, len(grun), grun))

        # "Please specify KCT (Exposure time + Dead time):")
        # args.kct = InputCallbackLoop.str(msg, 0.04, check=validity.float, what='KCT')

        for cube, kct in zip(grun, args.kct):
            cube.timing.kct = kct

    # ===========================================================================
    if args.flats or args.bias:

        args.combine = list(map(str.lower, args.combine))
        hows = 'day', 'daily', 'week', 'weekly'
        methods = 'sigma clipped',
        funcs = 'mean', 'median'
        vocab = hows + methods + funcs
        transmap = dict(grouper(hows, 2))
        understood, misunderstood = map(list, partition(vocab.__contains__, args.combine))
        if any(misunderstood):
            raise ValueError('Argument(s) {} for combine not understood.'
                             ''.format(misunderstood))
        else:
            understood = [transmap.get(u, u) for u in understood]

            how = next(filter(hows.__contains__, understood))
            func = next(filter(funcs.__contains__, understood))
            meth = next(filter(methods.__contains__, understood), '')

            args.combine = how
            args.fcombine = getattr(np, func)
            print('\nBias/Flat combination will be done by {}.'.format(' '.join([how, meth, func])))

             # TODO: sigma clipping ... even though it sucks

    # ===========================================================================
    if args.flats:

        #TODO full matching here ...

        # args.flats = parse.to_list(args.flats, imaccess, path=workdir, raise_error=1)
        args.flats = shocFlatFieldRun(filenames=args.flats, label='flat')

        # isolate the flat fields that match the science frames. only these will be processed
        match = args.flats.cross_check(args.sci, 'binning', 1)
        args.flats = args.flats[match]

        # check which are master flats


        # for flat in args.flats:
        #     flat._needs_flip = not flat.cross_check(args.sci[0], 'flip_state')

        # flag the flats that need to be subframed, based on the science frames which are subframed
        args.flats.flag_sub(args.sci)

        args.flats.print_instrumental_setup()

        # check which of the given flats are potentially master
        # is_master = [f.ndims == 2 for f in args.flats]

        # else:
        # print('WARNING: No flat fielding will be done!')

    # ===========================================================================
    if args.bias:
        # args.bias = parse.to_list(args.bias, imaccess, path=workdir, raise_error=1)
        args.bias = shocBiasRun(filenames=args.bias, label='bias')

        # match the biases for the science run
        match4sci = args.bias.cross_check(args.sci, ['binning', 'mode'], 0)
        # for bias in args.bias:
        #     bias._needs_flip = bias.cross_check(args.sci[0], 'flip_state')
            # NOTE: THIS MAY BE INEFICIENT IF THE FIRST CUBE IS THE ONLY ONE WITH A DIFFERENT FLIP STATE...
        #args.bias[match4sci].flag_sub(args.sci) ?
        args.bias.flag_sub(args.sci)
        args.bias[match4sci].print_instrumental_setup(description='(for science frames)')

        # match the biases for the flat run
        if args.flats:
            match4flats = args.bias.cross_check(args.flats, ['binning', 'mode'], -1)
            # args.bias4flats = args.bias[match4flats]
            # for bias in args.bias4flats:
            #     bias._needs_flip = bias.cross_check(args.flats[0], 'flip_state')

            # print table of bias frames
            args.bias[match4flats].print_instrumental_setup(description='(for flat fields)')
            match = match4sci & match4flats
        else:
            match = match4sci

        args.bias = args.bias[match]

        # check which of the given flats are potentially master
        # is_master = [f.ndims == 2 for f in args.flats]

    # else:
    # warn( 'No de-biasing will be done!' )

    # ===========================================================================
    if args.split:
        if args.outdir[0]:  # if an output directory is given
            args.outdir = os.path.abspath(args.outdir[0])
            if not os.path.exists(args.outdir):  # if it doesn't exist create it
                print('Creating reduced data directory {}.\n'.format(args.outdir))
                os.mkdir(args.outdir)

    # ===========================================================================
    # Handle header updating here

    # NOTE: somehow, this attribute gets set even though we can never read it due to a syntax error
    delattr(head_info, 'update-headers')

    hi = head_info
    hi.coords = None
    # join arguments since they are read as lists
    hi.object = ' '.join(hi.object)
    hi.ra = ' '.join(hi.ra)
    hi.dec = ' '.join(hi.dec)
    hi.date = ' '.join(hi.date)

    if args.update_headers:
        if hi.ra and hi.dec:
            iocheck(hi.ra, validity.RA, 1)
            iocheck(hi.dec, validity.DEC, 1)
            hi.coords = SkyCoord(ra=hi.ra, dec=hi.dec, unit=('h', 'deg'))  # , system='icrs'
        else:
            from pySHOC.utils import retrieve_coords_ra_dec
            hi.coords, hi.ra, hi.dec = retrieve_coords_ra_dec(hi.object)

        # TODO: maybe subclass SkyCoords to calculate this?
        def is_close(cooA, cooB, threshold=1e-3):
            return np.less([(cooA.ra - cooB.ra).value,
                            (cooA.dec - cooB.dec).value], threshold).all()

        for cube in args.sci:  # TODO: select instead of loop
            if cube.has_coords and hi.coords and not is_close(cube.coords, hi.coords):
                fmt = dict(style='hmsdms', precision=2, sep=' ', pad=1)
                warn('Supplied coordinates {} will supersede header coordinates {} in {}'
                     ''.format(hi.coords.to_string(**fmt), cube.coords.to_string(**fmt),
                               cube.filename()))
                cube.coords = hi.coords

        if not hi.date:
            # hi.date = args.sci[0].date#[c.date for c in args.sci]
            warn('Dates will be assumed from file creation dates.')

            # if not hi.filter:
            #     warn('Filter assumed as Empty')
            #     hi.filter = 'Empty'

            # if hi.epoch:
            #     iocheck(hi.epoch, validity.epoch, 1)
            # else:
            # warn('Assuming epoch J2000')
            # hi.epoch = 2000

            # if not hi.obs:
            # note('Assuming location is SAAO Sutherland observatory.')
            # hi.obs = 'SAAO'

            # if not hi.tel:
            #     note('Assuming telescope is SAAO 1.9m\n')   #FIXME: Don't have to assume for new data
            #     hi.tel = '1.9m'

    elif args.timing or args.split:
        # Need target coordinates for Barycentrization! Check the headers
        for cube in args.sci:  # TODO: select instead of loop
            if cube.coords is None:
                warn('Object coordinates not found in header for {}!\n'
                     'Barycentrization cannot be done without knowing target '
                     'coordinates!'.format(cube.filename()))

                # iocheck( hi.date, validity.DATE, 1 )
                # else:
                # warn( 'Headers will not be updated!' )

                # ===========================================================================
                # if args.timing and not hi.coords:
                # Target coordinates not provided / inferred from
                # warn( 'Barycentrization cannot be done without knowing target coordinates!' )

    if args.names:
        shocFlatFieldRun.nameFormat = names.flats
        shocBiasRun.nameFormat = names.bias
        shocSciRun.nameFormat = names.sci

    # ANIMATE

    return args, head_info, names
    # WARN IF FILE HAS BEEN TRUNCATED -----------> PYFITS DOES NOT LIKE THIS.....WHEN UPDATING:  ValueError: total size of new array must be unchanged



################################################################################
# Headers
################################################################################
def header_proc(run, do_update, head_info, verbose=True):
    """update the headers where necessary"""
    # TODO: if verbose=False - we can simply run the update below and skip the first loop over the run
    # TODO: #THIS SHOULD BE A METHOD OF THE SHOC_RUN CLASS??????
    # TODO: maybe trim all that spectroscopy cruft from the headers???

    section_header('Updating Fits headers')

    infoDict = get_header_info(do_update, head_info, run[0][0].header)
    # Check if updating header info is necessary
    update_dicts, update_all = [], {}
    ronTable = []
    for cube in run:
        # check which of the user supplied keyword need to be updated
        update_dict = cube.header.needs_update(infoDict)
        update_all.update(update_dict)

        # check if the read-noise, sensitivity, saturation values need to be added to headers
        ronDict = cube.header.get_readnoise_dict()
        if cube.header.needs_update(ronDict):
            row = [cube.get_filename()] + list(ronDict.values())
            ronTable.append(row)

    if update_all:
        print('\n\nUpdating all headers with the following shared information (where necessary):\n')
        print(repr(Header(update_all)))
        print()

    if len(ronTable):
        # TODO:  CAN BENEFIT HERE FROM STRUCTURED RUN, TO PRESENT INFO MORE CONSICELY FOR LARGE NUMBER OF INPUT CUBES
        print('Updating individual headers with the following:\n')
        table = sTable(ronTable,
                       title='Readout Noise',
                       col_headers=('Filename', ) + tuple(ronDict.keys()), #'SATURATION', 'RON', 'SENSITIVITY',
                       minimalist=True)
        print(table)


    verbose = True
    table = []
    for cube, info in zip(run, update_dicts):
        # convert old keywords
        if cube.header.has_old_keys():
            cube.header.convert_old_new(verbose=verbose)
            verbose = False

        # Readout noise and Sensitivity as taken from ReadNoiseTable
        # set RON, SENS, OBS-DATE (the RON and SENS may change through the run)
        details = cube.header.set_readnoise()
        table.append(details)

        # copy the cards in update_head to the hdu header and flush
        if info:
            print('Updating header for {}.'.format(cube.get_filename()))
            cube.header.update(info)
            cube.header['history'] = 'pySHOC.pipeline header updated'
        # update_head.check()
        cube.flush(output_verify='warn', verbose=1)  # SLOW!! AND UNECESSARY!!!!

        # return update_head


################################################################################################################
def sciproc(args, head_info, names):  # TODO: WELL THIS CAN NOW BE A METHOD OF shocRun CLASS

    section_header('Science frame processing')

    run = args.sci
    run.print_instrumental_setup()

    if args.w2f:
        outname = os.path.join(args.outdir, 'sci.txt')
        run.export_filenames(outname)

    if args.update_headers:
        header_proc(run, args.update_headers, head_info)

    # ===========================================================================
    # Timing
    if args.timing or args.split:
        section_header('Timing')
        run.set_times(head_info.coords)

    # combine flats
    needs_debias = run
    if args.flats:
        section_header('Flat field processing')

        is_2d = args.flats.check_singles()
        is_3d = ~is_2d

        masterflats = args.flats[is_2d]
        args.flats = args.flats[is_3d]

        matchdate = 'date' if args.combine == 'daily' else None
        threshold_warn = 1 if matchdate else None
        # names = cubes.magic_filenames(args.outdir)

        # FIXME: grouping not yet needed here!!! since we will flatten below anyway!!
        # TODO: add closest matching to shocRun.cross_check ???
        _, f_sr = run.match_and_group(args.flats, 'binning', matchdate, threshold_warn)
        needs_debias += f_sr.flatten()  # NOTE: does not preserve names / ObsClass / runClass

    # ===========================================================================
    # combine bias
    if args.bias:
        section_header('De-biasing')

        # images = args.bias
        is_2d = args.bias.check_singles()
        is_3d = ~is_2d
        masterbias = args.bias[is_2d]
        args.bias = args.bias[is_3d]

        # match & combine bias
        srun, sr_b = needs_debias.match_and_group(args.bias, ('binning', 'mode'), 'kct')
        # NOTE: it may not always be possible to *uniquely* match the science / flat frames to single
        # NOTE: bias frame.  There are a few option on how to resolve this. 1) use them all, or
        # NOTE: 2) pick one.  We use (1) as default.  In interactive mode we ask for input.
        # ambiguous = np.greater(list(map(len, b_sr.values())), 1)
        # if args.interactive:
        #     msg = ('Ambiguity in matching {} to {}! Cannot uniquely match files for {} {} '
        #            ' based on header info alone. Please select the most appropriate file among the following:')
        #     warn(msg)
        #     self.print_instrumental_setup()
        #     i = InputCallbackLoop.str('Ix?', 0, check=lambda x: int(x) < len(combined),
        #                               convert=int)

        # TODO: CALCULATE UNCERTAINTY ON CALIBRATION FRAMES
        sr_b_masters = sr_b.combined(np.median)
        sr_b_masters.writeout(args.outdir)                 #TODO: option for writing these to flats/masters/*.fits
        masterbias += sr_b_masters.flatten()

        if args.plot:
            job = mp.Process(target=masterbias[0].plot, args=())
            job.start()

        # TODO: subframing      q = s_sr.subframe( masterflats )
        # de-bias science / master flats
        srun.debias(sr_b_masters)

        # separate science / calibration
        idd = srun.flatten().identify()
        # set *srun* so the writeout below works when bool(args.flats) is False
        run = srun = idd.science
        args.flats = idd.get('flat', [])        # NOTE: now dark subtracted

    # ===========================================================================
    # Flat fielding
    if args.flats:

        # NOTE: in order to handle combining multiple cubes into a single frame (combine = 'weekly'), need structured run
        # combining each cube in flattened stack is NOT equivalent
        srun, f_sr = run.match_and_group(args.flats, 'binning', matchdate, threshold_warn)
        f_sr_cmb = f_sr.combined(median_scaled_median)
        cmb_flats = f_sr_cmb.flatten()

        # normalize master flats
        for stack in cmb_flats:
            stack.data /= np.mean(stack.data)
            #stack.flush(output_verify='warn', verbose=True)

        # write masterflats to file
        cmb_flats.writeout(args.outdir)

        # join calculated master flats with those given
        masterflats += cmb_flats

        # NOTE: need to match again since we are now including the given master flats
        srun, f_sr = run.match_and_group(masterflats, 'binning', matchdate, threshold_warn)
        srun.flatfield(f_sr)

    #     b_sr.magic_filenames(args.outdir)
            #
    #     masterbias = b_sr.compute_master(args.fcombine, load=True, w2f=args.w2f, outdir=args.outdir)
    #     # StructuredRun for master biases separated by 'binning','mode', 'kct'
    #     # s_sr.subframe(b_sr)
    #
    #     s_sr.debias(masterbias)
    #     # b_sr = bias_proc(masterbias, s_sr)
    #     # s_sr.writeout( suffix )
    #     # s_sr, _ = run.group_by( 'binning' )     #'filter'
    # else:
    #     s_sr, _ = run.group_by('binning')  # 'filter'
    #     masterbias = masterbias4flats = None

    # embed()

    if args.bias or args.flats:
        # the string to append to the filename base string to indicate flat fielded / de-biased
        suffix = ('b' if args.bias else '') + ('ff' if args.flats else '')
        nfns = srun.writeout(suffix=suffix)

        # Table for Calibrated Science Run
        title = 'Calibrated Science Run'
        title_props = {'text': 'bold', 'bg': 'green'}
        keys, attrs = run.zipper(('binning', 'mode', 'kct'))
        col_head = ('Filename',) + tuple(map(str.upper, keys))
        datatable = [(os.path.split(fn)[1],) + attr
                     for fn, attr in zip(nfns, attrs)]
        table = sTable(datatable, title, title_props, col_headers=col_head)
        print(table)

        if args.w2f:
            outname = os.path.join(args.outdir, 'sci.bff.txt')
            run.export_filenames(outname)




    # animate(sbff[0])

    # ===========================================================================
    # splitting the cubes
    if args.split:
        # User input for science frame output designation string
        if args.interactive:
            print('\n' * 2)
            msg = ('You have entered %i science cubes with %i different binnings'
                   ' (listed above).\nPlease enter a naming convension:\n' %
                   (len(args.sci), len(sbff)))
            names.science = InputCallbackLoop.str(msg, '{basename}', check=validity.trivial,
                                                  example='s{sep}{filter}{binning}{sep}')
            msg = 'Please enter a naming option:\n1] Sequential \n2] Number suffix\n'
            nm_option = InputCallbackLoop.str(msg, '2', check=lambda x: x in [1, 2])
            sequential = 1 if nm_option == 1 else 0
        else:
            sequential = 0

        run.magic_filenames(args.outdir)
        run.unpack(sequential, w2f=args.w2f)
    else:
        # One can do photometry without splitting the cubes!!
        # TODO: check w2f???
        # run.make_slices(suffix)
        run.export_times(with_slices=False)
        # run.make_obsparams_file(suffix)
        run.export_headers()

    return run


################################################################################################################
def setup():
    namespaces, parser = parse_input()
    args, head_info, names = namespaces
    # print(args)
    sanity_checks(args, parser)
    return process_args(namespaces)


################################################################################################################
if __name__ == '__main__':
    # RUN_DIR = os.getcwd()

    # class initialisation
    # bar = ProgressBar()                     #initialise progress bar
    # bar2 = ProgressBar(nbars=2)             #initialise the double progress bar

    args, head_info, names = setup()
    run = sciproc(args, head_info, names)


    # def goodbye():
    #     """switch back to original working directory"""
    #     os.chdir(RUN_DIR)
    #     print('Adios!')

    # import atexit
    #
    # atexit.register(goodbye)

    # main()
