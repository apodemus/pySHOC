#!/usr/bin/python3

#TODO:'''Integrate various components into fully automated pipeline'''

#TODO: IDENTIFICATION OF CALIBRATION FRAMES WITH HEADER KEYWORDS - will
#allow passing directories and discriminating previously reduces calib files

#TODO: logging

#TODO: LOCALIZE IMPORTS FOR PERFORMANCE GAIN / import in thread and set global??
print( 'Importing modules...' )
import time
from types import ModuleType
t1 = time.time()

def ticktock(s):
    t2 = time.time()
    if isinstance(s, ModuleType):
        s = s.__name__
    print('%s: %.3f s' %(s, t2-t1))

#from decor.profile import profile
#profiler = profile()
#@profiler.histogram
#def do_imports():


import os, sys, textwrap
import numpy as np
import itertools as itt
from collections import OrderedDict

from datetime import datetime
#from warnings import warn
from myio import warn, note

#WARNING: THIS IMPORT IS MEGA SLOW!! ~10s
from astropy.coordinates import SkyCoord

from pySHOC.core import (shocHeaderCard, shocHeader, shocRun, StructuredRun,
                         get_coords_ra_dec)
from pySHOC.io import (ValidityTests as validity,
                       Conversion as convert,
                       Input)

from myio import iocheck, parsetolist
from recipes.list import flatten
from recipes.iter import grouper, partition#, flatiter
from recipes.misc import getTerminalSize
from ansi.str import SuperString
from ansi.table import Table as sTable
from ansi.progress import ProgressBar

from IPython import embed

#do_imports()
#raise SystemExit

ticktock('modules')
print('Done!\n\n')

################################################################################
#Misc Function definitions
################################################################################
def section_header( msg, swoosh='=', _print=True ):
    width = getTerminalSize()[0]
    swoosh = swoosh * width
    msg = SuperString(msg).center(width)
    info = '\n'.join(['\n', swoosh, msg, swoosh, '\n' ])
    if _print:
        print( info )
    return info

def imaccess(filename):
    return True #i.e. No validity test performed!
    #try:
        #pyfits.open( filename )
        #return True
    #except BaseException as err:
        #print( 'Cannot access the file {}...'.format(repr(filename)) )
        #print( err )
        #return False



################################################################################
#Bias reductions
################################################################################
#TODO: #THIS SHOULD BE A METHOD OF THE SHOC_RUN CLASS
def bias_proc(m_bias_dict, sb_dict):             
    #NOTE: Bias frames here are technically dark frames taken at minimum possible
    # exposure time.  SHOC's dark current is (theoretically) constant with time,
    # so these may be used as bias frames.  
    # Also, there is therefore no need to normalised these to the science frame 
    # exposure times.
    
    '''
    Do the bias reductions on sciene data
    Parameters
    ----------
    mbias_dict : Dictionary with binning,filename key,value pairs for master biases
    sb_dict : Dictionary with binning,run key,value pairs for science data
    
    Returns
    ------
    Bias subtracted shocRun
    '''
    
    #fn_ls = []
    for attrs, master in m_bias_dict.items():
        if master is None:
            continue
        
        stacks = sb_dict[attrs]              #the cubes (as shocRun) for this attrs value
       
        #stacks_b = []
        msg = '\nDoing bias subtraction on the stack: '
        lm = len(msg)
        print(msg)
        for stack in stacks:
            print(' '*lm, stack.get_filename())
            
            header = stack[0].header
            header['BIASCORR'] = (True, 'Bias corrected')					 #Adds the keyword 'BIASCORR' to the image header to indicate that bias correction has been done
            hist = ('Bias frame {} subtracted at {}'
                    ).format(master.get_filename(), datetime.now())
            header.add_history(hist, before='HEAD') 	#Adds the filename and time of bias subtraction to header HISTORY
            
            #TODO: multiprocessing here...??
            #NOTE avoid inplace -= here due to potential numpy casting error for different types
            stack[0].data = stack[0].data - master[0].data

    sb_dict.label = 'science frames (bias subtracted)'
    return sb_dict
    
  
################################################################################
#Flat field reductions
################################################################################

#TODO: THIS SHOULD BE A METHOD OF THE SHOC_RUN CLASS
def flat_proc(mflat_dict, sb_dict):
    '''Do the flat field reductions
    Parameters
    ----------
    mflat_dict : Dictionary with binning,run key,value pairs for master flat images
    sb_dict : Dictionary with binning,run key,value pairs
    
    Returns
    ------
    Flat fielded shocRun
    '''
    
    #fn_ls = []
    for attrs, masterflat in mflat_dict.items():
        
        if isinstance(masterflat, shocRun):
            masterflat = masterflat[0]                  #HACK!!
        
        mf_data = masterflat[0].data    #pyfits.getdata(masterflat, memmap=True)
        
        if round(np.mean(mf_data), 1) != 1:
            raise ValueError('Flat field not normalised!!!!')
        
        stacks = sb_dict[attrs]                #the cubes for this binning value
        
        msg = '\nDoing flat field division on the stack: '
        lm = len(msg)
        print( msg, )
        for stack in stacks:
            print( ' '*lm + stack.get_filename() )
            
            header = stack[0].header
            #Adds the keyword 'FLATCORR' to the image header to indicate that 
            #flat field correction has been done
            header['FLATCORR'] = (True, 'Flat field corrected')
            hist = 'Flat field {} subtracted at {}'.format(
                masterflat.get_filename(), datetime.now())
            header.add_history(hist, before='HEAD')
            #Adds the filename used and time of flat field correction to header HISTORY
            
            stack[0].data /= mf_data                        #flat field division

    sb_dict.label = 'science frames (flat fielded)'
    return sb_dict
    
################################################################################
# Headers
################################################################################

def header_proc(run, args, head_info, _pr=True):
    #TODO: #THIS SHOULD BE A METHOD OF THE SHOC_RUN CLASS??????
    #TODO: maybe trim all that spectroscopy cruft from the headers???
    '''update the headers where necesary'''
    
    section_header('Updating Fits headers')
    
    update_head = shocHeader( )
    update_head.set_defaults(run[0][0].header)        #set the defaults for object info according to those (if available) in the header of the first cube
    
    exra = "(eg: '03:14:15' or '03 14 15')"
    exdec = "(eg: '+27:18:28.1' or '27 18 28.1')"
    table = [
        ('OBJECT', 'Object name / alias', '', validity.trivial, convert.trivial),
        ('OBJRA', 'Right Ascension', exra, validity.RA, convert.RA),
        ('OBJDEC', 'Declination', exdec, validity.DEC, convert.DEC),
        ('EPOCH', 'Coordinate epoch', 'eg: 2000', validity.float, convert.trivial),
        #TODO: FILTERA, FILTERB
        #('FILTER', 'Filter', '(WL for clear)', validity.trivial,  convert.trivial),
        #TODO: observer
        ('OBSERVAT', 'Observatory', '', validity.trivial, convert.trivial),
        #('RON', 'CCD Readout Noise',  '', validity.trivial, convert.trivial),
        ('TELESCOP', 'Telescope', '', validity.trivial, convert.trivial)
    ]

    keywords, description, example, check, conversion = zip(*table)
    #convert.ws2ds

    if args.update_headers:
        info = []
        askfor = []
        for kw in keywords:
            #match the terminal (argparse) input arguments with the keywords
            match = [key for key in head_info.__dict__.keys()
                        if key in kw.lower()]
            #print('match', match)
            if len(match):
                inf = getattr(head_info, match[0])    
                #terminal input value (or default) for this keyword
                if inf:
                    info.append(inf)
                    askfor.append(0)
                else:
                    #if terminal input info is empty this item will be asked for explicitly
                    info.append('')
                    askfor.append(1)
            else:
                #the keywords that don't have corresponding terminal input arguments
                info.append('')
                askfor.append(0)
        
    else:    
        info = ['']*len(table)  #['test', '23 22 22.33', '44 33 33.534', '2000', '2012 12 12', 'WL','saao', '','']    #
        askfor = [1,1,1,1,1,0,1]
    
    if sum(askfor):
        print("\nPlease enter the following information about the observations "
              "to populate the image header. If you enter nothing that item will"
              " not be updated.")  #PRINT FORMAT OPTIONS
    
    update_cards = [
        shocHeaderCard(key, val, comment, example=ex, check=vt, conversion=conv, askfor=ask) 
            for key, val, comment, ex, vt, conv, ask 
                in zip(keywords, info, description, example, check, conversion, askfor)
                ]
    update_head.extend(update_cards)
    
    print('\n\nUpdating all headers with the following information:\n')
    print(repr(update_head))
    print()
    
    said = False
    table = []
    for cube in run:
        #convert old keywords
        if cube.has_old_keys:
            from pySHOC.convert_keywords import KEYWORDS as kw_old_to_new
            if not said:
                print('The following header keywords will be renamed:')
                print('\n'.join(itt.starmap('{:35}--> {}'.format, kw_old_to_new)))
                print()
                said = True
            
            for old, new in kw_old_to_new:
                try:
                    cube[0].header.rename_keyword(old, new)     #FIXME hdu = cube[0]
                except ValueError as err:
                    warn('keyword {} not renamed'.format(old))
                    print(err)
        
        # Readout noise and Sensitivity as taken from ReadNoiseTable
        #set RON, SENS, OBS-DATE (the RON and SENS may change through the run)
        details = update_head.set_ron_sens_date(cube[0].header) 
        table.append(details)
        
        #Check if updating observation info is necessary #NOT WORKING!!!!!!!
        #copy the cards in update_head to the hdu header and flush
        update_head.check()
        update_head.update_to(cube) 
        #cube.set_name_dict()
        
    if _pr:
        #TODO:  CAN BENEFIT HERE FROM STRUCTURED RUN, TO PRESENT INFO MORE CONSICELY
        table = sTable(table, 
                       title = 'Readout Noise', 
                       col_headers=('RON', 'SENSITIVITY', 'SATURATION'), 
                       row_headers=run.get_filenames())
        print(table)
    
    return update_head #this will have the RON and SENSITIVITY of the last updated
    
    
    
    
################################################################################
# Science image pre-reductions
################################################################################

def match_closest(sci_run, calib_run, exact, closest=None, threshold_warn=7, _pr=1):
    '''
    Match the attributes between sci_run and calib_run.
    Matches exactly to the attributes given in exact, and as closely as possible to the  
    attributes in closest. Separates sci_run by the attributes in both exact and 
    closest, and builds an index dictionary for the calib_run which can later be used to
    generate a StructuredRun instance.
    Parameters
    ----------
    sci_run     :   The shocRun to which comparison will be done
    calib_run   :   shocRun which will be trimmed by matching
    exact       :   tuple or str. keywords to match exactly         
                    NOTE: No checks run to ensure calib_run forms a subset of
                    sci_run w.r.t. these attributes
    closest     :   tuple or str. keywords to match as closely as possible

    Returns
    ------
    s_sr :              StructuredRun of science frames separated 
    out_sr

    '''
    #===========================================================================
    def str2tup(keys):
        if isinstance(keys, str):
            keys = keys,          #a tuple
        return keys
    
    #===========================================================================
    msg = ('\nMatching {} frames to {} frames by:\tExact {};\t Closest {}\n'
          '').format(calib_run.label.upper(), sci_run.label.upper(), exact, 
                     repr(closest))
    print(msg)
    
    #create the StructuredRun for science frame and calibration frames
    exact, closest = str2tup(exact), str2tup(closest)
    sep_by = tuple( key for key in flatten([exact, closest]) if not key is None )   
    s_sr, sflag = sci_run.attr_sep( *sep_by )
    c_sr, bflag = calib_run.attr_sep( *sep_by )
    
    #Do the matching - map the science frame attributes to the calibration StructuredRun element with closest match
    #NOTE AT THE MOMENT THIS ONLY USES THE FIRST KEYWORD IN closest TO DETERMINE THE CLOSEST MATCH 
    lme = len(exact)
    _, sciatts = sci_run.zipper(sep_by)                       #sep_by key attributes of the sci_run
    _, calibatts = calib_run.zipper(sep_by)
    ssciatts = np.array(list(set(sciatts)), object)           #a set of the science frame attributes
    calibatts = np.array(list(set(calibatts)), object)
    sss = ssciatts.shape
    where_thresh = np.zeros((2*sss[0], sss[1]+1))       #state array to indicate where in data threshold is exceeded (used to colourise the table)
    
    runmap, attmap = {}, {}
    datatable = []
    
    for i, attrs in enumerate(ssciatts):
        lx = np.all( calibatts[:,:lme]==attrs[:lme], axis=1 )       #those calib cubes with same attrs (that need exact matching)
        delta = abs( calibatts[:,lme] - attrs[lme] )
        
        if ~lx.any():   #NO exact matches
            threshold_warn = False                                  #Don't try issue warnings below
            cattrs = (None,)*len(sep_by)
            crun = None
        else:
            lc = delta==delta[lx].min()
            l = lx & lc
            cattrs = tuple( calibatts[l][0] )
            crun = c_sr[cattrs]
        
        tattrs = tuple(attrs)                                           #array to tuple
        attmap[tattrs] = cattrs
        runmap[tattrs] = crun

        datatable.append( (str(s_sr[tattrs]), ) + tattrs )
        datatable.append( (str(crun),)          + cattrs )
        
        #Threshold warnings
        #FIXME:  MAKE WARNINGS MORE READABLE
        if threshold_warn:
            #type cast the attribute for comparison (datetime.timedelta for date attribute, etc..)
            deltatype = type(delta[0])
            threshold = deltatype(threshold_warn)
            if np.any(delta[l] > deltatype(0)):
                where_thresh[2*i:2*(i+1), lme+1] += 1
            
            #compare to threshold value
            if np.any(delta[l] > threshold):
                fns = ' and '.join(c_sr[cattrs].get_filenames())
                sci_fns = ' and '.join(s_sr[tattrs].get_filenames())
                msg = ('Closest match of {} {} in {}\n'
                       '\tto {} in {}\n'
                       '\texcedees given threshold of {}!!\n\n'
                        ).format(tattrs[lme], closest[0].upper(), fns, 
                                 cattrs[lme], sci_fns, threshold_warn)
                warn( msg )
                where_thresh[2*i:2*(i+1), lme+1] += 1
        
    out_sr = StructuredRun(runmap)
    out_sr.label = calib_run.label
    out_sr.sep_by = s_sr.sep_by
    
    if _pr:
        #Generate data table of matches
        col_head = ('Filename(s)',) + tuple( map(str.upper, sep_by) )
        where_row_borders = range(0, len(datatable)+1, 2)
        
        table = sTable(datatable,
                       title='Matches',
                       title_props=dict(text='bold', bg='light blue'),
                       col_headers=col_head,
                       where_row_borders=where_row_borders)
        
        #colourise           #TODO: highlight rows instead of colourise??
        unmatched = [None in row for row in datatable]
        unmatched = np.tile(unmatched, (len(sep_by)+1,1)).T
        states = where_thresh
        states[unmatched] = 3
        table.colourise(states, 'default', 'yellow', 202, {'bg':'red'})
        
        print('\nThe following matches have been made:')
        print(table )
        
    return s_sr, out_sr


#===============================================================================
def sciproc(args, head_info, names):                               #WELL THIS CAN NOW BE A METHOD OF shocRun CLASS
    
    section_header('Science frame processing')
    
    run = args.cubes
    run.print_instrumental_setup()
    
    if args.w2f:
        outname = os.path.join(args.outdir, 'cubes.txt')
        run.export_filenames( outname )
    
    if args.update_headers:
        updated_head = header_proc(run, args, head_info)
    
    #===========================================================================
    # Timing
    if args.timing or args.split:
        section_header('Timing')
        
    #if args.gps:
        #args.cubes.that_need_kct()
        #args.cubes.that_need_triggers().set_gps_triggers(args.gps)
        
    
    if args.timing or args.split:
        run.set_times(head_info.coords)
        #run.set_airmass( head_info.coords )
    
    #===========================================================================
    # De-biasing
    suffix = ''
    if args.bias:
        section_header('De-biasing')
        
        suffix += 'b'
        #the string to append to the filename base string to indicate bias subtraction
        s_sr, b_sr = match_closest(run, args.bias, ('binning','mode'), 'kct',
                                    threshold_warn=None)
        b_sr.magic_filenames(args.outdir)
        masterbias = b_sr.compute_master(args.fcombine, load=1, w2f=args.w2f, 
                                         outdir=args.outdir)
        #StructuredRun for master biases separated by 'binning','mode', 'kct' 
        #s_sr.subframe( b_sr )

        b_sr = bias_proc(masterbias, s_sr)
        #s_sr.writeout( suffix )
        #s_sr, _ = run.attr_sep( 'binning' )     #'filter'
    else:
        s_sr, _ = run.attr_sep('binning')     #'filter'
        masterbias = masterbias4flats = None
    
    #===========================================================================
    # Flat fielding
    if args.bias or args.flats:         section_header('Flat field processing')
    
    if args.flats:
        suffix += 'ff'
        #the string to append to the filename base string to indicate flat fielding
        
        matchdate = 'date'      if args.combine=='daily'     else None
        threshold_warn = 1      if matchdate                 else None
        
        s_sr, f_sr = match_closest(run, args.flats, 'binning', matchdate, 
                                   threshold_warn)
        f_sr.magic_filenames(args.outdir)
        
        if args.bias:
            if args.bias4flats:
                f_sr, b4f_sr = match_closest(f_sr.flatten(), args.bias4flats, 
                                             ('binning','mode'), 'kct')
                b4f_sr.magic_filenames(args.outdir, extension='.b4f.fits')
                
                embed()
                
                #TODO: CROSS CHECK THE bias4flats SO WE DON'T RECOMPUTE unncessary...
                
                masterbias4flats = b4f_sr.compute_master(args.fcombine, load=1,
                                                         outdir=args.outdir)
                #StructuredRun for master biases separated by 'binning','mode', 'kct' 
                #else:
                    #masterbias4flats = masterbias 
                    ##masterbias contains frames with right mode to to flat field debiasing
            else:
                masterbias4flats = None
        
        masterflats = f_sr.compute_master(args.fcombine, masterbias4flats,
                                          load=1, outdir=args.outdir)
        #q = s_sr.subframe( masterflats )
        #embed()
        #raise Exception
        
        masterflats, _ = masterflats.attr_sep( 'binning' )
        s_sr, _ = s_sr.attr_sep( 'binning' )
        s_sr = flat_proc( masterflats, s_sr)
        #StructuredRun of flat fielded science frames
    
    if args.bias or args.flats:
        
        nfns = s_sr.writeout(suffix)
        
        #Table for Calibrated Science Run
        title = 'Calibrated Science Run'
        title_props = { 'text':'bold', 'bg': 'green' }
        keys, attrs = run.zipper( ('binning', 'mode', 'kct') )
        col_head = ('Filename',) + tuple( map(str.upper, keys) )
        datatable = [(os.path.split(fn)[1],) + attr for fn, attr in zip(nfns, attrs)]
        table = sTable( datatable, title, title_props, col_head )
        print( table )
        
        if args.w2f:
            outname = os.path.join(args.outdir, 'cubes.bff.txt')
            run.export_filenames( outname )
            
    #animate(sbff[0])
    
    #raise ValueError
    #===========================================================================
    #splitting the cubes
    if args.split:
        #User input for science frame output designation string
        if args.interactive:    
            print('\n'*2)
            msg = ('You have entered %i science cubes with %i different binnings'
                   ' (listed above).\nPlease enter a naming convension:\n' %
                  (len(args.cubes), len(sbff)))
            names.science = Input.str(msg, '{basename}', check=validity.trivial,
                                      example='s{sep}{filter}{binning}{sep}')
            msg = 'Please enter a naming option:\n1] Sequential \n2] Number suffix\n'
            nm_option = Input.str(msg, '2', check=lambda x: x in [1,2])
            sequential = 1  if nm_option==1 else 0
        else:
            sequential = 0
        
        run.magic_filenames( args.outdir )
        run.unpack(sequential, w2f=args.w2f )								#THIS FUNCTION NEEDS TO BE BROADENED IF YOU THIS PIPELINE AIMS TO REDUCE MULTIPLE SOURCES....
    else:
        #One can do photometry without splitting the cubes!!
        #TODO: check w2f???
        #run.make_slices(suffix)
        run.export_times(with_slices=False)
        #run.make_obsparams_file(suffix)
        run.export_headers()
        
    return run
  


################################################################################
#MAIN
################################################################################
#TODO: PRINT SOME INTRODUCTORY STUFFS

def parse_input():
    #Parse sys.argv arguments from terminal input
    
    #global args, head_info, names
    
    # exit clause for script parser.exit(status=0, message='')
    from sys import argv
    import argparse
    
    #Main parser
    main_parser = argparse.ArgumentParser(
        description='Data reduction pipeline for SHOC.')

    #group = parser.add_mutually_exclusive_group()
    #main_parser.add_argument('-v', '--verbose', action='store_true')
    #main_parser.add_argument('-s', '--silent', action='store_true')
    main_parser.add_argument(
        '-i',
        '--interactive',
        action='store_true',
        default=False,
        dest='interactive',
        help='Run the script in interactive mode.  You will be prompted for input when necessary')
    
    #FIXME: merge -d & -c option in favour of positional argument
    
    main_parser.add_argument(
        '-d',
        '--dir',
        default=None,
        help='The data directory. Defaults to current working directory.')
    main_parser.add_argument(
        '-o',
        '--outdir',
        help=('The data directory where the reduced data is to be placed.'
              'Defaults to input directory'))
    main_parser.add_argument(
        '-w',
        '--write-to-file',
        nargs='?',
        const=True,
        default=True,
        dest='w2f',
        help=(
            'Controls whether the script creates txt list of the files created.'
            'Requires -c option. Optionally takes filename base string for txt lists.'
        ))
    main_parser.add_argument(
        '-c',
        '--cubes',
        nargs='+',
        type=str,
        help=(
            'Science data cubes to be processed.  Requires at least one argument.'
            'Argument can be explicit list of files, a glob expression, a txt list, or a directory.'
        ))
    main_parser.add_argument(
        '-b',
        '--bias',
        nargs='+',
        default=False,
        help=(
            'Bias subtraction will be done. Requires -c option. Optionally takes argument(s) which indicate(s)'
            'filename(s) that can point to'
            ' master bias / '
            'cube of unprocessed bias frames / '
            'txt list of bias frames / '
            'explicit list of bias frames.'))
    main_parser.add_argument(
        '-f',
        '--flats',
        nargs='+',
        default=False,
        help=(
            'Flat fielding will be done.  Requires -c option.  Optionally takes an argument(s) which indicate(s)'
            ' filename(s) that can point to either '
            'master flat / '
            'cube of unprocessed flats / '
            'txt list of flat fields / '
            'explicit list of flat fields.'))
    #main_parser.add_argument('-u', '--update-headers',  help = 'Update fits file headers.')
    main_parser.add_argument(
        '-s',
        '--split',
        nargs='?',
        const=True,
        default=False,
        help='Split (burst) the data cubes into single fits files. '
        'This is utterly unnecessary, inefficient, and unproductive.'
        'Requires -c option.')
    main_parser.add_argument(
        '-t',
        '--timing',
        nargs='?',
        const=True,
        default=True,
        help=(
            'Calculate the timestamps for data cubes. Note that time-stamping is '
            'done by default when the cubes are split.  The timing data will be '
            'written to a text files with the cube basename and extention ".time"'
        ))
    main_parser.add_argument(
        '-g',
        '--gps',
        nargs='+',
        default=None,
        help='GPS triggering times. Explicitly or listed in txt file')
    main_parser.add_argument(
        '-k',
        '--kct',
        default=None,
        nargs='+',
        help='Kinetic Cycle Time for External GPS triggering.')
    main_parser.add_argument(
        '-q',
        '--combine',
        nargs='+',
        default=['daily', 'median'],
        help="Specifies how the bias/flats will be combined. Options are daily/weekly mean/median.")
    args = argparse.Namespace()

    #mx = main_parser.add_mutually_exclusive_group                           #NEED PYTHON3.3 AND MULTIGROUP PATCH FOR THIS...  OR YOUR OWN ERROR ANALYSIS???

    #Header update parser
    #FIXME: why not in main parser????
    
    head_parser = argparse.ArgumentParser()
    head_parser.add_argument(
        'update-headers',
        nargs='?',
        help='Update fits file headers.')  #action=store_const?
    head_parser.add_argument(
        '-obj', '--object',
        nargs='*', help='',
        default=[''])
    head_parser.add_argument('-ra',
                             '--right-ascension',
                             nargs='*',
                             default=[''],
                             dest='ra',
                             help='')
    head_parser.add_argument('-dec',
                             '--declination',
                             nargs='*',
                             default=[''],
                             dest='dec',
                             help='')
    head_parser.add_argument('-epoch', '--epoch', default=None, help='')
    head_parser.add_argument(
        '-date', '--date', nargs='*',
        default=[''], help='')
    head_parser.add_argument('-filter', '--filter', default=None, help='')
    head_parser.add_argument('-tel',
                             '--telescope',
                             dest='tel',
                             default=None,
                             type=str,
                             help='')
    head_parser.add_argument(
        '-obs', '--observatory',
        dest='obs', default=None,
        help='')
    #head_parser.add_argument('-em', '--em-gain', dest='em', default=None, help='Electron Multiplying gain level')
    head_info = argparse.Namespace()

    #Name convension parser
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
            "[] brackets indicate optional parameters."))  #action=store_const?
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
    #TODO: get  parser.add_subparsers vibes working!
    #===========================================================================
    def groupargs(arg, currentarg=[None]):
        '''Groups the arguments in sys.argv for parsing.'''
        if arg in valid_commands:
            currentarg[0] = arg
        return currentarg[0]
    
    commandlines = [list(argz) for cmd, argz in itt.groupby(argv, groupargs)]   #Groups the arguments in sys.argv for parsing
    for vc in valid_commands:
        setattr(args, vc.replace('-','_'), vc in argv)
        if not vc in argv:
            commandlines.append([''])                                         #DAMN HACK!
    
    for cmds, parser, namespace in zip(commandlines, parsers, namespaces):
        parser.parse_args(cmds[1:], namespace=namespace)
    
    return namespaces, main_parser


################################################################################################################    
def process_args(namespaces):
    
    args, head_info, names = namespaces
    #===========================================================================
    if args.dir:
        args.dir = iocheck(args.dir, os.path.exists, 1)
        args.dir = os.path.abspath(args.dir)
    #elif os.path.isdir(args.cubes): TypeError: argument should be string, bytes or integer, not list
        #args.dir = args.cubes
    else:
        args.dir = os.getcwd()
        print('No input dir given. Using %r' % args.dir)
    
    if args.outdir:
        args.outdir = iocheck(args.outdir, os.path.exists, 1)
    else:
        args.outdir = args.dir
    
    #===========================================================================
    if args.cubes is None:
        args.cubes = args.dir       #no cubes explicitly provided will use list of all files in input directory
    ##else:
        ##if os.path.isdir(args.cubes):
            ##args.dir = args.cubes
    
    if args.cubes:
        args.cubes = parsetolist(args.cubes, 
                                 os.path.exists, 
                                 path=args.dir,
                                 abspaths=True,
                                 raise_error=1)
        
        if not len(args.cubes):
            raise ValueError('No data!!')
        
        #Initialize Run
        args.cubes = shocRun(filenames=args.cubes, label='science')
        
        for cube in args.cubes:             #DO YOU NEED TO DO THIS IN A LOOP?
            cube._needs_flip = not cube.check(args.cubes[0], 'flip_state')                                #self-consistency check for flip state of cubes #NOTE: THIS MAY BE INEFICIENT IF THE FIRST CUBE IS THE ONLY ONE WITH A DIFFERENT FLIP STATE...
        
    if args.split:
        if args.outdir[0]:    #if an output directory is given
            args.outdir = os.path.abspath(args.outdir[0])
            if not os.path.exists(args.outdir):  #if it doesn't exist create it
                print('Creating reduced data directory {}.\n'.format(args.outdir))
                os.mkdir(args.outdir)

    #===========================================================================
    if args.gps:
        args.timing = True      #Do timing if gps info given
        
        if len(args.gps)==1:
            #triggers give either as single trigger time string or filename of trigger list
            valid_gps = iocheck(args.gps[0], validity.RA, raise_error=-1)       #if valid single time this will return that same str else None
            if not valid_gps:
                args.gps = parsetolist(args.gps, validity.RA, 
                                       path=args.dir, 
                                       abspath=0, 
                                       sort=0, 
                                       raise_error=1)
        
        #at ths point args.gps is list of explicit time strings.  
        #Check if they are valid representations of time
        args.gps = [iocheck(g, validity.RA, raise_error=1, convert=convert.RA)
                        for g in args.gps]
        #Convert and set as cube attribute
        args.cubes.that_need_triggers().set_gps_triggers(args.gps)
        
        
        #if any cubes are GPS triggered on each individual frame
        grun = args.cubes.that_need_kct()
        if len(args.kct) == 1 and len(grun) != 1:
            warn('A single GPS KCT provided for multiple externally triggered runs. '
                 'Assuming this applies for all these cubes: {}'.format(grun))
            
        elif len(grun) != len(args.kct):
            raise ValueError('The number of GPS KCT values provided ({}) {} does '
                             'not match the required number ({}) for {}'
                             ''.format(len(args.kct), args.kct, len(grun), grun))
        
        #if len(grun) and args.kct is None:
            #msg = ("In 'External' triggering mode EXPOSURE stores the total "
                  #"accumulated exposure time, which is utterly useless. We need "
                  #"the actual exposure time - i hope you've written it down somewhere!! "
                  #"Please specify KCT (Exposure time + Dead time):")
            #args.kct = Input.str(msg, 0.04, check=validity.float, what='KCT')
        
        for cube, kct in itt.zip_longest(grun, args.kct, fill_value=args.kct[0]):
            cube.kct = kct
        
    #===========================================================================        
    #try:
        ##set the kct for the cubes
        #for stack in args.cubes:
            #_, stack.kct = stack.get_kct()
    
    #except AttributeError as err:       #error catch for gps triggering
        ##Annotate the traceback
        #msg = section_header('Are these GPS triggered frames??', 
                             #swoosh='!', _print=False)
        #err = type(err)( '\n\n'.join((err.args[0], msg)) )
        #raise err.with_traceback(sys.exc_info()[2])
    
    #===========================================================================        
    if args.flats or args.bias:
        args.combine = list(map(str.lower, args.combine))
        when = 'day', 'daily', 'week', 'weekly'
        how = 'mean', 'median'
        understanding = when + how
        transmap = dict(grouper(when, 2))
        understood, misunderstood= map(list,
                            partition(understanding.__contains__, args.combine))
        if any(misunderstood):
            raise ValueError('Argument(s) {} for combine not understood.'
                             ''.format(misunderstood))
        else:
            understood = [transmap.get(u,u) for u in understood]
            how, when = next(zip(*partition(how.__contains__, understood)))
            args.combine = when
            args.fcombine = getattr(np, how)
            print('\nBias/Flat combination will be done by {} {}.'
                  ''.format(when, how))
        
    #===========================================================================        
    if args.flats:
        args.flats = parsetolist(args.flats, imaccess, path=args.dir, raise_error=1)
        args.flats = shocRun(filenames=args.flats, label='flat')
        
        #isolate the flat fields that match the science frames. only these will 
        #be processed
        match = args.flats.check(args.cubes, 'binning', 1, 1)
        args.flats = args.flats[match]
        
        for flat in args.flats:
            flat._needs_flip = not flat.check(args.cubes[0], 'flip_state')
        
        #flag the flats that need to be subframed, based on the science frames
        #which are subframed
        args.flats.flag_sub(args.cubes)
        
        args.flats.print_instrumental_setup()
        
    #else:
        #print('WARNING: No flat fielding will be done!')
    
    #===========================================================================    
    if args.bias:
        args.bias = parsetolist(args.bias, imaccess, path=args.dir, raise_error=1)
        args.bias = shocRun(filenames=args.bias, label='bias')
        
        
        #match the biases for the flat run
        if args.flats:
            match4flats = args.bias.check( args.flats, ['binning', 'mode'], 0, 1)
            args.bias4flats = args.bias[match4flats]
            for bias in args.bias4flats:
                bias._needs_flip = bias.check( args.flats[0], 'flip_state' )
            
            print('Biases for flat fields: ')
            args.bias4flats.print_instrumental_setup()
            
        #match the biases for the science run
        match4sci = args.bias.check( args.cubes, ['binning', 'mode'], 0, 1)
        args.bias = args.bias[match4sci]
        for bias in args.bias:
            bias._needs_flip = bias.check(args.cubes[0], 'flip_state')                        #NOTE: THIS MAY BE INEFICIENT IF THE FIRST CUBE IS THE ONLY ONE WITH A DIFFERENT FLIP STATE...
        
        args.bias.flag_sub( args.cubes )
        
        #FIXME: unnecessary to repeat if same as for flats
        print('Biases for science frames: ')
        args.bias.print_instrumental_setup()
    
    #else:
        #warn( 'No de-biasing will be done!' )
    
    #===========================================================================
    #Handle header updating here
    
    #NOTE: somehow, this attribute gets set even though we can never read it due to a syntax error
    delattr(head_info, 'update-headers')
    
    hi = head_info
    hi.coords = None
    if args.update_headers:
        #print( hi )
        #for attr in ['object', 'ra', 'dec', 'date']:
        #join arguments since they are read as lists
        hi.object    = ' '.join(hi.object)
        hi.ra        = ' '.join(hi.ra)
        hi.dec       = ' '.join(hi.dec)
        hi.date      = ' '.join(hi.date)
        
        if hi.ra and hi.dec:
            iocheck(hi.ra, validity.RA, 1)
            iocheck(hi.dec, validity.DEC, 1)
            hi.coords = SkyCoord(ra=hi.ra, dec=hi.dec, unit=('h','deg'))  #, system='icrs'
        else:
            hi.coords, hi.ra, hi.dec = get_coords_ra_dec(hi.object)
        
        def is_close(cooA, cooB, threshold=1e-3):
            return np.less([(cooA.ra - cooB.ra).value, 
                            (cooA.dec - cooB.dec).value], threshold).all()
        
        for cube in args.cubes:       #TODO: select instead of loop
            if cube.has_coords:
                if not is_close(cube.coords, hi.coords):
                    warn('Supplied coordinates {} will supersede header '
                         'coordinates {} in {}'.format(hi.coords, cube.coords,
                                                       cube.filename()))
                    cube.coords = hi.coords
                
        if not hi.date:
            #hi.date = args.cubes[0].date#[c.date for c in args.cubes]
            warn('Dates will be assumed from file creation dates.')
        
        if not hi.filter:
            warn('Filter assumed as WL.')
            hi.filter = 'WL'
        
        if hi.epoch:
            iocheck(hi.epoch, validity.epoch, 1)
        else:    
            warn('Assuming epoch J2000')
            hi.epoch = 2000
        
        if not hi.obs:
            note('Assuming location is SAAO Sutherland observatory.') 
            hi.obs = 'SAAO'
        
        if not hi.tel:
            note('Assuming telescope in SAAO 1.9m telescope!\n')   #FIXME: Don't have to assume for new data
            hi.tel = '1.9m'
    
    elif args.timing or args.split:
        #Need target coordinates for Barycentrization! Check the headers
        for cube in args.cubes:   #TODO: select instead of loop
            if cube.coordinates is None:
                warn('Object coordinates not found in header for {}!\n' 
                     'Barycentrization cannot be done without knowing target ' 
                     'coordinates!'.format(cube.filename()))
        
        #iocheck( hi.date, validity.DATE, 1 )
    #else:
        #warn( 'Headers will not be updated!' )
    
    #===========================================================================
    #if args.timing and not hi.coords:
        #Target coordinates not provided / inferred from 
       # warn( 'Barycentrization cannot be done without knowing target coordinates!' )
    
    
    if args.names:
        shocRun.NAMES.flats = names.flats
        shocRun.NAMES.bias = names.bias
        shocRun.NAMES.sci = names.sci

    #ANIMATE
    
    return args, head_info, names
    #WARN IF FILE HAS BEEN TRUNCATED -----------> PYFITS DOES NOT LIKE THIS.....WHEN UPDATING:  ValueError: total size of new array must be unchanged
    

################################################################################################################
def sanity_checks(args, main_parser):
    
    #Sanity checks for mutually exclusive keywords
    args_dict = args.__dict__
    prix = main_parser.prefix_chars
    prix2 = prix * 2
    
    #no other keywords allowed in interactive mode
    disallowedkw = {'interactive': set(args_dict.keys()) - set(['interactive'])}
    if args.interactive:
        for key in disallowedkw['interactive']:
            if args_dict[key]:
                #TODO: ignore these with warning
                raise KeyError('%s (%s) option not allowed in interactive mode'
                               %(prix + key[0], prix2 + key) )
    
    
    #Sanity checks for non-interactive mode
    #any one of these need to be specified for an actionable outcome
    rqd = ['dir', 'cubes', 'bias', 'flats']
    if not any([args_dict[key] for key in rqd]):
        main_parser.print_help()
        raise ValueError('\nNo files specified! Please specify files to be processed')
                         #''.format('/'.join(rqd)))
    
    
    ##, 'split', 'timing', 'update_headers', 'names']  
            #raise ValueError('No action specified!\n'
                         #'Please specify one or more commands:'
                         #'-s, -t, update-headers, name, or -i for interactive mode')
    return

    #FIXME: needs work since default arguments apply
    
    #Sanity checks for mutually inclusive keywords. Any one (or more) of the
    #listed keywords are required => or
    required = (('cubes',            ['timing', 'flats', 'bias', 'update-headers']),
                #('bias',             ['cubes']),       #if no cubes specified, simply compute masters
                #('flats',            ['cubes']),
                ('update-headers',   ['cubes']),
                ('split',            ['cubes']),
                ('timing',           ['cubes', 'gps']),
                ('ra',               ['update-headers']),
                ('dec',              ['update-headers']),
                ('obj',              ['update-headers']),
                ('epoch',            ['update-headers', 'ra', 'dec']),
                ('date',             ['update-headers']),
                ('filter',           ['update-headers']),
                ('kct',              ['gps']),)
            #    ('combine',          ['bias', 'flats'])       )
    requiredkw = OrderedDict(required)
    for key in args_dict:
        if args_dict[key] and (key in requiredkw):                                        #if this option has required options
            if not any(args_dict.get(rqk) for rqk in requiredkw[key]):                    #if none of the required options for this option are given
                ks = prix2 + key                                          #long string for option which requires option(s)
                ks_desc = '%s (%s)' %(ks[1:3], ks)
                rqks = [prix2 + rqk for rqk in requiredkw[key]]           #list of long string for required option(s)
                rqks_desc = ' / '.join(['%s (%s)' %(rqk[1:3], rqk)
                                            for rqk in rqks])   
                raise KeyError('One or more of the following option(s) required'
                               ' with option {}: {}'.format(ks_desc, rqks_desc))
        

    
################################################################################################################
def setup():
    namespaces, parser = parse_input()
    args, head_info, names = namespaces
    print(args)
    sanity_checks(args, parser)
    return process_args(namespaces)

       






    
################################################################################################################
if __name__ == '__main__':
    RUN_DIR = os.getcwd()

    #class initialisation
    bar = ProgressBar()                     #initialise progress bar
    #bar2 = ProgressBar(nbars=2)             #initialise the double progress bar
    args, head_info, names = setup()

    run = sciproc(args, head_info, names)


    def goodbye():
        '''switch back to original working directory'''
        os.chdir(RUN_DIR)
        print('Adios!')

    import atexit
    atexit.register(goodbye)

#main()