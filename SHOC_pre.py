# -*- coding: utf-8 -*-

# execute this script in the iraf working directory, or in day directory
# does pre-reductions using pyfits and pyraf

#NECESARY IMPROVEMENTS
#option for verbose output / logfile / TRIM EXCECUTION OPTION

#PYFITS COMBINE ALGORITHM ----> REJECTION ALGORITHM / SIGMA CLIPPING

#NEED MULTIPROCESSING!!!!!!!!!!!!!!!!!!!!!!


def tsktsk(s):
    t2 = time.time()
    print(s, t2-t1)

print( 'Importing modules...' )
import time
t1 = time.time()

import numpy as np
t1 = time.time()
tsktsk(np)

t1 = time.time()
import pyfits
tsktsk(pyfits)
#import matplotlib.animation as ani
#import matplotlib.pyplot as plt
t1 = time.time()
import os
tsktsk(os)

t1 = time.time()
import datetime
tsktsk(datetime)

t1 = time.time()
from glob import glob
tsktsk(glob)

t1 = time.time()
import collections
tsktsk(collections)

t1 = time.time()
import re
tsktsk(re)

#t1 = time.time()
#from pyraf.iraf import imaccess
#tsktsk(imaccess)

from copy import copy

t1 = time.time()
from astropy.time import TimeDelta
tsktsk('astropy')

t1 = time.time()
from misc import *
tsktsk('misc')

t1 = time.time()
from airmass import Young94, altitude
from SHOC_readnoise import ReadNoiseTable
from SHOC_timing import Time
from SHOC_user_input import Input
from SHOC_user_input import ValidityTests as validity
from SHOC_user_input import Conversion as convert
tsktsk('SHOC modules')
print( 'Done!\n\n' )


ipshell = make_ipshell()

#################################################################################################################################################################################################################
#Function definitions
#################################################################################################################################################################################################################

#def in_ipython():
    #try:
        #return __IPYTHON__
    #except:
        #return False

#################################################################################################################################################################################################################
#if not in_ipython():
    #print('\n\nRunning in Python. Defining autocompletion functions.\n\n')
    #import readline
    #from completer import Completer
    
    #comp = Completer()
    #readline.parse_and_bind('tab: complete')
    ##readline.parse_and_bind('set match-hidden-files off')
    #readline.parse_and_bind('set skip-completed-text on')
    #readline.set_completer(comp.complete)                                                   #sets the autocomplete function to look for filename matches
#else:
    #print('\n\nRunning in IPython...\n\n')


#################################################################################################################################################################################################################  
#Bias reductions
#################################################################################################################################################################################################################
def bias_proc(m_bias_dict, sb_dict):             #THIS SHOULD BE A METHOD OF THE SHOC_RUN CLASS
    #NOTE: Bias frames here are technically dark frames taken at minimum possible exposure time.  SHOC's dark current is (theoretically) constant with time,so these may be used as bias frames.  
    #	   Also, there is therefore no need to normalised these to the science frame exposure times.
    
    '''Do the bias reductions on sciene data
    Parameters
    ----------
    mbias_dict : Dictionary with binning,filename key,value pairs for master biases
    sb_dict : Dictionary with binning,run key,value pairs for science data
    
    Returns
    ------
    Bias subtracted SHOC_Run
    '''
    
    fn_ls = []
    for attrs, master in m_bias_dict.items():
        #mb_data = #pyfits.getdata(mb_fn, memmap=True)
        
        stacks = sb_dict[attrs]              #the cubes (as SHOC_Run) for this attrs value
       
        stacks_b = []
        msg = '\nDoing bias subtraction on the stack: '
        lm = len(msg)
        print( msg )
        for stack in stacks:
            print( ' '*lm, stack.get_filename() )
            
            header = stack[0].header
            header['BIASCORR'] = ( True, 'Bias corrected' )					 #Adds the keyword 'BIASCORR' to the image header to indicate that bias correction has been done
            hist = 'Bias frame {} subtracted at {}'.format( master.get_filename(), datetime.datetime.now())
            header.add_history(hist, before='HEAD' ) 	#Adds the filename and time of bias subtraction to header HISTORY
            
            stack[0].data -= master[0].data
            
    sb_dict.label = 'science frames (bias subtracted)'
    return sb_dict
    
  
#################################################################################################################################################################################################################  
#Flat field reductions
#################################################################################################################################################################################################################

def flat_proc(mflat_dict, sb_dict):             #THIS SHOULD BE A METHOD OF THE SHOC_RUN CLASS
    '''Do the flat field reductions
    Parameters
    ----------
    mflat_dict : Dictionary with binning,run key,value pairs for master flat images
    sb_dict : Dictionary with binning,run key,value pairs
    
    Returns
    ------
    Flat fielded SHOC_Run
    '''
    
    fn_ls = []
    for attrs, masterflat in mflat_dict.items():
        
        if isinstance( masterflat, SHOC_Run):
            masterflat = masterflat[0]                  #HACK!!
        
        mf_data = masterflat[0].data            #pyfits.getdata(masterflat, memmap=True)
            
        if round(np.mean(mf_data), 1) != 1:
            raise ValueError('Flat field not normalised!!!!')
        
        stacks = sb_dict[attrs]                                       #the cubes for this binning value
        msg = '\nDoing flat field division on the stack: '
        lm = len(msg)
        print( msg, )
        for stack in stacks:
            print( ' '*lm + stack.get_filename() )
            
            header = stack[0].header
            header['FLATCORR'] = (True, 'Flat field corrected')					#Adds the keyword 'FLATCORR' to the image header to indicate that flat field correction has been done
            hist = 'Flat field {} subtracted at {}'.format( masterflat.get_filename(), datetime.datetime.now() )
            header.add_history(hist, before='HEAD' ) 	#Adds the filename used and time of flat field correction to header HISTORY
            
            stack[0].data /= mf_data                                                                            #flat field division

    sb_dict.label = 'science frames (flat fielded)'
    return sb_dict
    
#################################################################################################################################################################################################################  
# Headers
#################################################################################################################################################################################################################  

def header_proc( run ):             #THIS SHOULD BE A METHOD OF THE SHOC_RUN CLASS
    '''update the headers where necesary'''
    
    section_header( 'Updating Fits headers' )
    
    update_head = SHOCHeader( )
    update_head.set_defaults( run[0][0].header )        #set the defaults for object info according to those (if available) in the header of the first cube
    
    keywords = ['OBJECT', 'RA', 'DEC', 'EPOCH', 'FILTER', 'OBSERVAT', 'RON', 'SENSITIVITY']      #'DATE-OBS'                    #ADD TELESCOPE??
    description = ['Object designation', 'RA', 'Dec', 'Epoch of RA and Dec', 'Filter', 'Observatory', 'CCD Readout Noise', 'CCD Sensitivity' ]  #'Date of observation',
    example = [''," (eg: '03:14:15' or '03 14 15')", " (eg: '+27:18:28.1' or '27 18 28.1')", " eg: 2000", ' (WL for clear)', '', '', '']        #" (eg: '2011-12-13' or '2011 12 13') ",
    validity_test = [validity.trivial, validity.RA, validity.DEC, validity.float, validity.trivial] + [validity.trivial]*3                      #validity.DATE,
    conversion = [convert.trivial, convert.RA_DEC, convert.RA_DEC, convert.trivial, convert.trivial] + [convert.trivial]*3                      #convert.ws2ds,
    
    if args.update_headers:
        info = []
        askfor = []
        for kw in keywords:
            match = [key for key in head_info.__dict__.keys() if kw.lower().startswith(key)]    #match the terminal (argparse) input arguments with the keywords
            if len(match):
                inf = getattr( head_info, match[0] )    #terminal input value (or default) for this keyword
                if inf:
                    info.append( inf )
                    askfor.append( 0 )
                else:
                    info.append( '' )                   #if terminal input info is empty this item will be asked for explicitly
                    askfor.append( 1 )
            else:               #the keywords that don't have corresponding terminal input arguments
                info.append( '' )
                askfor.append( 0 )
        
    else:    
        info = ['']*9  #['test', '23 22 22.33', '44 33 33.534', '2000', '2012 12 12', 'WL','saao', '','']    #
        askfor = [1,1,1,1,1,0,0,0]
    
    if sum(askfor):
        print("\nPlease enter the following information about the observations to populate the image header. If you enter nothing that item will not be updated.")  #PRINT FORMAT OPTIONS
        
    update_cards = [ SHOCHeaderCard(key, val, comment, example=ex, validity_test=vt, conversion=conv, askfor=ask) 
                    for key, val, comment, ex, vt, conv, ask 
                    in zip(keywords, info, description, example, validity_test, conversion, askfor) ]
    
    update_head.extend( update_cards )
    #update_head['OBSERVAT'] = 
    
    print( 'Updating headers with the following information:' )
    print( update_head )
    print( )
    
    for hdu in run:
        update_head.set_ron_sens_date( hdu[0].header )     #set RON, SENS, OBSERVAT (the RON and SENS may change through the run)
        update_head.check()                                #Check if updating observation info is necessary                            #NOT WORKING!!!!!!!!!!!!!
        
        update_head.update_to( hdu )                       #copy the cards in update_head to the hdu header and flush
        #hdu.set_name_dict()
    
    return update_head
    
#################################################################################################################################################################################################################  
# Science image pre-reductions
#################################################################################################################################################################################################################

def match_closest(sci_run, calib_run, match_exact, match_closest=None, threshold_warn=7, _pr=1):
    '''
    Match the attributes between sci_run and calib_run.
    Matches exactly to the attributes given in match_exactly, and as closely as possible to the  
    attributes in match_closest. Separates sci_run by the attributes in both match_exact and 
    match_closest, and builds an index dictionary for the calib_run which can later be used to
    generate a StructuredRun instance.
    Parameters
    ----------
    sci_run :           The SHOC_Run to which comparison will be done
    calib_run :         SHOC_Run which will be trimmed by matching
    match_exact :       tuple or str. keywords to match exactly         NOTE: No checks run to ensure calib_run forms a subset of sci_run w.r.t. these attributes
    match_closest:      tuple or str. keywords to match as closely as possible

    Returns
    ------
    s_sr :              StructuredRun of science frames separated 
    out_sr

    '''
    #====================================================================================================
    def str2tup(keys):
        if isinstance(keys, str):
            keys = keys,          #a tuple
        return keys
    #====================================================================================================
    
    msg = ('\nMatching {} frames to {} frames by:\tExact {};\t Closest {}\n'
            ).format( calib_run.label.upper(), sci_run.label.upper(), match_exact, repr(match_closest) )
    print( msg )
    
    #create the StructuredRun for science frame and calibration frames
    match_exact, match_closest = str2tup(match_exact), str2tup(match_closest)
    sep_by = tuple( key for key in flatten([match_exact, match_closest]) if not key is None )   
    s_sr, sflag = sci_run.attr_sep( *sep_by )
    c_sr, bflag = calib_run.attr_sep( *sep_by )
    
    #Do the matching - map the science frame attributes to the calibration StructuredRun element with closest match
    lme = len(match_exact)                      #NOTE AT THE MOMENT THIS ONLY USES THE FIRST KEYWORD IN match_losest TO DETERMINE THE CLOSEST MATCH 
    _, sciatts = sci_run.zipper( sep_by )                       #sep_by key attributes of the sci_run
    _, calibatts = calib_run.zipper( sep_by )
    ssciatts = np.array( list(set(sciatts)), object )           #a set of the science frame attributes
    calibatts = np.array( list(set(calibatts)), object )
    sss = ssciatts.shape
    where_thresh = np.zeros((2*sss[0], sss[1]+1))       #state array to indicate where in data threshold is exceeded (used to colourise the table)
    
    runmap, attmap = {}, {}
    datatable = []
    
    for i, attrs in enumerate(ssciatts):
        lx = np.all( calibatts[:,:lme]==attrs[:lme], axis=1 )       #those calib cubes with same attrs (that need exact matching)
        delta = abs( calibatts[:,lme] - attrs[lme] )
        lc = delta==min(delta[lx])
        l = lx & lc
        
        tattrs = tuple(attrs)
        cattrs = tuple( calibatts[l][0] )
        attmap[tattrs] = cattrs
        runmap[tattrs] = c_sr[cattrs]

        datatable.append( ( str(s_sr[tattrs]), )  + tattrs )
        datatable.append( ( str(runmap[tattrs]), ) + attmap[tattrs] )
        
        #Threshold warnings
        if threshold_warn:
            deltatype = type(delta[0])
            threshold = deltatype(threshold_warn)  #type cast the attribute for comparison (datetime.timedelta for date attribute, etc..)
            if np.any( delta[l] > deltatype(0) ):
                where_thresh[2*i:2*(i+1),lme+1] += 1
                
            if np.any( delta[l] > threshold ):
                fns = ' and '.join( c_sr[cattrs].get_filenames() )
                sci_fns = ' and '.join( s_sr[tattrs].get_filenames() )
                msg = ( 'Closest match of {} {} in {} to {} in {} excedees given threshold of {}!!'
                        ).format( tattrs[lme], match_closest[0].upper(), fns, cattrs[lme], sci_fns, threshold_warn )
                warn( msg )
                where_thresh[2*i:2*(i+1),lme+1] += 1
        
    out_sr = StructuredRun( runmap )
    out_sr.label = calib_run.label
    out_sr.sep_by = s_sr.sep_by
    
    if _pr:
        #Generate data table of matches
        col_head = ('Filename(s)',) + tuple( map(str.upper, sep_by) )
        row_borders = range(1, len(datatable), 2)
        
        table = Table( datatable, col_head, row_borders=row_borders )
        table.colourise( where_thresh, 'default', 'yellow', 202 )
        
        print( '\nThe following matches have been made:' )
        print( table )
        
    return s_sr, out_sr

#====================================================================================================
def sciproc(run):                               #WELL THIS CAN NOW BE A METHOD OF SHOC_Run CLASS
    
    section_header( 'Science frame processing' )
    
    if args.w2f:
        run.export_filenames( 'cubes.txt' )
    
    if args.update_headers:
        updated_head = header_proc(run)
    
    #====================================================================================================
    # Timing
    
    if args.timing or args.split:       section_header( 'Timing' )
        
    if args.gps:
        run.set_gps_triggers( args.gps )

    if args.timing or args.split:
        run.set_times( head_info.coords )
        run.set_airmass( head_info.coords )
        run.export_times()
    
    #====================================================================================================
    # Debiasing
    
    suffix = ''
    if args.bias:
        
        section_header( 'Debiasing' )
        
        suffix += 'b'           #the string to append to the filename base string to indicate bias subtraction
        s_sr, b_sr = match_closest(run, args.bias, ('binning','mode'), 'kct')
        
        b_sr.magic_filenames()
        
        masterbias = b_sr.compute_master( load=1 )      #StructuredRun for master biases separated by 'binning','mode', 'kct' 
        b_sr = bias_proc(masterbias, s_sr)
        
        #s_sr.writeout( suffix )
        
        #run.reload( bls )
        #s_sr, _ = run.attr_sep( 'binning' )     #'filter'
    else:
        s_sr, _ = run.attr_sep( 'binning' )     #'filter'
        masterbias = masterbias4flats = None
    
    #====================================================================================================
    # Flat fielding
    if args.bias or args.flats:         section_header( 'Flat field processing' )
    
    if args.flats:
        suffix += 'ff'          #the string to append to the filename base string to indicate flat fielding
        
        matchdate = 'date'      if args.combine=='daily'     else None
        threshold_warn = 1      if matchdate                 else None
        
        s_sr, f_sr = match_closest(run, args.flats, 'binning', matchdate, threshold_warn)
        f_sr.magic_filenames()
        
        if args.bias:
            if args.bias4flats:
                f_sr, b4f_sr = match_closest(f_sr.flatten(), args.bias4flats, ('binning','mode'), 'kct')
                b4f_sr.magic_filenames()
                for attrs, brun in b4f_sr.items():
                    for cube in brun:
                        cube.filename_gen.basename += '.b4f'              #HACK to keep the filenames of biases for science run and flat run different
                
                masterbias4flats = b4f_sr.compute_master( load=1 )                        #StructuredRun for master biases separated by 'binning','mode', 'kct' 
                #else:
                    #masterbias4flats = masterbias           #masterbias contains frames with right mode to to flat field debiasing
            else:
                masterbias4flats = None
                
        masterflats = f_sr.compute_master( masterbias4flats, load=1 )
        masterflats, _ = masterflats.attr_sep( 'binning' )
        s_sr, _ = s_sr.attr_sep( 'binning' )
        
        s_sr = flat_proc( masterflats, s_sr)                       #returns StructuredRun of flat fielded science frames
    
    if args.bias or args.flats:
        
        nfns = s_sr.writeout( suffix )
        
        #Table for Calibrated Science Run
        keys, attrs = run.zipper( ('binning', 'mode', 'kct') )
        col_head = ('Filename',) + tuple( map(str.upper, keys) )
        datatable = [(fn, )+attr for fn, attr in zip(nfns, attrs)]
        table = Table( datatable, col_head )
        
        print( '\n\nCalibrated Science Run' )
        print( table )
        
        if args.w2f:
            run.export_filenames( 'cubes.bff.txt' )
            
    #animate(sbff[0])
    
    #raise ValueError
    #====================================================================================================
    #splitting the cubes
    if args.split:
        #User input for science frame output designation string
        if args.interactive:    
            print('\n'*2)
            names.science = Input.str('You have entered %i science cubes with %i different binnings (listed above).\nPlease enter a naming convension:\n' %(len(args.cubes), len(sbff)),
                                '{basename}', example='s{filter}{binning}{sep}',  validity_test=validity.trivial)
            
            nm_option = Input.str('Please enter a naming option:\n1] Sequential \n2] Number suffix\n', '2', validity_test=lambda x: x in [1,2] )
            sequential = 1  if nm_option==1 else 0
        else:
            sequential = 0
        
        run.magic_filenames( args.output_dir )
        run.unpack( sequential )								#THIS FUNCTION NEEDS TO BE BROADENED IF YOU THIS PIPELINE AIMS TO REDUCE MULTIPLE SOURCES....
 
    return run
  
#################################################################################################################################################################################################################
#Misc Function definitions
#################################################################################################################################################################################################################

#################################################################################################################################################################################################################    
def imaccess( filename ):
    return True #i.e. No validity test performed!
    #try:
        #pyfits.open( filename )
        #return True
    #except BaseException as err:
        #print( 'Cannot access the file {}...'.format(repr(filename)) )
        #print( err )
        #return False

###########################################################################q######################################################################################################################################      
def get_coords( obj_name ):
    ''' Attempts a SIMBAD Sesame query with the given object name. '''
    from astropy import coordinates as astcoo
    try: 
        print( '\nQuerying SIMBAD database for {}...'.format(repr(obj_name)) )
        coo = astcoo.name_resolve.get_icrs_coordinates( obj_name )
        ra = coo.ra.to_string( unit='h', precision=2, sep=' ', pad=1 )
        dec = coo.dec.to_string( precision=2, sep=' ', alwayssign=1, pad=1 )
        
        print( 'The following ICRS J2000.0 coordinates were retrieved:\nRA = {}, DEC = {}\n'.format(ra, dec) )
        return coo, ra, dec
        
    except BaseException as err:     #astcoo.name_resolve.NameResolveError
        print( 'ERROR in retrieving coordinates...' )
        print( err )
        return None, None, None

#################################################################################################################################################################################################################      

def catfiles(infiles, outfile):
    '''used to concatenate large files.  Works where bash fails due to too large argument lists'''
    with open(outfile, 'w') as outfp:
        for fname in infiles:
            with open(fname) as infile:
                for line in infile:
                    outfp.write(line)   

    
#################################################################################################################################################################################################################
def section_header( msg, swoosh='=' ):
    width = 100
    swoosh = swoosh * width
    msg = SuperString( msg ).center( width )
    
    print( '\n'.join( ['\n', swoosh, msg, swoosh, '\n' ] ) )

################################################################################################################
#MAIN
#TODO: PRINT SOME INTRODUCTORY STUFFS
################################################################################################################

#Parse sys.argv arguments from terminal input
def setup():
    global args, head_info, names
    
    # exit clause for script parser.exit(status=0, message='')
    from sys import argv
    import argparse
    from itertools import groupby
    
    #Main parser
    main_parser = argparse.ArgumentParser(description='Data reduction pipeline for SHOC.')
    
    #group = parser.add_mutually_exclusive_group()
    #main_parser.add_argument('-v', '--verbose', action='store_true')
    #main_parser.add_argument('-s', '--silent', action='store_true')
    main_parser.add_argument('-i', '--interactive', action='store_true', default=False, dest='interactive', help='Run the script in interactive mode.  You will be prompted for input when necessary')

    main_parser.add_argument('-d', '--dir', nargs=1, default=[os.getcwd()], dest='dir', 
                help = 'The data directory. Defaults to current working directory.')
    main_parser.add_argument('-o', '--output-dir', nargs=1, default=['ReducedData'], type=str, 
                help = ('The data directory where the reduced data is to be placed.' 
                        'Defaults to $PWD/ReducedData.'))
    main_parser.add_argument('-w', '--write-to-file', nargs='?', const=True, default=False, dest='w2f', 
                help = ('Controls whether the script creates txt list of the files created.' 
                        'Requires -c option. Optionally takes filename base string for txt lists.'))
    main_parser.add_argument('-c', '--cubes', nargs='+', type=str, 
                help = ('Science data cubes to be processed.  Requires at least one argument.'  
                        'Argument can be explicit list of files, a glob expression, a txt list, or a directory.'))
    main_parser.add_argument('-b', '--bias', nargs='+', default=False, 
                help = ('Bias subtraction will be done. Requires -c option. Optionally takes argument(s) which indicate(s)'
                        'filename(s) that can point to' 
                           ' master bias / '
                            'cube of unprocessed bias frames / '
                            'txt list of bias frames / '
                            'explicit list of bias frames.'))
    main_parser.add_argument('-f', '--flats', nargs='+', default=False, 
                help = ('Flat fielding will be done.  Requires -c option.  Optionally takes an argument(s) which indicate(s)'
                        ' filename(s) that can point to either '
                            'master flat / '
                            'cube of unprocessed flats / '
                            'txt list of flat fields / '
                            'explicit list of flat fields.'))
    #main_parser.add_argument('-u', '--update-headers',  help = 'Update fits file headers.')
    main_parser.add_argument('-s', '--split', nargs='?', const=True, default=False, 
                help = 'Split the data cubes. Requires -c option.')
    main_parser.add_argument('-t', '--timing', nargs='?', const=True, default=False, 
                help = 'Calculate the timestamps for data cubes. Note that time-stamping is done by default when the cubes are split.  The timing data will be written to text files with the cube basename and extention indicating the time format used.')
    main_parser.add_argument('-g', '--gps', nargs='+', default=None, 
                help = 'GPS triggering times. Explicitly or listed in txt file')
    main_parser.add_argument('-k', '--kct', default=None, 
                help = 'Kinetic Cycle Time for External GPS triggering.')
    main_parser.add_argument('-q', '--combine', type=str, default='daily', 
                help = "Specifies how the bias/flats will be combined. Options are 'daily' or 'weekly'.")
    args = argparse.Namespace()
    
    #mx = main_parser.add_mutually_exclusive_group                           #NEED PYTHON3.3 AND MULTIGROUP PATCH FOR THIS...  OR YOUR OWN ERROR ANALYSIS???
    
    #Header update parser
    head_parser = argparse.ArgumentParser()
    head_parser.add_argument('update-headers', nargs='?', help='Update fits file headers.') #action=store_const?
    head_parser.add_argument('-obj', '--object', nargs='*', help='')
    head_parser.add_argument('-ra', '--right-ascention', nargs='*', default=[''], dest='ra', help='')
    head_parser.add_argument('-dec', '--declination', nargs='*', default=[''], dest='dec', help='')
    head_parser.add_argument('-epoch', '--epoch', default=None, help='')
    head_parser.add_argument('-date', '--date', nargs='*', default=[''], help='')
    head_parser.add_argument('-filter', '--filter', default=None, help='')
    head_parser.add_argument('-obs', '--observatory', dest='obs', default=None, help='')
    head_info = argparse.Namespace()
    
    #Name convension parser
    name_parser = argparse.ArgumentParser()
    name_parser.add_argument('name', nargs='?',
                help = ("template for naming convension of output files.  "
                        "eg. 'foo{sep}{basename}{sep}{filter}[{sep}b{binning}][{sep}sub{sub}]' where"
                        "the options are: "
                        "basename - the original filename base string (no extention); "
                        "name - object designation (if given or in header); "
                        "sep - separator character(s); filter - filter band; "
                        "binning - image binning. eg. 4x4; "
                        "sub - the subframed region eg. 84x60. "
                        "[] brackets indicate optional parameters.")) #action=store_const?
    name_parser.add_argument('-fl', '--flats', nargs=1, default='f[{date}{sep}]{binning}[{sep}sub{sub}][{sep}filt{filter}]')
    name_parser.add_argument('-bi', '--bias', default='b[{date}{sep}]{binning}[{sep}m{mode}][{sep}t{kct}]', nargs=1) 
    name_parser.add_argument('-sc', '--science-frames', nargs=1, dest='sci', default='{basename}')
    names = argparse.Namespace()
    
    parsers = [main_parser, head_parser, name_parser]
    namespaces = [args, head_info, names]
    
    valid_commands = ['update-headers', 'names']
    #====================================================================================================
    def groupargs(arg, currentarg=[None]):
        '''Groups the arguments in sys.argv for parsing.'''
        if arg in valid_commands:
            currentarg[0] = arg
        return currentarg[0]
    
    commandlines = [ list(args) for cmd, args in groupby(argv, groupargs) ]   #Groups the arguments in sys.argv for parsing
    for vc in valid_commands:
        setattr(args, vc.replace('-','_'), vc in argv)
        if not vc in argv:
            commandlines.append( [''] )                                         #DAMN HACK!
    
    for cmds, parser, namespace in zip(commandlines, parsers, namespaces):
        parser.parse_args( cmds[1:], namespace=namespace )
    
    #Sanity checks for mutually exclusive keywords
    args_dict = args.__dict__
    disallowedkw = {'interactive': set(args_dict.keys()) - set(['interactive']),
                    }                                                                   #no other keywords allowed in interactive mode
    if args.interactive:
        for key in disallowedkw['interactive']:
            if args_dict[key]:
                raise KeyError( '%s (%s) option not allowed in interactive mode' %(main_parser.prefix_chars+key[0], main_parser.prefix_chars*2+key) )

    #Sanity checks for mutually inclusive keywords. Potentailly only one of the listed keywords required => or
    requiredkw = {      'cubes':                ['timing', 'flats', 'bias', 'update-headers'],
                        'bias':                 ['cubes'],
                        'flats':                ['cubes'],
                        'update-headers':       ['cubes'],
                        'split':                ['cubes'],
                        'timing':               ['cubes', 'gps'],
                        'ra':                   ['update-headers'],
                        'dec':                  ['update-headers'],
                        'obj':                  ['update-headers'],
                        'epoch':                ['update-headers', 'ra', 'dec'],
                        'date':                 ['update-headers'],
                        'filter':               ['update-headers'],
                        'kct':                  ['gps'],
                        'combine':              ['bias', 'flats']       }
    for key in args_dict:
        if args_dict[key] and key in requiredkw:                                        #if this option has required options
            if not any( rqk in args_dict for rqk in requiredkw[key] ):                    #if none of the required options for this option are given
                ks = main_parser.prefix_chars*2+key                                          #long string for option which requires option(s)
                ks_desc = '%s (%s)' %(ks[1:3], ks)
                rqks = [main_parser.prefix_chars*2+rqk for rqk in requiredkw[key]]           #list of long string for required option(s)
                rqks_desc = ' / '.join( ['%s (%s)' %(rqk[1:3], rqk) for rqk in rqks] )   
                raise KeyError( 'One or more of the following option(s) required with option {}: {}'.format(ks_desc, rqks_desc) )
        
    
    
    #Sanity checks for non-interactive mode
    any_rqd = ['cubes', 'bias', 'flats', 'split', 'timing', 'update_headers', 'names']  #any of these need to be specified for an action
    if not any([getattr(args, key) for key in any_rqd]):
        raise ValueError('No action specified!\n Please specify one or more commands: -s,-t, update-headers, name, or -i for interactive mode')
    #====================================================================================================
    if args.dir[0]:
        args.dir = validity.test(args.dir[0], os.path.exists, 1)

        args.dir = os.path.abspath(args.dir)
        os.chdir(args.dir)                                        #change directory  ??NECESSARY??
    
    #====================================================================================================
    if args.cubes is None:
        args.cubes = args.dir       #no cubes explicitly provided will use list of all files in input directory
    
    if args.cubes:
        args.cubes = validity.test_ls(args.cubes, imaccess, raise_error=1)      #imaccess????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
        if not len(args.cubes):
            raise ValueError( 'File {} contains no data!!'.format('?') )
        args.cubes = SHOC_Run( filenames=args.cubes, label='science' )
        
        for cube in args.cubes:             #DO YOU NEED TO DO THIS IN A LOOP?
            cube._needs_flip = not cube.check( args.cubes[0], 'flip_state' )                                #self-consistency check for flip state of cubes #NOTE: THIS MAY BE INEFICIENT IF THE FIRST CUBE IS THE ONLY ONE WITH A DIFFERENT FLIP STATE...
        
        args.cubes.print_instrumental_setup()
    
    if args.output_dir[0]:                                          #if an output directory is given
        args.output_dir = os.path.abspath( args.output_dir[0] )
        if not os.path.exists(args.output_dir):                              #if it doesn't exist create it
            print( 'Creating reduced data directory {}.\n'.format(args.output_dir) )
            os.mkdir( args.output_dir )
    
    #====================================================================================================
    if args.gps:
        
        if len( args.cubes )==1:
            test_func = validity.test
        else:
            if len(args.gps)==1:          #single gps trigger 
                if args.cubes.check_rollover_state():
                    print( ("\nA single GPS trigger provided. Run contain rolled over cubes. " +\
                            "Start time for rolled over cubes will be inferred from the length of the preceding cube(s).\n") )
                    test_func, args.gps  = validity.test, args.gps[0]
                else:
                    raise ValueError( 'Only {} GPS trigger given. Please provide {} for {}'.format( len(args.gps), len(arg.cubes), args.cubes ) )
            else:
                test_func = validity.test_ls
        
        args.gps = test_func( args.gps, validity.RA, raise_error=1)
        
        if np.any( [stack.trigger_mode=='External' for stack in args.cubes] ):
            if args.kct is None:
                msg = ( "'\nIn External triggering mode EXPOSURE stores the total exposure time which is utterly useless.\n" + \
                        "I need the kinetic cycle time - i hope you've written it down somewhere!!\n")
                args.kct = Input.str(msg+'Please specify KCT (Exposure time):', 0.04, validity_test=validity.float, what='KCT')
    
    #====================================================================================================        
    if not args.combine.lower() in ['daily','weekly']:
        warn( 'Argument {} for combine not understood.  Bias/Flat combination will be done daily.' )
        args.combine = 'daily'
    
    #====================================================================================================        
    if args.flats:
        args.flats = validity.test_ls(args.flats, imaccess, raise_error=1)
        args.flats = SHOC_Run( filenames=args.flats, label='flat' )
        
        match = args.flats.check(args.cubes, 'binning', 1, 1)
        args.flats = args.flats[match]                          #isolates the flat fields that match the science frames --> only these will be processed
        
        for flat in args.flats:
            flat._needs_flip = not flat.check( args.cubes[0], 'flip_state' )                            #NOTE: THIS MAY BE INEFICIENT IF THE FIRST CUBE IS THE ONLY ONE WITH A DIFFERENT FLIP STATE...
        
        args.flats.flag_sub( args.cubes )                       #flag the flats that need to be subframed, based on the science frames which are subframed
        
        args.flats.print_instrumental_setup()
    #else:
        #print( 'WARNING: No flat fielding will be done!' )
    
    #====================================================================================================    
    if args.bias:
        args.bias = validity.test_ls(args.bias, imaccess, raise_error=1)
        args.bias = SHOC_Run( filenames=args.bias, label='bias' )
        
        
        #match the biases for the flat run
        if args.flats:
            match4flats = args.bias.check( args.flats, ['binning', 'mode'], 0, 1)
            args.bias4flats = args.bias[match4flats]
            for bias in args.bias4flats:
                bias._needs_flip = bias.check( args.flats[0], 'flip_state' )
            
            print( 'Biases for flat fields: ')
            args.bias4flats.print_instrumental_setup()
            
        #match the biases for the science run
        match4sci = args.bias.check( args.cubes, ['binning', 'mode'], 1, 1)
        args.bias = args.bias[match4sci]
        for bias in args.bias:
            bias._needs_flip = bias.check( args.cubes[0], 'flip_state' )                        #NOTE: THIS MAY BE INEFICIENT IF THE FIRST CUBE IS THE ONLY ONE WITH A DIFFERENT FLIP STATE...
        
        args.bias.flag_sub( args.cubes )
    
        print( 'Biases for science frames: ' )
        args.bias.print_instrumental_setup()
        
        
    #else:
        #print( 'WARNING: No de-biasing will be done!' )
    
    #====================================================================================================
    delattr( head_info, 'update-headers')
    head_info.coords = None
    if args.update_headers:
        #print( head_info )
        #for attr in ['object', 'ra', 'dec', 'date']:
        head_info.object = ' '.join( head_info.object )
        head_info.ra = ' '.join( head_info.ra )
        head_info.dec = ' '.join( head_info.dec )
        head_info.date = ' '.join( head_info.date )
        
        if head_info.ra and head_info.dec:
            validity.test( head_info.ra, validity.RA, 1 )
            validity.test( head_info.dec, validity.DEC, 1 )
            head_info.coords = ICRS(ra=head_info.ra, dec=head_info.dec, unit=('h','deg'))
        else:
            coo, ra, dec = get_coords( head_info.object )
            head_info.ra = ra
            head_info.dec = dec
            head_info.coords = coo
        
        if not head_info.date:
            #head_info.date = args.cubes[0].date#[c.date for c in args.cubes]
            warn( 'Dates will be assumed from file creation dates.' )
        
        if not head_info.filter:
            warn( 'Filter assumed as WL.' )
            head_info.filter = 'WL'
        
        if head_info.epoch:
            validity.test( head_info.epoch, validity.epoch, 1)
        else:    
            warn( 'Assuming epoch J2000' )
            head_info.epoch = 2000
        
        if not head_info.obs:
            print( '\nAssuming SAAO Sutherland observatory\n' ) 
            head_info.obs = 'SAAO'
        
        #
        #validity.test( head_info.date, validity.DATE, 1 )
    #else:
        #warn( 'Headers will not be updated!' )
        
    
    if args.names:
        SHOC_Run.NAMES.flats = names.flats
        SHOC_Run.NAMES.bias = names.bias
        SHOC_Run.NAMES.sci = names.sci

    #ANIMATE
    

    #WARN IF FILE HAS BEEN TRUNCATED -----------> PYFITS DOES NOT LIKE THIS.....WHEN UPDATING:  ValueError: total size of new array must be unchanged
    
    
################################################################################################################
RUN_DIR = os.getcwd()
#RNT_filename = '/home/SAAO/hannes/work/SHOC_ReadoutNoiseTable_new'              
RNT_filename = '/home/hannes/iraf-work/SHOC_ReadoutNoiseTable_new'      #
RNT = ReadNoiseTable(RNT_filename)

#filenames, dirnames, txtnames = ['rar'],[],[]   #dir_info(args.dir)

#class initialisation
bar = ProgressBar()                     #initialise progress bar
bar2 = ProgressBar(nbars=2)             #initialise the double progress bar
setup()

#raise ValueError( 'STOPPING' )

run = sciproc( args.cubes )



def goodbye():
    os.chdir(RUN_DIR)
    print('Adios!')

import atexit
atexit.register( goodbye )

#main()

#NOTES FOR PHOTOMETRY

#1. ANIMATE
#2. SHIFTS?
#3. FIND STARS ---> PLOT SKY BOX ETC.
#4. MEASURE SKY SIGMA ---> ACROSS CUBE
#5. MEASURE PSF, elipticity, etc		--> ACROSS CUBE
#6. FIND STARS
#7. CHECK SATURATION
#8. PHOTOMETRY --> PLOTS APERTURES, SKYBOX ETC.


    
