
class Input(object):
    MAX_LS = 10
    #retrieve a string input from user
    #INPUT    : query - string query passed to raw_input
    #         : dflt - the default input option
    #OPTIONAL: verify - Boolean for whether to ask for verification of user input
    #        : validity_test - a function that returns True if the input is valid or False otherwise
    #        : example -
    #====================================================================================================
    @staticmethod
    def str(query, default=None, **kw):
        validity_test =  kw['validity_test']        if 'validity_test' in kw        else lambda x: True
        verify = kw['verify']                       if 'verify' in kw               else 0
        what = kw['what']                           if 'what' in kw                 else 'output string'
        example = kw['example']                     if 'example' in kw              else 0
        convert = kw['convert']                     if 'convert' in kw              else lambda x: x
        
        default_str = '' if default in ['', None] else ' or ENTER for the default %s.\n' %repr(default)
        repeat = 1
        while repeat:
            instr = input(query + default_str)
            if not instr:
                if default:
                    instr = default
                    repeat = 0
                else:
                    repeat = 0
                    pass                            #returns empty instr = ''
            else:
                if verify:
                    while not instr:
                        print("You have entered '%s' as the %s" %(instr, what))
                        #print "Example output filename:\t %s_0001.fits" %instr
                        new_instr = input('ENTER to accept or re-input.\n')
                        if new_instr == '':
                            break
                        else:
                            instr = new_instr
                if validity_test(instr):
                    repeat = 0
                else:
                    print('Invalid input!! Please try again.\n')
                    repeat = 1
        
        return convert( instr )
    
    #====================================================================================================
    @staticmethod
    def read_data_from_file(fn, N=None, pr=False):
        '''N - number of lines to read'''
        MAX_LS = Input.MAX_LS
        with open(fn,'r') as fp:
            if N:
                from itertools import islice
                fp = islice(fp, N)
            lsfromfile = [s for s in fp if s]                       #filters out empty lines 
        
        if pr:
            nbot = 3
            ls_trunc = lsfromfile[:MAX_LS-nbot] if len(lsfromfile) > MAX_LS else lsfromfile
            print('You have input the txt file {} containing the following:\n{}'.format(repr(fn), ''.join(ls_trunc)))                  #USE THE INPUT NAME HERE AS BASIS NAME FOR BIAS SUBTRACTED OUTPUT TXT LIST ETC...
            if len(lsfromfile) > MAX_LS:
                print( '.\n'*3 )
            if len(lsfromfile) > MAX_LS+nbot:
                print( ''.join(lsfromfile[-nbot:]) )
        
        lsfromfile = [q.strip('\n') for q in lsfromfile]
        return lsfromfile
    
    #====================================================================================================
    @staticmethod
    def list(query, validity_test, default=None, empty_allowed=0):
        ''' Retrieve a list of user input values and test validity of input according to input function validity_test
            INPUTS  : query - string query passed to input
                    : validity_test - a function that returns True if the input is valid or False otherwise
                    : default - the default option
                    : emty_allowed - whether or not an empty string can be returned by the function.'''
        if isinstance(default, list):
            default = ' ,'.join(default)
        repeat = True
        while repeat:
            if default in ['', None]:
                default_str = ''
            else:
                default_str = ' or ENTER for the default %s.\n' %repr(default)
        
            instr = input(query + default_str)                              #cmd
        
            if not instr:
                if default not in ['', None]:
                    instr = default                 #this means the default option needs to be a string
                elif empty_allowed:
                    return instr                    #returns an empty string
                else:
                    pass
                
            for delimiter in [' ', ',', ';']:
                inls = instr.split(delimiter)
                if len(inls) > 1:
                    inls = [s.strip(' ,;') for s in inls]
                    break
            
            #if inls[0]=='*all':                                                                            #NOT YET TESTED..................
                #inls = dirnames
                #break
            #print('VT')
            #print( validity_test )
            inls = ValidityTests.test_ls(inls, validity_test)
            repeat = not bool( inls )
        inls = [s.strip() for s in inls]
        return inls

#################################################################################################################################################################################################################
#Conversion functions for user input
#################################################################################################################################################################################################################
class Conversion(object):
    #====================================================================================================
    @staticmethod
    def trivial(instr):
        return instr
    
    #====================================================================================================
    @staticmethod
    def yn2TF(instr):
        return instr in ('y', 'Y')
    
    #====================================================================================================
    @staticmethod
    def RA_DEC(instr):
        if isinstance(instr, str):
            instr = instr.strip()
            if ':' in instr:
                return instr
            elif ' ' in instr:
                return Conversion.ws2cs(instr)
            else:
                return float(instr)
        
        elif isinstance(instr, float):
            return instr    
    
    #====================================================================================================
    @staticmethod
    def ws2cs(instr):
        '''convert from whitespace separated to colon separated triplets.'''
        h, m, s = instr.split()
        return '{:d}:{:02d}:{:02f}'.format( int(h), int(m), float(s) )
         
    
    #====================================================================================================
    @staticmethod
    def ws2ds(instr):
        '''convert from whitespace separated to dash separated triplets.'''
        if '-' in instr:
            return instr
        try:
            h, m, s = instr.split()
            return '{:d}-{:02d}-{:02f}'.format( int(h), int(m), float(s) )
        except ValueError:
            return float(instr)
    
#################################################################################################################################################################################################################
#Validity tests for user input
#################################################################################################################################################################################################################
class ValidityTests(object):
    MAX_CHECK = 25
    #====================================================================================================
    @staticmethod
    def trivial(instr):                                 #CHECK FOR INVALID CHARACTERS???????????
        return True
    
    #====================================================================================================
    @staticmethod
    def yn(instr):
        return instr in ['y','n','Y','N']
    
    #====================================================================================================
    @staticmethod
    def float(num_str):
        try:
            float(num_str)
            return True
        except ValueError:
            return False
    
    #====================================================================================================
    @staticmethod
    def int(num_str):
        try:
            int(num_str)
            return True
        except ValueError:
            return False
    
    #validity test function for the naming convention
    #====================================================================================================
    @staticmethod
    def name_convention(num_str):
        
        
        return valid

    #Validity tests for number triplet input
    #====================================================================================================
    @staticmethod
    def triplet(triplet):
        HMS = triplet.split()
        if len(HMS)==1:
            HMS = triplet.split(':')
        if len(HMS)==1:
            HMS = triplet.split('-')
        if len(HMS)==1:
            HMS = triplet.split('\t')
        
        try:
            h,m,s = [float(hms) for hms in HMS]
            valid = True
        except:
            h,m,s = [None]*3
            valid = False
        
        return valid, h,m,s

    #====================================================================================================
    @staticmethod
    def RA(RA):
        '''Validity tests for RA'''
        try:
            RA = float(RA)
            if RA >= 0. and RA < 24.:
                return True
            else:
                return False
        except (ValueError, TypeError): 
            valid, h,m,s = ValidityTests.triplet(RA)
            if valid:
                if any([h<0., h>23., m<0., m>59., s<0., s>=60]):
                    valid = False
            return valid

    #====================================================================================================
    @staticmethod
    def DEC(DEC):
        '''Validity tests for DEC'''
        try:
            DEC = float(DEC)
            if DEC > -90. and DEC < 90.:
                return True
            else:
                return False
        except ValueError: 
            valid, d,m,s = ValidityTests.triplet(DEC)
            if valid:
                if any([d<-90., d>90., m<0., m>59., s<0., s>=60]):
                    valid = False
            return valid
    
    #====================================================================================================
    @staticmethod
    def epoch(epoch):
        try:
            return 1850 < float(epoch) < 2050
        except ValueError:
            return False
    
    #====================================================================================================
    @staticmethod
    def DATE(DATE):
        '''Validity tests for DATE'''
        try:
            float(DATE)
            return True
        except ValueError: 
            valid, y,m,d = ValidityTests.triplet(DATE)                                   #WARN IF DIFFERENT FROM DATE IN FILENAME / FILE CREATION DATE IN HEADER
            if valid:
                try:
                    from datetime import date
                    date(year=int(y),month=int(m),day=int(d))
                except ValueError:
                    valid = False
        return valid
    
    #====================================================================================================
    @staticmethod
    def test(instr, validity_test, raise_error=0):
        '''Tests a input str for validity by calling validity_test on it.
        Returns None if an error was found or raises ValueError if raise_error is set.
        Returns the original list if input is valid'''
        if not validity_test(instr):
            msg = 'Invalid input!! %s \nPlease try again: ' %instr                                              #REPITITION!!!!!!!!!!!!
            if raise_error==1:
                raise ValueError( msg )
            elif raise_error==0:
                print( msg )
                return
            elif raise_error==-1:
                return 
        else:
            return instr
    
    #====================================================================================================
    @staticmethod
    def test_ls(inls, validity_test=None, **kw):
        '''
        Test a list input values for validity by calling validity_test on each value in the list.
        Parameters
        ----------
        inls :          Input can be one of the following :  dir name (str)
                                                          :  file name (str) for file which contains a list of the input filenames to be checked
                                                          :  file glob expression eg: '*.fits'
        validity_test : A function to call as a validity test.  Defaults to lambda x: True
        
        Keywords
        ----------
        max_check :     maximum number of inputs to run validity_test on.
        readlines :     the number of lines to read from the file. Read all lines if unspecified
        read_ext  :     Only files with this extension will be read from directory. (Somewhat redundant with glob)
        raise_error:    Boolean:  Whether to raise error when file is invalid.
        
        Returns
        -------
        None            if an error was found or raises ValueError if raise_error is set.
        inls            the original list / names in txt list if input is valid
        '''
        
        validity_test   = lambda x: True        if validity_test is None        else validity_test
        raise_error     = kw['raise_error']     if 'raise_error' in kw          else 0
        max_check       = kw['max_check']       if 'max_check' in kw            else 25
        readlines       = kw['readlines']       if 'readlines' in kw            else None
        read_ext        = kw['read_ext']        if 'read_ext' in kw             else '.fits'
        abspath         = kw['abspath']         if 'abspath' in kw              else 1
        
        
        if isinstance(inls, str):    inls = [inls]
        
        if len(inls)==1:         #and not inls[0].endswith('.fits')
            import os
            
            wildcards = '*?[]'
            has_wildcards = any([w in inls[0] for w in wildcards])
            
            if os.path.isdir( inls[0] ):                                #if the input is a directory
                path, _, inls = next( os.walk( inls[0] ) )              #list all the files in the directory
                inls = [fn for fn in inls if not fn.startswith('.') and fn.endswith(read_ext)]             #full path with hidden files ignored and only files with given extension
            
            elif has_wildcards:                                         #if the input is a glob expression
                from glob import glob
                path, _ = os.path.split( inls[0] )
                inls = glob(inls[0])
                #print( inls )
            
            elif not inls[0].endswith('.fits'):                                               #if the input is a text list with filenames
                try:
                    path, _ = os.path.split( inls[0] )
                    inls = Input.read_data_from_file( inls[0], readlines )
     
                except IOError:                                                                                            #SPECIFY!!!!!!!!!!!!!
                    msg = 'Invalid txt list: %s' %inls[0]
                    if raise_error:
                        raise ValueError( msg )
                    else:
                        print( msg )
                        return
            else: #single filename given
                path, _ = os.path.split( inls[0] )
                
        if abspath:
            inls = [ os.path.join(path, s) for s in inls ]
        
        i = 0
        badnames = []
        while i < len(inls):
            nm = inls[i]
            if not ValidityTests.test(nm, validity_test, raise_error):
                badnames.append( nm )
            i += 1
            if i >= max_check:
                print( 'WARNING: The input list is too long. No further entries will be checked for validity.\n' )
                break
                
        if len(badnames):
            return
        else:
            inls.sort()
            return inls