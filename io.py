from myio import parsetolist, iocheck, warn

class Input(object):
    MAX_LS = 10
    #retrieve a string input from user
    #INPUT    : query - string query passed to raw_input
    #         : dflt - the default input option
    #OPTIONAL: verify - Boolean for whether to ask for verification of user input
    #        : check - a function that returns True if the input is valid or False otherwise
    #        : example -
    #====================================================================================================
    @staticmethod
    def str(query, default=None, **kw):
        check           =       kw.get( 'check', lambda x: True )
        verify          =       kw.get( 'verify', 0 )
        what            =       kw.get( 'what', 'output string' )
        example         =       kw.get( 'example', 0 )
        convert         =       kw.get( 'convert', lambda x: x )
        
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
                if check(instr):
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
    def list(query, check, default=None, empty_allowed=0):
        ''' Retrieve a list of user input values and test validity of input according to input function check
            INPUTS  : query - string query passed to input
                    : check - a function that returns True if the input is valid or False otherwise
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
            #print( check )
            inls = ValidityTests.test_ls(inls, check)
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
    def test(instr, check, raise_error=0):
        warn( 'FIX THIS LAME FUNCTION!!' )
        return iocheck(instr, check, raise_error)
    
    #====================================================================================================
    @staticmethod
    def test_ls(inls, check=None, **kw):
        warn( 'FIX THIS LAME FUNCTION!!' )
        return parselist( inls, check, **kw)