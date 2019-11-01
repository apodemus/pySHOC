from recipes.io import iocheck, warn

class InputCallbackLoop(object):
    # TODO: class __call__ method that implements the main callback loop
    #retrieve input from user
    #INPUT    : query - string query passed to raw_input
    #         : default - the default input option
    #OPTIONAL: verify - Boolean for whether to ask for verification of user input
    #        : check - a function that returns True if the input is valid or False otherwise
    #        : example -
    #==========================================================================
    @staticmethod
    def str(query, default=None, **kw):
        """
        Callback loop that expects text input from user

        :param query:
        :param default:
        :param kw:
        :return:
        """
        verify = kw.get('verify', False)
        check = kw.get('check', lambda x: True)
        what = kw.get('what', 'output string')
        example = kw.get('example', '')
        example = '(eg: {})'.format(example) if example else ''
        convert = kw.get('convert', lambda x: x)

        default_str = ('(or hit return for the default %r)' % default
                       if bool(default) else '')
        repeat = True
        while repeat:
            msg = ' '.join(filter(None, (query, example, default_str, ': ')))
            instr = input(msg)
            if not instr and default:
                instr = default
                repeat = False
            else:
                if verify:
                    while not instr:
                        print('You have entered %r as the %s' % (instr, what))
                        new_instr = input('ENTER to accept or re-input.\n')
                        if new_instr == '':
                            break
                        else:
                            instr = new_instr

                repeat = not check(instr)
                if repeat:
                    print('Invalid input!! Please try again.\n')

        return convert(instr)

    #===========================================================================
    @staticmethod
    def list(query, check, default=None, empty_allowed=False):
        """
        Retrieve a list of user input values and test validity of input according
        to input function check

        :param query:           string query passed to input
        :param check:           a function that returns boolean for input validity
        :param default:         the default option
        :param empty_allowed:   whether or not an empty string can be returned by the function.
        :return:
        """

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
                    instr = default
                    # this means the default option needs to be a string
                elif empty_allowed:
                    return instr                    #returns an empty string
                else:
                    pass

            for delimiter in [' ', ',', ';']:
                inls = instr.split(delimiter)
                if len(inls) > 1:
                    inls = [s.strip(' ,;') for s in inls]
                    break

            #if inls[0]=='*all':   #NOT YET TESTED..................
                #inls = dirnames
                #break
            #print('VT')
            #print( check )
            inls = ValidityTests.test_ls(inls, check)
            repeat = not bool( inls )
        inls = [s.strip() for s in inls]
        return inls

################################################################################
#Conversion functions for user input
################################################################################
class Conversion(object):
    #===========================================================================
    @staticmethod
    def trivial(instr):
        return instr

    #===========================================================================
    @staticmethod
    def yn2TF(instr):
        return instr in ('y', 'Y')

    #===========================================================================
    @staticmethod
    def RA(val):
        return Angle(val, 'h').to_string(sep=':', precision=2)

    #===========================================================================
    @staticmethod
    def DEC(val):
        return Angle(val, 'deg').to_string(sep=':', precision=2)

    #===========================================================================
    @staticmethod
    def ws2cs(instr):
        """convert from whitespace separated to colon separated triplets."""
        h, m, s = instr.split()
        return '{:d}:{:02d}:{:02f}'.format( int(h), int(m), float(s) )


    #===========================================================================
    @staticmethod
    def ws2ds(instr):
        """convert from whitespace separated to dash separated triplets."""
        if '-' in instr:
            return instr
        try:
            h, m, s = instr.split()
            return '{:d}-{:02d}-{:02f}'.format( int(h), int(m), float(s) )
        except ValueError:
            return float(instr)

################################################################################
#Validity tests for user input
################################################################################

from astropy.coordinates.angles import Angle

class ValidityTests(object):
    MAX_CHECK = 25
    #===========================================================================
    @staticmethod
    def trivial(instr):    #CHECK FOR INVALID CHARACTERS???????????
        return True

    #===========================================================================
    @staticmethod
    def yn(instr):
        return instr in ['y','n','Y','N']

    #===========================================================================
    @staticmethod
    def float(num_str):
        try:
            float(num_str)
            return True
        except ValueError:
            return False

    #===========================================================================
    @staticmethod
    def int(num_str):
        try:
            int(num_str)
            return True
        except ValueError:
            return False

    #validity test function for the naming convention
    #===========================================================================
    @staticmethod
    def name_convention(num_str):


        return valid

    #Validity tests for number triplet input
    #===========================================================================
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

    #===========================================================================
    @staticmethod
    def RA(RA):
        """Validity tests for RA"""
        try:
            Angle(RA, 'h')
            return True
        except ValueError:
            return False

    #===========================================================================
    @staticmethod
    def DEC(DEC):
        """Validity tests for DEC"""
        try:
            dec = Angle(DEC, 'deg')
            return abs(dec) < Angle(90, 'deg')
        except ValueError:
            return False

    #===========================================================================
    @staticmethod
    def epoch(epoch):
        try:
            return 1850 < float(epoch) < 2050
        except ValueError:
            return False

    #===========================================================================
    @staticmethod
    def DATE(DATE):
        """Validity tests for DATE"""
        try:
            float(DATE)
            return True
        except ValueError:
            valid, y,m,d = ValidityTests.triplet(DATE)
            #TODO: warn if different from date in filename / file creation date in header?
            if valid:
                try:
                    from datetime import date
                    date(year=int(y),month=int(m),day=int(d))
                except ValueError:
                    valid = False
        return valid

    #===========================================================================
    @staticmethod
    def test(instr, check, raise_error=0):
        warn( 'FIX THIS LAME FUNCTION!!' )
        return iocheck(instr, check, raise_error)

    #===========================================================================
    @staticmethod
    def test_ls(inls, check=None, **kw):
        warn( 'FIX THIS LAME FUNCTION!!' )
        return parselist( inls, check, **kw)