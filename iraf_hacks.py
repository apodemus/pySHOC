import os
import subprocess

import numpy as np

from myio import warn


class IrafHacks():
    """Mixin for stupid iraf shit"""

    def make_slices(self, suffix):
        for i, cube in enumerate(self):
            cube.make_slices(suffix, i)

    def make_obsparams_file(self, suffix):
        for i, cube in enumerate(self):
            cube.make_obsparams_file(suffix, i)

    # ===========================================================================
    # def make_slices(self, suffix, i):
    #     # HACK! BECAUSE IRAF SUX
    #     # generate file with cube name and slices in iraf slice syntax.  This is a clever way of
    #     # sidestepping splitting the cubes and having thousands of fits files to deal with, but it
    #     # remains an ungly, unncessary, and outdated way of doing things.  IMHO (BRAAK!!!)
    #     # NOTE: This also means that the airmass correction done by phot is done with the airmass of
    #     # the first frame only... One has to generate an obsparams file for mkapfile COG with the
    #     # airmasses and observation times etc...
    #
    #     source = self.get_filename(1, 0, (suffix, 'fits'))
    #     link_to_short_name_because_iraf_sux(source, i, 'fits')
    #
    #     unpacked = self.get_filename(1, 0, (suffix, 'slice'))
    #     naxis3 = self.shape[-1]
    #
    #     self.real_slices = make_iraf_slicemap(source, naxis3, unpacked)
    #
    #     linkname = args.dir + '/s{}{}'.format(i, '.fits')
    #     slicefile = args.dir + '/s{}{}'.format(i, '.slice')
    #     slices = make_iraf_slicemap(linkname, naxis3, slicefile)
    #     # link_to_short_name_because_iraf_sux(unpacked, i, 'slice')
    #     self.link_slices = np.array(slices)
    #
    #     # filename = self.get_filename(0, 0, (suffix,'fits'))
    #     # self.real_slices = make_iraf_slicemap( filename, naxis3 )
    #
    # # ===========================================================================
    # def make_obsparams_file(self, suffix, count):
    #     slices = np.fromiter(map(os.path.basename, self.real_slices), 'U23')
    #     texp = np.ones(self.shape[-1]) * self.texp
    #     Filt = np.empty(self.shape[-1], 'U2')
    #     Filt[:] = head_info.filter
    #     data = np.array([slices, Filt, texp, self.timedata.airmass, self.timedata.utstr], object).T
    #     fmt = ('%-23s', '%-2s', '%-9.6f', '%-9.6f', '%s')
    #     filename = self.get_filename(1, 0, (suffix, 'obspar'))
    #     np.savetxt(filename, data, fmt, delimiter='\t')
    #
    #     # HACK! BECAUSE IRAF SUX
    #     slices = np.fromiter(map(os.path.basename, self.link_slices), 'U23')
    #     data = np.array([slices, Filt, texp, self.timedata.airmass, self.timedata.utstr], object).T
    #     filename = args.dir + '/s{}.obspar'.format(count)
    #     np.savetxt(filename, data, fmt, delimiter='\t')

        # link_to_short_name_because_iraf_sux(filename, count, 'obspar')



def make_iraf_slicemap(filename, N, savename=None):
    if len(filename) + len('[*,*,]') + len(str(N)) > 23:
        warn('Filenames + slice is larger that 23 characters! ==>  IRAF will truncate theses '
             'filenames when writing to database.  YES, IRAF SUCKS!!!' )
    slicemap = lambda i: '%s[*,*,%i]'%(filename, i)
    slices =list( map(slicemap, range(1,N+1)) )
    if not savename is None:
        np.savetxt( str(savename), slices, fmt='%s' )
    return list(slices)


def link_to_short_name_because_iraf_sux(filename, count, ext):
    # HACK! BECAUSE IRAF SUX.
    # FIXME: Integrate
    linkname = args.dir + '/s{}.{}'.format(count, ext)
    print('LINKING:', 'ln -f', os.path.basename(filename), os.path.basename(linkname))
    subprocess.call(['ln', '-f', filename, linkname])