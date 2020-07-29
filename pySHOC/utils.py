import copy
from astropy.io import fits


def print_timing_table(run):
    """prints the various timing keys in the headers"""
    from motley.table import Table

    keys = ['TRIGGER', 'DATE', 'DATE-OBS', 'FRAME',
            'GPSSTART', 'GPS-INT', 'KCT', 'EXPOSURE']
    tbl = []
    for obs in run:
        tbl.append([type(obs).__name__] + [obs.header.get(key, '--')
                                           for key in keys])

    return Table(tbl, chead=['TYPE'] + keys)



def combine_single_images(ims, func):  # TODO MERGE WITH shocObs.combine????
    """Combine a run consisting of single images."""
    header = copy(ims[0][0].header)
    data = func([im[0].data for im in ims], 0)

    header.remove('NUMKIN')
    header['NCOMBINE'] = (len(ims), 'Number of images combined')
    for i, im in enumerate(ims):
        imnr = '{1:0>{0}}'.format(3, i + 1)  # Image number eg.: 001
        comment = 'Contributors to combined output image' if i == 0 else ''
        header['ICMB' + imnr] = (im.get_filename(), comment)

    # uses the FilenameGenerator of the first image in the shocRun
    # outname = next( ims[0].filename_gen() )

    return fits.PrimaryHDU(data, header)  # outname
