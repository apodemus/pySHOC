
# std
from copy import copy

# third-party
from astropy.io import fits


def str2tup(keys):
    if isinstance(keys, str):
        keys = (keys, )
    return keys


def combine_single_images(ims, func):   
    """Combine a run consisting of single images."""
    
     # TODO MERGE WITH shocObs.combine????
    
    header = copy(ims[0][0].header)
    data = func([im[0].data for im in ims], 0)

    header.remove('NUMKIN')
    header['NCOMBINE'] = (len(ims), 'Number of images combined')
    for i, im in enumerate(ims):
        imnr = '{1:0>{0}}'.format(3, i + 1)  # Image number eg.: 001
        comment = 'Contributors to combined output image' if i == 0 else ''
        header[f'ICMB{imnr}'] = (im.get_filename(), comment)

    # uses the FilenameGenerator of the first image in the shocRun
    # outname = next( ims[0].filename_gen() )

    return fits.PrimaryHDU(data, header)  # outname
