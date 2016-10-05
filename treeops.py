#TODO: auto sort into source directories based on header info + make as dir struct

from pathlib import Path

def with_ext(root, extention):
    '''return all files in tree with given extension as list of Paths'''
    ext = extention.strip('.')
    return list(Path(root).rglob('*.{}'.format(ext)))

def get_fits(root):
    '''return all fits files in tree as Path object'''
    return with_ext(root, '.fits')

#from recipes.list import flatten

from pySHOC.core import SHOC_Run
from recipes.array import unique_rows

def unique_modes(root):
    
    fitsfiles = get_fits(root)
    run = SHOC_Run(filenames=fitsfiles)
    names, dattrs, vals = zip(*(stack.get_instrumental_setup(('binning', 'mode', 'gain'))
                                for stack in run))
    
    #Convert to strings so we can compare
    vals = np.array([list(map(str, v)) for v in vals])
    
    return unique_rows(vals)    #unique_rows(np.array(vals, dtype='U50'))


from obstools.fastfits import FITSFrame

def get_first_frames(root):
    data = {}
    for fits in get_fits(root):
        ff = FITSFrame(fits)
        data[fits] = ff[0]
    
    return data


import numpy as np
import matplotlib.pyplot as plt
def get_first_frames_png(root, clobber=False, verbose=True):
    
    for fpath, data in get_first_frames(root).items():
        #get first frame data
        ff = FITSFrame(fpath)
        data = ff[0]
        vmin, vmax = np.percentile(data, (2.25, 99.75))
        
        #
        fig = plt.figure(figsize=(8,8), frameon=False)
        ax = fig.add_axes([0,0,1,1], frameon=False)
        ax.imshow(data, origin='llc', 
                  cmap='gist_earth', vmin=vmin, vmax=vmax)
        
        #add filename to image
        fig.text(0.01, 0.99, fpath.name, 
                 color='w', va='top', size=12, fontweight='bold')
        
        fpng = fpath.with_suffix('.png')
        if not fpng.exists():
            if verbose:
                print('saving', str(fpng))
            fig.savefig(str(fpng))
        else:
            if verbose:
                print('not saving', str(fpng))
        #from pySHOC.treeops import get_first_frames_png
        
        
#def flagger