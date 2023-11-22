#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# std
import logging
import itertools as itt
from pathlib import Path

# third-party
from matplotlib import rc
from pyshoc import shocCampaign
from IPython.display import HTML, display
from pySHOC import MATCH_DARKS, MATCH_FLATS

# local
from scrawl.image import ImageDisplay
from obstools.stats import median_scaled_median


# setup logging
rootlog = logging.getLogger()
rootlog.setLevel(logging.INFO)

# disable line wrapping (so tables display nicely)
display(HTML("<style>div.output_area pre {white-space: pre;}</style>"))

# set the default colourmap
rc('image', cmap='cmr.dusk')


# In[4]:


# Load data
root_folder = Path('/media/Oceanus/UCT/Observing/data/sources/J1928-5001')
fig_folder = Path('/home/hannes/Documents/papers/dev/J1928/figures')
rc('savefig', directory=fig_folder)

run = shocCampaign.load(root_folder / 'SHOC/raw')
print(run)
run.pprint()


# In[ ]:


# Some of the frames are labelled incorrectly.
# OBSTYPE discovery + grouping
g = run.guess_obstype()
tables = g.pprint();  
# notice `shocHDU.guess_obstype` identifies frames correctly from the 
# distribution of pixel values, despite incorrect header info!


# In[ ]:


# Apply the guessed observation type labels (flat / dark / bad)
# Note this updates the attributes of the shocHDU's, but leaves the fits 
# headers untouched
for obstype, r in g.items():
    # set attributes on HDU objects - not yet in header
    r.attrs.set(obstype=itt.repeat(obstype))

# remove bad files
g.pop('bad', None)

# set the target name
run_src = g['object']
run_src.attrs.set(target=itt.repeat('CTCV J1928-5001'))

# add telescope info for old data.  We will need this later
for obstype in ['object', 'flat']:
    run = g[obstype]
    is74in = np.equal(run.attrs('telescope'), None)
    run[is74in].attrs.set(telescope=itt.repeat('74in'))

# print target observations
run_src.sort_by('date').pprint();   # todo: list by date; export to latex


# In[9]:


# Science images thumbnail grid 
fig, axes, _ = run_src.sort_by('date').thumbnails()
fig.set_size_inches(9, 7)
fig


# In[ ]:


# Match calibration frames

# DEBIAS
# need to debias flats & science observations
obs = g['flat'].join(g['object']) 
gobs, gdark = obs.match(g['dark'], *MATCH_DARKS, keep_nulls=False, report=True)

# compute master darks
mbias = gdark.merge_combine(np.median)

# Notice that i have previously placed all the correct dark and flat 
# observation files in the one folder. This is actually not necessary. If 
# you already have a folder where you store all your calibration data, you can
# read all those files and select the correct ones with a few lines of code 
# similar to what was done above. 
# For example:
# darks = shocCampaign.load_dir('/media/Oceanus/UCT/Observing/data/darks')
#_ = run_src.match(darks, *MATCH_DARKS, keep_nulls=False, report=True)


# In[17]:


# display
fig, *_ = mbias.to_list().sort_by('date').thumbnails(title=('date', 'readout'))
fig


# In[ ]:


# Science files are large and won't all fit into RAM, so needs careful handling.
# use `set_calibrators` to do calibration arithmetic on the fly when accessing 
# data via `calibrated` attribute
gobj = gobs.select_by(obstype='object')
gobj.set_calibrators(mbias)

# Flat fields are small enough volume that we can safely read them into RAM.
gflat = gobs.select_by(obstype='flat')
gflat = gflat.subtract(mbias)

# Match calibrated flat fields to science observations. This grouping will be 
# different to that of `gflat` above since we are now matching for closest 
# dates
gobj, gflat = g['object'].match(gflat.to_list(), *MATCH_FLATS, keep_nulls=False,
                                report=True)

# flat field: median scale each image, then median combine images
mflat = gflat.merge_combine(median_scaled_median)
# 
gobj.set_calibrators(flats=mflat)


# In[34]:


# display master flats
fig, *_ = mflat.to_list().sort_by('date').thumbnails(title='date')
fig


# In[35]:


# show image grid (calibrated images)

# gobj.set_calibrators(mbias) # mflat
orun =  gobj.to_list().sort_by('date')
fig, *_ = orun.thumbnails(calibrated=True)
fig


# In[ ]:


# The flats for 20150606.0300.fits seem to introduce artifacts rather than 
# remove them.  This is most likely due to small number statistics since 
# there are only 30 frames in the flat field observation..
orun =  gobj.to_list().sort_by('date')
orun[9].calibrated.flat = None

# Also don't have calibration images for 1.0m data, which can't be helped


# In[ ]:


# Image registration & mosaic

# align images
reg = orun.coalign(plot=False) 

# plot
mp = reg.mosaic(alpha=0.35, cmap='cmr.dusk', number_sources=True)
mp.fig


# In[43]:


def pixel_transform(i: int):
    # scale images by source counts for source 0 and median subtract
    image = reg[i]
    a = image.data / image.counts[sidx[i] == 0]
    return a - np.ma.median(a)

# get source indices per frame
sidx = reg.source_indices
# get mean image across stack
g, bs = reg.binned_statistic(image_func=pixel_transform, interpolate=True)

im = ImageDisplay(bs.statistic.T)
im.image.set_clim(0, 0.012)
im.figure.set_size_inches(8, 8)
im.figure

