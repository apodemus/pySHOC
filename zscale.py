import numpy as np
from scipy.optimize import leastsq

class Zscale(object):
    def __init__(self, **kw):
        #from copy import copy
        #data = copy(data) #create local copy of image data
        self.count = 0
        self.rejects = 0
        self.sigma_clip = kw['sigma_clip']      if 'sigma_clip' in kw   else 3.5
        self.maxiter = kw['maxiter']             if 'maxiter' in kw      else 10
        self.num_points = kw['num_points']      if 'num_points' in kw   else 1000

        mask = kw['mask'] if 'mask' in kw else None
        if not mask is None:
            if not mask.dtype is np.dtype('bool'):
                mask = mask.astype('bool')                #cast mask array as boolean
        self.mask = mask


    def apply_mask(self, data):
        '''Apply bad pixel mask if given. Return flattened array'''
        if self.mask is None:
            return data.ravel()
        else:
            assert(data.shape==self.mask.shape)
            return data[self.mask]

    def resample(self, data):
        '''Resample data without replacement.'''
        num_points = self.num_points
        if num_points > data.size:
            num_points = data.size//2                    #use at most half of the data points in the image
        elif num_points < 100:
            num_points = 100                             #use at least 100 of the data points in the image
        data = np.random.choice(data, size=num_points, replace=False)
        #return data sorted in ascending order
        return np.sort( data, axis=None )

    def range(self, data=None, **kw):
        '''Algorithm to determine colour limits for astronomical images based on zscale algorithm used by IRAF display task.'''
        if data is None: 
            data = self.data
        if self.count==0:
            data = self.apply_mask(data)
            data = self.resample(data)
            self.data_range = (data[0], data[-1])
            self.data_len = len(data)

        Il = len(data)
        Ir = np.arange(Il)
        Im = np.median(data)

        residuals = lambda param, data: \
                    np.abs( data - (param[0]*Ir + param[1]) )
        m0 = np.ptp(data)/Il
        fit, _ = leastsq(residuals, (m0,0), data)

        #Assume I is normally distributed and clip values outside acceptable confidence interval
        res = residuals(fit, data)
        clipped = res > res.std()*self.sigma_clip

        if np.any(clipped) and self.count<self.maxiter:
            self.count += 1
            self.rejects += sum(clipped)
            #print('rejects: ', self.rejects )
            if self.rejects > self.data_len//2:                 #if more than half the original datapoints are rejected return original data range
                return self.data_range
            else:
                return self.range(data[~clipped], **kw)
        else:
            contrast = kw['contrast'] if 'contrast' in kw else 1.
            midpoint = Il/2
            slope = fit[0]
            z1 = Im + (slope/contrast)*(1.-midpoint)
            z2 = Im + (slope/contrast)*(Il-midpoint)
            self.z1 = self.data_range[0] if z1 < self.data_range[0] else z1
            self.z2 = self.data_range[-1] if z2 > self.data_range[-1] else z2
            self.count = 0
            self.rejects = 0
            return self.z1, self.z2