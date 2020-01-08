if __name__ == '__main__':
    fitsfile = '/media/Oceanus/UCT/Observing/data/Feb_2017/J0614-2725/SHA_20170209.0006.bff.fits'

    w = wcs.WCS(naxis=2)

    # see: https://www.aanda.org/articles/aa/full/2002/45/aah3859/aah3859.html
    # for definitions
    # array location of the reference point in pixels
    w.wcs.crpix = [-234.75,                   8.3393]
    # coordinate increment at reference point
    w.wcs.cdelt = [-0.066667,                   0.066667]
    # coordinate value at reference point
    w.wcs.crval = [0, -90]
    # axis type
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    # rotation from stated coordinate type.
    w.wcs.set_pv([(2, 1, 45.0)])