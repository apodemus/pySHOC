"""
Helper functions to infer the WCS given a target name or coordinates of a object in the field
"""

import functools
import logging
import multiprocessing as mp
import itertools as itt

import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from astropy.coordinates import SkyCoord

from recipes.dict import AttrDict
from obstools.phot.find import sourceFinder
from decor.profiler.timers import timer

from .utils import retrieve_coords, get_dss

import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D


# WARNING: the optimization methods here are somewhat add hoc, and are not robust
# TODO: plug in an MCMC sampler.  then you can discover flip states and measure FoV!!!!!

def rotation_matrix_2D(theta):
    '''Rotation matrix'''
    cos = np.cos(theta)
    sin = np.sin(theta)
    return np.array([[cos, -sin],
                     [sin, cos]])


def transform(X, p):
    """rotate and translate"""
    rotm = rotation_matrix_2D(p[-1])
    Xnew = (rotm @ X.T).T + p[:2]
    return Xnew


def transform_yx(X, p):
    """rotate and translate"""
    rotm = rotation_matrix_2D(p[-1])
    Xnew = (rotm @ X[:, ::-1].T).T + p[1::-1]
    return Xnew[:, ::-1]


def objective1(cooref, coords, p, thresh=0.1):
    """Objective function that highlights small distances for crude grid search"""
    coonew = transform_yx(coords, p)
    ifd = cdist(cooref, coonew)
    # return np.percentile(ifd, thresh)
    return np.sum(ifd < thresh)


# def objective2(p, cooref, coords):
#     coonew = transform(coords, p)
#     return np.square(cooref - coonew).sum()

@timer
def gridsearch_mp(objective, args, grid):
    # grid search

    # r = np.empty(yres, xres)
    f = functools.partial(objective, *args)
    ndim, *rshape = grid.shape
    with mp.Pool() as pool:
        r = pool.map(f, grid.reshape(ndim, -1).T)
    pool.join()
    r = np.reshape(r, rshape)
    i, j = divmod(r.argmax(), rshape[0])
    return r, (i, j), grid[:, i, j]


def match_constellation(cooref, coo, thresh=0.2):
    # associate stars across frames by checking distances
    # this will probably only work if the frames are fairly well aligned
    dr = cdist(cooref, coo)

    # if thresh is None:
    #     thresh = np.percentile(dr, dr.size / len(coo))

    dr[(dr > thresh) | (dr == 0)] = np.nan

    ur, uc = [], []
    for i, d in enumerate(dr):
        dr[i, uc] = np.nan
        if ~np.isnan(d).all():
            # closest star not yet matched with another
            jmin = np.nanargmin(d)
            ur.append(i)
            uc.append(jmin)

    return ur, uc


def match_cube(self, filename, object_name=None, coords=None):
    from .core import shocObs

    cube = shocObs.load(filename, mode='update')  # ff = FitsCube(fitsfile)
    if coords is None:
        coords = cube.get_coords()
    if (coords is None):
        if object_name is None:
            raise ValueError('Need object name or coordinates')
        coords = retrieve_coords(object_name)

    image = np.fliplr(cube.data[:5].mean(0))
    fov = cube.get_FoV()


def plot_transformed_image(ax, image, fov, p=(0, 0, 0), frame=False, **kws):
    """"""
    # image = image / image.max()
    extent = np.c_[[0., 0.], fov]
    pixscale = np.divide(fov, image.shape)
    #extent -= 0.5 * pixscale[None].T  # adjust to pixel centers...

    # set colour limits
    vmin, vmax = np.percentile(image, (0.25, 99.75))
    kws.setdefault('vmin', vmin)
    kws.setdefault('vmax', vmax)

    # plot
    pl = ax.imshow(image, extent=extent.ravel(), **kws)
    # Rotate the image by setting the transform
    xy, theta = p[1::-1], p[-1]
    tr = pl.get_transform()
    tt = Affine2D().rotate(theta).translate(*xy)
    pl.set_transform(tt + tr)

    if bool(frame):
        from matplotlib.patches import Rectangle
        frame_kws = dict(fc='none', lw=1.5, ec='0.5')
        if isinstance(frame, dict):
            frame_kws.update(frame)
        r = Rectangle(xy - 0.5 * pixscale, *fov, np.degrees(theta),
                      **frame_kws)
        ax.add_patch(r)

    return pl


def plot_coords_nrs(cooref, coords):
    fig, ax = plt.subplots()

    for i, yx in enumerate(cooref):
        ax.plot(*yx[::-1], marker='$%i$' % i, color='r')

    for i, yx in enumerate(coords):
        ax.plot(*yx[::-1], marker='$%i$' % i, color='g')


class MatchImages():
    gridStep = 3 / 60  # arcmin
    searchFrac = 0.7

    def __init__(self, image, fov, **findkws):
        self._findkws = dict(snr=3.,
                             npixels=7,
                             edge_cutoff=3,
                             deblend=False,
                             flux_sort=False)  # defaults
        self._findkws.update(findkws)
        # NOTE: deblending is *off* since it may work in the high res image,
        # but not in the lower- which will cause errors below

        # data array
        self.data = np.asarray(image)

        # Detect stars in dss frame
        # snr, npix, edgecut, deblend, flux_sort
        coor, flxr, self.segm = sourceFinder(self.data, **self._findkws)
        self.coords = (coor / self.data.shape) * fov
        self.fov = np.array(fov)
        self.pixscale = self.fov / self.data.shape  # arcmin per pixel

        # self.match(image, fov)

        ashape = np.array(self.data.shape)
        self.grid = np.mgrid[tuple(map(slice, (0, 0), self.fov, 1j * ashape))]
        self.ogrid = np.array([self.grid[0, :, 0], self.grid[1, 0, :]])

    @timer
    def match_image(self, image, fov):
        # Detect stars in image
        coo, flxr, segmr = sourceFinder(image, **self._findkws)
        coo = (coo / image.shape) * fov

        # do grid search
        o = np.ones((2, 2))
        o[:, 1] = -1

        (ys, ye), (xs, xe) = self.fov * self.searchFrac * o
        xres = int((xs - xe) / self.gridStep)
        yres = int((ys - ye) / self.gridStep)
        grid = np.mgrid[ys:ye:complex(yres),
               xs:xe:complex(xres)]
        # add 0s for angle grid
        z = np.zeros(grid.shape[1:])[None]
        grid = np.r_[grid, z]
        logging.info("Doing search on (%.1f' x %.1f') (%d x %d pix) sky grid",
                     *fov, yres, xres)
        r, ix, pGs = gridsearch_mp(objective1, (self.coords, coo), grid)
        logging.debug('Grid search optimum: %s', pGs)

        # match patterns
        cooGs = transform_yx(coo, pGs)
        ir, ic = match_constellation(self.coords, cooGs)
        logging.info('Matched %d stars across images.', len(ir))

        # final alignment
        # pick a star to re-center coordinates on
        cooDSSsub = self.coords[ir]
        cooGsub = cooGs[ic]
        distDSS = cdist(cooDSSsub, cooDSSsub)
        ix = distDSS.sum(0).argmax()
        # translate to origin at star `ix`
        yx = y, x = (cooGsub - cooGsub[ix]).T
        vu = v, u = (cooDSSsub - cooDSSsub[ix]).T  # transform destination
        # calculate rotation angle
        thetas = np.arctan2(v * x - u * y, u * x + v * y)
        theta = np.median(np.delete(thetas, ix))
        # calc final offset in DSS coordinates
        rotm = rotation_matrix_2D(-theta)
        yxoff = yo, xo = np.median(cooDSSsub - (rotm @ coo[ic].T).T, 0)

        p = (yo, xo, theta)
        return p


class MatchDSS(MatchImages):
    _servers = ('poss2ukstu_blue', 'poss1_blue',
                'poss2ukstu_red', 'poss1_red',
                'poss2ukstu_ir',
                'all')

    def __init__(self, name_or_coords, fov=(3, 3), **findkws):
        """

        Parameters
        ----------
        name_or_coords: str
            name of object or coordinate string
        fov
            field of view in arcmin
        """

        if isinstance(name_or_coords, SkyCoord):
            coords = name_or_coords
        else:
            try:
                # first try interpret coords. eg. MatchDSS('06:14:51.7 -27:25:35.5', (3,3))
                coords = SkyCoord(name_or_coords, unit=('h', 'deg'))
            except:
                coords = retrieve_coords(name_or_coords)
                pass

        if (coords is None):
            raise ValueError('Need object name or coordinates')

        for serv in self._servers:
            self.hdu = get_dss(serv, coords.ra.deg, coords.dec.deg, fov)
            break

        # DSS data array
        data = self.hdu[0].data.astype(float)
        MatchImages.__init__(self, data, fov, **findkws)

        # save target coordinate position
        self.targetCoords = coords
        self.targetCoordsPix = np.divide(self.data.shape, 2) + 0.5



    def build_wcs(self, cube, p, telescope=None):
        """
        Create tangential plane wcs
        Parameters
        ----------
        p
        cube
        telescope

        Returns
        -------
        astropy.wcs.WCS instance

        """

        from astropy import wcs

        *yxoff, theta = p
        fov = cube.get_FoV(telescope)
        pxscl = fov / cube.ishape
        # transform target coordinates in DSS image to target in SHOC image
        h = self.hdu[0].header
        crpix = np.array([h['crpix1'], h['crpix2']])  # target object pixel coordinates
        crpixDSS = crpix - 0.5  # convert to pixel llc coordinates
        cram = crpixDSS / self.pixscale  # convert to arcmin
        rotm = rotation_matrix_2D(-theta)
        crpixSHOC = (rotm @ (cram - yxoff)) / pxscl
        # target coordinates in degrees
        xtrg = self.targetCoords
        # coordinate increment
        cdelt = pxscl / 60  # in degrees
        flip = np.array(cube.flip_state[::-1], bool)
        cdelt[flip] = -cdelt[flip]

        # see: https://www.aanda.org/articles/aa/full/2002/45/aah3859/aah3859.html for parameter definitions
        w = wcs.WCS(naxis=2)
        w.wcs.crpix = crpixSHOC  # array location of the reference point in pixels
        w.wcs.cdelt = cdelt  # coordinate increment at reference point
        w.wcs.crval = xtrg.ra.value, xtrg.dec.value  # coordinate value at reference point
        w.wcs.crota = np.degrees([-theta, -theta])  # rotation from stated coordinate type.
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]  # axis type

        return w

    def plot_coords_nrs(self, coords):
        fig, ax = plt.subplots()

        for i, yx in enumerate(self.coords):
            ax.plot(*yx[::-1], marker='$%i$' % i, color='r')

        for i, yx in enumerate(coords):
            ax.plot(*yx[::-1], marker='$%i$' % i, color='g')


class MosaicPlotter():
    def __init__(self, dss, use_aplpy=True):

        self.dss = dss
        self.use_aplpy = use_aplpy
        self.plots = AttrDict()
        self._counter = itt.count()
        self._fig = None
        self._ax = None
        self._lowlims = (0, 0)
        self._uplims = (0, 0)
        self._istate = 0

    def setup(self):
        if self.use_aplpy:
            import aplpy as apl
            self._ff = f = apl.FITSFigure(self.dss.hdu)
            self._ax = f._ax1
            self._fig = f._ax1.figure
            # f.add_grid()

        else:
            self._fig, self._ax = plt.subplots()

        self._fig.tight_layout()

    @property
    def fig(self):
        if self._fig is None:
            self.setup()
        return self._fig

    @property
    def ax(self):
        if self._ax is None:
            self.setup()
        return self._ax

    def _world2pix(self, p, fov):
        # convert fov to the DSS pixel coordinates (aplpy)
        if self.use_aplpy:
            scale_ratio = (fov / self.dss.fov)
            dsspix = scale_ratio * self.dss.data.shape
            y, x = p[:2] / self.dss.pixscale + 0.5
            return (y, x, p[-1]), dsspix
        return p, fov

    def plot_image(self, image=None, fov=None, p=(0, 0, 0), name=None, frame=False,
                   **kws):
        """"""
        if image is None:
            image = self.dss.data
            fov = self.dss.fov
            name = 'dss'

        if name is None:
            name = 'image%i' % next(self._counter)

        # convert fov to the DSS pixel coordinates (aplpy)
        p, fov = self._world2pix(p, fov)

        # plot
        # image = image / image.max()
        self.plots[name] = pl = \
            plot_transformed_image(self.ax, image, fov, p, frame, **kws)

        self.update_lims(p, fov)

    def get_corners(self, p, fov):
        """Get corners relative to DSS coordinates"""
        c = np.array([[0, 0], fov])  # lower left, upper right
        corners = np.c_[c[0], c[:, 0], c[1], c[::-1, 1]].T  # all 4 clockwise
        corners = transform_yx(corners[:, ::-1], p)
        return corners[:, ::-1]

    def update_lims(self, p, fov):

        corners = self.get_corners(p, fov)
        # if np.isnan(corners).any():
        #     print('SHIT!!!')
        #     return
        mins, maxs = np.c_[corners.min(0), corners.max(0)].T

        self._lowlims = np.min([mins, self._lowlims], 0)
        self._uplims = np.max([maxs, self._uplims], 0)

        ylim, xlim = np.c_[self._lowlims, self._uplims] + 1
        self.ax.set(xlim=xlim, ylim=ylim)

        # mins = np.min([mins, (0, 0)], 0)
        # maxs = np.max([maxs, self.fov], 0)
        # ylim, xlim = np.c_[mins, maxs]
        # return xlim , ylim

    def mosaic(self, images, fovs, ps, **kws):
        # mosaic plot

        show_dss = kws.pop('show_dss', True)
        alpha_magic = min(1. / (len(images) + show_dss), 0.5)
        alpha = kws.setdefault('alpha', alpha_magic)
        if show_dss:
            self.plot_image(**kws)

        for image, fov, p in zip(images, fovs, ps):
            self.plot_image(image, fov, p, **kws)

        n = len(self.plots)
        self.states = np.vstack([np.eye(n), np.ones(n) * alpha])
        self.fig.canvas.mpl_connect('scroll_event', self._scroll)

    def _scroll(self, event):
        try:
            # print(vars(event))
            if event.inaxes:

                self._istate += [-1, +1][event.button == 'up']
                self._istate %= len(self.plots) + 1  # wrap
                # print('state', self._istate)
                alphas = self.states[self._istate]
                for i, pl in enumerate(self.plots.values()):
                    pl.set_alpha(alphas[i])

                self.fig.canvas.draw()

        except Exception as err:
            print('Scroll failed:', str(err))


# def mosaic_aplpy(dss, images, fovs, ps, **kws):
#     import aplpy as apl
#
#     f = apl.FITSFigure(dss.hdu)
#     if kws.pop('show_dss', True):
#         f.show_grayscale(invert=True)
#
#     # coordinate grid
#     f.add_grid()
#     f.grid.set_color('0.5')
#
#     ax = f._ax1
#
#     for image, fov, p in zip(images, fovs, ps):
#         # get the size of the image in DSS pixels
#         scale_ratio = (fov / dss.fov)
#         dsspix = scale_ratio * dss.data.shape
#         y, x = p[:2] / dss.pixscale + 0.5
#         # plot image
#         pl = plot_image(ax, image, dsspix, (y, x, p[-1]), **kws)
#
#     return f




# infer offset and rotation of grids


# gridr = ndgrid.like(imr)
# rotm = rotation_matrix_2D(np.radians(p[-1]))
# gridn = gridr.swapaxes(0, -1) @ rotm + p[:2]

if __name__ == '__main__':
    fitsfile = '/media/Oceanus/UCT/Observing/data/Feb_2017/J0614-2725/SHA_20170209.0006.bff.fits'

    w = wcs.WCS(naxis=2)

    # see: https://www.aanda.org/articles/aa/full/2002/45/aah3859/aah3859.html for definitions
    w.wcs.crpix = [-234.75, 8.3393]  # array location of the reference point in pixels
    w.wcs.cdelt = [-0.066667, 0.066667]  # coordinate increment at reference point
    w.wcs.crval = [0, -90]  # coordinate value at reference point
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]  # axis type
    w.wcs.set_pv([(2, 1, 45.0)])  # rotation from stated coordinate type.


