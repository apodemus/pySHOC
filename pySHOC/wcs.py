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

from recipes.containers.dict import AttrDict
from obstools.phot.segmentation import SegmentationHelper
# from obstools.phot.segmentation import sourceFinder
# from motley.profiling.timers import timer

from .utils import retrieve_coords, get_dss, STScIServerError

import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D

from recipes.logging import LoggingMixin
from recipes.transformations.rotation import rotation_matrix_2d, rotate_2d


# WARNING: optimization methods here are somewhat add hoc. ie. not robust
# TODO: plug in an MCMC sampler.  then you can discover flip states and measure
#  FoV!!!!!


# TODO: might be faster to anneal sigma param rather than blind grid search.
#  test this.


def transform(X, p):  # todo rename roto_translation
    """rotate and translate"""
    rotm = rotation_matrix_2d(p[-1])
    Xnew = (rotm @ X.T).T + p[:2]
    return Xnew


def transform_yx(X, p):
    """rotate and translate"""
    rotm = rotation_matrix_2d(p[-1])
    Xnew = (rotm @ X[:, ::-1].T).T + p[1::-1]
    return Xnew[:, ::-1]


def objective0(xy_trg, xy, p, thresh=0.1):
    """
    Objective function that highlights small distances for crude grid search
    """
    xy_new = transform(xy, p)
    ifd = cdist(xy_trg, xy_new)
    # return np.percentile(ifd, thresh)
    return np.sum(ifd < thresh)


def objective1(xy_trg, xy, p):
    """
    Objective function that highlights small distances for crude grid search
    """
    xy_new = transform(xy, p)
    d = cdist(xy_trg, xy_new)
    return np.sum(1 / (d * d))


def prob_gmm(xy_trg, xy, σ):
    # *_, d = xy.shape
    _2σ2 = 2 * σ * σ
    return np.exp(-np.square(xy_trg[None] - xy[:, None]).sum(-1) / _2σ2).sum(-1)
    # / np.pow(np.pi * _2σ2, d)


def loglike_gmm(xy_trg, xy, σ):
    # add arbitrary offset to avoid nans!
    return np.log(prob_gmm(xy_trg, xy, σ) + 1).sum()


def objective_gmm(xy_trg, xy, σ, p):
    xy_new = transform(xy, p)
    # xy_new = xy + p[:2]
    return -loglike_gmm(xy_trg, xy_new, σ)


def objective_pix(img_trg, img, p):
    xy = transform(np.indices(img.shape[::-1]).reshape(2, -1).T, p)
    yx = xy.T[::-1]
    # print(yx.shape)
    l = ((0 < yx) & (yx < np.array(img.shape, ndmin=2).T)).all(0)
    ix = tuple(yx[:, l].round().astype(int))
    return np.square(img_trg[ix] - img.flatten()[l]).sum()


def find_objects(image, mask=False, background=None, snr=3., npixels=7,
                 edge_cutoff=None, deblend=False, dilate=0):
    seg = SegmentationHelper.detect(image, mask, background, snr, npixels,
                                    edge_cutoff, deblend, dilate)
    return seg, seg.com_bg(image),


def detect_measure(image, mask=False, background=None, snr=3., npixels=7,
                   edge_cutoff=None, deblend=False, dilate=0):
    seg = SegmentationHelper.detect(image, mask, background, snr, npixels,
                                    edge_cutoff, deblend, dilate)

    counts = seg.sum(image) - seg.median(image, [0]) * seg.areas
    return seg, seg.com_bg(image), counts


def dist_tril(coo, masked=False):
    """distance matrix with lower triangular region masked"""
    n = len(coo)
    sdist = cdist(coo, coo)  # pixel distance between stars
    ix = np.tril_indices(n, -1)
    # since the distance matrix is symmetric, ignore lower half
    if masked:
        sdist = np.ma.masked_array(sdist)
        sdist[ix] = np.ma.masked
    return sdist[ix]


# @timer
def gridsearch_mp(objective, args, grid):
    # grid search
    f = functools.partial(objective, *args)
    ndim, *rshape = grid.shape

    with mp.Pool() as pool:
        r = pool.map(f, grid.reshape(ndim, -1).T)
    pool.join()

    r = np.reshape(r, rshape)
    return r


def worker(func, args, input_, output, i, indexer, axes):
    indexer[axes] = i
    output[i] = func(*args, input_[tuple(indexer)])


def gridsearch_alt(func, args, grid, axes=..., output=None):
    # from joblib import Parallel, delayed

    # grid search
    if axes is not ...:
        axes = list(axes)
        out_shape = tuple(np.take(grid.shape, axes))
        indexer = np.full(grid.ndim, slice(None))

    if output is None:
        output = np.empty(out_shape)

    indices = np.ndindex(out_shape)
    # with Parallel(max_nbytes=1e3, prefer='threads') as parallel:
    #     parallel(delayed(worker)(func, args, grid, output, ix, indexer, axes)
    #              for ix in indices)
    # note: seems about twice as slow when testing for small datasets due to
    #  additional overheads

    with mp.Pool() as pool:
        pool.starmap(worker, ((func, args, grid, output, ix, indexer, axes)
                              for ix in indices))

    i, j = np.unravel_index(output.argmax(), out_shape)
    return output, (i, j), grid[:, i, j]


def identify_points(cooref, coo, thresh=0.2):
    # identify points matching points between sets by checking distances
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


# def match_cube(self, filename, object_name=None, coords=None):
#     from .core import shocObs
#
#     cube = shocObs.load(filename, mode='update')  # ff = FitsCube(fitsfile)
#     if coords is None:
#         coords = cube.get_coords()
#     if (coords is None):
#         if object_name is None:
#             raise ValueError('Need object name or coordinates')
#         coords = retrieve_coords(object_name)
#
#     image = np.fliplr(cube.data[:5].mean(0))
#     fov = cube.get_FoV()
#

def plot_transformed_image(ax, image, fov, p=(0, 0, 0), frame=False, **kws):
    """"""
    # image = image / image.max()
    extent = np.c_[[0., 0.], fov[::-1]]
    # pixel_size = np.divide(fov, image.shape)
    # extent -= 0.5 * pixel_size[None].T  # adjust to pixel centers...

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

        # - 0.5 * pixel_size
        r = Rectangle(xy, *fov[::-1], np.degrees(theta),
                      **frame_kws)
        ax.add_patch(r)

    return pl


def plot_coords_nrs(cooref, coords):
    fig, ax = plt.subplots()

    for i, yx in enumerate(cooref):
        ax.plot(*yx[::-1], marker='$%i$' % i, color='r')

    for i, yx in enumerate(coords):
        ax.plot(*yx[::-1], marker='$%i$' % i, color='g')


class ImageRegistration(LoggingMixin):
    gridStep = 3 / 60  # arcmin
    searchFrac = 0.7

    def __init__(self, image, fov, **find_kws):
        """

        Parameters
        ----------
        image:
        fov:
            Field of view for the image. Order of the dimensions should be
            the same as that of the image
        find_kws
        """
        self._find_kws = dict(snr=3.,
                              npixels=7,
                              edge_cutoff=3,
                              deblend=False,
                              )  # defaults
        self._find_kws.update(find_kws)

        # data array
        self.data = np.asarray(image)

        # Detect stars in dss frame
        self.segm, coords, self.counts = detect_measure(self.data,
                                                        **self._find_kws)
        self.fov = np.array(fov)[::-1]
        # pixel size in arcmin xy
        self.pixel_size = np.divide(fov, self.data.shape)[::-1]

        # FIXME: is it better to keep coordinates at pixel scale ??
        #  and even keep yx order to avoid lots of flippy flippy
        self.xy = coords[:, ::-1] * self.pixel_size
        # self.counts = counts

        #
        # ashape = np.array(self.data.shape)
        # self.grid = np.mgrid[tuple(map(slice, (0, 0), self.fov, 1j * ashape))]
        # self.ogrid = np.array([self.grid[0, :, 0], self.grid[1, 0, :]])

    def to_pixel_coords(self, xy):
        # internal coordinates are in arcmin origin at (0,0) for image
        return np.divide(xy, self.pixel_size)

    def match_image(self, image, fov, rotation=0., return_coords=False,
                    plot=False):
        """

        Parameters
        ----------
        image
        fov
        rotation
        return_coords
        plot

        Returns
        -------

        """
        # solve internally in xy coords
        pGs, xy = self.match_image_brute(image, fov, rotation, True, plot)
        # et tu brute??

        # TODO: try match on images directly?

        # final answer via gradient descent
        σ = 0.03  # width of gaussian kernel for gmm
        f = functools.partial(objective_gmm, self.xy, xy, σ)
        result = minimize(f, pGs)
        if result.success:
            p = result.x
        else:
            p = pGs

        #  return p in yx (image) coordinates
        pyx = np.r_[p[1::-1], p[-1]]
        if return_coords:
            return pyx, transform(xy, p)[:, ::-1]
        return pyx

    def match_image_brute(self, image, fov, rotation=0., return_coords=False,
                          plot=False):  # TODO step_size, sigma_gmm params??
        segmr, coo, counts = detect_measure(image, **self._find_kws)
        xy = (coo * fov / image.shape)[:, ::-1]
        fov = fov[::-1]

        g, r, ix, pGs = self._match_image_brute(xy, fov, self.gridStep,
                                                rotation, objective_gmm)
        pGs[-1] = rotation

        if plot:
            # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
            from graphical.imagine import ImageDisplay

            ggfig, ax = plt.subplots()
            ax.scatter(*self.xy.T, self.counts / self.counts.max() * 200)
            ax.plot(*transform(xy, pGs).T, 'r*')

            ext = np.r_[g[1, 0, [0, -1]],
                        g[0, [0, -1], 0]]
            im = ImageDisplay(r, extent=ext)
            im.ax.plot(*pGs[1::-1], 'ro', ms=15, mfc='none', mew=2)
            # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

        if return_coords:
            return pGs, xy
        return pGs

    # @timer
    def _match_image_brute(self, xy, fov, step_size, rotation=0,
                           objective=objective_gmm, σ=0.03):
        # do grid search

        # create grid
        xy0 = x0, y0 = -fov * self.searchFrac
        x1, y1 = self.fov - xy0

        xres = int((x1 - x0) / step_size)
        yres = int((y1 - y0) / step_size)
        grid = np.mgrid[x0:x1:complex(xres), y0:y1:complex(yres)]

        # add 0s for angle grid
        z = np.full((1,) + grid.shape[1:], rotation)
        grid = np.r_[grid, z]
        self.logger.info(
                "Doing search on (%.1f' x %.1f') (%d x %d pix) sky grid",
                *fov, yres, xres)

        # parallel
        r = gridsearch_mp(objective, (self.xy, xy, σ), grid)
        ix = (i, j) = np.unravel_index(r.argmin(), r.shape)
        pGs = grid[:, i, j]
        self.logger.debug('Grid search optimum: %s', pGs)
        return grid, r, ix, pGs

    def _match_final(self, xy):

        from scipy.spatial.distance import cdist
        from recipes.containers.list import where_duplicate

        # transform
        xy = self.to_pixel_coords(xy)
        ix = tuple(xy.round().astype(int).T)[::-1]
        labels = self.segm.data[ix]
        #
        use = np.ones(len(xy), bool)
        # ignore stars not detected in dss, but detected in sample image
        use[labels == 0] = False
        # ignore labels that are detected as 2 (or more) sources in the
        # sample image, but only resolved as 1 in dss
        for w in where_duplicate(labels):
            use[w] = False

        assert use.sum() > 3, 'Not good enough'

        d = cdist(self.to_pixel_coords(self.xy), xy[use])
        iq0, iq1 = np.unravel_index(d.argmin(), d.shape)
        xyr = self.to_pixel_coords(self.xy[iq0])
        xyn = xy[use] - xyr

        xyz = self.to_pixel_coords(self.xy[labels[use] - 1]) - xyr
        a = np.arctan2(*xyz.T[::-1]) - np.arctan2(*xyn.T[::-1])
        return np.median(a)


class ImageRegistrationDSS(ImageRegistration):
    _servers = ('poss2ukstu_blue', 'poss1_blue',
                'poss2ukstu_red', 'poss1_red',
                'poss2ukstu_ir',
                'all')

    def __init__(self, name_or_coords, fov=(3, 3), **find_kws):
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
                # first try interpret coords.
                # eg. ImageRegistrationDSS('06:14:51.7 -27:25:35.5', (3, 3))
                coords = SkyCoord(name_or_coords, unit=('h', 'deg'))
            except:
                coords = retrieve_coords(name_or_coords)
                pass

        if coords is None:
            raise ValueError('Need object name or coordinates')

        for serv in self._servers:
            try:
                self.hdu = get_dss(serv, coords.ra.deg, coords.dec.deg, fov)
                break
            except STScIServerError as err:
                self.logger.warning('Failed to retrieve image from server: '
                                    '%s', serv)

        # DSS data array
        data = self.hdu[0].data.astype(float)
        find_kws.setdefault('deblend', True)
        ImageRegistration.__init__(self, data, fov, **find_kws)

        # save target coordinate position
        self.targetCoords = coords
        self.targetCoordsPix = np.divide(self.data.shape, 2) + 0.5

    def get_labels(self, xy):
        # transform
        ix = tuple(self.to_pixel_coords(xy).round().astype(int).T)[::-1]
        return self.segm.data[ix]

        #
        use = np.ones(len(xy), bool)
        # ignore stars not detected in dss, but detected in sample image
        use[labels == 0] = False
        # ignore labels that are detected as 2 (or more) sources in the
        # sample image, but only resolved as 1 in dss
        for w in where_duplicate(labels):
            use[w] = False
        return labels, use

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
        crpix = np.array(
                [h['crpix1'], h['crpix2']])  # target object pixel coordinates
        crpixDSS = crpix - 0.5  # convert to pixel llc coordinates
        cram = crpixDSS / self.pixel_size  # convert to arcmin
        rotm = rotation_matrix_2d(-theta)
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
        w.wcs.crota = np.degrees(
                [-theta, -theta])  # rotation from stated coordinate type.
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]  # axis type

        return w

    def plot_coords_nrs(self, coords):
        fig, ax = plt.subplots()

        for i, yx in enumerate(self.xy):
            ax.plot(*yx[::-1], marker='$%i$' % i, color='r')

        for i, yx in enumerate(coords):
            ax.plot(*yx[::-1], marker='$%i$' % i, color='g')


class MosaicPlotter(object):
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
            self._ff = ff = apl.FITSFigure(self.dss.hdu)
            self._ax = ff.ax
            self._fig = ff.ax.figure
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
            y, x = p[:2] / self.dss.pixel_size + 0.5
            return (y, x, p[-1]), dsspix
        return p, fov

    def plot_image(self, image=None, fov=None, p=(0, 0, 0), name=None,
                   frame=False, **kws):
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
        """Get corners relative to DSS coordinates. yx coords"""
        c = np.array([[0, 0], fov])  # lower left, upper right yx
        corners = np.c_[c[0], c[:, 1], c[1], c[::-1, 0]].T  # / clockwise yx
        corners = transform_yx(corners, p)
        return corners[:, ::-1]  # return xy !

    def update_lims(self, p, fov):

        corners = self.get_corners(p, fov)
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
        cmap_dss = kws.pop('cmap_dss', 'Greys')
        cmap = kws.pop('cmap', None)
        alpha_magic = min(1. / (len(images) + show_dss), 0.5)
        alpha = kws.pop('alpha', alpha_magic)

        if show_dss:
            self.plot_image(**kws, cmap=cmap_dss, alpha=1)

        for image, fov, p in zip(images, fovs, ps):
            self.plot_image(image, fov, p, cmap=cmap, **kws)

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
#         y, x = p[:2] / dss.pixel_size + 0.5
#         # plot image
#         pl = plot_image(ax, image, dsspix, (y, x, p[-1]), **kws)
#
#     return f


# infer offset and rotation of grids


# gridr = ndgrid.like(imr)
# rotm = rotation_matrix_2d(np.radians(p[-1]))
# gridn = gridr.swapaxes(0, -1) @ rotm + p[:2]

if __name__ == '__main__':
    fitsfile = '/media/Oceanus/UCT/Observing/data/Feb_2017/J0614-2725/SHA_20170209.0006.bff.fits'

    w = wcs.WCS(naxis=2)

    # see: https://www.aanda.org/articles/aa/full/2002/45/aah3859/aah3859.html for definitions
    w.wcs.crpix = [-234.75,
                   8.3393]  # array location of the reference point in pixels
    w.wcs.cdelt = [-0.066667,
                   0.066667]  # coordinate increment at reference point
    w.wcs.crval = [0, -90]  # coordinate value at reference point
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]  # axis type
    w.wcs.set_pv([(2, 1, 45.0)])  # rotation from stated coordinate type.
