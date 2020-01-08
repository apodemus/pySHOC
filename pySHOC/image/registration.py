"""
Helper functions to infer World Coordinate System given a target name or
coordinates of a object in the field. This is done by matching the image
with the DSS image for the same field via image registration.  A number of
methods are implemented for doing this:
  point cloud drift
  matching via locating dense cluster of displacements between points
  direct image-to-image matching
  brute force search with gaussian mixtures on points
"""

import functools as ftl
import logging
import multiprocessing as mp
import itertools as itt

import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from astropy.coordinates import SkyCoord

from recipes.containers.dict_ import AttrDict
from obstools.phot.segmentation import SegmentationHelper

from pySHOC.utils import retrieve_coords, get_dss, STScIServerError

import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D

from recipes.logging import LoggingMixin
from recipes.transformations.rotation import rotation_matrix_2d

from scipy.stats import binned_statistic_2d

from motley.profiling.timers import timer
from recipes.introspection.utils import get_module_name

# TODO: might be faster to anneal sigma param rather than blind grid search.
#  test this.

logger = logging.getLogger(get_module_name(__file__))


def roto_translate(X, p):
    """rotate and translate"""
    # https://en.wikipedia.org/wiki/Rigid_transformation
    rotm = rotation_matrix_2d(p[-1])
    Xnew = (rotm @ X.T).T + p[:2]
    return Xnew


def roto_translate2(X, xy_off, theta=0):
    """rotate and translate"""
    if theta:
        rotm = rotation_matrix_2d(theta)
        X = (rotm @ X.T).T
    return X + xy_off


def roto_translate_yx(X, p):
    """rotate and translate"""
    rotm = rotation_matrix_2d(p[-1])
    Xnew = (rotm @ X[:, ::-1].T).T + p[1::-1]
    return Xnew[:, ::-1]


def check_transforms(yx, p):
    z0 = roto_translate_yx(yx, p)

    pxy = np.r_[p[1::-1], p[-1]]
    z1 = roto_translate(yx[:, ::-1], pxy)[:, ::-1]

    return np.allclose(z0, z1)


def normalize_image(image, centre=np.ma.median, scale=np.ma.std):
    """Recenter and scale"""
    image = image - centre(image)
    if scale:
        return image / scale(image)
    return image


def get_sample_image(hdu, stat='median', depth=5):
    # get sample image
    n = int(np.ceil(depth // hdu.timing.t_exp))

    logger.info(f'Computing {stat} of {n} images (exposure depth of '
                f'{float(depth):.1f} seconds) for sample image from '
                f'{hdu.filepath.name!r}')

    sampler = getattr(hdu.sampler, stat)
    image = hdu.calibrated(sampler(n, n))
    return normalize_image(image, scale=False)


def objective_pix(target, values, yx, bins, p):
    """Objective for direct image matching"""
    yx = roto_translate_yx(yx, p)
    bs = binned_statistic_2d(*yx.T, values, 'mean', bins)
    rs = np.square(target - bs.statistic)
    return np.nansum(rs)


# def match_pixels(self, image, fov, p0):
#     """Match pixels directly"""
#     sy, sx = self.data.shape
#     dx, dy = self.fov
#     hx, hy = 0.5 * self.pixel_size
#     by, bx = np.ogrid[-hx:(dx + hx):complex(sy + 1),
#                       -hy:(dy + hy):complex(sx + 1)]
#     bins = by.ravel(), bx.ravel()
#
#     sy, sx = image.shape
#     dx, dy = fov
#     yx = np.mgrid[0:dx:complex(sy), 0:dy:complex(sx)].reshape(2, -1).T  # [::-1]
#
#     target = normalize_image(self.data)
#     values = normalize_image(image).ravel()
#     result = minimize(ftl.partial(objective_pix, target, values, yx, bins),
#                       p0)
#     return result


# def objective0(xy_trg, xy, p, thresh=0.1):
#     """
#     Objective function that highlights small distances for crude grid search
#     """
#     xy_new = transform(xy, p)
#     ifd = cdist(xy_trg, xy_new)
#     # return np.percentile(ifd, thresh)
#     return np.sum(ifd < thresh)
#
#
# def objective1(xy_trg, xy, p):
#     """
#     Objective function that highlights small distances for crude grid search
#     """
#     xy_new = transform(xy, p)
#     d = cdist(xy_trg, xy_new)
#     return np.sum(1 / (d * d))


def prob_gmm(xy_trg, xy, σ):
    # *_, d = xy.shape
    # f = 2 * σ * σ  #  / np.pow(np.pi * f, d)
    return np.exp(
            -np.square(xy_trg[None] - xy[:, None]).sum(-1) / (2 * σ * σ)
    ).sum(-1)


def loglike_gmm(xy_trg, xy, σ):
    # add arbitrary offset to avoid nans!
    return np.log(prob_gmm(xy_trg, xy, σ) + 1).sum()


def objective_gmm(xy_trg, xy, σ, p):
    """Objective for gaussian mixture model"""
    xy_new = roto_translate(xy, p)
    # xy_new = xy + p[:2]
    return -loglike_gmm(xy_trg, xy_new, σ)


def objective_gmm_yx(yx_trg, yx, σ, p):
    """Objective for gaussian mixture model"""
    yx_new = roto_translate_yx(yx, p)
    # xy_new = xy + p[:2]
    return -loglike_gmm(yx_trg, yx_new, σ)


def objective_gmm2(xy_trg, xy, σ, xy_off, theta=0):
    xy_new = roto_translate2(xy, xy_off, theta)
    return -loglike_gmm(xy_trg, xy_new, σ)


def objective_gmm3(x_trg, x, sigma, xy_off, theta=0):
    xy, counts = x
    xy_new = roto_translate2(xy, xy_off, theta)
    return -loglike_gmm(x_trg, xy_new, sigma)


# def find_objects(image, mask=False, background=None, snr=3., npixels=7,
#                  edge_cutoff=None, deblend=False, dilate=0):
#     seg = SegmentationHelper.detect(image, mask, background, snr, npixels,
#                                     edge_cutoff, deblend, dilate)
#     return seg, seg.com_bg(image),


def detect_measure(image, mask=False, background=None, snr=3., npixels=5,
                   edge_cutoff=None, deblend=False, dilate=0):
    #
    seg = SegmentationHelper.detect(image, mask, background, snr, npixels,
                                    edge_cutoff, deblend, dilate)

    counts = seg.sum(image) - seg.median(image, [0]) * seg.areas
    return seg, seg.com_bg(image), counts


def offset_disp_cluster(xy0, xy1, dmax=0.05, bins=25):
    # find offset (translation) between frames by looking for the dense
    # cluster of inter-frame point-to-point displacements that represent the
    # xy offset between the two images. Cannot handle rotations

    points = (xy0[None] - xy1[:, None]).reshape(-1, 2).T
    vals, xe, ye = np.histogram2d(*points, bins)
    # todo probably raise if there is no significantly dense cluster
    i, j = np.unravel_index(vals.argmax(), vals.shape)
    l = ((xe[i] < points[0]) & (points[0] <= xe[i + 1]) &
         (ye[j] < points[1]) & (points[1] <= ye[j + 1]))
    sub = points[:, l].T
    yay = sub[np.sum(cdist(sub, sub) > dmax, 0) < (len(sub) // 2)]
    return np.mean(yay, 0)


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
def gridsearch_mp(objective, grid, args, **kws):
    # grid search
    f = ftl.partial(objective, *args, **kws)
    ndim, *rshape = grid.shape

    with mp.Pool() as pool:
        r = pool.map(f, grid.reshape(ndim, -1).T)
    pool.join()

    return np.reshape(r, rshape)


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


def cross_index(coo_trg, coo, dmax=0.2):
    # identify points matching points between sets by checking distances
    # this will probably only work if the frames are fairly well aligned
    dr = cdist(coo_trg, coo)

    # if thresh is None:
    #     thresh = np.percentile(dr, dr.size / len(coo))

    dr[(dr > dmax) | (dr == 0)] = np.nan

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
        ax.add_patch(
                Rectangle(xy, *fov[::-1], np.degrees(theta), **frame_kws)
        )

    return pl


def plot_coords_nrs(cooref, coords):
    fig, ax = plt.subplots()

    for i, yx in enumerate(cooref):
        ax.plot(*yx[::-1], marker='$%i$' % i, color='r')

    for i, yx in enumerate(coords):
        ax.plot(*yx[::-1], marker='$%i$' % i, color='g')


def display_multitab(images, fovs, params, coords):
    from graphing.multitab import MplMultiTab
    from graphing.imagine import ImageDisplay

    import more_itertools as mit

    ui = MplMultiTab()
    for i, (image, fov, p, yx) in enumerate(zip(images, fovs, params, coords)):
        xy = yx[:, ::-1]  # roto_translate_yx(yx, np.r_[-p[:2], 0])[:, ::-1]
        ex = mit.interleave((0, 0), fov)
        im = ImageDisplay(image, extent=list(ex))
        im.ax.plot(*xy.T, 'kx', ms=5)
        ui.add_tab(im.figure)
        plt.close(im.figure)
        # if i == 1:
        #    break
    ui.show()


from recipes.misc import duplicate_if_scalar


class RegisteredImage(object):
    """helper class for image registration"""

    def __init__(self, data, fov):
        # data array
        self.data = np.asarray(data)
        self.fov = np.array(fov)[::-1]

        # pixel size in arcmin xy
        self.pixel_size = self.fov / self.data.shape


#

class ImageRegistration(LoggingMixin):  # SourceDetectionMixin??
    searchFrac = 1.0
    _dflt_find_kws = dict(snr=3.,
                          npixels=5,
                          # edge_cutoff=3,
                          deblend=False)

    # FIXME: is it better to keep coordinates at pixel scale ??
    #  and even keep yx order to avoid lots of flippy flippy

    # @classmethod
    # def from_image(cls, fov, **find_kws):

    @staticmethod
    def detect(image, fov, **kws):
        # find stars
        seg, coo, counts = detect_measure(image, **kws)
        yx = (coo * fov / image.shape)
        return seg, yx, counts

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

        # defaults
        for k, v in self._dflt_find_kws.items():
            find_kws.setdefault(k, v)

        # data array
        self.data = np.asarray(image)
        self.fov = np.array(fov)[::-1]
        # pixel size in arcmin xy
        self.pixel_size = self.fov / self.data.shape

        # Detect stars in dss frame
        self.seg, yx, self.counts = detect_measure(self.data, **find_kws)
        self.yx = yx * self.pixel_size
        self.xy = self.yx[:, ::-1]

        # containers for matched images
        # self.fovs = []
        # self.images = []
        # self.detections = []

        # ashape = np.array(self.data.shape)
        # self.grid = np.mgrid[tuple(map(slice, (0, 0), self.fov, 1j * ashape))]
        # self.ogrid = np.array([self.grid[0, :, 0], self.grid[1, 0, :]])

    def to_pixel_coords(self, xy):
        # internal coordinates are in arcmin origin at (0,0) for image
        return np.divide(xy, self.pixel_size)

    def match_points(self, yx, rotation=0.):
        # todo: method
        if rotation is None:
            raise NotImplementedError
        else:
            # get shift (translation) via peak detection in displacements (outer
            # difference) between point clouds
            dyx = offset_disp_cluster(self.yx, yx)
            p = np.r_[dyx, rotation]

        # match images directly
        # mr = self.match_pixels(image, fov[::-1], p)
        return p, yx

    def match_image(self, image, fov, rotation=0.):
        """
        Search heuristic for image offset and rotation.


        Parameters
        ----------
        image
        fov
        rotation

        Returns
        -------

        """

        # detect
        seg, yx, counts = self.detect(image, fov)

        # aggregate
        # self.fovs.append(fov[::-1])
        # self.images.append(image)
        # self.detections.append(seg)

        return self.match_points(yx, rotation)

        # pGs = self.match_points_brute(yx, fov, rotation, plot=plot)
        # et tu brute??
        #
        # # final match gradient descent
        # p = self._match_points_gd(xy, pGs, fit_angle)
        # if p is None:
        #     p = pGs
        #
        # #  return p in yx (image) coordinates
        # # pyx = np.r_[p[1::-1], p[-1]]
        # # if return_coords:
        # #     return pyx#, xy[:, ::-1]  # roto_translate(xy, p)
        # return pyx

    def match_pixels(self, image, fov, p0):
        """Match pixels directly"""
        sy, sx = self.data.shape
        dx, dy = self.fov
        hx, hy = 0.5 * self.pixel_size
        by, bx = np.ogrid[-hx:(dx + hx):complex(sy + 1),
                 -hy:(dy + hy):complex(sx + 1)]
        bins = by.ravel(), bx.ravel()

        sy, sx = image.shape
        dx, dy = fov
        yx = np.mgrid[:dx:complex(sy), :dy:complex(sx)].reshape(2, -1).T

        return minimize(
                ftl.partial(objective_pix,
                            normalize_image(self.data),
                            normalize_image(image).ravel(),
                            yx, bins),
                p0)

    def match_image_brute(self, image, fov, rotation=0., step_size=0.05,
                          return_coords=False, plot=False, sigma_gmm=0.03):

        seg, yx, counts = self.find_stars(image, fov)
        return self.match_points_brute(yx, fov, rotation, step_size,
                                       return_coords, plot, sigma_gmm)

    @timer
    def match_points_brute(self, yx, fov, rotation=0., step_size=0.05,
                           plot=False, sigma_gmm=0.03):
        # TODO: this method may not be needed if you choose anneal on sigma_gmm
        # grid search
        g, r, ix, pGs = self._match_points_brute_yx(yx, fov, step_size,
                                                    rotation, objective_gmm,
                                                    sigma_gmm)
        pGs[-1] = rotation

        if plot:
            from graphing.imagine import ImageDisplay

            # plot xy coords
            ggfig, ax = plt.subplots()
            ax.scatter(*self.xy.T, self.counts / self.counts.max() * 200)
            ax.plot(*roto_translate_yx(yx, pGs).T[::-1], 'r*')

            ext = np.r_[g[1, 0, [0, -1]],
                        g[0, [0, -1], 0]]
            im = ImageDisplay(r, extent=ext)
            im.ax.plot(*pGs[1::-1], 'ro', ms=15, mfc='none', mew=2)

        return pGs

    def _match_points_brute_yx(self, yx, fov, step_size, rotation=0.,
                               objective=objective_gmm_yx, *args, **kws):

        # create grid
        yx0 = y0, x0 = -fov * self.searchFrac
        y1, x1 = self.fov - yx0

        xres = int((x1 - x0) / step_size)
        yres = int((y1 - y0) / step_size)
        grid = np.mgrid[y0:y1:complex(yres),
               x0:x1:complex(xres)]

        # add 0s for angle grid
        z = np.full((1,) + grid.shape[1:], rotation)
        grid = np.r_[grid, z]  # FIXME: silly to stack rotation here..
        self.logger.info(
                "Doing search on (%.1f' x %.1f') (%d x %d pix) sky grid",
                *fov, yres, xres)

        # parallel
        r = gridsearch_mp(objective, grid, (self.yx, yx) + args, **kws)
        ix = (i, j) = np.unravel_index(r.argmin(), r.shape)
        pGs = grid[:, i, j]
        self.logger.debug('Grid search optimum: %s', pGs)
        return grid, r, ix, pGs

    def _match_points_brute(self, xy, fov, step_size, rotation=0.,
                            objective=objective_gmm, *args, **kws):

        # create grid
        xy0 = x0, y0 = -fov * self.searchFrac
        x1, y1 = self.fov - xy0

        xres = int((x1 - x0) / step_size)
        yres = int((y1 - y0) / step_size)
        grid = np.mgrid[x0:x1:complex(xres),
               y0:y1:complex(yres)]

        # add 0s for angle grid
        z = np.full((1,) + grid.shape[1:], rotation)
        grid = np.r_[grid, z]  # FIXME: silly to stack rotation here..
        self.logger.info(
                "Doing search on (%.1f' x %.1f') (%d x %d pix) sky grid",
                *fov, yres, xres)

        # parallel
        r = gridsearch_mp(objective, grid, (self.xy, xy) + args, **kws)
        ix = (i, j) = np.unravel_index(r.argmin(), r.shape)
        pGs = grid[:, i, j]
        self.logger.debug('Grid search optimum: %s', pGs)
        return grid, r, ix, pGs

    @timer
    def _match_points_gd(self, xy, p0, fit_angle=True, sigma_gmm=0.03):
        """
        Match points with gradient descent on Gaussian mixture model likelihood

        Parameters
        ----------
        xy: array
            xy coordinates of image features (center of mass of stars)
        p0: array-like
            parameter starting values
        sigma_gmm: float
             standard deviation of gaussian kernel for gmm

        Returns
        -------

        """

        if fit_angle:
            f = ftl.partial(objective_gmm, self.xy, xy, sigma_gmm)
        else:
            theta = p0[-1]
            xy = roto_translate2(xy, [0, 0], theta)
            f = ftl.partial(objective_gmm2, self.xy, xy, sigma_gmm)
            p0 = p0[:2]

        result = minimize(f, p0)
        if not result.success:
            return None

        if fit_angle:
            return result.x

        # noinspection PyUnboundLocalVariable
        return np.r_[result.x, theta]

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

    def display(self, show_coms=True, number=True, cmap=None, marker_color='r'):
        """
        Display the image and the coordinates of the stars

        Returns
        -------

        """
        from graphing.imagine import ImageDisplay
        import more_itertools as mit

        ex = mit.interleave((0, 0), self.fov)
        im = ImageDisplay(self.data, cmap=cmap, extent=list(ex))

        if show_coms:
            if number:
                s = []
                for i, xy in enumerate(self.xy):
                    s.extend(
                            im.ax.plot(*xy, marker='$%i$' % i,
                                       color=marker_color)
                    )
            else:
                s, = im.ax.plot(*self.xy.T, 'rx')

        return im, s


class ImageRegistrationDSS(ImageRegistration):
    """

    """
    #
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
        fov: int or 2-tuple
            field of view in arcmin
        """

        #
        fov = duplicate_if_scalar(fov)

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

    def build_wcs(self, hdu, p, telescope=None):
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
        fov = hdu.get_fov(telescope)
        pxscl = fov / hdu.shape[-2:]
        # transform target coordinates in DSS image to target in SHOC image
        h = self.hdu[0].header
        # target object pixel coordinates
        crpix = np.array([h['crpix1'], h['crpix2']])
        crpixDSS = crpix - 0.5  # convert to pixel llc coordinates
        cram = crpixDSS / self.pixel_size  # convert to arcmin
        rotm = rotation_matrix_2d(-theta)
        crpixSHOC = (rotm @ (cram - yxoff)) / pxscl
        # target coordinates in degrees
        xtrg = self.targetCoords
        # coordinate increment
        cdelt = pxscl / 60  # in degrees
        flip = np.array(hdu.flip_state[::-1], bool)
        cdelt[flip] = -cdelt[flip]

        # see:
        # https://www.aanda.org/articles/aa/full/2002/45/aah3859/aah3859.html
        # for parameter definitions
        w = wcs.WCS(naxis=2)
        # array location of the reference point in pixels
        w.wcs.crpix = crpixSHOC
        # coordinate increment at reference point
        w.wcs.cdelt = cdelt
        # coordinate value at reference point
        w.wcs.crval = xtrg.ra.value, xtrg.dec.value
        # rotation from stated coordinate type.
        w.wcs.crota = np.degrees([-theta, -theta])
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]  # axis type

        return w

    def plot_coords_nrs(self, coords):
        fig, ax = plt.subplots()

        for i, yx in enumerate(self.xy):
            ax.plot(*yx[::-1], marker='$%i$' % i, color='r')

        for i, yx in enumerate(coords):
            ax.plot(*yx[::-1], marker='$%i$' % i, color='g')


class MosaicPlotter(object):  # todo better name
    """Plot the results from image registration run"""

    default_cmap_dss = 'Greys'
    alpha_cycle_value = 0.65

    def __init__(self, dss, use_aplpy=True):

        self.dss = dss
        self.use_aplpy = use_aplpy

        self.names = []
        self.params = []
        self.fovs = []

        self.art = AttrDict()  # art
        self.image_label = None

        self._counter = itt.count()
        self._ff = None
        self._fig = None
        self._ax = None
        self._low_lims = (np.inf, np.inf)
        self._up_lims = (-np.inf, -np.inf)
        self._idx_active = 0
        self.alpha_cycle = []

        # connect image scrolling
        self.fig.canvas.mpl_connect('scroll_event', self._scroll_safe)
        self.fig.canvas.mpl_connect('button_press_event', self.reset)

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

    def _world2pix(self, p, fov):  # FIXME: this name is inappropriate
        # convert fov to the DSS pixel coordinates (aplpy)
        if self.use_aplpy:
            # scale_ratio = (fov / self.dss.fov)
            dsspix = (fov / self.dss.fov) * self.dss.data.shape
            y, x = p[:2] / self.dss.pixel_size + 0.5
            return (y, x, p[-1]), dsspix
        return p, fov

    def plot_image(self, image=None, fov=None, p=(0, 0, 0), name=None,
                   frame=False, **kws):
        """

        Parameters
        ----------
        image
        fov
        p
        name
        frame
        kws

        Returns
        -------

        """
        update = True
        if image is None:
            image = self.dss.data
            fov = self.dss.fov
            header = self.dss.hdu[0].header
            name = ' '.join(filter(None, map(header.get, ('ORIGIN', 'FILTER'))))
            update = False

        if name is None:
            name = 'image%i' % next(self._counter)

        # convert fov to the DSS pixel coordinates (aplpy)
        p, fov = self._world2pix(p, fov)
        self.params.append(p)
        self.fovs.append(fov)

        # plot
        # image = image / image.max()
        self.art[name] = plot_transformed_image(self.ax, image, fov, p, frame,
                                                **kws)
        self.names.append(name)
        if update:
            self.update_axes_limits(p, fov)

    def mosaic(self, images, fovs, params, names=(), **kws):
        # mosaic plot

        show_dss = kws.pop('show_dss', True)
        cmap_dss = kws.pop('cmap_dss', self.default_cmap_dss)
        cmap = kws.pop('cmap', None)
        alpha_magic = min(1. / (len(images) + show_dss), 0.5)
        alpha = kws.pop('alpha', alpha_magic)

        if show_dss:
            self.plot_image(**kws, cmap=cmap_dss, alpha=1)

        for image, fov, p, name in itt.zip_longest(images, fovs, params, names):
            self.plot_image(image, fov, p, name, cmap=cmap, alpha=alpha, **kws)

        n = len(self.art)
        self.alpha_cycle = np.vstack([np.eye(n) * self.alpha_cycle_value,
                                      np.ones(n) * alpha])
        if show_dss:
            self.alpha_cycle[:, 0] = 1

    def get_corners(self, p, fov):
        """Get corners relative to DSS coordinates. xy coords anti-clockwise"""
        c = np.array([[0, 0], fov])  # lower left, upper right yx
        corners = np.c_[c[0], c[:, 1], c[1], c[::-1, 0]].T  # / clockwise yx
        corners = roto_translate_yx(corners, p)
        return corners[:, ::-1]  # return xy !

    def update_axes_limits(self, p, fov):
        corners = self.get_corners(p, fov)
        self._low_lims = np.min([corners.min(0), self._low_lims], 0)
        self._up_lims = np.max([corners.max(0), self._up_lims], 0)
        xlim, ylim = list(zip(self._low_lims, self._up_lims))
        self.ax.set(xlim=xlim, ylim=ylim)

    def _scroll(self, event):
        """
        This method allows you to scroll through the images in the mosaic
        using the mouse.
        """
        if event.inaxes and len(self.art):
            # set alphas
            self._idx_active += [-1, +1][event.button == 'up']
            self._idx_active %= (len(self.art) + 1)  # wrap
            alphas = self.alpha_cycle[self._idx_active]
            for i, pl in enumerate(self.art.values()):
                pl.set_alpha(alphas[i])
                if i == self._idx_active:
                    # position -1 represents the original image
                    z = -1 if (i == -1) else 1
                    pl.set_zorder(z)
                else:
                    pl.set_zorder(0)

            if self.image_label is not None:
                self.image_label.remove()

            # set tiles
            if self._idx_active != len(self.art):
                # position -1 represents the original image
                name = f'{self._idx_active}: {self.names[self._idx_active]}'
                p = self.params[self._idx_active]
                yx = np.atleast_2d([self.fovs[self._idx_active][0], 0])
                xy = roto_translate_yx(yx, p)[0, ::-1]
                self.image_label = self.ax.text(*xy, name, color='w',
                                                rotation=np.degrees(p[-1]),
                                                rotation_mode='anchor',
                                                va='top')

            # redraw
            self.fig.canvas.draw()

    def _scroll_safe(self, event):
        try:
            # print(vars(event))
            self._scroll(event)

        except Exception as err:
            import traceback

            print('Scroll failed:')
            traceback.print_exc()
            print('len(names)', len(self.names))
            print('self._idx_active', self._idx_active)

            self.image_label = None
            self.fig.canvas.draw()

    def reset(self, event):
        # reset original alphas
        if event.button == 2:
            self._idx_active = 0
            alphas = self.alpha_cycle[0]
            for i, pl in enumerate(self.art.values()):
                pl.set_alpha(alphas[i])
                pl.set_zorder(0)

        # redraw
        self.fig.canvas.draw()