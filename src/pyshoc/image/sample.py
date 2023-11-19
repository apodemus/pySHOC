
from obstools.image.orient import ImageOrienter
from obstools.image.sample import BootstrapResample


# class ImageOrienter(ImageOrientBase):
#     def __init__(self, hdu):

#
#         super().__init__(hdu, x=hdu.readout.isEM)


class ResampleFlip(BootstrapResample, ImageOrienter):
    """
    This class ensures that all SHOC images have the same orientation
    relative to the sky, assist doing image arithmetic.
    """

    def __init__(self, data, sample_size=None, subset=None, axis=0, flip=()):
        BootstrapResample.__init__(self, data, sample_size, subset, axis)
        ImageOrienter.__init__(self, flip)

    def draw(self, n=None, subset=None):
        return BootstrapResample.draw(self, n, subset)[self.orient]
