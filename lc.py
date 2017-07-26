import numpy as np

def _make_ma(data):
    if data is None:
        data = []

    if np.ndim(np.squeeze(data)) > 1:
        raise ValueError('Signal should be at most 1 dimensional, not %iD'
                         % np.ndim(data))

    mask = False
    if np.ma.isMA(data):
        mask = data.mask
    mask |= (np.isnan(data) | np.isinf(data))

    data = np.ma.asarray(data)
    data.mask = mask
    return data



class LightCurve(object):

    @property
    def parts(self):
        return self.t, self.signal, self.std

    def __init__(self, *args):

        # signals only
        if len(args) == 1:
            signal, = args
            t = []   # No time given, use index array ?
            std = []  # No errors given
        # times & signals given
        elif len(args) == 2:  # No errors given
            t, signal = args
            std = []
        # times, signals, errors given
        elif len(args) == 3:
            t, signal, std = args
        else:
            raise ValueError('Invalid number of arguments: %i' %len(args))

        # handle if signal is LightCurve instance
        if isinstance(signal, self.__class__):
            self.t, self.signal, self.std = signal.parts
            return

        # store data internally as masked array
        self.signal = _make_ma(signal)
        self.std = _make_ma(std)
        if len(self.std):
            assert len(self.signal) == len(self.std)

        if len(t) == 0:
            t = np.arange(len(self.signal))
        self.t = np.asanyarray(t)


    # def __str__(self):
    #     'TODO'

    def __len__(self):
        return len(self.t)

    def __iter__(self):
        raise NotImplementedError

    def __pos__(self):
        return self

    def __neg__(self):
        return self.__class__(self.t, -self.signal, self.std)

    def __abs__(self):
        return self.__class__(self.t, abs(self.signal), self.std)

    def __add__(self, o):
        o = self.__class__(o)
        raise NotImplementedError
        assert len(self) == len(o)


        # TODO: handle constants, arrays, etc

        t, r, e = self.parts
        to, ro, eo = o.parts

        # TODO: check that times overlap

        if len(t) != len(to):
            raise NotImplementedError

            # can't add directly...have to rebin?
            # NOTE: this also induces uncertainty in T
            tn, deltat = self._retime(o)
            t = tn + deltat / 2  # bin centers
            i = np.searchsorted(t, tn)  # indeces to use
            io = np.searchsorted(to, tn)
            r, e = r[i], e[i]
            ro, eo = ro[io], eo[io]

        R = r + ro
        E = np.sqrt(np.square((e, eo)).sum(0))  # add errors in quadrature

        return self.__class__(t, R, E)

    def __sub__(self, o):
        return self + -o

    # def _retime(self, o):
    #     t, r, e = self.parts
    #     to, ro, eo = o.parts
    #
    #     mode0 = get_deltat_mode(t)
    #     mode1 = get_deltat_mode(to)
    #
    #     if mode0 == mode1:  # cool, time steps are at least identical
    #         # now find the overlapping bits
    #         t0 = min(t.min(), to.min())
    #         t1 = max(t.max(), to.max())
    #
    #         first = np.argmin((t[0], to[0]))  # which has the earliest start?
    #         if first:
    #             t, to = to, t  # swap them around, so that t starts first
    #             r, ro = ro, r
    #             e, eo = eo, e
    #
    #         # choose bins in such a way that the times from both arrays in a single bin have minimal seperation
    #         offset = abs(t - to[0]).argmin()
    #         b0 = t[0] - (to[0] - t[offset]) / 2  # first bin starts here
    #         Nbins = np.ceil((t1 - t0) / mode0)
    #         bin_edges = b0 + np.arange(Nbins + 1) * mode0
    #
    #         h, b = np.histogram(np.r_[t, to], bin_edges)
    #         tn = b[h == 2]  # bin edges for new time bins
    #     else:
    #         'rebin one of them??'
    #
    #     return tn, mode0

    def __mul__(self, o):
        o = self.__class__(o)
        raise NotImplementedError

    def __truediv__(self, o):
        o = self.__class__(o)
        raise NotImplementedError

    def __pow__(self, o):
        o = self.__class__(o)
        raise NotImplementedError

    def __lt__(self, o):
        raise NotImplementedError

    def __le__(self, o):
        raise NotImplementedError

    def __eq__(self, o):
        raise NotImplementedError

    def __ne__(self, o):
        raise NotImplementedError

    def __ge__(self, o):
        raise NotImplementedError

    def __gt__(self, o):
        raise NotImplementedError





if __name__ == '__main__':
    # tests
    _make_ma([])


