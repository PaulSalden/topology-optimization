import numpy as np

def nurbs(shape, params):
    if shape == "square":
        U = np.array([0., 0., 1., 1.])
        V = np.array([0., 0., 1., 1.])
        cpoints = np.array([
            [[0., 0.], [1., 0.]],
            [[0., 1.], [1., 1.]]
        ])
        weights = np.array([
            [[1.], [1.]],
            [[1.], [1.]]
        ])
        plotrange = (0, 1, 0, 1)

    elif shape == "rectangle":
        ar = params["aspectratio"]
        p = params["p"]
        q = params["q"]

        U = np.repeat(np.array([0., 1.]), p + 1)
        V = np.repeat(np.array([0., 1.]), q + 1)

        x = np.linspace(0, ar, p + 1)
        y = np.linspace(0, 1, q + 1)
        xx, yy = np.meshgrid(x, y)

        cpoints = np.dstack((xx, yy))
        weights = np.ones(cpoints.shape[:-1])

        plotrange = (0, ar, 0, 1)

    elif shape == "roundbase":
        ar = params["aspectratio"]

        U = np.array([0., 0., 0., 1., 1., 1.])
        V = np.array([0., 0., 0., 1., 1., 2., 2., 2.])

        r = 0.1 # radius of support
        p = 0.4 # vertical distance from support edge to domain edge

        m = p + r
        xs = [r, (r + ar) / 2, ar]
        ys = [p, p / 2, 0]

        cpoints = np.array([[[0, y], [x, y], [x, m], [x, 1 - y], [0, 1 - y]] for x, y in zip(xs, ys)])
        cpoints = np.swapaxes(cpoints, 0, 1)
        weights = np.array([[1., 1. / np.sqrt(2), 1., 1. / np.sqrt(2), 1.]] * 3).T


        plotrange = (0, ar, 0, 1)

    elif shape == "wheel":
        U = np.array([0., 0., 0., 1., 1., 1.])
        V = np.array([0., 0., 0., 1., 1., 1.])

        r = 0.05 # radius of "axis"

        locs = [r, (1 - r) / 2, 1]

        cpoints = np.array([[[l, 0], [l, l], [0, l]] for l in locs])
        cpoints = np.swapaxes(cpoints, 0, 1)
        weights = np.array([[1., 1. / np.sqrt(2), 1.]] * 3).T

        plotrange = (0, 1, 0, 1)

    else:
        return None

    return U, V, cpoints, weights, plotrange

def load(cpoints, nel0, nel1, ltype, params):
    free = np.ones_like(cpoints, dtype=np.bool)

    # first construct load arrays for elements on each side
    t0 = np.zeros((nel0, 2))
    t1 = np.zeros((nel1, 2))
    t2 = np.zeros_like(t0)
    t3 = np.zeros_like(t1)

    if ltype == "cantilever" and "ratio" in params:
        free[:, 0, :] = False

        nzero = int((1 - params["ratio"]) / 2 * nel1)
        nforce = nel1 - 2 * nzero

        t1[nzero:nzero + nforce, 1] = -1

    elif ltype == "cantileverflip" and "ratio" in params:
        free[:, -1, :] = False

        nzero = int((1 - params["ratio"]) / 2 * nel1)
        nforce = nel1 - 2 * nzero

        t3[nzero:nzero + nforce, 1] = -1

    elif ltype == "mbb":
        free[:, 0, 0] = False
        free[0, -1, 1] = False

        t2[0, 1] = -1

    elif ltype == "uniaxial":
        free[:, 0, :] = False

        nzero = int((1 - params["ratio"]) / 2 * nel1)
        nforce = nel1 - 2 * nzero

        t1[nzero:nzero + nforce, 0] = 1

    elif ltype == "linear":
        free[:, 0, 0] = False
        free[0, 0, :] = False

        l = t1.shape[0]
        for i in range(l):
            t1[i, 1] = 0.5 - float(i) / l

    elif ltype == "wheel":
        free[-1, :, 0] = False
        free[0, -1, 1] = False

        t3[:, 1] = 1

    '''# convert load arrays to span all elements
    t0 = np.zeros((nel0 * nel1, 2))
    t1 = np.zeros_like(t0)
    t2 = np.zeros_like(t0)
    t3 = np.zeros_like(t0)

    t0[:nel0, :] = t0side
    t1[nel0 - 1::nel0, :] = t1side
    t2[-nel0:, :] = t2side
    t3[::nel0, :] = t3side'''

    return free, (t0, t1, t2, t3)
