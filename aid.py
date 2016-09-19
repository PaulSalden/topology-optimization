import numpy as np
import scipy.sparse as sp
import matplotlib, sympy
matplotlib.use('Agg')
import matplotlib.pyplot as pl
import algorithms as al

def plotdomain(p, q, nel0, nel1, C0, C1, P, W, plotpoints, plotwidth, plotrange, plotfile):
    """
    Generate an illustration of the domain
    """
    plotheight = 1. * plotwidth / (plotrange[1] - plotrange[0]) * (plotrange[3] - plotrange[2])

    pgf_with_rc_fonts = {
        "font.family": "Arial",
        "font.serif": [],
        "font.sans-serif": ["Arial"],
        "font.size": 8,
        "figure.dpi": 600,
    }
    matplotlib.rcParams.update(pgf_with_rc_fonts)

    fig = pl.figure(figsize=(plotwidth, plotheight))

    # plot control points
    radius = float(min(abs(plotrange[1] - plotrange[0]), abs(plotrange[3] - plotrange[2]))) / 40
    pointweights = np.dstack((P, W / np.amax(W)))
    points = set([tuple(point) for point in pointweights.reshape(-1, 3).tolist()])
    for point in points:
        circle = pl.Circle(point[:2], radius, color=pl.cm.Blues(point[2]))
        fig.gca().add_artist(circle)

    # plot surface
    l = plotpoints.size

    B0 = al.bernstein(p, plotpoints)
    B1 = al.bernstein(q, plotpoints)
    N0 = np.einsum("ijl, kl", C0, B0)
    N1 = np.einsum("ijl, kl", C1, B1)

    X = np.zeros((l, 2 * (nel0 + 1) * (nel1 + 1)))
    Y = np.zeros_like(X)
    el = 0
    for j in range(nel1):
        for i in range(nel0):
            # "horizontal" line
            N = np.reshape(np.einsum("i, jk -> ijk", N1[j, :, 0], N0[i, ...]), (-1, l))
            R = N * W[el, :]
            R /= np.sum(R, axis=0, keepdims=True)
            points = np.einsum("ik, ij", P[el, ...], R)
            X[:, 2 * el] = points[:, 0]
            Y[:, 2 * el] = points[:, 1]

            # "vertical" line
            N = np.reshape(np.einsum("ik, j -> ijk", N1[j, ...], N0[i, :, 0]), (-1, l))
            R = N * W[el, :]
            R /= np.sum(R, axis=0, keepdims=True)
            points = np.einsum("ik, ij", P[el, ...], R)
            X[:, 2 * el + 1] = points[:, 0]
            Y[:, 2 * el + 1] = points[:, 1]

            # add "horizontal" end
            if j == nel1 - 1:
                N = np.reshape(np.einsum("i, jk -> ijk", N1[j, :, -1], N0[i, ...]), (-1, l))
                R = N * W[el, :]
                R /= np.sum(R, axis=0, keepdims=True)
                points = np.einsum("ik, ij", P[el, ...], R)
                spot = 2 * (nel0 * nel1 + i)
                X[:, spot] = points[:, 0]
                Y[:, spot] = points[:, 1]

            # add "vertical" end
            if i == nel0 - 1:
                N = np.reshape(np.einsum("ik, j -> ijk", N1[j, ...], N0[i, :, -1]), (-1, l))
                R = N * W[el, :]
                R /= np.sum(R, axis=0, keepdims=True)
                points = np.einsum("ik, ij", P[el, ...], R)
                spot = 2 * (nel0 * nel1 + j) + 1
                X[:, spot] = points[:, 0]
                Y[:, spot] = points[:, 1]

            el += 1

    pl.plot(X, Y, 'k--')  # cmap=pl.cm.viridis_r)
    # pl.colorbar()
    pl.axis('equal')
    pl.axis(plotrange)
    #pl.show()
    pl.savefig(plotfile, bbox_inches="tight", dpi="figure")



# ---------- CONTINUOUS ----------

def airystress(X, Y, P, a, b, E, nu):
    """
    Calculate strain energy values based on an Airy stress function
    """
    x1, x2 = sympy.symbols("x_1, x_2")

    sigma = sympy.Matrix([
        [3 * P / (2 * a**3 * b) * x1 * x2],
        [0],
        [3 * P / (4 * a * b) * (1 - (x2**2) / (a**2))]
    ])

    C = 3 * P / (4 * E * a**3 * b)
    epsilon = sympy.Matrix([
        [2 * C * x1 * x2],
        [-2 * nu * C * x1 * x2],
        [2 * (1 + nu) * C * (a**2 - x2**2)]
    ])

    SE = sympy.lambdify((x1, x2), sigma.dot(epsilon))

    return SE(X, Y)

def gradplot(X, Y, nel0, nel1, graddata, airydata, plotpoints, plotwidth, plotrange, plotfile):
    """
    Plot the gradient.
    """
    l = plotpoints.size

    Z = np.zeros_like(X)
    el = 0
    for j in range(nel1):
        for i in range(nel0):
            irange = slice(i * l, (i + 1) * l)
            jrange = slice(j * l, (j + 1) * l)

            Z[jrange, irange] = graddata[el, ...]

            el += 1

    Z += airydata
    #Z = airydata

    plotheight = 1. * plotwidth / (plotrange[1] - plotrange[0]) * (plotrange[3] - plotrange[2])

    pgf_with_rc_fonts = {
        "font.family": "Arial",
        "font.serif": [],
        "font.sans-serif": ["Arial"],
        "font.size": 8,
        "figure.dpi": 600,
    }
    matplotlib.rcParams.update(pgf_with_rc_fonts)

    pl.figure(figsize=(plotwidth, plotheight))
    pl.contourf(X, Y, Z, 10, cmap=pl.cm.Blues)#cmap=pl.cm.viridis_r)
    pl.colorbar()
    pl.axis('equal')
    pl.axis(plotrange)
    #pl.show()
    pl.savefig(plotfile, bbox_inches="tight", dpi="figure")



# ---------- ELEMENT WISE ----------

def elementlimits(nel0, nel1, w, h):
    XA = np.empty(nel0 * nel1)
    XB = np.empty_like(XA)
    YA = np.empty_like(XA)
    YB = np.empty_like(XA)

    el = 0
    for j in range(nel1):
        for i in range(nel0):
            XA[el] = i * w
            XB[el] = (i + 1) * w
            YA[el] = j * h
            YB[el] = (j + 1) * h

            el += 1

    return XA, XB, YA, YB

def airystressint(XA, XB, YA, YB, P, a, b, E, nu):
    """
    Calculate integrals of strain energy values based on an Airy stress function
    """
    x1, x2, xa, xb, ya, yb = sympy.symbols("x_1, x_2 x_a x_b y_a y_b")

    sigma = sympy.Matrix([
        [3 * P / (2 * a**3 * b) * x1 * x2],
        [0],
        [3 * P / (4 * a * b) * (1 - (x2**2) / (a**2))]
    ])

    C = 3 * P / (4 * E * a**3 * b)
    epsilon = sympy.Matrix([
        [2 * C * x1 * x2],
        [-2 * nu * C * x1 * x2],
        [2 * (1 + nu) * C * (a**2 - x2**2)]
    ])

    SE = sigma.dot(epsilon)
    SEi = sympy.lambdify((xa, xb, ya, yb), SE.integrate((x1, xa, xb), (x2, ya, yb)))

    return SEi(XA, XB, YA, YB)

def elgradplot(X, Y, nel0, nel1, graddata, airydata, plotpoints, plotwidth, plotrange, plotfile):
    """
    Plot the gradient.
    """
    l = plotpoints.size

    Z = np.zeros_like(X)
    el = 0
    for j in range(nel1):
        for i in range(nel0):
            irange = slice(i * l, (i + 1) * l)
            jrange = slice(j * l, (j + 1) * l)

            #Z[jrange, irange] = graddata[el, ...] + airydata[el, ...]
            Z[jrange, irange] = airydata[el, ...]

            el += 1

    plotheight = 1. * plotwidth / (plotrange[1] - plotrange[0]) * (plotrange[3] - plotrange[2])

    pgf_with_rc_fonts = {
        "font.family": "Arial",
        "font.serif": [],
        "font.sans-serif": ["Arial"],
        "font.size": 8,
        "figure.dpi": 600,
    }
    matplotlib.rcParams.update(pgf_with_rc_fonts)

    pl.figure(figsize=(plotwidth, plotheight))
    pl.contourf(X, Y, Z, 10, cmap=pl.cm.Blues)#cmap=pl.cm.viridis_r)
    pl.colorbar()
    pl.axis('equal')
    pl.axis(plotrange)
    #pl.show()
    pl.savefig(plotfile, bbox_inches="tight", dpi="figure")

