import numpy as np
import scipy.sparse as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pl
import algorithms as al
import os.path

def shape(p, q, C0, C1, nel0, P, W, intpoints0, intpoints1):
    """
    Shape function routine.
    """
    # set up bernstein basis and derivative values
    B0 = al.bernstein(p, intpoints0)
    B1 = al.bernstein(q, intpoints1)
    dBdt0 = al.bernstein_deriv(p, intpoints0)
    dBdt1 = al.bernstein_deriv(q, intpoints1)

    # compute unidirectional element bpsline basis values
    #   (element * function * int. point)
    N0 = np.einsum("ijl, kl", C0, B0)
    N1 = np.einsum("ijl, kl", C1, B1)
    dNdt0 = np.einsum("ijl, kl", C0, dBdt0)
    dNdt1 = np.einsum("ijl, kl", C1, dBdt1)

    # compute element basis functions and derivatives
    arrshape = np.array(N0.shape) * np.array(N1.shape)
    R = np.reshape(np.einsum("jln, ikm", N0, N1), arrshape) * W
    w = R.sum(axis=1, keepdims=True)
    R /= w

    dRdt = np.reshape(np.einsum("jln, ikm", dNdt0, N1), arrshape) * W
    dRds = np.reshape(np.einsum("jln, ikm", N0, dNdt1), arrshape) * W
    #   (element * function * int. point * deriv. param.)
    dRdT = np.concatenate((dRdt[..., np.newaxis], dRds[..., np.newaxis]), axis=-1)
    dwdT = dRdT.sum(axis=1, keepdims=True)
    dRdT = (dRdT - R[..., np.newaxis] * dwdT) / w[..., np.newaxis]

    # calculate Jacobian
    dXdT = np.einsum("imk, imjl -> ijkl", P, dRdT)
    #   (element * int. point)
    J = abs(np.linalg.det(dXdT))

    # determine physical derivatives
    dTdX = np.linalg.inv(dXdT)
    #   (element * function * int. point * phys. deriv. dim.)
    dRdX = np.einsum("ijkm, ikml -> ijkl", dRdT, dTdX)

    '''# repeat for just dimension 0
    R0 = np.tile(N0, (N1.shape[0], N1.shape[1], 1)) * W
    w = R0.sum(axis=1, keepdims=True)
    R0 /= w
    dRdt0 = np.tile(dNdt0, (N1.shape[0], N1.shape[1], 1)) * W
    dwdt = dRdt0.sum(axis=1, keepdims=True)
    dRdt0 = (dRdt0 - R0 * dwdt) / w
    dXdt0 = np.einsum("imk, imj -> ijk", P, dRdt0)
    #   slight abuse of notation
    J0 = np.linalg.norm(dXdt0, axis=-1)

    # repeat for just dimension 1
    R1 = np.repeat(np.repeat(N1, N0.shape[0], axis=0), N0.shape[1], axis=1) * W
    w = R1.sum(axis=1, keepdims=True)
    R1 /= w
    dRdt1 = np.repeat(np.repeat(dNdt1, N0.shape[0], axis=0), N0.shape[1], 1) * W
    dwdt = dRdt1.sum(axis=1, keepdims=True)
    dRdt1 = (dRdt1 - R1 * dwdt) / w
    dXdt1 = np.einsum("imk, imj -> ijk", P, dRdt1)
    J1 = np.linalg.norm(dXdt1, axis=-1)'''

    # repeat for borders
    def bslice(side, w):
        return {
            0: slice(None, w),
            1: slice(w - 1, None, w),
            2: slice(-w, None),
            3: slice(None, None, w),
        }[side]

    def bordershape(side, N, dNdt):
        ind = bslice(side, nel0), bslice(side, p + 1), slice(None)
        Pb = P[ind]
        Wb = W[ind]

        R = N * Wb
        w = R.sum(axis=1, keepdims=True)
        R /= w

        dRdt = dNdt * Wb
        dwdt = dRdt.sum(axis=1, keepdims=True)
        dRdt = (dRdt - R * dwdt) / w
        dXdt = np.einsum("imk, imj -> ijk", Pb, dRdt)
        #   slight abuse of notation
        J = np.linalg.norm(dXdt, axis=-1)

        return R, J

    R0, J0 = bordershape(0, N0, dNdt0)
    R1, J1 = bordershape(1, N1, dNdt1)
    R2, J2 = bordershape(2, N0, dNdt0)
    R3, J3 = bordershape(3, N1, dNdt1)

    return R, (R0, R1, R2, R3), dRdX, J, (J0, J1, J2, J3)

def force(n, m, IDE, IEN0, IEN1, intweights0, intweights1, ndofs, Rb, Jb, t, forceval):
    """
    Force vector routine.
    """
    def sideforce(side, intweights, ind):
        wJ = np.einsum("j, ij -> ij", intweights, Jb[side])
        activelength = np.sum(wJ[np.any(t[side], axis=1)])

        dat = np.reshape(np.einsum("ik, ijl, il -> ijk", t[side], Rb[side], wJ), -1)
        if activelength:
            dat *= forceval[side] / activelength

        row = np.reshape(IDE[ind], -1)

        return row, dat

    row0, dat0 = sideforce(0, intweights0, IEN1 == 0)
    row1, dat1 = sideforce(1, intweights1, IEN0 == n - 1)
    row2, dat2 = sideforce(2, intweights0, IEN1 == m - 1)
    row3, dat3 = sideforce(3, intweights1, IEN0 == 0)

    '''# right side
    dat1 = np.einsum("ik, ij -> ijk", t1, wR1)
    ind = IEN0 == n - 1
    row1 = np.reshape(IDE[ind], -1)
    activelength = np.sum((t1.any(axis=1)[ind, np.newaxis] * wJ1)[ind])
    dat1 = np.reshape(dat1[ind], -1)

    # top side
    dat2 = np.einsum("ik, ij -> ijk", t2, wR0)
    ind = IEN1 == m - 1
    row2 = np.reshape(IDE[ind], -1)
    dat2 = np.reshape(dat2[ind], -1)

    # left side
    dat3 = np.einsum("ik, ij -> ijk", t3, wR1)
    ind = IEN0 == 0
    row3 = np.reshape(IDE[ind], -1)
    dat3 = np.reshape(dat3[ind], -1)'''

    # combine sides
    row = np.concatenate((row0, row1, row2, row3))
    dat = np.concatenate((dat0, dat1, dat2, dat3))
    ind = row != -1
    row = row[ind]
    dat = dat[ind]

    return sp.coo_matrix((dat, (row, np.zeros_like(row))), (ndofs, 1)).tocsr()

def computeB(R, dRdX):
    """
    Compute the B matrix.
    """
    Bi = np.zeros((R.shape[0], R.shape[1], R.shape[2], 3, 2))
    Bi[..., 0, 0] = dRdX[..., 0]
    Bi[..., 1, 1] = dRdX[..., 1]
    Bi[..., 2, 0] = dRdX[..., 1]
    Bi[..., 2, 1] = dRdX[..., 0]

    B = np.zeros((R.shape[0], R.shape[2], 3, 2 * R.shape[1]))
    for i in range(R.shape[1]):
        B[..., i * 2:i * 2 + 2] = Bi[:, i, ...]

    return B

def stiffness(IEN, IDE, ndofs, stiffdata):
    """
    Stiffness matrix routine.
    """
    dat = np.reshape(stiffdata, (IEN.shape[0], -1))

    row = np.reshape(IDE, (IEN.shape[0], -1))
    col = np.repeat(row, row.shape[1], axis=1)
    row = np.tile(row, (1, row.shape[1]))

    ind = np.logical_and(row != -1, col != -1)

    dat = dat[ind]
    row = row[ind]
    col = col[ind]

    return sp.coo_matrix((dat, (row, col)), (ndofs, ndofs)).tocsr()

def projectionmatrix(nd, md, IENd, intweights, Rd, J, darr):
    """
    Projection matrix routine.
    """
    amatrices = np.einsum("l, ijl, ikl, il -> ijk", intweights, Rd, Rd, J)

    dat = np.reshape(np.sum(amatrices[darr, ...], axis=1), -1)

    row = np.reshape(np.tile(IENd, (1, IENd.shape[1])), -1)
    col = np.reshape(np.repeat(IENd, IENd.shape[1], axis=1), -1)

    matsize = nd * md
    return sp.coo_matrix((dat, (row, col)), (matsize, matsize)).tocsr()

def projectionvector(nd, md, IENd, intweights, Rd, J, Khat, prho, prhodiff, u, darr):
    """
    Projection vector routine.
    """
    u1d = np.reshape(u, (J.shape[0], -1))
    Chat = np.einsum("ik, ijkl, il -> ij", u1d, Khat, u1d)
    compliance = np.einsum("i, ji, ji, ji ->", intweights, prho, Chat, J)

    amatrices = np.einsum("k, ik, ik, ijk, ik -> ij", intweights, prhodiff, Chat, Rd, J)

    dat = -np.reshape(np.sum(amatrices[darr, ...], axis=1), -1)
    row = np.reshape(IENd, -1)

    matsize = nd * md
    return compliance, sp.coo_matrix((dat, (row, np.zeros_like(row))), (matsize, 1)).tocsr()

def designarray(nel0, nel1, tarefine, sarefine):
    """
    Design/analysis conversion array routine.
    """
    ndel0 = nel0 / tarefine
    ndel = ndel0 * nel1 / sarefine
    darr = np.empty((ndel, tarefine * sarefine), dtype=np.int)

    for _del in range(ndel):
        i = _del % ndel0
        j = _del / ndel0

        iels = np.arange(i * tarefine, (i + 1) * tarefine)
        jels = np.arange(j * sarefine, (j + 1) * sarefine)

        ii, jj = np.meshgrid(iels, jels)

        darr[_del, :] = np.reshape(jj * nel0 + ii, -1)

    return darr

def designbasis(p, q, nel0, nel1, Cd0, Cd1, Wd, intpoints0, intpoints1, tarefine, sarefine, darr):
    """
    Design basis for analysis routine.
    """
    # determine integration points for design elements
    tpoints = np.empty(intpoints0.size * tarefine)
    spoints = np.empty(intpoints1.size * sarefine)
    for i in range(tarefine):
        tpoints[i * intpoints0.size:(i + 1) * intpoints0.size] = (intpoints0 + i) / tarefine
    for j in range(sarefine):
        spoints[j * intpoints1.size:(j + 1) * intpoints1.size] = (intpoints1 + j) / sarefine

    # determine design basis values
    B0 = al.bernstein(p, tpoints)
    B1 = al.bernstein(q, spoints)
    N0 = np.einsum("ijl, kl", Cd0, B0)
    N1 = np.einsum("ijl, kl", Cd1, B1)
    arrshape = (-1, N0.shape[1] * N1.shape[1], N1.shape[2], N0.shape[2])
    dbasis = np.reshape(np.einsum("jln, ikm", N0, N1), arrshape) * Wd[..., np.newaxis]
    dbasis /= np.sum(dbasis, axis=1, keepdims=True)

    # split values up over analysis elements
    Rd = np.empty((nel0 * nel1, dbasis.shape[1], intpoints0.size * intpoints1.size))
    for _del in range(darr.shape[0]):
        for el, pos in enumerate(darr[_del, :]):
            i = el % tarefine
            j = el / tarefine

            irange = slice(i * intpoints0.size, (i + 1) * intpoints0.size)
            jrange = slice(j * intpoints1.size, (j + 1) * intpoints1.size)

            vals = dbasis[_del, :, jrange, irange]
            Rd[pos, ...] = np.reshape(vals, (dbasis.shape[1], -1))

    return Rd

def update(volume, volfrac, rho, g, alpha, densfilter=None):
    """
    Density update routine.
    """
    rhoold = rho

    # Lagrange multiplier
    Lambda = 0

    # iterate while global volume constraint is violated
    while True:
        # update while satisfying local constraints
        rho = np.clip(rhoold - alpha * (g + Lambda), 1E-6, 1)

        if densfilter:
            rho = densfilter(rho)

        deltaLambda = max(0, (volume(rho) - volfrac) / alpha)
        
        if deltaLambda == 0:
            break

        Lambda += deltaLambda

    return rho, rhoold

def plotprepare(p, q, nel0, nel1, C0, C1, P, W, plotpoints):
    """
    Build re-usable X and Y arrays for plotting.
    """
    l = plotpoints.size

    B0 = al.bernstein(p, plotpoints)
    B1 = al.bernstein(q, plotpoints)
    N0 = np.einsum("ijl, kl", C0, B0)
    N1 = np.einsum("ijl, kl", C1, B1)
    arrshape = np.array(N0.shape) * np.array(N1.shape)
    Rp = np.reshape(np.einsum("jln, ikm", N0, N1), arrshape) * W
    Rp /= np.sum(Rp, axis=1, keepdims=True)
    physdata = np.reshape(np.einsum("ilk, ilj -> ijk", P, Rp), (-1, l, l, 2))

    X = np.zeros((nel1 * l, nel0 * l))
    Y = np.zeros_like(X)
    el = 0
    for j in range(nel1):
        for i in range(nel0):
            irange = slice(i * l, (i + 1) * l)
            jrange = slice(j * l, (j + 1) * l)

            X[jrange, irange] = physdata[el, ..., 0]
            Y[jrange, irange] = physdata[el, ..., 1]

            el += 1

    return Rp, X, Y

def plot(X, Y, nel0, nel1, densdata, plotpoints, plotwidth, plotrange, plotfile):
    """
    Plot the density.
    """
    l = plotpoints.size

    Z = np.zeros_like(X)
    el = 0
    for j in range(nel1):
        for i in range(nel0):
            irange = slice(i * l, (i + 1) * l)
            jrange = slice(j * l, (j + 1) * l)

            Z[jrange, irange] = densdata[el, ...]

            el += 1

    plotheight = 1. * plotwidth / (plotrange[1] - plotrange[0]) * (plotrange[3] - plotrange[2])

    #pgf_with_rc_fonts = {
    #    "text.usetex": True,
    #    "pgf.rcfonts": False,
    #}
    pgf_with_rc_fonts = {
        "font.family": "Arial",
        "font.serif": [],
        "font.sans-serif": ["Arial"],
        "font.size": 8,
        "figure.dpi": 600,
    }
    matplotlib.rcParams.update(pgf_with_rc_fonts)

    pl.figure(figsize=(plotwidth, plotheight))
    Z = np.clip(Z, 0, 1)
    pl.contourf(X, Y, Z, 10, cmap=pl.cm.Blues)#cmap=pl.cm.viridis_r)
    #pl.colorbar()
    pl.axis('equal')
    pl.axis(plotrange)
    #pl.show()
    pl.savefig(plotfile, bbox_inches="tight", dpi="figure")

def filenames(base, name, plotext):
    density_path = "dens/"
    history_path = "hist/"
    pickle_ext = ".p"

    files = (
        os.path.join(base, name + plotext),
        os.path.join(base, density_path, name + pickle_ext),
        os.path.join(base, history_path, name + pickle_ext)
    )

    for file in files:
        if not os.path.exists(os.path.dirname(file)):
            os.makedirs(os.path.dirname(file))

    return files