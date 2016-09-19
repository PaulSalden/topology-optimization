import sys, os.path, json, pickle, datetime
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sl
import geometry as gm
import algorithms as al
import routines as ro
import aid as ai

def replot(inputfile):
    # ---------- SETUP ----------

    # handle input of data
    with open(inputfile) as datafile:
        data = json.load(datafile)

    geometry = data["geometry"]
    U, V, cpoints, weights, plotrange = gm.nurbs(geometry["shape"], geometry["params"])
    tdrefine, sdrefine = geometry["refine"]
    optparams = data["optimization"]
    tarefine, sarefine = optparams["refine"]

    # assume open knot vectors
    p = np.count_nonzero(U == U[0]) - 1
    q = np.count_nonzero(V == V[0]) - 1

    # apply refinement
    projcpoints = np.dstack((cpoints * np.atleast_3d(weights), weights))
    projcpoints, Ud, Vd = al.refine(projcpoints, p, q, U, V, tdrefine, sdrefine)
    designweights = projcpoints[:, :, -1]
    projcpoints, U, V = al.refine(projcpoints, p, q, Ud, Vd, tarefine, sarefine)
    cpoints = projcpoints[..., :-1] / projcpoints[..., -1:]
    weights = projcpoints[:, :, -1]

    # geometric properties
    n = U.size - p - 1
    m = V.size - q - 1
    nd = Ud.size - p - 1
    md = Vd.size - q - 1
    nel0 = len(set(U)) - 1
    nel1 = len(set(V)) - 1

    # build localization tools
    C0 = al.bezier_extract(U, p, n)
    C1 = al.bezier_extract(V, q, m)
    Cd0 = al.bezier_extract(Ud, p, nd)
    Cd1 = al.bezier_extract(Vd, q, md)
    INN, IEN = al.build_inc_ien(U, V, p, q, n, m)
    INNd, IENd = al.build_inc_ien(Ud, Vd, p, q, nd, md)

    # convenience arrays
    IEN0 = INN[IEN][..., 0]
    IEN1 = INN[IEN][..., 1]
    IENd0 = INNd[IENd][..., 0]
    IENd1 = INNd[IENd][..., 1]

    # obtain shape arrays
    P = cpoints[IEN1, IEN0, :]
    W = weights[IEN1, IEN0, np.newaxis]

    # determine design/analysis conversion array and design basis
    darr = ro.designarray(nel0, nel1, tarefine, sarefine)
    Wd = designweights[IENd1, IENd0, np.newaxis]

    # retrieve filenames
    plot = data["plot"]
    plotfile, densfile, histfile = ro.filenames(optparams["filepath"], optparams["filename"],
                                                plot["extension"])

    # set up a starting density
    rhod = pickle.load(open(densfile, "rb"))

    # prepare x and y arrays for plotting
    plotpoints = np.linspace(0, 1 - 1E-6, plot["npoints"])
    Rdummy, X, Y = ro.plotprepare(p, q, nel0, nel1, C0, C1, P, W, plotpoints)

    # calculate basis functions for plotting
    Rp = ro.designbasis(p, q, nel0, nel1, Cd0, Cd1, Wd, plotpoints, plotpoints, tarefine, sarefine, darr)

    # plot density
    densdata = np.reshape(np.einsum("il, ilj -> ij", rhod, Rp), (-1, plot["npoints"], plot["npoints"]))
    ro.plot(X, Y, nel0, nel1, densdata, plotpoints, plot["width"], plotrange, plotfile)
#
#
#
if __name__ == "__main__":
    args = sys.argv
    if len(args) == 2 and os.path.isfile(args[1]):
        replot(args[1])

    # temp
    else:
        replot("input.txt")
