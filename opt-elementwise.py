import sys, os.path, json, pickle, datetime
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sl
import geometry as gm
import algorithms as al
import routines as ro
import aid as ai

def optimize(inputfile):
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
    projcpoints, U, V = al.refine(projcpoints, p, q, U, V, tdrefine, sdrefine)
    neld0 = len(set(U)) - 1
    neld1 = len(set(V)) - 1
    projcpoints, U, V = al.refine(projcpoints, p, q, U, V, tarefine, sarefine)
    cpoints = projcpoints[..., :-1] / projcpoints[..., -1:]
    weights = projcpoints[:, :, -1]

    # geometric properties
    n = U.size - p - 1
    m = V.size - q - 1
    nel0 = len(set(U)) - 1
    nel1 = len(set(V)) - 1

    # build localization tools
    C0 = al.bezier_extract(U, p, n)
    C1 = al.bezier_extract(V, q, m)
    INN, IEN = al.build_inc_ien(U, V, p, q, n, m)

    # handle loading situation
    load = data["load"]
    free, t = gm.load(cpoints, nel0, nel1, load["type"], load["params"])
    ndofs, ID = al.build_id(INN, free)

    # convenience arrays
    IEN0 = INN[IEN][..., 0]
    IEN1 = INN[IEN][..., 1]
    IDE = ID[IEN]

    # determine integration points and weights (map from [-1, 1] to [0, 1])
    intpoints0, intweights0 = np.polynomial.legendre.leggauss(p + 1)
    intpoints1, intweights1 = np.polynomial.legendre.leggauss(q + 1)
    intpoints0 = (intpoints0 + 1.) / 2.
    intpoints1 = (intpoints1 + 1.) / 2.
    intweights0 /= 2
    intweights1 /= 2
    intweights = np.reshape(np.einsum("j, i", intweights0, intweights1), -1)

    # obtain shape arrays
    P = cpoints[IEN1, IEN0, :]
    W = weights[IEN1, IEN0, np.newaxis]
    R, Rb, dRdX, J, Jb = ro.shape(p, q, C0, C1, nel0, P, W, intpoints0, intpoints1)
    # it would be more efficient to work with a Rd for plotting as well

    # set up force vector for analysis
    f = ro.force(n, m, IDE, IEN0, IEN1, intweights0, intweights1, ndofs, Rb, Jb, t, load["force"])

    # calculate the "B matrices"
    B = ro.computeB(R, dRdX)

    # compute the "ground D matrix", assuming plain stress
    material = data["material"]
    E = material["E"]
    nu = material["nu"]
    Dhat =  E / (1 - nu*nu) * np.array([
        [1, nu, 0],
        [nu, 1, 0],
        [0, 0, (1 - nu) / 2.]
    ])

    # calculate the "ground" stiffness matrices
    Khat = np.einsum("l, ilmj, mn, ilnk, il -> ijk", intweights, B, Dhat, B, J)

    # set up design/analysis conversion array and starting density
    darr = ro.designarray(nel0, nel1, tarefine, sarefine)
    if optparams["startdens"]:
        rhod = pickle.load(open(optparams["startdens"], "rb"))
    else:
        rhod = np.full(neld0 * neld1, optparams["volfrac"])

    # define a way to calculate the normalized volume in use
    elementvolume = np.einsum("j, ij -> i", intweights, J)
    elementvolume = np.sum(elementvolume[darr], axis=1)
    elementvolume /= np.sum(elementvolume)
    def volume(rhod):
        return np.einsum("i, i", rhod, elementvolume)

    # prepare x and y arrays for plotting
    plot = data["plot"]
    plotpoints = np.linspace(0, 1 - 1E-6, plot["npoints"])
    Rp, X, Y = ro.plotprepare(p, q, nel0, nel1, C0, C1, P, W, plotpoints)

    # define filter method
    if optparams["rmin"]:
        H, Hs = al.density_filter(neld0, neld1, optparams["rmin"])

        def densfilter(x):
            return H.dot(x / Hs)
    else:
        def densfilter(x):
            return x

    # retrieve filenames
    plotfile, densfile, histfile = ro.filenames(optparams["filepath"], optparams["filename"],
                                                plot["extension"])



    # ---------- OPTIONAL DOMAIN PLOT ----------
    if "domain" in optparams:
        ai.plotdomain(p, q, nel0, nel1, C0, C1, P, W, plotpoints, plot["width"], plotrange, plotfile)
        sys.exit()

    # ---------- OPTIMIZATION ----------

    iteration = 0
    compliances = []
    fullit = (optparams["pmax"] - 1) / optparams["pperit"]
    starttime = datetime.datetime.now()

    rho = np.zeros(P.shape[0])
    rho[darr] = rhod[:, np.newaxis]
    alpha = None

    while True:
        # compute and assemble stiffness matrix
        simpfac = min(1 + optparams["pperit"] * iteration, optparams["pmax"])
        simpmargin = 1E-6
        prho = (1 - simpmargin) * rho ** simpfac + simpmargin
        stiffdata = np.einsum("i, ijk -> ijk", prho, Khat)
        K = ro.stiffness(IEN, IDE, ndofs, stiffdata)

        # solve for displacement (ensuring symmetry for efficiency)
        K = (K + K.transpose()) / 2.
        umat = sl.spsolve(K, f)
        u = umat[IDE]
        u[IDE == -1] = 0

        # determine compliance and gradient
        #eps = np.einsum("ijkl, il -> ijk", B, np.reshape(u, (IEN.shape[0], -1)))
        prhodiff = (1 - simpmargin) * simpfac * rho ** (simpfac - 1)
        u1d = np.reshape(u, (IEN.shape[0], -1))
        Chat = np.einsum("ij, ijk, ik -> i", u1d, Khat, u1d)
        compliance = np.einsum("i, i", prho, Chat)
        compliances.append(compliance)
        # alternatively, K could be used to determine compliance
        g = -prhodiff * Chat

        # sum up gradients per "design element" and adjust original values accordingly
        gd = densfilter(np.sum(g[darr], axis=1))

        """if iteration > 20:
            densdata = np.reshape(np.einsum("il, ilj -> ij", g, Rp), (-1, 10, 10))
            densdata[np.abs(densdata) < 1E-3*np.abs(densdata).mean()] = 0
            densdata = np.clip(densdata, 0, 3*densdata.mean())
            ro.plot(X, Y, p, q, nel0, nel1, densdata)"""

        #g[np.abs(g) < 1E-3 * np.abs(g).mean()] = 0

        # update density
        if not alpha:
            # step size
            change_ratio = 0.1
            #alpha = np.abs(rho.mean() / g.mean() * change_ratio)
            alpha = np.abs(np.linalg.norm(rhod) / np.linalg.norm(gd) * change_ratio)
        rhod, rhodold = ro.update(volume, optparams["volfrac"], rhod, gd, alpha, densfilter)
        rho[darr] = rhod[:, np.newaxis]

        # temp
        print datetime.datetime.now().strftime("%H:%M:%S"), iteration, compliance

        # provide stop conditions
        if iteration >= optparams["maxit"] - 1:
            break
        if iteration >= fullit + 30:
            if np.max(np.absolute(rhod - rhodold)) <= 1E-3:
                break
            #if np.average(compliances[-10:]) >= np.average(compliances[-30:]):
            #    break

        iteration += 1

    # temp
    totaltime = datetime.datetime.now() - starttime
    print "done in {} seconds".format(totaltime.total_seconds())

    # back up element density, write compliance history
    pickle.dump(rhod, open(densfile, "wb"))
    pickle.dump({"compliances": compliances, "n_it": iteration, "time": totaltime},
                open(histfile, "wb"))

    # ---------- OPTIONAL GRADIENT PLOT ----------
    #     use only after first iteration
    if "airy" in optparams:
        g[darr] = gd
        graddata = np.tile(g[:, np.newaxis, np.newaxis], (1, plot["npoints"], plot["npoints"]))
        XA, XB, YA, YB = ai.elementlimits(nel0, nel1, 1. * geometry["params"]["aspectratio"] / nel0, 1. / nel1)
        effE = E * (optparams["volfrac"] ** 2)
        airydata = ai.airystressint(XA, XB, YA - 0.5, YB - 0.5, load["force"][3], 0.5, 1, effE, nu)
        airydata = np.tile(airydata[:, np.newaxis, np.newaxis], (1, plot["npoints"], plot["npoints"]))
        ai.elgradplot(X, Y, nel0, nel1, graddata, airydata, plotpoints, plot["width"], plotrange, plotfile)
        sys.exit()

    # plot density
    densdata = np.tile(rho[:, np.newaxis, np.newaxis], (1, plot["npoints"], plot["npoints"]))
    ro.plot(X, Y, nel0, nel1, densdata, plotpoints, plot["width"], plotrange, plotfile)
#
#
#
if __name__ == "__main__":
    args = sys.argv
    if len(args) == 2 and os.path.isfile(args[1]):
        optimize(args[1])

    # temp
    else:
        optimize("input.txt")
