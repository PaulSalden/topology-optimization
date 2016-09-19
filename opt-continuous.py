import sys, os.path, json, pickle, datetime
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sl
import geometry as gm
import algorithms as al
import routines as ro
import aid as ai

def optimize(inputfile, mode=None):
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

    # handle loading situation
    load = data["load"]
    free, t = gm.load(cpoints, nel0, nel1, load["type"], load["params"])
    ndofs, ID = al.build_id(INN, free)

    # convenience arrays
    IEN0 = INN[IEN][..., 0]
    IEN1 = INN[IEN][..., 1]
    IENd0 = INNd[IENd][..., 0]
    IENd1 = INNd[IENd][..., 1]
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

    # define a way to calculate the normalized volume in use
    domainvolume = np.einsum("i, ji ->", intweights, J)
    def volume(rho):
        return np.einsum("i, jk, jki, ji", intweights, rho, R, J) / domainvolume

    # set up force vector for analysis
    '''wJ0 = np.einsum("j, ij -> ij", intweights0, J0)
    wJ1 = np.einsum("j, ij -> ij", intweights1, J1)
    wR0 = np.einsum("ijk, ik -> ij", R0, wJ0)
    wR1 = np.einsum("ijk, ik -> ij", R1, wJ1)'''
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
    Khat = np.einsum("ijmk, mn, ijnl-> ijkl", B, Dhat, B)

    # determine design/analysis conversion array and design basis
    darr = ro.designarray(nel0, nel1, tarefine, sarefine)
    Wd = designweights[IENd1, IENd0, np.newaxis]
    Rd = ro.designbasis(p, q, nel0, nel1, Cd0, Cd1, Wd, intpoints0, intpoints1, tarefine, sarefine, darr)

    # set up a starting density
    if optparams["startdens"]:
        rhod = pickle.load(open(optparams["startdens"], "rb"))
    else:
        rhod = np.full(P.shape[:-1], optparams["volfrac"])

    # compute A matrix for gradient determination
    A = ro.projectionmatrix(nd, md, IENd, intweights, Rd, J, darr)

    # prepare x and y arrays for plotting
    plot = data["plot"]
    plotpoints = np.linspace(0, 1 - 1E-6, plot["npoints"])
    Rdummy, X, Y = ro.plotprepare(p, q, nel0, nel1, C0, C1, P, W, plotpoints)

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

    rhovals = np.einsum("il, ilj -> ij", rhod, Rd)
    gd = np.empty_like(rhod)
    alpha = None

    while True:
        # compute and assemble stiffness matrix
        simpfac = min(1 + optparams["pperit"] * iteration, optparams["pmax"])
        simpmargin = 1E-6
        prho = (1 - simpmargin) * rhovals ** simpfac + simpmargin
        stiffdata = np.einsum("l, il, iljk, il -> ijk", intweights, prho, Khat, J)
        K = ro.stiffness(IEN, IDE, ndofs, stiffdata)

        # solve for displacement (ensuring symmetry for efficiency)
        K = (K + K.transpose()) / 2.
        umat = sl.spsolve(K, f)
        u = umat[IDE]
        u[IDE == -1] = 0

        # determine compliance and matrix/vector for gradient computation
        #eps = np.einsum("ijkl, il -> ijk", B, np.reshape(u, (IEN.shape[0], -1)))
        prhodiff = (1 - simpmargin) * simpfac * rhovals ** (simpfac - 1)
        compliance, b = ro.projectionvector(nd, md, IENd, intweights, Rd, J, Khat, prho, prhodiff, u, darr)
        compliances.append(compliance)
        # alternatively, K could be used to determine compliance

        # solve for gradient
        A = (A + A.transpose()) / 2.
        gdmat = sl.spsolve(A, b)
        gd[darr, ...] = gdmat[IENd][:, np.newaxis, :]

        # update density
        if not alpha:
            # step size
            change_ratio = 1
            #alpha = np.abs(rho.mean() / g.mean() * change_ratio)
            alpha = np.abs(np.linalg.norm(rhod) / np.linalg.norm(gd) * change_ratio)
        rhod, rhodold = ro.update(volume, optparams["volfrac"], rhod, gd, alpha)
        rhovals = np.einsum("il, ilj -> ij", rhod, Rd)

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

    # back up density, write compliance history
    pickle.dump(rhod, open(densfile, "wb"))
    pickle.dump({"compliances": compliances, "n_it": iteration, "time": totaltime},
                open(histfile, "wb"))

    # calculate basis functions for plotting
    Rp = ro.designbasis(p, q, nel0, nel1, Cd0, Cd1, Wd, plotpoints, plotpoints, tarefine, sarefine, darr)

    # ---------- OPTIONAL GRADIENT PLOT ----------
    #     use only after first iteration
    if "airy" in optparams:
        graddata = np.reshape(np.einsum("il, ilj -> ij", gd, Rp), (-1, plot["npoints"], plot["npoints"]))
        effE = E * (optparams["volfrac"] ** 2)
        airydata = ai.airystress(X, Y-0.5, load["force"][3], 0.5, 1, effE, nu)
        ai.gradplot(X, Y, nel0, nel1, graddata, airydata, plotpoints, plot["width"], plotrange, plotfile)
        sys.exit()

    # plot density
    densdata = np.reshape(np.einsum("il, ilj -> ij", rhod, Rp), (-1, plot["npoints"], plot["npoints"]))
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