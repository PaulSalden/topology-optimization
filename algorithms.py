import numpy as np
import scipy.sparse as sp

def bernstein(p, t):
    """
    Evaluate Bernstein polynomials.

    Args:
        p: polynomial order
        t: locations array

    Returns:
        polynomial values (locations x functions)

    Raises:
        -
    """
    out = np.ones((t.size, 1))

    for n in range(p):
        a = np.zeros((t.size, n + 2))
        a[:, :-1] = out

        b = np.zeros_like(a)
        b[:, 1:] = out

        t = t.reshape((-1, 1))
        out = (1 - t) * a + t * b

    return out

def bernstein_deriv(p, t):
    """
    Evaluate Bernstein polynomial derivatives.

    Args:
        p: polynomial order
        t: locations array

    Returns:
        polynomial derivative values (locations x functions)
    """
    out = bernstein(p - 1, t)

    a = np.zeros((t.size, p + 1))
    a[:, :-1] = out

    b = np.zeros_like(a)
    b[:, 1:] = out

    return p * (b - a)

def refine(cpoints, p, q, U, V, tfac, sfac):
    """
    Refine knot vectors and compute control points such that geometry remains the same.

    Args:
        cpoints: control points array (second dir. x first dir. x physical dir.)
        p: polynomial order in first direction
        q: polynomial order in second direction
        U: knot vector in first direction
        V: knot vector in second direction
        tfac: refinement factor in first direction
        sfac: refinement factor in second direction

    Returns:
        new cpoints, U and V (same structures)
    """
    def alphamatrix(ncp, U, p, t, k):
        def alpha(A):
            if A <= k - p:
                return 1
            if A >= k + 1:
                return 0
            return (t - U[A]) / (U[A + p] - U[A])

        alphas = np.empty(ncp + 1)
        for i in range(len(alphas)):
            alphas[i] = alpha(i)
        
        out = np.zeros((ncp + 1, ncp))
        out[:-1, :] = np.diag(alphas[:-1])
        out[1:, :] += np.diag(1 - alphas[1:])

        return out

    # insert knots in first direction
    Uold = np.copy(U)
    for i in range(len(Uold) - 1):
        dist = Uold[i + 1] - Uold[i]
        if dist == 0:
            continue

        for inc in np.linspace(0, dist, tfac, endpoint=False)[1:]:
            t = Uold[i] + inc
            k = np.searchsorted(U, t) - 1
            almat = alphamatrix(cpoints.shape[1], U, p, t, k)
            cpoints = np.einsum("ij, kjl -> kil", almat, cpoints)
            U = np.insert(U, k + 1, t)

    # insert knots in second direction
    Vold = np.copy(V)
    for i in range(len(Vold) - 1):
        dist = Vold[i + 1] - Vold[i]
        if dist == 0:
            continue

        for inc in np.linspace(0, dist, sfac, endpoint=False)[1:]:
            s = Vold[i] + inc
            k = np.searchsorted(V, s) - 1
            almat = alphamatrix(cpoints.shape[0], V, q, s, k)
            cpoints = np.einsum("ij, jkl -> ikl", almat, cpoints)
            V = np.insert(V, k + 1, s)

    return cpoints, U, V

def bezier_extract(U, p, n):
    """
    Perform Bezier extraction according to Borden et al. algorithm 1.

    Args:
        U: knot vector
        p: polynomial order
        n: number of basis functions

    Returns:
        extraction operators ("element NURBS coordinate" x (operator matrix))
    """
    # a-1 b-1 i-1 mult-0 j-1 r-0 save-0 s-0 k-1
    a = p
    b = a + 1
    C = np.tile(np.identity(p + 1), (len(set(U)) - 1, 1, 1))
    #C = np.tile(np.identity(p + 1), (n, 1, 1))

    el = 0
    while b < len(U) - 1:
        i = b

        # count multiplicity of the knot at location b
        while b < len(U) - 1 and U[b + 1] == U[b]: b += 1
        mult = b - i + 1

        if mult < p:
            # use (10) to compute the alphas
            numer = U[b] - U[a]
            alphas = [0] * (p - mult)
            for j in reversed(range(mult, p)):
                alphas[j - mult] = numer / (U[a + j + 1] - U[a])
            r = p - mult
            # update the matrix coefficients for r new knots
            for j in range(r):
                save = r - j
                s = mult + j + 1
                for k in reversed(range(s, p + 1)):
                    alpha = alphas[k - s]
                    # the following line corresponds to (9)
                    C[el, :, k] = alpha * C[el, :, k] + (1.0 - alpha) * C[el, :, k - 1]
                if b < len(U) - 1:
                    # update overlapping coefficients of the next operator
                    C[el + 1, save - 1:j + save + 1, save - 1] = C[el, p - j - 1:p + 1, p]
            # finished with the current operator
        # article claims this loop should be inside the previous one...
        if b < len(U) - 1:
            # update indices for the next operator
            a = b
            b += 1

        el += 1

    return C

def build_inc_ien(U, V, p, q, n, m):
    """
    Build the INC and IEN arrays corresponding to Cotrell algorithm 7.

    Args:
        p: polynomial order in first direction
        q: polynomial order in second direction
        n: number of basis functions in first direction
        q: number of basis functions in second direction

    Returns:
        INN, IEN arrays
    """
    nel = (n - p) * (m - q)                  # number of elements
    nnp = n * m                              # number of global basis functions
    nen = (p + 1) * (q + 1)                  # number of local basis functions

    INN = np.zeros((nnp, 2), dtype=np.int)   # NURBS coordinates array
    IEN = np.zeros((nel, nen), dtype=np.int) # connectivity array

    e = 0
    A = 0

    for j in range(m):
        for i in range(n):
            INN[A][:] = i, j

            if i >= p and j >= q:
                for jloc in range(q + 1):
                    for iloc in range(p + 1):
                        B = A - jloc * n - iloc

                        b = nen - (jloc * (p + 1) + iloc + 1)

                        IEN[e][b] = B

                e += 1

            A += 1

    # extra: filter out "empty" elements
    nc0 = INN[IEN][:, -1, 0]
    nc1 = INN[IEN][:, -1, 1]

    filter = np.logical_and(U[nc0] != U[nc0 + 1], V[nc1] != V[nc1 + 1])
    IEN = IEN[filter]

    return INN, IEN

def build_id(INN, free):
    """
    Build the ID array as defined in Cotrell.

    Args:
        INN: INN array
        free: array with shape of control points, indicating
            degrees of freedom

    Returns:
        number of free degrees of freedom, ID array
            (functions x physical directions)
    """
    ID = np.array(free[INN[:, 1], INN[:, 0], :], dtype=np.int)

    P = 0
    for i in np.nditer(ID, op_flags=['readwrite']):
        if i:
            i[...] = P
            P += 1
        else:
            i[...] = -1

    return P, ID

def density_filter(nel0, nel1, rmin):
    """
    Prepare the density filter matrix and vector.

    Args:
        nel0: number of elements in first direction
        nel1: number of elements in second direction
        rmin: filter radius

    Returns:
        H matrix, Hs vector
    """
    iH = np.ones(nel0 * nel1 * (2 * int(rmin) + 1) ** 2)
    jH = np.ones_like(iH)
    sH = np.zeros_like(iH)
    k = 0
    for j0 in range(nel1):
        for i0 in range(nel0):
            e0 = j0 * nel0 + i0
            for j1 in range(max(j0 - int(rmin), 0), min(j0 + int(rmin) + 1, nel1)):
                for i1 in range(max(i0 - int(rmin), 0), min(i0 + int(rmin) + 1, nel0)):
                    e1 = j1 * nel0 + i1
                    iH[k] = e0
                    jH[k] = e1
                    sH[k] = max(0, rmin - np.sqrt((i0 - i1) ** 2 + (j0 - j1) ** 2))
                    k += 1
    H = sp.coo_matrix((sH, (iH, jH))).tocsr()
    Hs = np.array(H.sum(1))[:, 0]

    return H, Hs



if __name__ == "__main__":
    p = 2
    t = np.array([0.2, 0.5, 0.7])
    
    print "bernstein and derivative"
    print bernstein(p, t)
    print bernstein_deriv(p, t)

    U = np.array([0., 0., 0., 0.5, 0.5, 1., 1., 1.])
    n = 5

    print "\nextraction operators"
    print bezier_extract(U, p, n)

    V = np.array([0., 0., 1., 1.])
    q = 1
    m = 2

    INN, IEN = build_inc_ien(U, V, p, q, n, m)
    print "\nINN and IEN"
    print INN
    print IEN

    free = np.array([
        [[1, 1], [1, 1], [0, 1], [1, 1], [1, 0]],
        [[0, 1], [1, 1], [1, 0], [0, 1], [1, 1]],
        ])

    print "\ndegrees of freedom and ID"
    N, ID = build_id(INN, free)
    print N
    print ID

    cpoints = np.array([
        [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]],
        [[0, 1], [1, 1], [2, 1], [3, 1], [4, 1]],
        ])

    print "\nrefine"
    cpoints, U, V = refine(cpoints, p, q, U, V, 2, 2)
    print cpoints
    print U
    print V

    nel0 = 5
    nel1 = 5
    rmin = 2

    print "\nfilter"
    H, Hs = density_filter(nel0, nel1, rmin)
    print H
    print Hs
