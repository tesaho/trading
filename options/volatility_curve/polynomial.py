"""
polynomial curve fit
"""

import numpy as np

def curveFit(x, y, degree, size):

    N = size
    n = degree

    ## array that stores the values of sigma(xi), sigma(xi^2), sigma(xi^3)... sigma(xi^2n)
    X = np.zeros((2*n+1))
    for i in range(2*n):
        for j in range(N):
            X[i] += np.power(x[j], i)

    ## B is a normal matrix (n, n+1)
    ## last column is kept empty
    B = np.zeros((n+1, n+2))
    for i in range(n):
        for j in range(n):
            B[i, j] = X[i+j]

    ## Y array to store values of sigma(yi), sigma(xi*yi), sigma(xi^2*yi)... sigma(xi^n*yi)
    Y = np.zeros((n+1))
    for i in range(n):
        for j in range(N):
            Y[i] += np.power(x[j], i)*y[j]

    ## load values of Y as last column of B
    B[:,-1] = Y

    # n_ = n+1 because the Gaussian Elimination part below was for n equations,
    # but here n is the degree of polynomial and for n degree we get n+1 equations
    n_ = n + 1
    for i in range(n_):
        for k in range(i+1, n_):
            if B[i, i] < B[k, i]:
                for j in range(n_+1):
                    temp = B[i, j]
                    B[i, j] = B[k, j]
                    B[k, j] = temp

    # perform gaussian elimination
    for i in range(n_-1):
        for k in range(i+1, n_):
            # make the elements below the pivot elements equal to zero or eliminate the variables
            t = B[k, i]*1.0 / B[i, i]
            for j in range(n_+1):
                B[k, j] = B[k, j] - t*B[i, j]

    # back substitution
    a = np.zeros((n_))
    curveParam = []
    for i in reversed(range(n_-1)):
        # make the variable to be calculated equal to the rhs of the last equation
        a[i] = B[i, -1]
        for j in range(n_):
            if j != i:
                # then subtract all the lhs values except the coefficient of the
                # variable whose value is being calculated
                a[i] -= B[i, j]*a[j]
        a[i] /= B[i, i]
        curveParam.append(a[i])

    # reverse curveParam so that c[0] == c0, .. c[-1] = c_n * x**n
    return curveParam[::-1]

def interpolateCurve(x, curveParam):

    return np.polyval(curveParam, x)
