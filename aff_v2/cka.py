""" import math
import numpy as np


def centering(K):
    n = K.shape[0]
    unit = np.ones([n, n])
    I = np.eye(n)
    H = I - unit / n

    return np.dot(np.dot(H, K), H)


def rbf(X, sigma=None):
    GX = np.dot(X, X.T)
    KX = np.diag(GX) - GX + (np.diag(GX) - GX).T
    if sigma is None:
        mdist = np.median(KX[KX != 0])
        sigma = math.sqrt(mdist)
    KX *= - 0.5 / (sigma * sigma)
    KX = np.exp(KX)
    return KX


def kernel_HSIC(X, Y, sigma):
    return np.sum(centering(rbf(X, sigma)) * centering(rbf(Y, sigma)))


def linear_HSIC(X, Y):
    L_X = np.dot(X, X.T)
    L_Y = np.dot(Y, Y.T)
    return np.sum(centering(L_X) * centering(L_Y))


def linear_CKA(X, Y):
    hsic = linear_HSIC(X, Y)
    var1 = np.sqrt(linear_HSIC(X, X))
    var2 = np.sqrt(linear_HSIC(Y, Y))

    return hsic / (var1 * var2)


def kernel_CKA(X, Y, sigma=None):
    hsic = kernel_HSIC(X, Y, sigma)
    var1 = np.sqrt(kernel_HSIC(X, X, sigma))
    var2 = np.sqrt(kernel_HSIC(Y, Y, sigma))

    return hsic / (var1 * var2) """

import numpy as np

def cka_wide(X, Y):
    """
    Calculate CKA for two matrices. This algorithm uses a Gram matrix 
    implementation, which is fast when the data is wider than it is 
    tall.

    This implementation is inspired by the one in this colab:
    https://colab.research.google.com/github/google-research/google-research/blob/master/representation_similarity/Demo.ipynb#scrollTo=MkucRi3yn7UJ

    Note that we use center the features rather than the Gram matrix
    because we think the latter is tricky and mysterious. It only works for 
    linear CKA though (we only implement linear CKA throughout).
    """     
    X = X.copy()
    Y = Y.copy()

    X -= X.mean(0)
    Y -= Y.mean(0)

    XXT = X @ X.T
    YYT = Y @ Y.T

    # We use reshape((-1,)) instead of ravel() to ensure this is compatible
    # with numpy and pytorch tensors.
    top = (XXT.reshape((-1,)) * YYT.reshape((-1,))).sum()
    bottom = np.sqrt((XXT ** 2).sum() * (YYT ** 2).sum())
    c = top / bottom

    return c


def cka_tall(X, Y):
    """
    Calculate CKA for two matrices.
    """
    X = X.copy()
    Y = Y.copy()

    X -= X.mean(0)
    Y -= Y.mean(0)
            
    XTX = X.T @ X
    YTY = Y.T @ Y
    YTX = Y.T @ X

    # Equation (4)
    top = (YTX ** 2).sum()
    bottom = np.sqrt((XTX ** 2).sum() * (YTY ** 2).sum())
    c = top / bottom

    return c

def cka(X, Y):
    """
    Calculate CKA for two matrices.

    CKA has several potential implementations. The naive implementation is 
    appropriate for tall matrices (more examples than features), but this 
    implementation uses lots of memory and it slow when there are many more 
    features than examples. In that case, which often happens with DNNs, we 
    prefer the Gram matrix variant.
    """
    if X.shape[0] < X.shape[1]:
        return cka_wide(X, Y)
    else:
        return cka_tall(X, Y)