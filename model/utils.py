import numpy as np

def l1Norm(x):
    """
        Computes the l1 norm of x
    """
    return np.sum(np.abs(x))

def l2Norm(x):
    """
        Computes the l2 norm of x
    """
    return np.sqrt(np.sum(x**2))

def r2Compute(u, v, x):
    """
        Computes the r square score of the data
        x in the projection defined by u and v
        :param u: Vector of size x
        :param v: Vector of size x
        :param x: Matrix of size N * x
        :return r2: r square score of the N points projected in (u,v)
    """
    xProj = np.dot(x, u.T).flatten()
    yProj = np.dot(x, v.T).flatten()
    projectedPoints = np.array([xProj,yProj]).T
    SSRes = np.sum((projectedPoints[:, 1] - projectedPoints[:, 0])**2)
    SSTot = np.sum((projectedPoints[:, 1] - np.mean(projectedPoints[:, 1]))**2)
    
    return 1 - SSRes/(SSTot + 10**(-5)) # Add epsilon to avoid nan
