"""
    Translation of the R code from Maria
"""

import numpy as np
import pandas as pd
from scipy.linalg import eig
from numpy.linalg import matrix_rank
from model.utils import l1Norm, l2Norm, r2Compute
from joblib import Parallel, delayed 
from multiprocessing import cpu_count
from model.caaObject import CAA

def softThreshold(Pw, l):
    w = np.sign(Pw)*np.maximum(0, np.abs(Pw)-l)
    return w / l2Norm(w)

def updateW(Co, ci, w):
    """
        Solves the convex problem for one iteration

        Arguments:
            Co {Matrix Features x Features} -- Correlation Matrix
            ci {float} -- Sparsity of the given dimension
            w {Array features} -- Projection Axis

        Raises:
            Exception -- If not converge raises an exemption after 10000 tries

        Returns:
            Array features -- Converged w
    """
    Pw = np.dot(w, Co)
    absW = np.abs(w)
    lambda1 = np.max(np.abs(Pw[absW > 0]/absW[absW > 0]))
    wOutput = softThreshold(Pw, lambda1*absW)

    if l1Norm(wOutput) <= ci:
        return wOutput
    else:
        lambda2_min, lambda2_max = 0., np.max(Pw[absW == 0])
        while lambda2_max - lambda2_min > 1e-5:
            lambda2 = (lambda2_min + lambda2_max) / 2.
            wOutput = softThreshold(Pw, lambda2 + lambda1*absW)

            if l1Norm(wOutput) < ci or np.sum(wOutput != 0) <= 1:
                lambda2_max = lambda2
            else:
                lambda2_min = lambda2

        return wOutput

def computeProjection(Co, uInit, vInit, c1, c2):
    """
        Computes a projection maximizing the correlation

        Arguments:
            Co {Matrix Features x Features} -- Correlation Matrix
            uInit {Array Features} -- Projection Axis 1
            vInit {Array Features} -- Projection Axis 2
            c1 {float} -- Sparsity of the 1 dimension
            c2 {float} -- Sparsity of the 2 dimension

        Raises:
            Exception -- If not converge raises an exemption after 10000 tries

        Returns:
            (Array Features, Array Features) -- Axis of projections
    """
    uPrev, vPrev = uInit, vInit
    for _ in range(1000):
        uUpdate = updateW(Co, c1, vPrev)
        vUpdate = updateW(Co, c2, uUpdate)

        if l2Norm(uUpdate - uPrev) < 1e-5 and l2Norm(vUpdate - vPrev) < 1e-5:
            return uUpdate, vUpdate
        
        uPrev, vPrev = uUpdate, vUpdate
    else:
        raise Exception("No convergence")

def CAAComputation(dataPoints, penalty1, penalty2, maxProj = None, minr2 = None, scale = True, doubleInit = True, orthogonality = False):
    """
        This function allows to compute the Caa on a given dataset
        
        Arguments:
            dataPoints {Matrix Points x Features} -- The data on which to compute Caa
            penalty1 {1/Features <= float <= 1} -- Penalty on first dimension projection 
            penalty2 {1/Features <= float <= 1} -- Penalty on second dimension projection
                -> Higher it is, Less sparse is the dimension
        
        Keyword Arguments:
            maxProj {int} -- Maximum number of projections to extract (default: {None} - Extract the number of features)
            minr2 {float <= 1} -- Minimum rsquare to take into account the projections (default: {None} - Take all rSquare into account)
            scale {bool} -- Normalize the given data (default: {True})
            doubleInit {bool} -- Double initialization of the projections (default: {True})
            orthogonality {bool} -- Force orthogonality between projections (default : {False})

        Returns:
            CAA -- Object containing all the projections for the given data
    """
    row, features = dataPoints.shape
    
    assert(1./features <= penalty1 and penalty1 <= 1)
    assert(1./features <= penalty2 and penalty2 <= 1)
    assert(minr2 is None or minr2 <= 1)

    if maxProj is None:
        maxProj = features

    if scale:
        std = np.std(dataPoints, axis = 0)
        std[std == 0] = 1
        X = (dataPoints - np.mean(dataPoints, axis = 0)) / std
    else:
        X = dataPoints

    Co = np.matmul(X.T, X) / row
    uList, vList, rSquare, dList = [], [], [], []

    # Remove diagonal values to avoid max
    Co[np.diag_indices_from(Co)] = 0

    for _ in range(maxProj):
        maxCorr = np.unravel_index(np.argmax(np.abs(Co), axis=None), Co.shape)

        u, v = np.zeros((1, features)), np.zeros((1, features))
        u[0, maxCorr[0]] = 1.
        v[0, maxCorr[1]] = 1.

        if doubleInit:
            c1 = c2 = 0.5 * np.sqrt(features)
            try:
                u, v = computeProjection(Co, u, v, c1, c2)
            except:
                return CAA(uList, vList, dList, rSquare, penalty1, penalty2, dataPoints)

        c1 = penalty1 * np.sqrt(features)
        c2 = penalty2 * np.sqrt(features)
        try:
            u, v = computeProjection(Co, u, v, c1, c2)
        except:
            return CAA(uList, vList, dList, rSquare, penalty1, penalty2, dataPoints)
        d = np.dot(np.dot(u,Co),v.T).flatten()
        r = r2Compute(u, v, X).flatten()
        
        # Append values to the list
        if minr2 is None or r >= minr2:
            uList.append(u)
            vList.append(v)
            dList.append(d)
            rSquare.append(r)

        # Update Correlation Matrix
        Co -= d * (np.matmul(u.T,v) + np.matmul(v.T,u))
        if orthogonality:
            selection = np.ones_like(Co)
            notNull = (np.abs(u) + np.abs(v) != 0).flatten()
            selection[notNull,:] = 0
            selection[:,notNull] = 0
            Co[selection == 0] = 0
        
    return CAA(uList, vList, dList, rSquare, penalty1, penalty2, dataPoints)

def gridSearchCaa(dataPoints, maxIteration = 10, parallel = True, toMax = lambda caa: np.max(caa.rs)):
    """
        Computes a gridsearch over the penalty in order to maximize the d
    
        Arguments:
            dataPoints {Matrix Points x Features} -- The data on which to compute Caa
        
        Keyword Arguments:
            maxIteration {int} -- Number of iteration to train (default: {10})
    """
    _, features = dataPoints.shape
    linspace, maxD, caaRes = np.linspace(1./features, 1., maxIteration), None, None
    
    if parallel:
        caas = Parallel(n_jobs = int(cpu_count()*0.75))(delayed(CAAComputation)(dataPoints, pen1, pen2) for i, pen1 in enumerate(linspace) for pen2 in linspace[:i])
        caas = [caa for caa in caas if caa.size() > 0]
        if len(caas) > 0:
            caaRes = [toMax(caa) for caa in caas]
            sortIndex = np.argsort(caaRes)[::-1]
            caaRes = caas[sortIndex[0]]
    else:
        for i, pen1 in enumerate(linspace):
            for pen2 in linspace[:i]:
                caa = CAAComputation(dataPoints, pen1, pen2)
                if caa.size() > 0 and (maxD is None or toMax(caa) > maxD): 
                    maxD = toMax(caa)
                    caaRes = caa
    return caaRes
    
class CAAModel():

    def __init__(self, number_cell = None, classes = None, **args_caa_grid_search):
        """
            Initialize a caa hash model
            Project on different caa dimension
            And return a constant number of cells

            Arguments:
                window {int / or time} -- Moving window to compute moving CAA
                number_cell {int} -- Number of cell to create
                classes {list} -- Class on which to compute separate caa (default: {None} -- All class)
        """
        self.caas = {}
        self.number_cell = number_cell
        self.classes = classes
        self.args_caa_grid_search = args_caa_grid_search

    def fit(self, x, y = None):
        """
            Fit the caa on the set of points
            
            Arguments:
                x {Array} -- Set of points
                y {Array} -- Labels (if class is indicated)
        """
        # Delete previous CAAs
        self.caas = {}
        if isinstance(x, pd.DataFrame):
            x = x.values

        if (self.classes is not None) and (y is not None):
            for c in self.classes:
                self.caas[c] = gridSearchCaa(x[y == c], **self.args_caa_grid_search)
        else:
            self.caas[-1] = gridSearchCaa(x, **self.args_caa_grid_search)
        return self
    
    def fit_transform(self, x, y = None):
        """
            Fit the caa on the set of points
            And transforms it
            
            Arguments:
                x {Array} -- Set of points
        """
        self.fit(x, y)
        return self.transform(x)

    def transform(self, x):
        """
            Transforms the set of points by projecting on the caa
            
            Arguments:
                x {Array} -- Set of points
        """
        assert self.caas != {}, "CAA not trained"

        transformed = []
        for c in self.caas:
            transformed.append(self.caas[c].projectPoints(x, self.number_cell))
        return np.concatenate(transformed, axis = 1)
