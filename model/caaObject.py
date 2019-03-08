"""
    CAA Objects which allow an easy manipulation and display of the projections
"""

import numpy as np
from numpy.linalg import pinv
import seaborn as sns
import matplotlib.pyplot as plt

from model.utils import *

class CAA:
    """
        Objects to keep a CAA
    """
    def __init__(self, US, VS, DS, error, penalty1, penalty2, trainingData):
        """
            US, VS: Linear Combinations
            DS : Score (related to the r square)
            mea, std : Necessary to recentred new data
            error : R squared observed on the training data

            trainingData : Needed for computation of caa distance
        """
        # Used for reconstructiong matrices
        self.US = US
        self.VS = VS

        self.projections = []
        self.ds = DS
        self.rs = error
        self.penalty1 = penalty1
        self.penalty2 = penalty2
        self.trainingData = trainingData

        for i in range(len(error)):
            self.projections.append(Projection(US[i], VS[i], error[i], DS[i], self))
        self.mean = np.mean(trainingData, axis = 0)
        self.std = np.std(trainingData, axis = 0)
        self.std[self.std == 0] = 1

    def size(self):
        """
            Returns the number of projection
        """
        return len(self.projections)

    def plot(self, header, show = True):
        """
            Plots the matrix correlation
            header: Name of the different features of original data
        """
        corr = 0
        for i, p in enumerate(self.projections):
            corr = np.add(corr, p.d * p.getCorrelation())
            p.plot(header, show = False, title="Projection {}".format(i))

        plt.figure("Explained correlation")
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        color = sns.diverging_palette(220, 10, as_cmap=True)
        sns.heatmap(corr, mask=mask, vmin=-1, vmax=1, square=True, cmap=color, xticklabels=header, yticklabels=header)
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        if show:
            plt.show()

    def getCorrelation(self):
        """
            Computes the correlation structure of the caa
        """
        corr = 0
        for p in self.projections:
            corr = np.add(corr, p.d * p.getCorrelation())
        return corr

    def projectPoints(self, dataPoints, k):
        """
            Projects the given points in each projection of the CAA
            
            Arguments:
                dataPoints {Array (_ x len(mean))} -- Matrix of points to project
        """
        assert(dataPoints.shape[1] == len(self.mean))
        return np.concatenate([proj.projectPoints(dataPoints) for proj in self.projections[:k]], axis = 1)

    def inverseTransform(self, projectedPoints = None, k = None):
        """
            Inverse the CAA transformation to transform the given
            projections into data in the original space

            Keyword Arguments:
                projectedPoints {Array (_ x 2*k)} -- Matrix of projected points (default: {None : Rebuilt data used for CAA computation})
                k {Int} -- Used the k first projections if k > 0 else the k last (default: {None : Use all})
        """
        if k is None:
            k = len(self.US)
        if projectedPoints is None:
            projectedPoints = self.projectPoints(self.trainingData, k) # Data already normalized when projected

        assert(projectedPoints.shape[1] == 2*k)
        # Computes the inverse of the transformation
        transform = np.concatenate([(proj.d * proj.u, proj.d * proj.v) for proj in self.projections[:k]], axis=0).reshape((2*k,len(self.mean)))

        # Pseudo inverse
        inv = pinv(transform)
        res = np.matmul(projectedPoints, inv.T) * self.std + self.mean

        return res

    def __str__(self):
        res = "CAA"
        for p in self.projections:
            res += "\n" + p.__str__()
        return res


class Projection:
    """
        It is a projection of a caa
        Caracterized by two axes of projection (sparse subset of features) highly correlated
    """
    def __init__(self, u, v, e, d, caa):
        """
            u, v : Linear Combinations
            e : rsquared on training
            d : Score
            caa : From which caa is come (needed in order to have mean and std of
                original data)
        """
        self.u = u
        self.v = v
        self.e = e
        self.d = d
        self.caaFather = caa

    @classmethod
    def distance(cls, proj1, proj2):
        """
            Computes the distances between two projections
            => No rsquare otherwise not a distance
        """
        r = min(l2Norm(proj1.u - proj2.v) + l2Norm(proj2.u - proj1.v),
            l2Norm(proj1.u - proj2.u) + l2Norm(proj1.v - proj2.v))
        return r

    def projectPoints(self, points, normalize = True):
        """
            Project points on the projection
        """
        if normalize:
            points = (points - self.caaFather.mean) / self.caaFather.std
        xtest = np.dot(points, self.u.T).flatten()
        ytest = np.dot(points, self.v.T).flatten()
        projectedPoints = np.array([xtest,ytest]).T

        return projectedPoints

    def rSquareProjection(self, points, applyMax = False):
        """
            Computes the rsquare if we project the given points in this projection
        """
        if len(points) == 0:
            return 0.
        projectedPoints = self.projectPoints(points)
        SSRes = np.sum((projectedPoints[:, 1] - projectedPoints[:, 0])**2)
        SSTot = np.sum((projectedPoints[:, 1] - np.mean(projectedPoints[:, 1]))**2)
        if applyMax:
            return max(0., 1. - SSRes/(SSTot + 0.00001))
        else:
            return 1 - SSRes/(SSTot + 0.00001)

    def getCorrelation(self):
        """
            Computes the correlation structure of the current projection
        """
        corr = np.matmul(self.u.T,self.v) + np.matmul(self.v.T,self.u)
        return corr

    def plot(self, header, show = True, title = None):
        """
            Plots the projection
        """
        corr = self.getCorrelation()
        plt.figure(title)
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        color = sns.diverging_palette(220, 10, as_cmap=True)
        sns.heatmap(corr, mask=mask, vmin=-1, vmax=1, square=True, cmap=color, xticklabels=header, yticklabels=header)
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        if show:
            plt.show()

    def plotProjection(self, points):
        """
            Plots the scatter points of the given points
        """
        projected = self.projectPoints(points)
        plt.figure("Projections Points")
        plt.scatter(projected[:,0], projected[:,1])
        plt.plot(projected[:,0], projected[:,0], color="red", ls=":")
        plt.show()

    def __str__(self):
        res = "Projection \n"
        res += "\t u => {}\n".format(self.u)
        res += "\t v => {}\n".format(self.v)
        res += "\t d => {}".format(self.d)
        return res
