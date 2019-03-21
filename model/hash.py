"""
Use the CAA in order to project the points
Then computes for each projections the r2

This model is inspired by random projections classification models
"""
import numpy as np
from model.caa import CAAModel

class HashCAA(CAAModel):

    def transform(self, x):
        """
            Transforms the set of points by projecting on the caa
            And computing the r2
            
            Arguments:
                x {Array} -- Set of points
        """
        assert self.caas != {}, "CAA not trained"
        projections = []
        for c in self.caas:
            projections.extend(self.caas[c].projections[:self.number_cell] if self.number_cell is not None else self.caa.projections)
        return np.array([p.rSquareProjection(x) for p in projections])
