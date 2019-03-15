"""
Use the CAA in order to project the points
Then computes for each projections the r2

This model is inspired by random projections classification models
"""
from model.caa import CAAModel

class HashCAA(CAAModel):

    def transform(self, x):
        """
            Transforms the set of points by projecting on the caa
            And computing the r2
            
            Arguments:
                x {Array} -- Set of points
        """
        assert self.caa is not None, "CAA not trained"
        projections = self.caa.projections[:self.number_cell] if self.number_cell is not None else self.caa.projections
        return [p.rSquareProjection(x) for p in projections]
