"""
Use the CAA in order to project the points
Then computes for each projections the r2

This model is inspired by random projections classification models
"""
from model.utils import r2Compute
from model.caa import CAAModel
import pandas as pd

class HashCAA(CAAModel):

    def transform(self, x):
        """
            Transforms the set of points by projecting on the caa
            And computing the r2
            
            Arguments:
                x {Array} -- Set of points
        """
        assert self.caa is not None, "CAA not trained"
        rproj = {}
        projections = self.caa.projections[:self.number_cell] if self.number_cell is not None else self.caa.projections
        for i, p in enumerate(projections):
            # Project all points
            projPoints = pd.DataFrame(p.projectPoints(x), index = x.index, columns = ["x", "y"])

            # Computes quantity for R squares
            mean = projPoints["y"].rolling(self.window, min_periods = 1).mean()
            SSRes = (projPoints["y"] - projPoints["x"])**2
            SSTot = (projPoints["y"] - mean)**2

            projPoints = pd.DataFrame({"SSRes": SSRes, "SSTot": SSTot}, index = x.index)
            projPoints = projPoints.rolling(self.window, min_periods = 1).sum()
            rproj[i] = 1 - projPoints["SSRes"]/(projPoints["SSTot"] + 10**(-5))

        return pd.concat(rproj, axis=1)
