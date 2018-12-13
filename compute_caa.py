# Sample program to load 2 datasets and compare them using CAA projections
# Datasets should have the same shape/structure and contain all numeric features only

import pandas as pd
from model.caa import CAAComputation
import sys

# Read datasets
data = pd.read_csv(sys.argv[1], header=0)
data2 = pd.read_csv(sys.argv[2], header=0)

# Compute Global CAA
caa = CAAComputation(data, 0.25, 0.25)
caa.plot(data.columns, True)

# Project both sets of points on CAA projections learnt from data
caa.displayPointsOnProjections(data2, data)

caa = CAAComputation(data2, 0.25, 0.25)
caa.plot(data2.columns, True)
# Project both sets of points on CAA projections learnt from data2
caa.displayPointsOnProjections(data2, data)
