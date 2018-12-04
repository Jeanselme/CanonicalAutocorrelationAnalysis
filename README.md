# CanonicalAutocorrelationAnalysis
This project is a python implementation of the orignal work by Maria De Arteaga : [CAE](https://github.com/mariadea/CAE)

## Project
### Model
Contains all the code for the CAA model: computation and display.

### Test
Reproduces the test of the original paper

## Dependencies
Code tested with python 2 and 3 with the libraries in `requirement.txt`

## Example
```
import pandas as pd
from caa import CAAComputation

# Open and clean data
data = pd.read_csv("data.csv")

# Compute Global CAA
caa = CAAComputation(data, 0.35, 0.35)
print(caa)
caa.plot(data.columns, True)
```
