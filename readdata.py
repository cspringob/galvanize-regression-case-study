import numpy as np
import pandas as pd

mydata = pd.read_csv('data/Train.csv')

print(mydata.shape)

print(mydata.SalesID.nunique())
print(mydata.MachineID.nunique())
print(mydata.ModelID.nunique())
print(mydata.YearMade.nunique())
