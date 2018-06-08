import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

df = pd.read_csv("iris_data_tb.csv", skiprows=1, names=['Sample', 'Type', 'PW', 'PL', 'SW', 'SL'])

type0 = df[df.Type == 0]
type1 = df[df.Type == 1]
type2 = df[df.Type == 2]

print(type0)
print(type1)
print(type2)

