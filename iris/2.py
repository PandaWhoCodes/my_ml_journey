import pandas as pd

df = pd.read_csv("iris_data_tb.csv", skiprows=1,skipinitialspace=True, names=['Sample', 'Type', 'PW', 'PL', 'SW', 'SL'])

type0 = df[df.Type == 0]
type1 = df[df.Type == 1]
type2 = df[df.Type == 2]


def get_mean_vector(A):
    mean_vector=[]
    for i in A.iterrows():
        sum = (i.PW + i.PL + i.SW + i.SL) / 4
        mean_vector.append(sum)
    return mean_vector

type0_vectors = get_mean_vector(type0)
type1_vectors = get_mean_vector(type1)
type2_vectors = get_mean_vector(type2)

# print(type0.PW)