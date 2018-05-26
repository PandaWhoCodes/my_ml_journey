import csv
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def read_csv(file_name):
    with open(file_name, 'r') as f:
        csv_reader = csv.reader(f)
        return list(csv_reader)


def convert():
    df = pd.read_csv("dataset.csv", skiprows=1, names=['Sample', 'Type', 'PW', 'PL', 'SW', 'SL'])
    # Dropping the sample number from dataframe
    df.drop(['Sample'], axis=1)
    features = ['Type', 'PW', 'PL', 'SW', 'SL']
    # Standardizing the values with respect to column PL
    x = df.loc[:, features].values
    y = df.loc[:, ['PL']].values
    x = StandardScaler().fit_transform(x)
    # Finding the PCA using n =1
    pca = PCA(n_components=1)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data=principalComponents
                               , columns=['principal component 1'])
    print(principalDf)


convert()
