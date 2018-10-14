import matplotlib.pyplot as plt
import numpy
import numpy as np
import math

"""-------------------------------------------------
	Name : calCovariance()
	Input = 4*n matrix, mean of Iris data
	Output = 4*4 covariance matrix
	Description : calculate covariance for each component
-------------------------------------------------"""


def calCovariance(Iris, mean):
    sepal_length = 0
    sepal_width = 0
    petal_length = 0
    petal_width = 0

    IrisCovarianceMatrix = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    # covariance of sepal_length
    for i in range(0, len(Iris)):
        if i % 4 == 0:
            sepal_length = sepal_length + ((Iris[i] - mean[0]) * (Iris[i] - mean[0]))
        elif i % 4 == 1:
            sepal_width = sepal_width + ((Iris[i - 1] - mean[0]) * (Iris[i] - mean[1]))
        elif i % 4 == 2:
            petal_length = petal_length + ((Iris[i - 2] - mean[0]) * (Iris[i] - mean[2]))
        elif i % 4 == 3:
            petal_width = petal_width + ((Iris[i - 3] - mean[0]) * (Iris[i] - mean[3]))

    sepal_length = sepal_length / ((len(Iris) / 4) - 1)
    sepal_width = sepal_width / ((len(Iris) / 4) - 1)
    petal_length = petal_length / ((len(Iris) / 4) - 1)
    petal_width = petal_width / ((len(Iris) / 4) - 1)

    IrisCovarianceMatrix[0][0] = sepal_length
    IrisCovarianceMatrix[0][1] = sepal_width
    IrisCovarianceMatrix[0][2] = petal_length
    IrisCovarianceMatrix[0][3] = petal_width

    sepal_length = 0
    sepal_width = 0
    petal_length = 0
    petal_width = 0

    # covariance of sepal_width
    for i in range(0, len(Iris)):
        if i % 4 == 0:
            sepal_length = sepal_length + ((Iris[i + 1] - mean[1]) * (Iris[i] - mean[0]))
        elif i % 4 == 1:
            sepal_width = sepal_width + ((Iris[i] - mean[1]) * (Iris[i] - mean[1]))
        elif i % 4 == 2:
            petal_length = petal_length + ((Iris[i - 1] - mean[1]) * (Iris[i] - mean[2]))
        elif i % 4 == 3:
            petal_width = petal_width + ((Iris[i - 2] - mean[1]) * (Iris[i] - mean[3]))

    sepal_length = sepal_length / ((len(Iris) / 4) - 1)
    sepal_width = sepal_width / ((len(Iris) / 4) - 1)
    petal_length = petal_length / ((len(Iris) / 4) - 1)
    petal_width = petal_width / ((len(Iris) / 4) - 1)

    IrisCovarianceMatrix[1][0] = sepal_length
    IrisCovarianceMatrix[1][1] = sepal_width
    IrisCovarianceMatrix[1][2] = petal_length
    IrisCovarianceMatrix[1][3] = petal_width

    sepal_length = 0
    sepal_width = 0
    petal_length = 0
    petal_width = 0

    # covariance of petal_length
    for i in range(0, len(Iris)):
        if i % 4 == 0:
            sepal_length = sepal_length + ((Iris[i + 2] - mean[2]) * (Iris[i] - mean[0]))
        elif i % 4 == 1:
            sepal_width = sepal_width + ((Iris[i + 1] - mean[2]) * (Iris[i] - mean[1]))
        elif i % 4 == 2:
            petal_length = petal_length + ((Iris[i] - mean[2]) * (Iris[i] - mean[2]))
        elif i % 4 == 3:
            petal_width = petal_width + ((Iris[i - 1] - mean[2]) * (Iris[i] - mean[3]))

    sepal_length = sepal_length / ((len(Iris) / 4) - 1)
    sepal_width = sepal_width / ((len(Iris) / 4) - 1)
    petal_length = petal_length / ((len(Iris) / 4) - 1)
    petal_width = petal_width / ((len(Iris) / 4) - 1)

    IrisCovarianceMatrix[2][0] = sepal_length
    IrisCovarianceMatrix[2][1] = sepal_width
    IrisCovarianceMatrix[2][2] = petal_length
    IrisCovarianceMatrix[2][3] = petal_width

    sepal_length = 0
    sepal_width = 0
    petal_length = 0
    petal_width = 0

    # covariance of petal_width
    for i in range(0, len(Iris)):
        if i % 4 == 0:
            sepal_length = sepal_length + ((Iris[i + 3] - mean[3]) * (Iris[i] - mean[0]))
        elif i % 4 == 1:
            sepal_width = sepal_width + ((Iris[i + 2] - mean[3]) * (Iris[i] - mean[1]))
        elif i % 4 == 2:
            petal_length = petal_length + ((Iris[i + 1] - mean[3]) * (Iris[i] - mean[2]))
        elif i % 4 == 3:
            petal_width = petal_width + ((Iris[i] - mean[3]) * (Iris[i] - mean[3]))

    sepal_length = sepal_length / ((len(Iris) / 4) - 1)
    sepal_width = sepal_width / ((len(Iris) / 4) - 1)
    petal_length = petal_length / ((len(Iris) / 4) - 1)
    petal_width = petal_width / ((len(Iris) / 4) - 1)

    IrisCovarianceMatrix[3][0] = sepal_length
    IrisCovarianceMatrix[3][1] = sepal_width
    IrisCovarianceMatrix[3][2] = petal_length
    IrisCovarianceMatrix[3][3] = petal_width

    IrisCovarianceMatrix = np.array(IrisCovarianceMatrix)

    print(IrisCovarianceMatrix)

    return IrisCovarianceMatrix


"""-------------------------------------------------
	Name : cal_mean()
	Input = 4*n matrix
	Output = 1*4
	Description : calculate mean for each component
-------------------------------------------------"""


def cal_mean(Iris):
    sepal_length_mean = 0
    sepal_width_mean = 0
    petal_length_mean = 0
    petal_width_mean = 0

    Iris_mean = [0, 0, 0, 0]

    for i in range(0, len(Iris)):
        if i % 4 == 0:
            sepal_length_mean = sepal_length_mean + Iris[i]
        elif i % 4 == 1:
            sepal_width_mean = sepal_width_mean + Iris[i]
        elif i % 4 == 2:
            petal_length_mean = petal_length_mean + Iris[i]
        elif i % 4 == 3:
            petal_width_mean = petal_width_mean + Iris[i]

    sepal_length_mean = sepal_length_mean / (len(Iris) / 4)
    sepal_width_mean = sepal_width_mean / (len(Iris) / 4)
    petal_length_mean = petal_length_mean / (len(Iris) / 4)
    petal_width_mean = petal_width_mean / (len(Iris) / 4)

    Iris_mean[0] = sepal_length_mean
    Iris_mean[1] = sepal_width_mean
    Iris_mean[2] = petal_length_mean
    Iris_mean[3] = petal_width_mean

    Iris_mean = np.array(Iris_mean)

    print(Iris_mean)

    return Iris_mean


"""-------------------------------------------------
	Name : discriminantFunctions()
	Input = 2*n matrix, 2*1 mean matrix, 2*2 covarianceMatrix, value of vi0
	Output = value(gi)
	Description : Calculated using discriminantFunctions
-------------------------------------------------"""


def discriminantFunctions(IrisTestData, meanMatrix, covarianceMatrix, vi0):
    cal1 = np.dot(np.dot(np.transpose(IrisTestData), -0.5 * np.linalg.inv(covarianceMatrix)), IrisTestData)
    cal2 = np.dot(np.transpose(np.dot(np.linalg.inv(covarianceMatrix), meanMatrix)), IrisTestData)

    result = cal1 + cal2 + vi0
    return result


"""-------------------------------------------------
	Name : calVi0()
	Input = 2*1 mean matrix, 2*2 covarianceMatrix
	Output = value of vi0
	Description : Calculate the required value for the discriminantFunction
-------------------------------------------------"""


def calVi0(meanMatrix, covarianceMatrix):
    vi0 = -0.5 * np.dot(np.dot(np.transpose(meanMatrix), np.linalg.inv(covarianceMatrix)), meanMatrix) - 0.5 * np.log2(
        numpy.linalg.det(covarianceMatrix))

    return vi0


"""-------------------------------------------------
	Name : plotTheTraingData()
	Input = three of Iris (2*n matrix)
	Description : plot the each iris traing data in graph
-------------------------------------------------"""


def plotTheTraningData(setosaData, versicolorData, virginicaData):
    setosaPlot = np.zeros([2, 40])
    versicolorPlot = np.zeros([2, 40])
    virginicaPlot = np.zeros([2, 40])

    count = 0
    for i in range(0, 160):
        if i % 4 == 0:
            setosaPlot[0][count] = setosaData[i]
            setosaPlot[1][count] = setosaData[i + 1]
            versicolorPlot[0][count] = versicolorData[i]
            versicolorPlot[1][count] = versicolorData[i + 1]
            virginicaPlot[0][count] = virginicaData[i]
            virginicaPlot[1][count] = virginicaData[i + 1]
            count += 1

    plt.axis([4, 8.1, 1.5, 5])
    plt.plot(setosaPlot[0], setosaPlot[1], 'bo')
    plt.plot(versicolorPlot[0], versicolorPlot[1], 'b^')
    plt.plot(virginicaPlot[0], virginicaPlot[1], 'bs')


"""-------------------------------------------------
	Name : plotTheTestData()
	Input = three of Iris (2*n matrix)
	Description : plot the each iris test data in graph
-------------------------------------------------"""


def plotTheTestData(setosaData, versicolorData, virginicaData):
    setosaPlot = np.zeros([2, 10])
    versicolorPlot = np.zeros([2, 10])
    virginicaPlot = np.zeros([2, 10])

    count = 0
    for i in range(0, 40):
        if i % 4 == 0:
            setosaPlot[0][count] = setosaData[count][0]
            setosaPlot[1][count] = setosaData[count][1]
            versicolorPlot[0][count] = versicolorData[count][0]
            versicolorPlot[1][count] = versicolorData[count][1]
            virginicaPlot[0][count] = virginicaData[count][0]
            virginicaPlot[1][count] = virginicaData[count][1]
            count += 1

    plt.plot(setosaPlot[0], setosaPlot[1], 'ko')
    plt.plot(versicolorPlot[0], versicolorPlot[1], 'k^')
    plt.plot(virginicaPlot[0], virginicaPlot[1], 'ks')


"""-------------------------------------------------
	Name : plotThedecisionBoundaries()
	Input = three of Iris g value
	Description : plot the each decision boundary in graph
-------------------------------------------------"""


def plotThedecisionBoundaries(g1, g2, g3):
    xx, yy = np.meshgrid(np.arange(4, 8.1, 0.05), np.arange(1, 6, 0.05))

    plt.contour(xx, yy, g1 - g2, [0], colors='b')
    plt.contour(xx, yy, g2 - g3, [0], colors='g')
    plt.contour(xx, yy, g3 - g1, [0], colors='r')


"""-------------------------------------------------
	Name : calMahalanobisDistance()
	Input = 2*n matrix, 2*1 mean matrix, 2*2 covariance matrix
	Output = expression of Z
	Description : calculate and plot the each mahalanobis distance in graph
-------------------------------------------------"""


def calMahalanobisDistance(Iris, meanVector, covarianceMatrix):
    irisMBPlot = np.zeros([2, 40])

    count = 0
    for i in range(0, 160):
        if i % 4 == 0:
            irisMBPlot[0][count] = Iris[i]
            irisMBPlot[1][count] = Iris[i + 1]
            count += 1

    xx, yy = np.meshgrid(np.arange(4, 8.1, 0.05), np.arange(1, 6, 0.05))

    matrix = np.linalg.inv(covarianceMatrix)

    temp1 = (meanVector[0] - xx) * (matrix[0][0]) + (meanVector[1] - yy) * (matrix[1][0])
    temp2 = (meanVector[0] - xx) * (matrix[0][1]) + (meanVector[1] - yy) * (matrix[1][1])

    Z = np.sqrt(temp1 * (meanVector[0] - xx) + temp2 * (meanVector[1] - yy)) - 2
    plt.contour(xx, yy, Z, [0], colors='y')

    return Z


"""-------------------------------------------------
	Name : calGListofMD()
	Input = 2*n matrix, 2*1 mean matrix, 2*2 covariance matrix
	Output = g value list of each class 
	Description : Generate list to determine test data
-------------------------------------------------"""


def calGListofMD(Iris, meanVector, covarianceMatrix):
    gList = np.zeros((1, 10))
    matrix = np.linalg.inv(covarianceMatrix)

    for i in range(0, len(Iris)):
        temp1 = (meanVector[0] - Iris[i][0]) * (matrix[0][0]) + (meanVector[1] - Iris[i][1]) * (matrix[1][0])
        temp2 = (meanVector[0] - Iris[i][0]) * (matrix[0][1]) + (meanVector[1] - Iris[i][1]) * (matrix[1][1])

        Z = np.sqrt(temp1 * (meanVector[0] - Iris[i][0]) + temp2 * (meanVector[1] - Iris[i][1])) - 2
        gList[0][i] = Z

    return gList


"""-------------------------------------------------
	Name : plotClassifyTestData()
	Input = 2*n matrix of each class, g value list of eace class, state num
	Output = 3*3 confusion matrix
	Description : Use the boundary to display data in the correct area
					and calculate confusion matrix
-------------------------------------------------"""


def plotClassifyTestData(Iris1, Iris2, Iris3, gList1, gList2, gList3, num):
    cf1, cf2, cf3 = 0, 0, 0
    matrix = np.zeros(3)
    for i in range(0, 10):
        if num == 0:
            if gList1[0][i] - gList1[1][i] > 0:
                cf2 += 1
                plt.plot(Iris1[i][0], Iris1[i][1], 'ro')
            elif gList1[0][i] - gList1[2][i] > 0:
                cf3 += 1
                plt.plot(Iris1[i][0], Iris1[i][1], 'ro')
            else:
                cf1 += 1
        elif num == 1:
            if gList2[1][i] - gList2[0][i] > 0:
                cf1 += 1
                plt.plot(Iris2[i][0], Iris2[i][1], 'r^')
            elif gList2[1][i] - gList2[2][i] > 0:
                cf3 += 1
                plt.plot(Iris2[i][0], Iris2[i][1], 'r^')
            else:
                cf2 += 1
        elif num == 2:
            if gList3[2][i] - gList3[0][i] > 0:
                cf1 += 1
                plt.plot(Iris3[i][0], Iris3[i][1], 'rs')
            elif gList3[2][i] - gList3[1][i] > 0:
                cf2 += 1
                plt.plot(Iris3[i][0], Iris3[i][1], 'rs')
            else:
                cf3 += 1

    matrix[0], matrix[1], matrix[2] = cf1, cf2, cf3
    return matrix


# -------read file, init variables, calculate confusion matrix
if __name__ == "__main__":
    File = open("iris_data_tb.txt", 'r')
    lines = File.readlines()
    features = []

    # variable of Iris Info 
    setosa = []
    versicolor = []
    virginica = []

    # mean of Iris Info
    setosaMeanVector = []
    versicolorMeanVector = []
    virginicaMeanVector = []

    # covariance matrix of Iris Info
    setosaCovarianceMatrix = []
    versicolorCovarianceMatrix = []
    virginicaCovarianceMatrix = []

    for line in lines:
        word = line.split("\t")
        features.append(word)

    File.close()

    for i in range(0, len(features)):
        if features[i][1] == "0" or features[i][1] == "0\n":
            for j in range(0, 4):
                setosa.append(float(features[i][j]))
        elif features[i][1] == "1" or features[i][1] == "1\n":
            for j in range(0, 4):
                versicolor.append(float(features[i][j]))
        elif features[i][1] == "2" or features[i][1] == "2\n":
            for j in range(0, 4):
                virginica.append(float(features[i][j]))

    print("----------------- project #1-1 -----------------")
    print("----setosa Mean----")
    setosaMeanVector = cal_mean(setosa)
    print("----versicolor Mean----")
    versicolorMeanVector = cal_mean(versicolor)
    print("----virginica Mean----")
    virginicaMeanVector = cal_mean(virginica)

    print("----setosa Covariance Matrix----")
    setosaCovarianceMatrix = calCovariance(setosa, setosaMeanVector)
    print("----versicolor Covariance Matrix----")
    versicolorCovarianceMatrix = calCovariance(versicolor, versicolorMeanVector)
    print("----virginica Covariance Matrix----")
    virginicaCovarianceMatrix = calCovariance(virginica, virginicaMeanVector)

    # --------------------Iris_test.dat.txt read------------------
    File = open("iris_data_tb.txt", 'r')
    lines = File.readlines()

    # each value of test data set
    setosaTestData = np.zeros((10, 4))
    versicolorTestData = np.zeros((10, 4))
    virginicaTestData = np.zeros((10, 4))

    featuresTestData = []

    confusionMatrix = np.zeros((3, 3))

    for line in lines:
        word = line.split("\t")
        featuresTestData.append(word)

    File.close()

    # save 
    for i in range(0, len(featuresTestData)):
        if featuresTestData[i][1] == "0" or featuresTestData[i][1] == "0\r\n":
            for j in range(0, 4):
                setosaTestData[i][j] = (float(featuresTestData[i][j]))
        elif featuresTestData[i][1] == "1" or featuresTestData[i][1] == "1\r\n":
            for j in range(0, 4):
                versicolorTestData[i - 10][j] = (float(featuresTestData[i][j]))
        elif featuresTestData[i][1] == "2" or featuresTestData[i][1] == "2\r\n":
            for j in range(0, 4):
                virginicaTestData[i - 20][j] = (float(featuresTestData[i][j]))

    # calculate vi0
    setosaVi0 = calVi0(setosaMeanVector, setosaCovarianceMatrix)
    versicolorVi0 = calVi0(versicolorMeanVector, versicolorCovarianceMatrix)
    virginicaVi0 = calVi0(virginicaMeanVector, virginicaCovarianceMatrix)

    # calculate confusion matrix
    for i in range(0, len(setosaTestData)):
        gx1 = discriminantFunctions(setosaTestData[i], setosaMeanVector, setosaCovarianceMatrix, setosaVi0)
        gx2 = discriminantFunctions(setosaTestData[i], versicolorMeanVector, versicolorCovarianceMatrix, versicolorVi0)
        gx3 = discriminantFunctions(setosaTestData[i], virginicaMeanVector, virginicaCovarianceMatrix, virginicaVi0)

        if gx1 - gx2 < 0:
            confusionMatrix[0][1] += 1
        elif gx1 - gx3 < 0:
            confusionMatrix[0][2] += 1
        else:
            confusionMatrix[0][0] += 1

    for i in range(0, len(versicolorTestData)):
        gx1 = discriminantFunctions(versicolorTestData[i], setosaMeanVector, setosaCovarianceMatrix, setosaVi0)
        gx2 = discriminantFunctions(versicolorTestData[i], versicolorMeanVector, versicolorCovarianceMatrix,
                                    versicolorVi0)
        gx3 = discriminantFunctions(versicolorTestData[i], virginicaMeanVector, virginicaCovarianceMatrix, virginicaVi0)

        if gx2 - gx1 < 0:
            confusionMatrix[1][0] += 1
        elif gx2 - gx3 < 0:
            confusionMatrix[1][2] += 1
        else:
            confusionMatrix[1][1] += 1

    for i in range(0, len(virginicaTestData)):
        gx1 = discriminantFunctions(virginicaTestData[i], setosaMeanVector, setosaCovarianceMatrix, setosaVi0)
        gx2 = discriminantFunctions(virginicaTestData[i], versicolorMeanVector, versicolorCovarianceMatrix,
                                    versicolorVi0)
        gx3 = discriminantFunctions(virginicaTestData[i], virginicaMeanVector, virginicaCovarianceMatrix, virginicaVi0)

        if gx3 - gx1 < 0:
            confusionMatrix[2][0] += 1
        elif gx3 - gx2 < 0:
            confusionMatrix[2][1] += 1
        else:
            confusionMatrix[2][2] += 1

    print("----confusion Matrix----")
    print(confusionMatrix)

    print("----------------- project #1-2 -----------------")
    # ---------------------project #1-2---------------------
    setosa2fMeanVector = setosaMeanVector[0:2]
    versicolor2fMeanVector = versicolorMeanVector[0:2]
    virginica2fMeanVector = virginicaMeanVector[0:2]

    setosa2fCovariance = np.zeros((2, 2))
    versicolor2fCovariance = np.zeros((2, 2))
    virginica2fCovariance = np.zeros((2, 2))

    setosa2fTestData = np.zeros((10, 2))
    versicolor2fTestData = np.zeros((10, 2))
    virginica2fTestData = np.zeros((10, 2))

    for i in range(0, 10):
        setosa2fTestData[i][0] = setosaTestData[i][0]
        setosa2fTestData[i][1] = setosaTestData[i][1]
        versicolor2fTestData[i][0] = versicolorTestData[i][0]
        versicolor2fTestData[i][1] = versicolorTestData[i][1]
        virginica2fTestData[i][0] = virginicaTestData[i][0]
        virginica2fTestData[i][1] = virginicaTestData[i][1]

    for i in range(0, 2):
        setosa2fCovariance[i][0] = setosaCovarianceMatrix[i][0]
        setosa2fCovariance[i][1] = setosaCovarianceMatrix[i][1]
        versicolor2fCovariance[i][0] = versicolorCovarianceMatrix[i][0]
        versicolor2fCovariance[i][1] = versicolorCovarianceMatrix[i][1]
        virginica2fCovariance[i][0] = virginicaCovarianceMatrix[i][0]
        virginica2fCovariance[i][1] = virginicaCovarianceMatrix[i][1]

    plotTheTraningData(setosa, versicolor, virginica)
    plotTheTestData(setosaTestData, versicolorTestData, virginicaTestData)

    print("----mean of setosa 2features----")
    print(setosa2fMeanVector)

    print("----mean of versicolor 2features----")
    print(versicolor2fMeanVector)

    print("----mean of virginica 2features----")
    print(virginica2fMeanVector)

    print("----covariance setosa of 2fefatures----")
    print(setosa2fCovariance)

    print("----covariance versicolor of 2fefatures----")
    print(versicolor2fCovariance)

    print("----covariance virginica of 2fefatures----")
    print(virginica2fCovariance)

    g1 = calMahalanobisDistance(setosa, setosa2fMeanVector, setosa2fCovariance)
    g2 = calMahalanobisDistance(versicolor, versicolor2fMeanVector, versicolor2fCovariance)
    g3 = calMahalanobisDistance(virginica, virginica2fMeanVector, virginica2fCovariance)

    plotThedecisionBoundaries(g1, g2, g3)

    setosa2fGList = np.zeros((3, 10))
    versicolor2fGList = np.zeros((3, 10))
    virginica2fGList = np.zeros((3, 10))

    confusionMatrix2f = np.zeros((3, 3))

    setosa2fGList[0] = calGListofMD(setosa2fTestData, setosa2fMeanVector, setosa2fCovariance)
    setosa2fGList[1] = calGListofMD(setosa2fTestData, versicolor2fMeanVector, versicolor2fCovariance)
    setosa2fGList[2] = calGListofMD(setosa2fTestData, virginica2fMeanVector, virginica2fCovariance)
    versicolor2fGList[0] = calGListofMD(versicolor2fTestData, setosa2fMeanVector, setosa2fCovariance)
    versicolor2fGList[1] = calGListofMD(versicolor2fTestData, versicolor2fMeanVector, versicolor2fCovariance)
    versicolor2fGList[2] = calGListofMD(versicolor2fTestData, virginica2fMeanVector, virginica2fCovariance)
    virginica2fGList[0] = calGListofMD(virginica2fTestData, setosa2fMeanVector, setosa2fCovariance)
    virginica2fGList[1] = calGListofMD(virginica2fTestData, versicolor2fMeanVector, versicolor2fCovariance)
    virginica2fGList[2] = calGListofMD(virginica2fTestData, virginica2fMeanVector, virginica2fCovariance)

    confusionMatrix2f[0] = plotClassifyTestData(setosa2fTestData, versicolor2fTestData, virginica2fTestData,
                                                setosa2fGList, versicolor2fGList, virginica2fGList, 0)
    confusionMatrix2f[1] = plotClassifyTestData(setosa2fTestData, versicolor2fTestData, virginica2fTestData,
                                                setosa2fGList, versicolor2fGList, virginica2fGList, 1)
    confusionMatrix2f[2] = plotClassifyTestData(setosa2fTestData, versicolor2fTestData, virginica2fTestData,
                                                setosa2fGList, versicolor2fGList, virginica2fGList, 2)

    print("----Confusion Matrix 2f of test data")
    print(confusionMatrix2f)

    plt.show()
