"""
A multilayered perceptron to tell you if -
You will attend a tie session based on these three things
1. Interest
2. Availability
3. Alignment with business
"""

import numpy as np
from random import seed
from random import randrange

train = np.array([[1, 1, 1, 1], [1, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, 0]])
weights = []


def pretty_print(arr):
    """
    takes a numpy array and outputs the meaning of each 0 and 1
    """
    if arr[0] == 0:
        print("Interest: No")
    else:
        print("Interest: Yes")
    if arr[1] == 0:
        print("Availability: No")
    else:
        print("Availability: Yes")
    if arr[2] == 0:
        print("Alignment with business: No")
    else:
        print("Alignment with business: Yes")
    if arr[3] == 0:
        print("Will be attending the session?: No")
    else:
        print("Will be attending the session?: Yes")


def predict(row, weights):
    activation = weights[0]
    for i in range(len(row) - 1):
        activation += weights[i] * row[i]
    return 1.0 if activation >= 0.0 else 0.0


# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


def train_weights(train, l_rate, n_epoch):
    weights = [0.0 for i in range(len(train[0]))]
    for epoch in range(n_epoch):
        # print(weights)
        # print(train)
        for row in train:
            prediction = predict(row, weights)
            # print(row[-1])
            error = row[-1] - prediction
            weights[0] = weights[0] + l_rate * error
            for i in range(len(row) - 1):
                weights[i + 1] = weights[i + 1] + l_rate * error * row[i]

    return weights


def perceptron(train, test, l_rate, n_epoch):
    predictions = list()
    weights = train_weights(train, l_rate, n_epoch)
    # for row in test:
    prediction = predict(test, weights)
    predictions.append(prediction)
    return (predictions)


def get_input():
    print("Enter the weightage you give to each crieteria in % ex- 30 30 40")
    print("""    1. Interest
    2. Available
    3. Alignment with business""")
    a = list(map(int, input().split()))
    for percentages in a:
        weights.append(percentages / 100)

    print("For the event: Applied AI- Enter 1 for yes abd 0 for no")
    a = int(input("Interested:"))
    b = int(input("Available:"))
    c = int(input("Is aligned with my business:"))
    test = [a, b, c, 0]
    return test


# seed(1)
n_folds = 3
l_rate = 0.01
n_epoch = 5000
# test = [1, 0, 1, 0]
test = get_input()
print(perceptron(train, test, l_rate, n_epoch))
