"""
A multilayered perceptron to tell you if -
You will attend a tie session based on these three things
1. Interest
2. Availability
3. Alignment with business
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


def make_plot(errors):
    ax = pd.DataFrame({'Error %': errors}).plot(marker='', color='olive', linewidth=2, linestyle='dashed')
    ax.set_xlabel('Iterations * 1000')
    ax.set_ylabel('Error %')
    plt.show()
    print("\n\n")


class NeuralNetwork():
    def __init__(self):
        # Seed the random number generator, so it generates the same numbers
        # every time the program runs.
        np.random.seed(1)

        # We model a single neuron, with 3 input connections and 1 output connection.
        # We assign random weights to a 3 x 1 matrix, with values in the range -1 to 1
        # and mean 0.
        self.synaptic_weights = 2 * np.random.random((4, 1)) - 1

    # The Sigmoid function, which describes an S shaped curve.
    # We pass the weighted sum of the inputs through this function to
    # normalise them between 0 and 1.
    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    # It indicates how confident we are about the existing weight.
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time.
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        errors = []
        for iteration in range(number_of_training_iterations):
            # Pass the training set through our neural network (a single neuron).
            output = self.think(training_set_inputs)

            # Calculate the error (The difference between the desired output
            # and the predicted output).
            error = training_set_outputs - output
            # print(error)

            # Multiply the error by the input and again by the gradient of the Sigmoid curve.
            # This means less confident weights are adjusted more.
            # This means inputs, which are zero, do not cause changes to the weights.
            adjustment = np.dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

            # Adjust the weights.
            self.synaptic_weights += adjustment
            if (iteration % 1000 == 0):
                # print("error after %s iterations: %s" % (iteration, str(np.mean(np.abs(error)))))
                errors.append(np.mean(np.abs(error)))
        print("Training Complete")
        return errors

    # The neural network thinks.
    def think(self, inputs):
        # Pass inputs through our neural network (our single neuron).
        return self.__sigmoid(np.dot(inputs, self.synaptic_weights))


if __name__ == "__main__":
    # Intialise a single neuron neural network.
    neural_network = NeuralNetwork()

    # print("Random starting synaptic weights: ")
    # print(neural_network.synaptic_weights)

    # The training set. We have 4 examples, each consisting of 3 input values
    # and 1 output value.
    # Remember:
    # You will attend a tie session based on these three things
    # 1. Interest
    # 2. Availability
    # 3. Alignment with business

    training_set_inputs = np.array([[1, 1, 1, 1], [1, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, 0]])
    training_set_outputs = np.array([[1, 1, 0, 0]]).T

    # Train the neural network using a training set.
    # Do it 10,000 times and make small adjustments each time.
    errors = neural_network.train(training_set_inputs, training_set_outputs, 20000)

    # print("New synaptic weights after training: ")
    # print(neural_network.synaptic_weights)

    # Test the neural network with a new
    test = [0, 0, 1, 1]
    pretty_print(test)
    # print("Considering new situation %s -> ?: " % test)
    print("Will attend score: ", end='')
    print(neural_network.think(np.array(test))[0])
    make_plot(errors)
