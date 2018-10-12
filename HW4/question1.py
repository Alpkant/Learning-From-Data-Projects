import numpy as np
import matplotlib.pyplot as plt
import random
from math import exp
from sys import exit

def main():
    numberOfIteration = int(input("How many iteration do you want?"))
    learningRate      = float(input("What learning rate do you want?"))
    #I shuffled the data
    x   = np.loadtxt('dataTrain.csv', delimiter=',',skiprows=1, dtype=int)
    np.random.shuffle(x)

    #64 features and label matrixes
    dataset = x[:,:64]
    labels  = x[:,64:65]

    inputNumber  = len(dataset[0])
    outputNumber = len(labels)

    #Initialize the multilayer model, hidden layer number is 10
    network = initialize_network(inputNumber, 10, 10)
    #Train network with 0.1 learning rate and 20 loops
    train_network(network, x, learningRate, numberOfIteration, 10)
    true  = 0
    false = 0
    for row in x:
        prediction = predict(network, row)
        if row[-1] == prediction:
            true+=1
        else:
            false += 1
    print("Accuracy = ", true*100/(true+false) ,"%")



def initialize_network(n_inputs, n_hidden, n_outputs):
    #Create dictionary
    network = list()
    # For hidden layer weights add realy small weights
    hidden = [{'w':[random.uniform(-0.1, 0.1) for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden)
    # For output layer weights add realy small weights
    output = [{'w':[random.uniform(-0.1, 0.1) for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output)
    return network


def predict(network, test):
    outputs = forward(network, test)
    return outputs.index(max(outputs)) #Predict as it belongs class with biggest probability



# Sigmoid function for activation of the neuron
def sigmoid(x):
    if x<0:
        a = exp(x)
        return a / (1 + a)
    else:
        return 1 / (1 + exp(-x))

#Derivation of the sigmoid function
def sigmoid_derivative(x):
    return x * (1.0 - x)

# Calculate sum of activations of the neurons by using weights
def activation(weights, data):
    sumOfProduct = weights[-1] #Last weight
    for i in range(len(weights)-1):
        sumOfProduct += weights[i] * data[i] # weight*data
    return sumOfProduct

# Calculate the outputs of the each neurons with  new weights
def forward(network, row):
    inputs = row
    for i in network:
        modifiedVals = []
        for j in i:
            j['out'] = sigmoid (activation(j['w'], inputs))
            modifiedVals.append(j['out'])
        inputs = modifiedVals

    return inputs



def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        #len değeri printle
        if i != len(network)-1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['w'][j] * neuron['deltaw'])
                errors.append(error) #Error of neuron
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['out'])

        for j in range(len(layer)):
            neuron = layer[j]
            neuron['deltaw'] = errors[j] * sigmoid_derivative(neuron['out']) # Error that multiplied by sigmoid derivative






# Update network weights with error
# It doesn't return anything since it directly updates the dictionary
def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [j['out'] for j in network[i - 1]]
            for j in network[i]:
                for k in range(len(inputs)):
                    j['w'][k] += l_rate * j['deltaw'] * inputs[k]

                j['w'][-1]+= l_rate * j['deltaw']


def train_network(network, train, l_rate, iterations, n_outputs):
    for iteration in range(iterations): # For number of iterations that user want
        sum_error = 0
        for row in train:
            outputs = forward(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1 # Make bias unit as 1
            sum_error -= sum([ expected[i]*np.log(outputs[i]) for i in range(n_outputs)]) # Cross entropy for error
            backward_propagate_error(network, expected) # According to the error
            update_weights(network, row, l_rate)
        if iteration <= 9 or iteration == 49 or iteration==99 or iteration == 199 :
            print("Iteration", iteration+1 , "Cross entropy error : " , sum_error)


main()
