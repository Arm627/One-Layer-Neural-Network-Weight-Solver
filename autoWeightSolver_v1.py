import numpy as np 
import tensorflow as tf
import time, heapq

class Activations:
    '''
    Inverse Activations Class
    '''
    def inverseSigmoid(self, z: np.ndarray) -> np.ndarray:
        '''
        -z: numpy array computing; -ln(1/z - 1). Domain: 0 < z < 1. Range: All Real Numbers
        Graph: https://www.desmos.com/calculator/1sooukdxx0
        '''
        return -np.log(1/z - 1)

    def inverseSoftmax(self, z: np.ndarray) -> np.ndarray:
        '''
        -z: numpy array computing the inverse softmax. ln(x) + C

        C = 1 because through trial and error seems best results
        '''
        return np.add(np.log(z), 1)

    def inverseLeakyReLU(self, z: np.ndarray) -> np.ndarray:
        '''
        -z: numpy array computing the inverse Leaky ReLU. if z > 0: z else: z/0.01

        Note: really bad results for LRELU
        '''
        return np.where(z > 0, z, z / 0.01)  

class AutoWeightSolver:
    '''
    AutoWeightSolver

    theta^T = inverseSigmoid(y_actual)*x^(-1)
    where thetha^T is hopefully the matrix weights for associated neural network.
    y_actual and x are nxn matrix where n is the number of dense features.

    Note: right now assuming 2d array
    '''
    def __init__(self, x: np.ndarray, y_actual: np.ndarray, add_bias=True):
        '''
        -x: the inputs
        -y_actual: the wanted predicted values
        '''
        self.x = x
        self.y_actual = y_actual
        self.add_bias = add_bias
        self.activations = Activations()

    def weightSolver(self, activation='sigmoid'):
        if activation == 'sigmoid':
            activation = self.activations.inverseSigmoid
        elif activation == 'softmax':
            activation = self.activations.inverseSoftmax
        elif activation == 'lrelu':
            activation = self.activations.inverseLeakyReLU

        if self.add_bias == True:
            average_weights = np.zeros(shape=(self.y_actual[0].shape[0], self.x[0].shape[1]+1))
        else:
            average_weights = np.zeros(shape=(self.y_actual[0].shape[0], self.x[0].shape[1]))

        for i in range(len(self.x)):
            sigmoid_y = activation(self.y_actual[i])
            average_weights = np.add(self.matrixDivision(self.x[i], sigmoid_y), average_weights)
        average_weights = np.divide(average_weights, len(self.x))
        self.theta = average_weights
        if self.add_bias == True:
            self.bias = average_weights[:, 0]
            self.weights = average_weights[:, 0 : -1]
        else:
            self.bias = None 
            self.weights = self.theta

    def matrixDivision(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.add_bias == True:
            div = np.divide(x, y)
            if np.argmax(div.shape) == 0:
                div = np.append(div, np.zeros((1, div.shape[1])), 0)
            elif np.argmax(div.shape) == 1:
                div = np.append(div, np.zeros((div.shape[0], 1)), 1)
            return div
        else:
            return np.divide(x, y)

    def evaluate(self, pred: np.ndarray) -> np.ndarray:
        '''
        -pred: input to be predicted
        '''
        if self.add_bias == True:
            if self.weights.shape[1] == pred.T.shape[0]:
                return np.add(np.matmul(self.weights, pred.T).flatten(), self.bias)
            else:
                return np.add(np.matmul(self.weights, pred).flatten(), self.bias)
        else:
            if self.weights.shape[1] == pred.T.shape[0]:
                return np.matmul(self.weights, pred.T)
            else:
                return np.matmul(self.weights, pred)