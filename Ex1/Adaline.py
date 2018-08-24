from math import exp
import numpy as np

def sigmoid(x):
    return (2 / (1 + exp(-x))) - 1

def step(x):
    if(x > 0):
        return int(1)
    return int(0)

THETA = 1.0
ETA = 0.5

class Adaline:

    def __init__(self, n, fun=sigmoid):
        self.actv = fun
        self.w = np.zeros(n+1).astype(np.float64)

    def __str__(self):
        return ("Size: " + str(self.w.shape) + "\nW: " + str(self.w))

    def run(self, x, t):
        y = np.sum(x * self.w)
        return self.actv(y)

    def train(self, x, t):
        y = self.run(x, t)
        e = t - y
        if(e != 0):
            self.w += ETA*e*x
        return e

