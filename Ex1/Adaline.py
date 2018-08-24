from math import exp
import numpy as np

# Funcao sigmoide
def sigmoid(x):
    return (2 / (1 + exp(-x))) - 1

# Funcao degrau
def step(x):
    if(x > 0):
        return int(1)
    return int(0)

THETA = 1.0     # Constante theta
ETA = 0.5       # Constate ete, velocidade de aprendizado

class Adaline:

    def __init__(self, n, fun=sigmoid):
        self.actv = fun
        self.w = np.zeros(n+1).astype(np.float64)

    def __str__(self):
        return ("Size: " + str(self.w.shape) + "\nW: " + str(self.w))

    # Verificar a saida y dado input x
    def run(self, x):
        y = np.sum(x * self.w)
        return self.actv(y)

    # Treinar dado input x e saida esperada t
    def train(self, x, t):
        y = self.run(x)
        e = t - y
        if(e != 0):
            self.w += ETA*e*x
        return e

