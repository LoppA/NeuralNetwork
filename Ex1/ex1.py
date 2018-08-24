import Adaline as ada
import numpy as np

# Inicializacao do perceptron
per = ada.Adaline(25, ada.sigmoid)

# Leitura do input de test para A
f = open('A_input_test.txt')
A_input_test = np.ones((3, 26)).astype(np.float64)
A_input_test[0][0] = 1
A_input_test[1][0] = 1
A_input_test[2][0] = 1
i = 0
j = 1
for line in f:
    values = list(map(int, line.split()))

    if(len(values) == 0):
        i += 1
        j = 1
    else:
        for val in values:
            A_input_test[i][j] = val
            j+=1
A_out_test = np.array([1, 1, 1])

# Leitura do input de treino para A
f = open('A_input_train.txt')
A_input_train = np.ones((3, 26)).astype(np.float64)
A_input_train[0][0] = 1
A_input_train[1][0] = 1
A_input_train[2][0] = 1
i = 0
j = 1
for line in f:
    values = list(map(int, line.split()))

    if(len(values) == 0):
        i += 1
        j = 1
    else:
        for val in values:
            A_input_train[i][j] = val
            j+=1
A_out_train = np.array([1, 1, 1])

# Leitura do input de test para A invertido
f = open('notA_input_test.txt')
notA_input_test = np.ones((3, 26)).astype(np.float64)
notA_input_test[0][0] = 1
notA_input_test[1][0] = 1
notA_input_test[2][0] = 1
i = 0
j = 1
for line in f:
    values = list(map(int, line.split()))

    if(len(values) == 0):
        i += 1
        j = 1
    else:
        for val in values:
            notA_input_test[i][j] = val
            j+=1
notA_out_test = np.array([-1, -1, -1])

# Leitura do input de treino para A invertido
f = open('notA_input_train.txt')
notA_input_train = np.ones((3, 26)).astype(np.float64)
notA_input_train[0][0] = 1
notA_input_train[1][0] = 1
notA_input_train[2][0] = 1
i = 0
j = 1
for line in f:
    values = list(map(int, line.split()))

    if(len(values) == 0):
        i += 1
        j = 1
    else:
        for val in values:
            notA_input_train[i][j] = val
            j+=1
notA_out_train = np.array([-1, -1, -1])

f.close()

# Algoritmo LMS
EPS = 1e-9
err = 3
while(err > EPS):
    err = 0

    # Treinando com os modelos de A
    n = A_input_train.shape[0]
    for i in range(n):
        differ = per.train(A_input_train[i], A_out_train[i])
        err += differ*differ

    # Treinando com os modelos de A invertido
    n = notA_input_train.shape[0]
    for i in range(n):
        differ = per.train(notA_input_train[i], notA_out_train[i])
        err += differ*differ

# Impressao do perceptron
print('\n', per, sep='')

# Testando para os casos de teste de A
n = A_input_test.shape[0]
for i in range(n):
    print("A test", i, per.run(A_input_test[i]))

# Testando para os casos de teste de A invertido
n = notA_input_test.shape[0]
for i in range(n):
    print("Not A test", i, per.run(notA_input_test[i]))
