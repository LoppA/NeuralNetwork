import Adaline as ada
import numpy as np

per = ada.Adaline(25, ada.sigmoid)

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

EPS = 1e-9
err = 3
while(err > EPS):
    err = 0

    n = A_input_train.shape[0]
    for i in range(n):
        differ = per.train(A_input_train[i], A_out_train[i])
        err += differ*differ

    n = notA_input_train.shape[0]
    for i in range(n):
        differ = per.train(notA_input_train[i], notA_out_train[i])
        err += differ*differ

print('\n', per, sep='')

n = A_input_test.shape[0]
for i in range(n):
    print("A test", i, per.run(A_input_test[i], A_out_test[i]))

n = notA_input_test.shape[0]
for i in range(n):
    print("Not A test", i, per.run(notA_input_test[i], A_out_test[i]))
