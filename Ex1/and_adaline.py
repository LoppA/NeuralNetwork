import Adaline as ada
import numpy as np

per = ada.Adaline(2, ada.step)

and_input = np.array([ [1, 0, 0],
                       [1, 0, 1],
                       [1, 1, 0],
                       [1, 1, 1] ]).astype(np.float64)

and_output = np.array( [0, 0, 0, 1] ).astype(np.int)

n = and_input.shape[0]
cnt = 4
while(cnt != 0):
    cnt = 0

    for i in range(n):
        differ = per.train(and_input[i], and_output[i])
        cnt += differ*differ

print('\n', per, sep='')
