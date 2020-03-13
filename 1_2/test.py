import tensorflow.compat.v1 as tf
import numpy as np

l={'a':[2,23],'b':[1,2,2,3,3,3,43],'z':[3,3,3,3]}

# print((l.keys()))
# for i in range(len(l.keys())):
for i in range(2):
    with open('history.txt', 'a') as f:
        # f.truncate()
        for keys, values in l.items():
            f.write(keys+'\t')
            for i in range(len(l[keys])):
                f.write(str(l[keys][i])+'\t')
            f.write('\n')
