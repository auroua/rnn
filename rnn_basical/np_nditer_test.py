import numpy as np

if __name__ == '__main__':
    a = np.arange(6).reshape(2, 3)
    for x in np.nditer(a):
        print(x)

    print('################')
    for x in np.nditer(a, op_flags=['readwrite']):
        x[...] = 2*x
    print(a)

    print('###################')
    a = np.arange(6).reshape(2, 3)
    it = np.nditer(a, flags=['f_index'])
    while not it.finished:
        print("%d <%d>" % (it[0], it.index))
        it.iternext()

    it = np.nditer(a, flags=['multi_index'])
    while not it .finished:
        print("%d <%s>" % (it[0], it.multi_index))
        it.iternext()