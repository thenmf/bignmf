import numpy as np
import pandas as pd

class ToyMatrix:
    def __init__(self, method='a'):
        if method == 'a':
            self.toy_matrix_a()
        elif method == 'b':
            self.toy_matrix_b()
        elif method == 'c':
            self.toy_matrix_c()
        elif method == 'm':
            self.toy_matrix_multilple()

    def toy_matrix_a(self, m=40, n=30, k=2, seed=0):
        np.random.seed(seed)
        self.W = np.random.random((n, k))
        self.H = np.ones((k, m))
        self.H[0, :10] = 0.5
        self.H[1, 10:] = 0.2
        self.X = np.dot(self.W, self.H)

    def toy_matrix_b(self, m=40, n=30, k=3, seed=0):
        np.random.seed(seed)
        self.W = np.random.random((n, k))
        self.H = np.zeros((k, m))
        self.H[0, :10] = 0.5
        self.H[1, 10:20] = 0.1
        self.H[2, 20:] = 0.3
        self.X = np.dot(self.W, self.H)

    def toy_matrix_c(self, m=100, n=40, k=7, seed=0):
        np.random.seed(seed)
        self.W = np.random.random((n, k))
        self.H = np.zeros((k, m))
        self.H[0, :10] = 0.5
        self.H[1, 10:20] = 0.1
        self.H[2, 20:25] = 0.3
        self.H[3, 25:45] = 0.452
        self.H[4, 45:60] = 0.8
        self.H[5, 60:67] = 0.59
        self.H[6, 67:] = 0.702
        np.random.shuffle(self.H)
        self.X = np.dot(self.W, self.H)

    def toy_matrix_multilple(self):

        w = np.ones((1, 10))
        h1 = np.ones((1, 30))
        h2 = np.ones((1, 40))
        h3 = np.ones((1, 50))

        self.W = np.zeros((30, 3))
        self.H = {}
        self.x = {}

        self.H['a'] = np.zeros((3, 90))
        self.H['b'] = np.zeros((3, 120))
        self.H['c'] = np.zeros((3, 150))

        self.W[0:10, 0:1] = w.T
        self.W[10:20, 1:2] = w.T
        self.W[20:30, 2:3] = w.T

        self.H['a'][0:1, 0:30] = h1
        self.H['a'][1:2, 30:60] = h1
        self.H['a'][2:3, 60:90] = h1

        self.H['b'][0:1, 0:40] = h2
        self.H['b'][1:2, 40:80] = h2
        self.H['b'][2:3, 80:120] = h2

        self.H['c'][0:1, 0:50] = h3
        self.H['c'][1:2, 50:100] = h3
        self.H['c'][2:3, 100:150] = h3

        X1 = np.dot(self.W, self.H['a'])
        X2 = np.dot(self.W, self.H['b'])
        X3 = np.dot(self.W, self.H['c'])
        np.random.rand(3, 2)

        self.x['a'] = X1 + 0.5 * np.random.rand(30, 90)
        self.x['b'] = X2 + 0.5 * np.random.rand(30, 120)
        self.x['c'] = X3 + 0.5 * np.random.rand(30, 150)
        self.x = {k:pd.DataFrame(self.x[k], columns=list(range(self.x[k].shape[1]))) for k in self.x}
        # plt.imshow(self.XX1, cmap='hot', interpolation='nearest')
        # plt.colorbar()
        # plt.show()

        # Permutations
        # p = np.random.permutation(30)
        # p1 = np.random.permutation(90)
        # p2 = np.random.permutation(120)
        # p3 = np.random.permutation(150)

        # RXX1, RXX2, RXX3 = XX1, XX2, XX3
        # for i in range(30):
        #     RX1[0:i, :] = XX1[p[i], :]
        #     RX2[0:i, :] = XX2[p[i], :]
        #     RX3[0:i, :] = XX3[p[i], :]
        #
        # for i in range(90):
        #     RXX1[:, 0:i] = RX1[:, 0:p1[i]]
        #
        # for i in range(120):
        #     RXX2[:, 0:i] = RX2[:, 0:p2[i]]
        #
        # for i in range(150):
        #     RXX3[:, 0:i] = RX3[:, 0:p3[i]]

    def cluster(self):
        return np.argmax(self.H, 0)
