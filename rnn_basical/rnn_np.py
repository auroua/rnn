import numpy as np


class RNNNumpy(object):
    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        self.U = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        self.V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))
        self.hidden_bias = np.random.uniform(-1, 1, (1, hidden_dim))
        self.output_bias = np.random.uniform(-1, 1, (1, word_dim))

    def forward_propagation(self, x):
        # input x is the index of the word, not a one_hot vector
        T = len(x)
        h_state = np.zeros((T+1, self.hidden_dim))
        output = np.zeros((T, self.word_dim))
        for i in range(T):
            h_state[i+1] = np.tanh(self.U[:, x[i]] + np.dot(self.W, h_state[i]))
            output[i] = self.softmax(np.dot(self.V, h_state[i+1]))
        return h_state, output

    def predict(self, x):
        s, o = self.forward_propagation(x)
        index = np.argmax(o, axis=1)
        return index

    # y is a collection of sentences
    def calculate_total_loss(self, x, y):
        L = 0
        for i in range(len(y)):
            s, o = self.forward_propagation(x[i])
            correct_predict = o[np.arange(len(y[i])), y[i]]
            L += -1*np.sum(np.log(correct_predict))
        return L

    def calculate_loss(self, x, y):
        N = np.sum((len(y_i) for y_i in y))
        return self.calculate_total_loss(x, y)/N

    def bptt(self, x, y):
        T = len(y)
        s, o = self.forward_propagation(x)
        dLdU = np.zeros((self.hidden_dim, self.word_dim))
        dLdV = np.zeros((self.word_dim, self.hidden_dim))
        dLdW = np.zeros((self.hidden_dim, self.hidden_dim))
        delta_o = o
        delta_o[np.arange(len(y)), y] -= 1
        # o = T * word_dim
        for t in np.arange(T):
            dLdV += np.outer(delta_o[t], s[t+1].T)

    def softmax(self, o):
        exp_o = np.exp(o)
        sum_o = np.sum(exp_o)
        return exp_o/sum_o


if __name__ == '__main__':
    x_train = np.load('../data/rnn_1/x_train.npy')
    y_train = np.load('../data/rnn_1/y_train.npy')
    np.random.seed(10)
    rnn = RNNNumpy(8000)
    # Limit to 1000 examples to save time
    print("Expected Loss for random predictions: %f" % np.log(8000))
    print("Actual loss: %f" % rnn.calculate_loss(x_train[:1000], y_train[:1000]))