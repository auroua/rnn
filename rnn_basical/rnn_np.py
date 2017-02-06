import numpy as np
import operator
from datetime import datetime
import sys

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
            h_state[i] = np.tanh(self.U[:, x[i]] + np.dot(self.W, h_state[i-1]))
            output[i] = self.softmax(np.dot(self.V, h_state[i]))
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
        # one sentence pre time
        T = len(y)
        s, o = self.forward_propagation(x)
        dLdU = np.zeros((self.hidden_dim, self.word_dim))
        dLdV = np.zeros((self.word_dim, self.hidden_dim))
        dLdW = np.zeros((self.hidden_dim, self.hidden_dim))
        delta_o = o
        delta_o[np.arange(len(y)), y] -= 1
        # o = T * word_dim
        for t in np.arange(T)[::-1]:
            # the parameter of outer should be column vector
            dLdV += np.outer(delta_o[t], s[t].T)
            delta_t = self.V.T.dot(delta_o[t]) * (1 - s[t]**2)
            for bptt_step in np.arange(max(0, t-self.bptt_truncate), t+1)[::-1]:
                dLdW += np.outer(delta_t, s[bptt_step-1])
                # dLdU[:, x[bptt_step]] += delta_t
                dLdU += np.outer(delta_t, self.one_hot(x[bptt_step]))
                # update delta for next step dL/dz at t-1
                delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step-1]**2)
        return [dLdU, dLdV, dLdW]

    def gradient_check(self, x, y, h=0.001, error_threshold=0.01):
        bptt_gradients = self.bptt(x, y)
        model_parameter = ['U', 'V', 'W']
        for pidx, pname in enumerate(model_parameter):
            parameter = operator.attrgetter(pname)(self)
            print("Performing gradient check for parameter %s with size %d." % (pname, np.prod(parameter.shape)))
            it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                ix = it.multi_index
                original_value = parameter[ix]
                parameter[ix] = original_value + h
                gradplus = self.calculate_total_loss([x], [y])
                parameter[ix] = original_value - h
                gradminus = self.calculate_total_loss([x], [y])
                estimated_gradient = (gradplus - gradminus) / (2.0*h)
                parameter[ix] = original_value
                backprop_gradient = bptt_gradients[pidx][ix]
                # calculate The relative error: (|x - y|/(|x| + |y|))
                relative_error = np.abs(backprop_gradient - estimated_gradient) / (np.abs(backprop_gradient) + np.abs(estimated_gradient))
                if relative_error > error_threshold:
                    print("Gradient Check ERROR: parameter=%s ix=%s" % (pname, ix))
                    print("+h Loss: %f" % gradplus)
                    print("-h Loss: %f" % gradminus)
                    print("Estimated_gradient: %f" % estimated_gradient)
                    print("Backpropagation gradient: %f" % backprop_gradient)
                    print("Relative Error: %f" % relative_error)
                    return
                it.iternext()
            print("Gradient check for parameter %s passed." % (pname))

    def sgd_step(self, x, y, learning_rate):
        dLdU, dLdV, dLdW = self.bptt(x, y)
        self.U -= learning_rate*dLdU
        self.V -= learning_rate*dLdV
        self.W -= learning_rate*dLdW


    def one_hot(self, x):
        one_hot_vec = np.zeros((self.word_dim, 1))
        one_hot_vec[x] = 1
        return one_hot_vec

    def softmax(self, o):
        exp_o = np.exp(o)
        sum_o = np.sum(exp_o)
        return exp_o/sum_o


# Outer SGD Loop
# - model: The RNN model instance
# - X_train: The training data set
# - y_train: The training data labels
# - learning_rate: Initial learning rate for SGD
# - nepoch: Number of times to iterate through the complete dataset
# - evaluate_loss_after: Evaluate the loss after this many epochs
def train_with_sgd(model, X_train, y_train, learning_rate=0.005, nepoch=100, evaluate_loss_after=5):
    # We keep track of the losses so we can plot them later
    losses = []
    num_examples_seen = 0
    for epoch in range(nepoch):
        # Optionally evaluate the loss
        if (epoch % evaluate_loss_after == 0):
            loss = model.calculate_loss(X_train, y_train)
            losses.append((num_examples_seen, loss))
            time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print("%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss))
            # Adjust the learning rate if loss increases
            if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                learning_rate = learning_rate * 0.5
                print("Setting learning rate to %f" % learning_rate)
            sys.stdout.flush()
        # For each training example...
        for i in range(len(y_train)):
            # One SGD step
            model.sgd_step(X_train[i], y_train[i], learning_rate)
            num_examples_seen += 1

# generate sentence
def generate_sentence(model, word_to_index, index_to_word, sentence_start_token, sentence_end_token, unknown_token):
    # We start the sentence with the start token
    new_sentence = [word_to_index[sentence_start_token]]
    # Repeat until we get an end token
    while not new_sentence[-1] == word_to_index[sentence_end_token]:
        # the return value of forward_propagation is list
        next_word_probs = model.forward_propagation(new_sentence)
        sampled_word = word_to_index[unknown_token]
        # We don't want to sample unknown words
        while sampled_word == word_to_index[unknown_token]:
            # 1 numbers of experiments
            samples = np.random.multinomial(1, next_word_probs[-1])
            sampled_word = np.argmax(samples)
        new_sentence.append(sampled_word)
    sentence_str = [index_to_word[x] for x in new_sentence[1:-1]]
    return sentence_str




if __name__ == '__main__':
    x_train = np.load('../data/rnn_1/x_train.npy')
    y_train = np.load('../data/rnn_1/y_train.npy')
    np.random.seed(10)
    grad_check_vocab_size = 8000
    rnn = RNNNumpy(grad_check_vocab_size)
    # rnn.gradient_check([0, 1, 2, 3], [1, 2, 3, 4])
    # Limit to 1000 examples to save time
    # print("Expected Loss for random predictions: %f" % np.log(8000))
    # print("Actual loss: %f" % rnn.calculate_loss(x_train[:1000], y_train[:1000]))

    train_with_sgd(rnn, x_train[:100], y_train[:100])

    num_sentences = 10
    senten_min_length = 7

    for i in range(num_sentences):
        sent = []
        # We want long sentences, not sentences with one or two words
        # while len(sent) > senten_min_length:
        #     sent = generate_sentence(rnn)
        # print(" ".join(sent))