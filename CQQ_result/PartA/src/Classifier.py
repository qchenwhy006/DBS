import numpy as np
from base import BaseMLP
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import seaborn as sns
import random


def relu(z):
    """
    Args:
        z: (batch_size, hidden_size)
    """
    flag = (z <= 0)
    z[flag] = 0
    return z


def derivation_relu(z):
    flag = (z <= 0)
    z[flag] = 0
    z[~flag] = 1
    return z


def sigmoid(z):
    """
    Args:
        z: (batch_size, hidden_size)
    """
    return 1 / (1 + np.exp(-z))


def derivation_sigmoid(z):
    return sigmoid(z) * (1 - sigmoid(z))


def tanh(z):
    """
    Args:
        z: (batch_size, hidden_size)
    """
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))


def derivation_tanh(z):
    return 1 - tanh(z) ** 2


def softmax(z):
    """
    Args:
        z: (batch_size, output_size)
    Returns:
        (batch_size, output_size)
    """
    max_row = np.max(z, axis=-1, keepdims=True)
    tmp = z - max_row
    return np.exp(tmp) / np.sum(np.exp(tmp), axis=-1, keepdims=True)


def softmax_cross_entropy(logits, y):
    """
    Args:
        logits: (batch_size, output_size)
        y: (batch_size, ) True label
    return:
        a: (batch_size, output_size)
        loss: scalar
    """
    n = logits.shape[0]
    a = softmax(logits)
    scores = a[range(n), y]
    loss = -np.sum(np.log(scores)) / n
    return a, loss


def derivation_softmax_cross_entropy(logits, y):
    """
    Args:
        logits: (batch_size, output_size)，
        y: (batch_size, )

    Return:
        \frac {\partial C}{\partial z^L}
        (batch_size, output_size)
    """
    n = logits.shape[0]
    a = softmax(logits)
    a[range(n), y] -= 1
    return a


def load_batches(x, y, batch_size):
    """
    Shuffle the data and divide it into a batch
    Args:
        x: (num_samples, input_size)
        y: (num_samples, )
        batch_size:
    Returns:
        batches_x: (batch_size, input_size)
        batches_y: (batch_size, )
    """
    n = len(x)
    shuffle_idx = random.sample(range(n), n)
    X = x[shuffle_idx]
    Y = y[shuffle_idx]

    batches_x = [X[i: i + batch_size] for i in range(0, n, batch_size)]
    batches_y = [Y[i: i + batch_size] for i in range(0, n, batch_size)]

    return batches_x, batches_y


class NeuralNetworkClassifier(BaseMLP):
    """
    fully-connected neural network
    Attributions:
        sizes: list, Each element in list is the number of neurons in each layer, including the input and output layers
        weights: list, Each element in list is the weight of one layer of the neural network
        bias: list, Each element in list is a bias of one layer of the neural network
    """
    __slots__ = ('batch_size', 'learning_rate', 'max_iter', 'hidden_layer_sizes', 'random_state', 'momentum', 'sizes', 'weights', 'bias')

    def __init__(self, batch_size, learning_rate, max_iter, hidden_layer_sizes, random_state, momentum):
        super().__init__(batch_size, learning_rate, max_iter, hidden_layer_sizes, random_state, momentum)

        self.sizes = len(self.hidden_layer_sizes)
        self.weights = [np.random.randn(i, j) for i, j in zip(self.hidden_layer_sizes[:-1], self.hidden_layer_sizes[1:])]
        self.bias = [np.random.randn(1, j) for j in self.hidden_layer_sizes[1:]]

    def forward(self, x):
        """
        For inference, no softmax probabilization for forward propagation
        x: (batch_size, input_size)
        """
        a = x
        for weight, bias in zip(self.weights[:-1], self.bias[:-1]):
            z = np.dot(a, weight) + bias
            a = relu(z)
        logits = np.dot(a, self.weights[-1]) + self.bias[-1]
        return logits

    def backward(self, x, y):
        """
        Args:
            x: (batch_size, input_size)
            y: (batch_size, )
        returns:
            dws: list
            dbs: list
        """
        dws = [np.zeros((i, j)) for i, j in zip(self.hidden_layer_sizes[:-1], self.hidden_layer_sizes[1:])]
        dbs = [np.zeros((1, j)) for j in self.hidden_layer_sizes[1:]]

        zs = []
        _as = []

        a = x
        _as.append(a)
        for weight, bias in zip(self.weights[:-1], self.bias[:-1]):
            z = np.dot(a, weight) + bias
            zs.append(z)
            a = relu(z)
            _as.append(a)

        logits = np.dot(a, self.weights[-1]) + self.bias[-1]
        zs.append(logits)
        a, loss = softmax_cross_entropy(logits, y)
        _as.append(a)

        dl = derivation_softmax_cross_entropy(logits, y)
        n = len(x)

        dws[-1] = np.dot(_as[-2].T, dl) / n
        dbs[-1] = np.sum(dl, axis=0, keepdims=True) / n

        for i in range(2, self.sizes):
            dl = np.dot(dl, self.weights[-i + 1].T) * derivation_relu(zs[-i])
            dws[-i] = np.dot(_as[-i - 1].T, dl) / n
            dbs[-i] = np.sum(dl, axis=0, keepdims=True) / n

        return loss, dws, dbs

    def fit(self, x, y):
        val_accs = []
        X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=self.random_state)
        gradient_ws = [np.random.randn(i, j) for i, j in zip(self.hidden_layer_sizes[:-1], self.hidden_layer_sizes[1:])]
        gradient_bs = [np.random.randn(1, j) for j in self.hidden_layer_sizes[1:]]


        for epoch in range(self.max_iter):
            x_batches, y_batches = load_batches(X_train, y_train, self.batch_size)

            for i, (x, y) in enumerate(zip(x_batches, y_batches)):
                if not self.momentum:  # Mini_Batch
                    loss, dws, dbs = self.backward(x, y)
                    self.weights = [weight - self.learning_rate * dw for weight, dw in zip(self.weights, dws)]
                    self.bias = [bias - self.learning_rate * db for bias, db in zip(self.bias, dbs)]
                else:  # Momentum
                    loss, dws, dbs = self.backward(x, y)
                    gradient_ws = [self.momentum * gradient_w + self.learning_rate * dw for gradient_w, dw in zip(gradient_ws, dws)]
                    gradient_bs = [self.momentum * gradient_b + self.learning_rate * db for gradient_b, db in zip(gradient_bs, dbs)]
                    self.weights = [weight - gradient_w for weight, gradient_w in zip(self.weights, gradient_ws)]
                    self.bias = [bias - gradient_b for bias, gradient_b in zip(self.bias, gradient_bs)]

                if i % 1 == 0:
                    print("Epoch {}, batch {}, loss {}".format(epoch, i, loss))

            x_batches, y_batches = load_batches(X_val, y_val, self.batch_size)

            corrects = 0
            n = len(X_val)
            for i, (x, y) in enumerate(zip(x_batches, y_batches)):
                logits = self.forward(x)
                correct = np.sum(np.argmax(logits, axis=-1) == y)
                corrects += correct
            acc = corrects / n
            val_accs.append(acc)
            print("Epoch {}, acc {}/{}={}".format(epoch, corrects, n, acc))
        print("Accuracy of all epochs on the validation set:", val_accs)
        print("Best Accuracy: ", np.max(val_accs))

    def predict(self, x):
        logits = self.forward(x)
        pred_y = np.argmax(logits, axis=-1)
        return pred_y


def make_plot(X, y, plot_name, XX=None, YY=None, preds=None, dark=False):
    if (dark):
        plt.style.use('dark_background')
    else:
        sns.set_style("whitegrid")
    plt.figure(figsize=(16,12))
    axes = plt.gca()
    axes.set(xlabel="$x_1$", ylabel="$x_2$")
    plt.title(plot_name, fontsize=30)
    plt.subplots_adjust(left=0.20)
    plt.subplots_adjust(right=0.80)
    if XX is not None and YY is not None and preds is not None:
        plt.contourf(XX, YY, preds.reshape(XX.shape), 25, alpha=1, cmap=plt.cm.Spectral)
        plt.contour(XX, YY, preds.reshape(XX.shape), levels=[.5], cmap="Greys", vmin=0, vmax=.6)

    plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), s=40, cmap=plt.cm.Spectral, edgecolors='none')


def main():
    x, y = make_moons(n_samples=2000, noise=0.2, random_state=100)
    make_plot(x, y, "Classification Dataset Visualization ")
    plt.show()
    X, X_test, Y, y_test = train_test_split(x, y, test_size=0.2, random_state=1314)
    model = NeuralNetworkClassifier(batch_size=16, learning_rate=0.01, max_iter=100, hidden_layer_sizes=[2, 25, 50, 25, 2], random_state=1314, momentum=0.1)
    model.fit(X, Y)
    y_pred = model.predict(X_test)
    make_plot(X_test, y_pred, 'Test Dataset Visualization')
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
