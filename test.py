import pytest
import numpy as np

from layersTest import Linear, ReLU, BatchNorm, SoftmaxCrossEntropyLoss
from networkTest import Network


def BN_forward(x, beta):
    return x - np.mean(x, axis=0, keepdims=True) + beta


def test_dBeta(n_tests):
    '''
    Args:
        n_tests: number of randomly configured tests to run
    Asserts 1 == (y(beta+epsilon) - y(beta-epsilon)) / (2*epsilon)
    and therefore dBeta ~= (loss(beta+epsilon) - loss(beta-epsilon)) / (2*epsilon),
    validating dBeta = np.sum(dY, axis=0, keepdims=True)
    '''
    for i in range(n_tests):
        X = np.random.rand(2, 2)
        epsilon = np.random.random() / 10
        beta = np.random.random()
        b1 = beta + epsilon
        b2 = beta - epsilon
        assert np.divide(np.subtract(BN_forward(X, b1), BN_forward(X, b2)), 2 * epsilon).all() == 1


def test_dX(seed, i1, i2, epsilon):
    '''
    Args:
        seed: random seed for network initializations
        i1, i2: 0 to 128
        epsilon: Value to subtract and add to BN layer input X at i1, i2
    Returns:
        delta: difference between dX and numerical dX
    Asserts dX ~= (loss(X+epsilon[@i1,i2]) - loss(X-epsilon[@i1,i2])) / (2*epsilon),
    validating dX = (len(dY) * dY - dbeta - x_hat * np.sum(dY * x_hat, axis=0)) / len(dY)
    Creates three networks with BN layer to perform training, calculation, and comparison.
    '''
    delta_thresh = 10 ** -8
    n_classes = 10
    dim = 784
    inputs, labels = load_normalized_mnist_data()

    print("Original network")
    np.random.seed(seed)
    net = Network(learning_rate = 1e-3)
    net.add_layer(Linear(dim, 128))
    net.add_layer(BatchNorm(128, 0, i1, i2))
    net.add_layer(ReLU())
    net.add_layer(Linear(128, n_classes))
    net.set_loss(SoftmaxCrossEntropyLoss())
    loss, dx = train_network(net, inputs, labels, 1)

    print("\nNetwork with subtraction")
    np.random.seed(seed)
    inputs, labels = load_normalized_mnist_data()
    netMinus = Network(learning_rate = 1e-3)
    netMinus.add_layer(Linear(dim, 128))
    netMinus.add_layer(BatchNorm(128, -1 * epsilon, i1, i2))
    netMinus.add_layer(ReLU())
    netMinus.add_layer(Linear(128, n_classes))
    netMinus.set_loss(SoftmaxCrossEntropyLoss())
    minus_loss = train_network(netMinus, inputs, labels, 1) [0]

    print("\nNetwork with addition")
    np.random.seed(seed)
    inputs, labels = load_normalized_mnist_data()
    netPlus = Network(learning_rate = 1e-3)
    netPlus.add_layer(Linear(dim, 128))
    netPlus.add_layer(BatchNorm(128, epsilon, i1, i2))
    netPlus.add_layer(ReLU())
    netPlus.add_layer(Linear(128, n_classes))
    netPlus.set_loss(SoftmaxCrossEntropyLoss())
    plus_loss = train_network(netPlus, inputs, labels, 1) [0]

    print ("\ndX used in backprop:", dx[i1][i2])
    num_dx = (plus_loss - minus_loss) / 2 / epsilon
    print ("Numerical dX:", num_dx)
    delta = abs(num_dx - dx[i1][i2])
    assert delta < delta_thresh
    return delta


def load_normalized_mnist_data():
    '''
    Loads and normalizes the MNIST data. Reads the data from
        data/mnist_train.csv
        data/mnist_test.csv
    These can be downloaded from https://pjreddie.com/projects/mnist-in-csv/
    Returns two dictionaries, input and labels
    Each has keys 'train', 'val', 'test' which map to numpy arrays
    '''
    data = np.loadtxt('data/mnist_train.csv', dtype=int, delimiter=',')
    test_data = np.loadtxt('data/mnist_test.csv', dtype=int, delimiter=',')

    inputs = dict()
    labels = dict()

    train_data = data[:128]
    train_inputs = train_data[:, 1:]

    val_data = data[128:128]
    val_inputs = val_data[:, 1:]

    test_inputs = test_data[:, 1:]

    mean = np.mean(train_inputs)
    std = np.std(train_inputs)

    inputs['train'] = (train_inputs - mean)/std
    inputs['val'] = (val_inputs - mean)/std
    inputs['test'] = (test_inputs - mean)/std

    labels['train'] = train_data[:, 0]
    labels['val'] = val_data[:, 0]
    labels['test'] = test_data[:, 0]

    return inputs, labels

def train_network(network, inputs, labels, n_epochs, batch_size=128):
    '''
    Trains a network for n_epochs
    Args:
        network (Network): The neural network to be trained
        inputs (dict): Dictionary with keys 'train' and 'val' mapping to numpy
                       arrays
        labels (dict): Dictionary with keys 'train' and 'val' mapping to numpy
                       arrays
        n_epochs (int): Specifies number of epochs trained for
        batch_size (int): Number of samples in a minibatch
    Returns:
        avg_train_loss: average training loss of the last epoch
        dx: dx from the batch norm layer of the last batch of the last epoch
    '''
    train_inputs = inputs['train']
    train_labels = labels['train']

    n_train = train_inputs.shape[0]

    dx = 0

    # Train network
    for epoch in range(n_epochs):
        order = np.random.permutation(n_train)
        num_batches = n_train // batch_size
        train_loss = 0

        start_idx = 0
        while start_idx < n_train:
            end_idx = min(start_idx+batch_size, n_train)
            idxs = order[start_idx:end_idx]
            mb_inputs = train_inputs[idxs]
            mb_labels = train_labels[idxs]
            this_loss, dx = network.train(mb_inputs, mb_labels)
            train_loss += this_loss
            start_idx += batch_size

        avg_train_loss = train_loss/num_batches
        print("Loss", avg_train_loss)

    return avg_train_loss, dx

# Call tests
print("Testing dBeta...\n")
test_dBeta(20)
print("Passed\n")
print("Testing dX:\n")
for i1 in range(14, 15): # choose test params
    for i2 in range(98, 99):
        print("Difference:", test_dX(42, i1, i2, 0.05), "\n")
print("Passed all")
