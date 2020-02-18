# mnist-numpy
A basic fully connected network, with and without batch normalization, implemented purely in NumPy and trained on the MNIST dataset.

## Experiments
The MNIST dataset is split into 50000 train, 10000 validation and 10000 test samples. All splits are normalized using the statistics of the training split (using the global mean and standard deviation, not per pixel).

Two networks are trained:
(1) The same as (2) but with batch normalization before each layer. There are batch normalization layers immediately before each ReLU activation, as well as immediately after the each ReLU activation to improve test accuracy, though this does not reduce covariate shift (Ioffe, Szegedy; arxiv.org/abs/1502.03167). Removing the batch normalization layers from before the ReLUs can actually further improve test accuracy.
(2) The second network has 2 fully connected layers with ReLU activations. The first hidden layer has 256 units and the second 128 units. The network is initialized with Xavier-He initialization.

The networks are trained for 50 epochs with vanilla minibatch SGD and learning rate 1e-3. The final accuracies on the test set are about 0.94.


## Code structure:
### layers.py
Contains classes that represent layers for different transformations. Each class has a forward and a backward method that define a transformation and its gradient. The class keeps track of the variables defining the transformation and the variables needed to calculate the gradient. The file also contains a class that defines the softmax cross entropy loss.

### network.py
Defines Network, a configurable class representing a sequential neural network with any combination of layers. Network has a train function that performs minibatch SGD.

### main.py
Data loading, training and validation scripts. Running it trains the networks described in experiments. For loading the data it expects two files "data/mnist_train.csv" and "data/mnist_test.csv". These can be downloaded from https://pjreddie.com/projects/mnist-in-csv/. To run use "python3 main.py".

### test.py
Numerical gradient checking for the backward pass of the batch normalization layer using the approximation df/dx ~= (f(x + epsilon) - f(x - epsilon)) / (2 * epsilon). The gradient for beta was exactly validated, and the gradient for dX was accurate within 10^-8, throughout a wide range of testing parameters.  To run, edit testing parameters if desired and use "python3 test.py". Details on the numerical gradient checking:
- dBeta: In the backward pass of batch normalization, dBeta was the column sum of dY, indicating that dY/dBeta is an identity.  With numerical gradient checking on Y(beta) on random epsilons, betas, and dY's, dY/dBeta was confirmed to be a matrix of     ones.
- dX: Three similar networks were run, each with one batch normalization layer, one epoch, and one batch of 128. In the backward pass of batch normalization, dX was calculated to be (len(dY) * dY - dbeta - x_hat * np.sum(dY * x_hat,  axis=0)) / len(dY). So the first network computed and recorded this supposed dX. The second network subtracted epsilon from   an input at a certain index for the batch normaization layer, generating a slightly different loss, and the third network instead added epsilon     to this input, generating a slighly different loss. The difference of these last two losses was taken and divided by       2 and epsilon, giving the numerical gradient dX, which ended up within 10^-8 of dX (at the index of epsilon addition and subtraction in the last two networks) from the first network.

### layersTest.py
Modified version of layers.py for testing. The BatchNorm layer here takes parameters to allow epsilon to be added/subtracted to a certain index of input during the forward pass. This allows the loss to change from the subtraction iteration to the addition iteration by 2 * epsilon * (dX estimated by the backward pass).

### networkTest.py
Modified version of network.py for testing. In addition to the training loss, the train method here outputs dx computed in the backward pass through the batch normalization layer of the network.
