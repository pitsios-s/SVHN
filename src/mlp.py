from __future__ import print_function

import tensorflow as tf
import scipy.io
import numpy as np


def one_hot_encode(data, length):
    """Creates a one-hot encoding vector
        Args:
            data: The data to be converted
            length: The length of the one-hot encoded vectors
        Returns:
            An array of one-hot encoded items
    """
    n = data.shape[0]
    one_hot = np.zeros(shape=(data.shape[0], length))
    for s in range(n):
        temp = [0 * v for v in range(0, length)]

        num = data[s][0]
        if num == 10:
            temp[0] = 1
        else:
            temp[num] = 1

        one_hot[s] = temp

    return one_hot


def flatten_data(data):
    """Flattens an image of size n * n * 3, into a an array of size N * 1, where N = n * n * 3
        Args:
            data: The array to be flattened 
        Returns:
            A flattened array
    """
    n = data.shape[3]
    flattened = np.zeros(shape=(n, 1024))
    for s in range(n):
        flattened[s] = rgb2gray(data[:, :, :, s]).flatten().astype(np.float32)

    return flattened


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


# Parameters
learning_rate = 0.001
training_epochs = 30
batch_size = 100
display_step = 1


# Network Parameters
n_hidden_1 = 256  # 1st layer number of features
n_hidden_2 = 256  # 2nd layer number of features
n_input = 1024  # SVHN data input (img shape: 32*32*1)
n_classes = 10  # SVHN total classes (0-9 digits)


svhn_train = scipy.io.loadmat("../res/train_32x32.mat")
svhn_train_data = flatten_data(svhn_train['X'])
svhn_train_labels = one_hot_encode(svhn_train['y'], n_classes)


svhn_test = scipy.io.loadmat("../res/test_32x32.mat")
svhn_test_data = flatten_data(svhn_test['X'])
svhn_test_labels = one_hot_encode(svhn_test['y'], n_classes)


# tf Graph input
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])


# TRY MORE OPTIONS (gamma, uniform, multinomial)
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


# Create model
def multilayer_perceptron(x, w, b):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, w['h1']), b['b1'])
    layer_1 = tf.nn.relu(layer_1)  # TRY MORE ACTIVATION FUNCTIONS (tanh, leaky_relu, sigmoid)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, w['h2']), b['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, w['out']) + b['out']

    return out_layer


# Construct model
pred = multilayer_perceptron(X, weights, biases)
# Test model
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=Y))
# TRY MORE OPTIMIZERS (ada, adagrad, ... )
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


# Initializing the variables
init = tf.global_variables_initializer()


# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        avg_acc = 0.
        total_batch = int(svhn_train['X'].shape[3] / batch_size)

        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = svhn_train_data[i * batch_size:(i + 1) * batch_size], \
                               svhn_train_labels[i * batch_size:(i + 1) * batch_size]
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_x, Y: batch_y})
            batch_acc = accuracy.eval({X: batch_x, Y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
            avg_acc += batch_acc / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            test_acc = accuracy.eval({X: svhn_test_data, Y: svhn_test_labels})
            print(
                "Epoch:",
                '%04d' % (epoch+1),
                "cost=",
                "{:.9f}".format(avg_cost),
                "average_train_accuracy=",
                "{:.6f}".format(avg_acc),
                "test_accuracy=",
                "{:.6f}".format(test_acc)
            )
    print("Optimization Finished!")
