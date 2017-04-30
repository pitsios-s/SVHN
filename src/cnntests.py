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
training_iters = 50000
batch_size = 100
display_step = 100

# Network Parameters
n_input = 1024  # SVHN data input (img shape: 32*32)
n_classes = 10  # SVHN total classes (0-9 digits)
dropout = 0.75  # Dropout, probability to keep units                  ##### play

# Get Data
svhn_train = scipy.io.loadmat("train_32x32.mat")
svhn_train_data = flatten_data(svhn_train['X'])
svhn_train_labels = one_hot_encode(svhn_train['y'], n_classes)

svhn_test = scipy.io.loadmat("test_32x32.mat")
svhn_test_data = flatten_data(svhn_test['X'])
svhn_test_labels = one_hot_encode(svhn_test['y'], n_classes)

# tf Graph input
X = tf.placeholder(tf.float32, [None, n_input])
Y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)


# Create some wrappers for simplicity
def conv2d(x, w, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    # other option is average pooling (or combination)
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


# Create model
def conv_net(x, w, b, d):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 32, 32, 1])
    #
    # Convolution Layer
    conv1 = conv2d(x, w['wc1'], b['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)
    #
    # Convolution Layer
    conv2 = conv2d(conv1, w['wc2'], b['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)
    #
    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, w['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, w['wd1']), b['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, d)
    #
    # Output, class prediction
    out = tf.add(tf.matmul(fc1, w['out']), b['out'])

    return out


# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([8 * 8 * 64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = conv_net(X, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    i = 0
    # Keep training until reach max iterations
    while step * batch_size < training_iters:

        batch_x, batch_y = svhn_train_data[i * 100:(i + 1) * 100], svhn_train_labels[i * 100:(i + 1) * 100]
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={X: batch_x, Y: batch_y, keep_prob: dropout})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={X: batch_x, Y: batch_y, keep_prob: dropout})
            # Calculate accuracy for 256 mnist test images
            t_acc = accuracy.eval({X: svhn_test_data, Y: svhn_test_labels, keep_prob: dropout})
            print(
                "Iter " + str(step * batch_size) +
                ", Minibatch Loss= " + "{:.6f}".format(loss) +
                ", Training Accuracy= " + "{:.5f}".format(acc) +
                ", Testing Accuracy= " + "{:.5f}".format(t_acc)
            )
        step += 1
    i += 1
    print("Optimization Finished!")

# useful http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/
