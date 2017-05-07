import tensorflow as tf
from svhn import SVHN

# Parameters
learning_rate = 0.001
iterations = 50000
batch_size = 50
display_step = 1000

# Network Parameters
channels = 3
image_size = 32
n_classes = 10
dropout = 0.9
hidden = 128
depth_1 = 16
depth_2 = 32
depth_3 = 64
filter_size = 5
normalization_offset = 0.0  # beta
normalization_scale = 1.0  # gamma
normalization_epsilon = 0.001  # epsilon


def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def bias_variable(shape):
    return tf.Variable(tf.constant(1.0, shape=shape))


def convolution(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME")


def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


# Load data
svhn = SVHN("../res", n_classes, use_extra=True, gray=False)

# Create the model
X = tf.placeholder(tf.float32, [None, image_size, image_size, channels])
Y = tf.placeholder(tf.float32, [None, n_classes])


# Weights & Biases
weights = {
    "layer1": weight_variable([filter_size, filter_size, channels, depth_1]),
    "layer2": weight_variable([filter_size, filter_size, depth_1, depth_2]),
    "layer3": weight_variable([filter_size, filter_size, depth_2, depth_3]),
    "layer4": weight_variable([image_size // 8 * image_size // 8 * depth_3, hidden]),
    "layer5": weight_variable([hidden, n_classes])
}

biases = {
    "layer1": bias_variable([depth_1]),
    "layer2": bias_variable([depth_2]),
    "layer3": bias_variable([depth_3]),
    "layer4": bias_variable([hidden]),
    "layer5": bias_variable([n_classes])
}


def normalize(x):
    """ Applies batch normalization """
    mean, variance = tf.nn.moments(x, [1, 2, 3], keep_dims=True)
    return tf.nn.batch_normalization(x, mean, variance, normalization_offset, normalization_scale,
                                     normalization_epsilon)


def cnn(x):
    # Batch normalization
    x = normalize(x)

    # Convolution 1 -> RELU -> Max Pool
    convolution1 = convolution(x, weights["layer1"])
    hidden1 = tf.nn.relu(convolution1 + biases["layer1"])
    hidden2 = max_pool(hidden1)

    # Convolution 2 -> RELU -> Max Pool
    convolution2 = convolution(hidden2, weights["layer2"])
    hidden3 = tf.nn.relu(convolution2 + biases["layer2"])
    hidden4 = max_pool(hidden3)

    # Convolution 3 -> RELU -> Max Pool
    convolution3 = convolution(hidden4, weights["layer3"])
    hidden5 = tf.nn.relu(convolution3 + biases["layer3"])
    hidden6 = max_pool(hidden5)

    # Fully Connected Layer
    shape = hidden6.get_shape().as_list()
    reshape = tf.reshape(hidden6, [-1, shape[1] * shape[2] * shape[3]])
    hidden7 = tf.nn.relu(tf.matmul(reshape, weights["layer4"]) + biases["layer4"])

    # Dropout Layer
    keep_prob_constant = tf.placeholder(tf.float32)
    dropout_layer = tf.nn.dropout(hidden7, keep_prob_constant)

    return tf.matmul(dropout_layer, weights["layer5"]) + biases["layer5"], keep_prob_constant


# Build the graph for the deep net
y_conv, keep_prob = cnn(X)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=y_conv))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(iterations):
        offset = (i * batch_size) % (svhn.train_examples - batch_size)
        batch_x = svhn.train_data[offset:(offset + batch_size)]
        batch_y = svhn.train_labels[offset:(offset + batch_size)]

        _, c = sess.run([optimizer, cost], feed_dict={X: batch_x, Y: batch_y, keep_prob: dropout})

        if i % display_step == 0:
            train_accuracy = accuracy.eval(feed_dict={X: batch_x, Y: batch_y, keep_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy / batch_size))

    # Test the model by measuring it's accuracy
    correct_predictions = 0
    test_iterations = svhn.test_examples / batch_size + 1
    for i in range(test_iterations):
        batch_x, batch_y = (svhn.test_data[i * batch_size:(i + 1) * batch_size],
                            svhn.test_labels[i * batch_size:(i + 1) * batch_size])
        correct_predictions += accuracy.eval(feed_dict={X: batch_x, Y: batch_y, keep_prob: 1.0})
    print('Test accuracy %g' % (correct_predictions / svhn.test_examples))
