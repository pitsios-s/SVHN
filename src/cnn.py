import tensorflow as tf
from svhn import SVHN

# Parameters
learning_rate = 0.001
iterations = 30000
batch_size = 50
display_step = 1000

# Network Parameters
channels = 3
image_size = 32
n_classes = 10
dropout = 0.95
hidden = 128
depth_1 = 16
depth_2 = 32
depth_3 = 64
filter_size = 5

svhn = SVHN("../res", n_classes)


# Create the model
X = tf.placeholder(tf.float32, [None, image_size, image_size, channels])
Y = tf.placeholder(tf.float32, [None, n_classes])


# Weights & Biases
weights = {
    "layer1": tf.Variable(tf.truncated_normal([filter_size, filter_size, channels, depth_1], stddev=0.1)),
    "layer2": tf.Variable(tf.truncated_normal([filter_size, filter_size, depth_1, depth_2], stddev=0.1)),
    "layer3": tf.Variable(tf.truncated_normal([filter_size, filter_size, depth_2, depth_3], stddev=0.1)),
    "layer4": tf.Variable(tf.truncated_normal([image_size // 8 * image_size // 8 * depth_3, hidden], stddev=0.1)),
    "layer5": tf.Variable(tf.truncated_normal([hidden, n_classes], stddev=0.1))
}

biases = {
    "layer1": tf.Variable(tf.constant(1.0, shape=[depth_1])),
    "layer2": tf.Variable(tf.constant(1.0, shape=[depth_2])),
    "layer3": tf.Variable(tf.constant(1.0, shape=[depth_3])),
    "layer4": tf.Variable(tf.constant(1.0, shape=[hidden])),
    "layer5": tf.Variable(tf.constant(1.0, shape=[n_classes]))
}


def deepnn(x):
    # Convolution 1 and RELU
    convolution1 = tf.nn.conv2d(x, weights["layer1"], [1, 1, 1, 1], padding='SAME')
    hidden1 = tf.nn.relu(convolution1 + biases["layer1"])
    # Max Pool
    hidden2 = tf.nn.max_pool(hidden1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Convolution 2 and RELU
    convolution2 = tf.nn.conv2d(hidden2, weights["layer2"], [1, 1, 1, 1], padding='SAME')
    hidden3 = tf.nn.relu(convolution2 + biases["layer2"])
    # Max Pool
    hidden4 = tf.nn.max_pool(hidden3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Convolution 3 and RELU
    convolution3 = tf.nn.conv2d(hidden4, weights["layer3"], [1, 1, 1, 1], padding='SAME')
    hidden5 = tf.nn.relu(convolution3 + biases["layer3"])
    # Max Pool
    hidden6 = tf.nn.max_pool(hidden5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Fully Connected Layer
    shape = hidden6.get_shape().as_list()
    reshape = tf.reshape(hidden6, [-1, shape[1] * shape[2] * shape[3]])
    hidden7 = tf.nn.relu(tf.matmul(reshape, weights["layer4"]) + biases["layer4"])

    # Dropout Layer
    keep_prob = tf.placeholder(tf.float32)
    dropout_layer = tf.nn.dropout(hidden7, keep_prob)

    return tf.matmul(dropout_layer, weights["layer5"]) + biases["layer5"], keep_prob


# Build the graph for the deep net
y_conv, keep_prob = deepnn(X)

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
