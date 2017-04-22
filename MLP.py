
# coding: utf-8

# In[1]:


from __future__ import print_function

# Import SVHN data
from tensorflow.examples.tutorials.mnist import input_data                          ##### CHANGE
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)                       ## CHANGE

import scipy.io
mat = scipy.io.loadmat('file.mat') # load file


# In[23]:

get_ipython().magic(u'matplotlib notebook')
get_ipython().magic(u'pylab inline')
import matplotlib.pyplot as plt

fig1 = plt.figure()

for i in range(9):    
    ax = fig1.add_subplot(191+i)
    ax.clear()
    ax.imshow(mnist.train.images[i].reshape(28, 28), cmap='gray')
    print(mnist.train.labels[i])


# In[24]:

import tensorflow as tf


# Parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1

# Network Parameters
n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
n_input = 1024 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)



# In[25]:


# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])


# In[26]:

weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),                    ########## TRY MORE OPTIONS (gamma, uniform, multinomial)
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


# In[31]:


# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)                                                    ##### TRY MORE ACTIVATION FUNCTIONS (tanh, leaky_relu, sigmoid)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Construct model
pred = multilayer_perceptron(x, weights, biases)
# Test model
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


# In[32]:


# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)          ##### TRY MORE OPTIMIZERS (ada, adagrad, ... )


# In[33]:

# Initializing the variables
init = tf.global_variables_initializer()


# In[36]:

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        avg_acc = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
            batch_acc = accuracy.eval({x: batch_x, y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
            avg_acc += batch_acc / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            test_acc = accuracy.eval({x: mnist.test.images, y: mnist.test.labels})
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



# In[ ]:



