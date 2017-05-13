
# coding: utf-8

# In[63]:

from collections import Counter
import numpy as np
import scipy.io as sio
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
def __one_hot_encode(n_classes, data):
    """Creates a one-hot encoding vector
        Args:
            data: The data to be converted
        Returns:
            An array of one-hot encoded items
    """
    n = data.shape[0]
    one_hot = np.zeros(shape=(data.shape[0], n_classes))
    for s in range(n):
        temp = np.zeros(n_classes)

        num = data[s][0]
        if num == 10:
            temp[0] = 1
        else:
            temp[num] = 1

        one_hot[s] = temp

    return one_hot

def __store_data(data, num_of_examples, gray):
    d = []

    for i in range(num_of_examples):
        if gray:
            d.append(__rgb2gray(data[:, :, :, i]))
        else:
            d.append(data[:, :, :, i])

    return np.asarray(d)

def __rgb2gray (rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
 


# In[65]:

n_classes = 10
train = sio.loadmat("E:/Program Files/MATLAB/R2013a/bin/PRO/pro/train_32x32.mat")
train_labels = __one_hot_encode(n_classes,train['y'])
train_examples = train['X'].shape[3]
train_data =__store_data(train['X'].astype("float32") / 128.0 - 1, train_examples, False)

# Load Test Set
test = sio.loadmat("E:/Program Files/MATLAB/R2013a/bin/PRO/pro/test_32x32.mat")
test_labels = __one_hot_encode(n_classes, test['y'])
test_examples = test['X'].shape[3]
test_data = __store_data(test['X'].astype("float32") / 128.0 - 1, test_examples, False)

#Load Extra Set
extra = sio.loadmat("E:/Program Files/MATLAB/R2013a/bin/PRO/pro/extra_32x32.mat")
extra_labels = __one_hot_encode(n_classes, extra['y'])
extra_examples = extra['X'].shape[3]
 


# In[66]:

trainlabels = pd.DataFrame.from_dict(train['y'],orient="columns") #load .mat to dataframe
testlabels = pd.DataFrame.from_dict(test['y'],orient="columns") #load .mat to dataframe
extralabels = pd.DataFrame.from_dict(extra['y'],orient="columns") #load .mat to dataframe
trainlabels.columns=['Labels'] #rename column
testlabels.columns=['Labels'] #rename column
extralabels.columns=['Labels'] #rename column


# In[67]:

labels, values = zip(*Counter(trainlabels.Labels).items())
indexes = np.arange(len(labels))
plt.bar(indexes, values, 0.8)
plt.xticks(indexes, labels)
plt.xlabel('Train data', fontsize=14, color='black')
plt.show()


# In[68]:

labels, values = zip(*Counter(testlabels.Labels).items())
indexes = np.arange(len(labels))
plt.bar(indexes, values, 0.8)
plt.xticks(indexes, labels)
plt.xlabel('Test data', fontsize=14, color='black')
plt.show()


# In[69]:

labels, values = zip(*Counter(extralabels.Labels).items())
indexes = np.arange(len(labels))
plt.bar(indexes, values, 0.8)
plt.xticks(indexes, labels)
plt.xlabel('Extra data', fontsize=14, color='black')
plt.show()


# In[ ]:




# In[ ]:



