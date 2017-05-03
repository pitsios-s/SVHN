
# coding: utf-8

# In[1]:

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
 


# In[2]:

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


# In[7]:

labels = pd.DataFrame.from_dict(train['y'],orient="columns") #load .mat to dataframe
data = pd.DataFrame.from_dict(train['X'],orient="columns") #load .mat to dataframe
labels.columns=['Labels'] #rename column
print(data) 


# In[6]:

labels.Labels.plot.hist()


# In[ ]:



