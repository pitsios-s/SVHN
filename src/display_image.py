import scipy.io as sio
from matplotlib import pyplot as plt
import numpy as np


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

# Read .mat file
mat = sio.loadmat("../res/train_32x32.mat")
mat_gray = rgb2gray(mat['X'][:, :, :, 0])


# Display first image
plt.imshow(mat['X'][:, :, :, 0], interpolation='nearest')
plt.show()


# Display first image gray-scale
plt.imshow(mat_gray, cmap="gray", interpolation="nearest")
plt.show()
