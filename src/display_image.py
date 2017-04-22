import scipy.io
from matplotlib import pyplot as plt

# Read .mat file
mat = scipy.io.loadmat('file.mat')

# Disaply first image
plt.imshow(mat['X'][:, :, :, 0], interpolation='nearest')
plt.show()
