import scipy.io
from matplotlib import pyplot as plt

# Read .mat file
mat = scipy.io.loadmat("../res/test_32x32.mat")

# Display first image
plt.imshow(mat['X'][:, :, :, 0], interpolation='nearest')
plt.show()

print (mat['y'].shape)
