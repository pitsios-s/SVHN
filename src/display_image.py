import scipy.io
from matplotlib import pyplot as plt

mat = scipy.io.loadmat('file.mat')

plt.imshow(data, interpolation='nearest')
plt.show()
