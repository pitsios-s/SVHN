import numpy as np
import scipy.io as sio


class SVHN:

    def __init__(self, file_path, n_classes, use_extra=False, gray=False):
        self.n_classes = n_classes

        # # Load Train Set
        train = sio.loadmat(file_path + "/train_32x32.mat")
        self.train_labels = self.__one_hot_encode(train['y'])
        self.train_examples = train['X'].shape[3]
        self.train_data = self.__store_data(train['X'].astype("float32"), self.train_examples, gray)

        # Load Test Set
        test = sio.loadmat("../res/test_32x32.mat")
        self.test_labels = self.__one_hot_encode(test['y'])
        self.test_examples = test['X'].shape[3]
        self.test_data = self.__store_data(test['X'].astype("float32"), self.test_examples, gray)

        # Load Extra dataset as additional training data if necessary
        if use_extra:
            extra = sio.loadmat(file_path + "/extra_32x32.mat")
            self.train_labels = np.append(self.train_labels, self.__one_hot_encode(extra['y']), axis=0)
            extra_examples = extra['X'].shape[3]
            self.train_examples += extra_examples
            self.train_data = np.append(self.train_data, self.__store_data(extra['X'].astype("float32"),
                                                                           extra_examples, gray), axis=0)
            # shuffle values
            idx = np.arange(self.train_data.shape[0])
            self.train_data = self.train_data[idx]
            self.train_labels = self.train_labels[idx]

    def __one_hot_encode(self, data):
        """Creates a one-hot encoding vector
            Args:
                data: The data to be converted
            Returns:
                An array of one-hot encoded items
        """
        n = data.shape[0]
        one_hot = np.zeros(shape=(data.shape[0], self.n_classes))
        for s in range(n):
            temp = np.zeros(self.n_classes)

            num = data[s][0]
            if num == 10:
                temp[0] = 1
            else:
                temp[num] = 1

            one_hot[s] = temp

        return one_hot

    def __store_data(self, data, num_of_examples, gray):
        d = []

        for i in range(num_of_examples):
            if gray:
                d.append(self.__rgb2gray(data[:, :, :, i]))
            else:
                d.append(data[:, :, :, i])

        return np.asarray(d)

    def __rgb2gray(self, rgb):
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
