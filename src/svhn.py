import numpy as np
import scipy.io as sio


class SVHN:

    def __init__(self, file_path, n_classes, n_input):
        self.n_classes = n_classes
        self.n_input = n_input

        # Load Train Set
        train = sio.loadmat(file_path + "/train_32x32.mat")
        self.train_data = self.flatten_data(train['X'])
        self.train_labels = self.one_hot_encode(train['y'])
        self.train_examples = train['X'].shape[3]

        # Load Test Set
        test = sio.loadmat("../res/test_32x32.mat")
        self.test_data = self.flatten_data(test['X'])
        self.test_labels = self.one_hot_encode(test['y'])
        self.test_examples = test['X'].shape[3]

        self._epochs_completed_train = 0
        self._index_in_epoch_train = 0
        self._epochs_completed_test = 0
        self._index_in_epoch_test = 0
        self._train_data = self.train_data
        self._train_labels = self.train_labels
        self._test_data = self.test_data
        self._test_labels = self.test_labels

    def one_hot_encode(self, data):
        """Creates a one-hot encoding vector
            Args:
                data: The data to be converted
            Returns:
                An array of one-hot encoded items
        """
        n = data.shape[0]
        one_hot = np.zeros(shape=(data.shape[0], self.n_classes))
        for s in range(n):
            temp = [0 * v for v in range(0, self.n_classes)]

            num = data[s][0]
            if num == 10:
                temp[0] = 1
            else:
                temp[num] = 1

            one_hot[s] = temp

        return one_hot

    def flatten_data(self, data):
        """Flattens an image of size n * n * 3, into a an array of size N * 1, where N = n * n * 3
            Args:
                data: The array to be flattened 
            Returns:
                A flattened array
        """
        n = data.shape[3]
        flattened = np.zeros(shape=(n, self.n_input))
        for s in range(n):
            flattened[s] = self.rgb2gray(data[:, :, :, s]).flatten().astype(np.float32)

        return flattened

    def rgb2gray(self, rgb):
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

    def next_train_batch(self, batch_size, shuffle=True):
        start = self._index_in_epoch_train

        # Shuffle for the first epoch
        if self._epochs_completed_train == 0 and start == 0 and shuffle:
            perm0 = np.arange(self.train_examples)
            np.random.shuffle(perm0)
            self._train_data = self.train_data[perm0]
            self._train_labels = self.train_labels[perm0]

        # Go to the next epoch
        if start + batch_size > self.train_examples:
            # Finished epoch
            self._epochs_completed_train += 1

            # Get the rest examples in this epoch
            rest_num_examples = self.train_examples - start
            images_rest_part = self._train_data[start:self.train_examples]
            labels_rest_part = self._train_labels[start:self.train_examples]

            # Shuffle the data
            if shuffle:
                perm = np.arange(self.train_examples)
                np.random.shuffle(perm)
                self._train_data = self.train_data[perm]
                self._train_labels = self.train_labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch_train = batch_size - rest_num_examples
            end = self._index_in_epoch_train
            images_new_part = self._train_data[start:end]
            labels_new_part = self._train_labels[start:end]
            return np.concatenate((images_rest_part, images_new_part), axis=0), \
                np.concatenate((labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch_train += batch_size
            end = self._index_in_epoch_train
            return self._train_data[start:end], self._train_labels[start:end]

    def next_test_batch(self, batch_size, shuffle=True):
        start = self._index_in_epoch_test

        # Shuffle for the first epoch
        if self._epochs_completed_test == 0 and start == 0 and shuffle:
            perm0 = np.arange(self.test_examples)
            np.random.shuffle(perm0)
            self._test_data = self.test_data[perm0]
            self._test_labels = self.test_labels[perm0]

        # Go to the next epoch
        if start + batch_size > self.test_examples:
            # Finished epoch
            self._epochs_completed_test += 1

            # Get the rest examples in this epoch
            rest_num_examples = self.test_examples - start
            images_rest_part = self._test_data[start:self.test_examples]
            labels_rest_part = self._test_labels[start:self.test_examples]

            # Shuffle the data
            if shuffle:
                perm = np.arange(self.test_examples)
                np.random.shuffle(perm)
                self._test_data = self.test_data[perm]
                self._test_labels = self.test_labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch_test = batch_size - rest_num_examples
            end = self._index_in_epoch_test
            images_new_part = self._test_data[start:end]
            labels_new_part = self._test_labels[start:end]
            return np.concatenate((images_rest_part, images_new_part), axis=0), \
                np.concatenate((labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch_test += batch_size
            end = self._index_in_epoch_test
            return self._test_data[start:end], self._test_labels[start:end]
