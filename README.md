# SVHN
Street View House Numbers (SVHN) is a real-world dataset containing images of house numbers taken from Google's street view.
This repository contains the source code needed to built machine learning algorithms that can "recognize" the numbers on the images.

For this purpose, we will use deep learning and neural network techniques through Tensorflow.

The dataset can be downloaded from here: http://ufldl.stanford.edu/housenumbers/

There are 2 formats available, one with full images and one with cropped. In our case we used the second one.

In order for the code to run, the following libraries are needed, which can be installed via pip:
* numpy
* matplotlib
* scipy
* tensorflow (either with GPU support enabled, or not. In our case we used the lib with the GPU support)

The source code resided under the `src` directory and contains the following files:
* `display_image.py`: A small script to load the dataset and display the images in RGB and grayscale format

* `svhn.py`: A class for loading and manipulating the dataset

* `mlp.py`: Contains the code needed to build a Multilayer Perceptron baseline classifier. The highest accuracy achieved with this, was around 86%

* `cnn.py`: Contains the code needed to build a deep convolution network. We were able to achieve an accuracy of 94.5%

Finally, for the code to run properly, the data files should be placed in a directory called `res`, at the same level of `src` directory. The dataset names are "train_32x32.mat" for the training set, "test_32x32.mat" for the test set and "extra_32x32.mat" for an additional training set. The third file is optional, but if used, it dramatically increases prediction accuracy.
