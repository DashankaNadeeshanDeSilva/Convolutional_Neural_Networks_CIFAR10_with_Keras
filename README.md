# Convolutional_Neural_Networks_CIFAR10_with_Keras

This script explains a convolutional neural network with 10 convolution layers and 1 dense layer on CIFAR10 data.



**CIFAR 10 Dataset**

1. data: a 10000x3072 numpy array of uint8s. Each row of the array stores a 32x32 color (RGB) image.
2. labels: a list of 10000 numbers in the range 0-9. The number at index i indicates the
label of the ith image in the array data. The dataset contains another file, called
batches.meta. It too contains a Python dictionary object. It has the following entries:
3. label_names: a 10-element list which gives meaningful names to the numeric labels
in the labels array described above. For example, label_names[0] == "airplane",
label_names[1] == "automobile", etc.
