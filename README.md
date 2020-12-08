# Convolutional_Neural_Networks_CIFAR10_with_Keras

This script explains a convolutional neural network with 10 convolution layers and 1 dense layer on [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) data.



**CIFAR 10 Dataset**

1. data: a 10000x3072 numpy array of uint8s. Each row of the array stores a 32x32 color (RGB) image.
2. labels: a list of 10000 numbers in the range 0-9. The number at index i indicates the label of the ith image in the array data.
3. label_names: a 10-element list which gives meaningful names to the numeric labels in the labels array 

**Main components of the CNN Architecture**

1. Convolutional layer: CNN are used to recognize images by transforming the original image through layers to a class score.
2. Max pooling: Its function is to progressively reduce the spatial size of the representation to reduce the amounts of parameters and computation in the network. Pooling layer operates on each feature map independently.
3. Flatten layer: When you finish editing all the layers in your image, you can merge or flatten layers to reduce the file size. Flattening combines all the layers into a single background layer.
4. Dense layer: dense layer is simply a layer where each unit or neuron is connected to each neuron in the next layer

**CNN Architecture**

10-Convolution layers (32,32,64,64,128,128,256,256,512,512)

1-Dense layer

