# cell_counting_v2

Here I provide a new version of code for cell counting.
Please download the synthetic cell dataset here :

http://www.robots.ox.ac.uk/~vgg/research/counting/

Related papers:

[1] Microscopy Cell Counting with Fully Convolutional Regression Networks.

https://www.robots.ox.ac.uk/~vgg/publications/2015/Xie15/weidi15.pdf

[2] U-Net: Convolutional Networks for Biomedical Image Segmentation

https://arxiv.org/abs/1505.04597

In [1,2] both architectures are fully convolutional, one is for cell counting and detection, while the other one is for cell segmentation.
The architectures consist of a down-sampling path followed by an up-sampling path.
During the first several layers, the structure resembles the cannonical classification CNN.
In the second half of the architecture, the spatial resolution gets back to the original size with upsampling operations.

As everybody knows, deep learning is an extremely fast developing field, both papers were published over a year ago.
Thus, though I think the idea shown in the paper is valuable, there is no reason why people should continue using the exact networks.
So, here I provide an updated version of the paper[1]. Batch normalization is used after each linear convolution to make the networks easier to train.

The code is based on Keras(https://github.com/fchollet/keras) with Tensorflow as backend, this is just a very naive version without any tuning on any dataset,
however, it has already performed as good as the networks reported in [1]. (It gives me error about 1-4 cells/image)

So, if anybody is interested in the cell counting task, please use this version, which can make your life much much easier. 
