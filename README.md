# cell_counting_v2

Here I provide a new version of code for cell counting.

Related papers:

[1] Microscopy Cell Counting with Fully Convolutional Regression Networks.
https://www.robots.ox.ac.uk/~vgg/publications/2015/Xie15/weidi15.pdf

[2] U-Net: Convolutional Networks for Biomedical Image Segmentation
https://arxiv.org/abs/1505.04597

In [1,2] both architectures are fully convolutional, one is for cell counting and detection, while the other one is for cell segmentation.
The architectures consist of a down-sampling path (left) followed by an up-sampling path (right).
During the first several layers, the structure resembles the cannonical classification CNN.
In the second half of the architecture, the spatial resolution got back to the original size with upsampling operations.

As everybody knows, deep learning is an extremely fast developing field, both papers were published over a year ago.
So, here I provide an updated version of the paper[1].

Batch normalization is used after each linear convolution to make the networks easier to train, 
if anybody is interested in the cell counting task, please use this version, which can make your life much much easier.
