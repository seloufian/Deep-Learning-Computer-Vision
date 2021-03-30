<div>
  <h2 align="center"><a href="https://cs231n.github.io">CS231n: Convolutional Neural Networks for Visual Recognition</a></h2>
  <h2 align="center"><a href="https://cs231n.github.io/assignments2020/assignment2/">Assignment 2 (2020)</a></h3>
</div>

# Goals

In this assignment you will practice writing backpropagation code, and training Neural Networks and Convolutional Neural Networks. The goals of this assignment are as follows:

- Understand **Neural Networks** and how they are arranged in layered architectures.
- Understand and be able to implement (vectorized) **backpropagation**.
- Implement various **update rules** used to optimize Neural Networks.
- Implement **Batch Normalization** and **Layer Normalization** for training deep networks.
- Implement **Dropout** to regularize networks.
- Understand the architecture of **Convolutional Neural Networks** and get practice with training them.
- Gain experience with a major deep learning framework: **PyTorch**.

# Questions

## Q1: Fully-connected Neural Network

The notebook [``FullyConnectedNets.ipynb``](https://github.com/seloufian/Deep-Learning-Computer-Vision/blob/master/cs231n/assignment2/FullyConnectedNets.ipynb) will introduce you to our modular layer design, and then use those layers to implement fully-connected networks of arbitrary depth. To optimize these models you will implement several popular update rules.

## Q2: Batch Normalization

In notebook [``BatchNormalization.ipynb``](https://github.com/seloufian/Deep-Learning-Computer-Vision/blob/master/cs231n/assignment2/BatchNormalization.ipynb) you will implement batch normalization, and use it to train deep fully-connected networks.

## Q3: Dropout

The notebook [``Dropout.ipynb``](https://github.com/seloufian/Deep-Learning-Computer-Vision/blob/master/cs231n/assignment2/Dropout.ipynb) will help you implement Dropout and explore its effects on model generalization.

## Q4: Convolutional Networks

In the IPython Notebook [``ConvolutionalNetworks.ipynb``](https://github.com/seloufian/Deep-Learning-Computer-Vision/blob/master/cs231n/assignment2/ConvolutionalNetworks.ipynb) you will implement several new layers that are commonly used in convolutional networks.

## Q5: PyTorch on CIFAR-10

For this last part, you will be working in PyTorch, a popular and powerful deep learning framework. Open up [``PyTorch.ipynb``](https://github.com/seloufian/Deep-Learning-Computer-Vision/blob/master/cs231n/assignment2/PyTorch.ipynb). There, you will learn how the framework works, culminating in training a convolutional network of your own design on CIFAR-10 to get the best performance you can.
