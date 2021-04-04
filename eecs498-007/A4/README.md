<div>
  <h2 align="center"><a href="https://web.eecs.umich.edu/~justincj/teaching/eecs498/">EECS 498-007 / 598-005: Deep Learning for Computer Vision</a></h2>
  <h2 align="center"><a href="https://web.eecs.umich.edu/~justincj/teaching/eecs498/FA2020/assignment4.html">Assignment 4 (2020)</a></h3>
</div>

# Goals

From this assignment forward, you will use autograd in PyTorch to perform backpropgation for you. This will enable you to easily build complex models without worrying about writing code for the backward pass by hand.

The goals of this assignment are:

- Understand how autograd can help automate gradient computation.
- See how to use PyTorch Modules to build up complex neural network architectures.
- Understand and implement recurrent neural networks.
- See how recurrent neural networks can be used for image captioning.
- Understand how to augment recurrent neural networks with attention.
- Use image gradients to synthesize saliency maps, adversarial examples, and perform class visualizations.
- Combine content and style losses to perform artistic style transfer.

# Questions

## Q1: PyTorch Autograd

The notebook [``pytorch_autograd_and_nn.ipynb``](pytorch_autograd_and_nn.ipynb) will introduce you to the different levels of abstraction that PyTorch provides for building neural network models. You will use this knowledge to implement and train Residual Networks for image classification.

## Q2: Image Captioning with Recurrent Neural Networks

The notebook [``rnn_lstm_attention_captioning.ipynb``](rnn_lstm_attention_captioning.ipynb) will walk you through the implementation of vanilla recurrent neural networks (RNN) and Long Short Term Memory (LSTM) RNNs. You will use these networks to train an image captioning model. You will then augment your implementation to perform spatial attention over image regions while generating captions.

## Q3: Network Visualization

The notebook [``network_visualization.ipynb``](network_visualization.ipynb) will walk you through the use of image gradients for generating saliency maps, adversarial examples, and class visualizations.

## Q4: Style Transfer

In the notebook [``style_transfer.ipynb``](style_transfer.ipynb), you will learn how to create images with the artistic style of one image and the content of another image.
