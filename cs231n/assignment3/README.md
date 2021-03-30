<div>
  <h2 align="center"><a href="https://cs231n.github.io">CS231n: Convolutional Neural Networks for Visual Recognition</a></h2>
  <h2 align="center"><a href="https://cs231n.github.io/assignments2020/assignment3/">Assignment 3 (2020)</a></h3>
</div>

# Goals

In this assignment, you will implement recurrent neural networks and apply them to image captioning on the Microsoft COCO data. You will also explore methods for visualizing the features of a pretrained model on ImageNet, and use this model to implement Style Transfer. Finally, you will train a Generative Adversarial Network to generate images that look like a training dataset!

The goals of this assignment are as follows:

- Understand the architecture of **recurrent neural networks (RNNs)** and how they operate on sequences by sharing weights over time.
- Understand and implement both **Vanilla RNNs** and **Long-Short Term Memory (LSTM)** networks.
- Understand how to combine convolutional neural nets and recurrent nets to implement an **image captioning** system.
- Explore various applications of **image gradients**, including **saliency maps**, **fooling images**, **class visualizations**.
- Understand and implement techniques for image **style transfer**.
- Understand how to train and implement a **Generative Adversarial Network (GAN)** to produce images that resemble samples from a dataset.

# Questions

## Q1: Image Captioning with Vanilla RNNs

The notebook [``RNN_Captioning.ipynb``](https://github.com/seloufian/Deep-Learning-Computer-Vision/blob/master/cs231n/assignment3/RNN_Captioning.ipynb) will walk you through the implementation of an image captioning system on MS-COCO using vanilla recurrent networks.

## Q2: Image Captioning with LSTMs

The notebook [``LSTM_Captioning.ipynb``](https://github.com/seloufian/Deep-Learning-Computer-Vision/blob/master/cs231n/assignment3/LSTM_Captioning.ipynb) will walk you through the implementation of Long-Short Term Memory (LSTM) RNNs, and apply them to image captioning on MS-COCO.

## Q3: Network Visualization: Saliency maps, Class Visualization, and Fooling Images

The notebook [``NetworkVisualization-PyTorch.ipynb``](https://github.com/seloufian/Deep-Learning-Computer-Vision/blob/master/cs231n/assignment3/NetworkVisualization-PyTorch.ipynb) will introduce the pretrained SqueezeNet model, compute gradients with respect to images, and use them to produce saliency maps and fooling images.

## Q4: Style Transfer

In the notebook [``StyleTransfer-PyTorch.ipynb``](https://github.com/seloufian/Deep-Learning-Computer-Vision/blob/master/cs231n/assignment3/StyleTransfer-PyTorch.ipynb) you will learn how to create images with the content of one image but the style of another.

## Q5: Generative Adversarial Networks

In the notebook [``Generative_Adversarial_Networks_PyTorch.ipynb``](https://github.com/seloufian/Deep-Learning-Computer-Vision/blob/master/cs231n/assignment3/Generative_Adversarial_Networks_PyTorch.ipynb) you will learn how to generate images that match a training dataset, and use these models to improve classifier performance when training on a large amount of unlabeled data and a small amount of labeled data.
