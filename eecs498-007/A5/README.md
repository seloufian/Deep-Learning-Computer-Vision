<div>
  <h2 align="center"><a href="https://web.eecs.umich.edu/~justincj/teaching/eecs498/">EECS 498-007 / 598-005: Deep Learning for Computer Vision</a></h2>
  <h2 align="center"><a href="https://web.eecs.umich.edu/~justincj/teaching/eecs498/FA2020/assignment5.html">Assignment 5 (2020)</a></h3>
</div>

# Goals

In this assignment you will implement two different object detection systems.

The goals of this assignment are:

- Learn about the object detection pipeline.
- Understand how to build an anchor-based single-stage object detectors.
- Understand how to build a two-stage object detector that combines a region proposal network with a recognition network.

# Questions

## Q1: Single-Stage Detector

The notebook [``single_stage_detector_yolo.ipynb``](https://github.com/seloufian/Deep-Learning-Computer-Vision/blob/master/eecs498-007/A5/single_stage_detector_yolo.ipynb) will walk you through the implementation of a fully-convolutional single-stage object detector similar to YOLO (Redmon et al, CVPR 2016). You will train and evaluate your detector on the [PASCAL VOC 2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/index.html) object detection dataset.

## Q2: Two-Stage Detector

The notebook [``two_stage_detector_faster_rcnn.ipynb``](https://github.com/seloufian/Deep-Learning-Computer-Vision/blob/master/eecs498-007/A5/two_stage_detector_faster_rcnn.ipynb) will walk you through the implementation of a two-stage object detector similar to Faster R-CNN (Ren et al, NeurIPS 2015). This will combine a fully-convolutional Region Proposal Network (RPN) and a second-stage recognition network.
