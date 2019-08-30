# Flow-Motion-Depth

This is the project page of the paper **"Flow-Motion and Depth Network for Monocular Stereo and Beyond''**. This project page contians:

* the implementation of the method,

* the GTA-SfM tools and generated dataset.

## The proposed method

In this work we propose a method that sloves monocular stereo and can further fuse depth information from multiple target images. The inputs and outputs of the method can be illustrated in the figure below. Given a source image, and one or many target images, the proposed method estimates the optical flow and relative poses between each source-target pair. The depth map of the source image is also estimated by fusing optical flow and pose information.

<p align="center">
<img src="fig/input_output.png" alt="input_output" width = "623" height = "300">
</p>

