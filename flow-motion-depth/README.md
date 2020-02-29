# Flow-Motion-Depth Code

Here, we provide codes to implement the network.

* ```correlation.py``` is the correlation implementation. It includes normal correlation from PWCNet and the proposed epipolar correlation.

* ```flow_motion_net.py``` takes images to generate optical flows and camera motion vectors (in angle-axis representation).

* ```flow2depth.py``` generates the proposed triangulate layer.

* ```depth_net.py``` takes image, optical flow, triangulate layer to generate the depth maps.

* ```gen_depth_and_motion.py``` is an example script to generate all estimations from images.
