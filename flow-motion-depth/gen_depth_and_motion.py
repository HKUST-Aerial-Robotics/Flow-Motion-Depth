import torch

# my functions
from flow_motion_net import FlowMotionNet
from depth_net import DepthNet
from flow2depth import Flow2Depth

flow_motion_net = FlowMotionNet()
deptnet = DepthNet(flow_motion_net.last_layer_size)
f2d = Flow2Depth(H = 128, W = 160)

# laod you left and right image
left_image = torch.zeros((1,3,256,320))
right_image = torch.zeros((1,3,256,320))
cat_img = torch.cat([left_image,right_image], dim = 1)

with torch.no_grad():
    # flows are estimated optical flow 
    # motions are estimated camera motion vector
    flows, motion = flow_motion_net(cat_img)

    # Rs, Ts is the estimated camera pose
    Rs, Ts = flow_motion_net.get_motion(motion[0])

    # triangle is the proposed triangulate layer
    triangle = f2d(Rs, Ts, flows[0])

    # depths is the estimated depth maps 
    depths = deptnet(left_image, flow_motion_net.last_layer, flows[0], triangle)

# depth maps are in log space, use exp() to recover
depths = torch.exp(depths[0])
