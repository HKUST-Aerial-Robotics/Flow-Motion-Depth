import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class Flow2Depth(nn.Module):
    def __init__(self, H, W):
        """
        this takes the optical flow, camera pose to generate the triangulation layer
        input: maxd --- displacement alone epipolar line (for correlation. default: 4)
               mind --- displacement perpendicular to epipolar line (for correlation. default: 4)
        """
        super(Flow2Depth, self).__init__()
        self.H = H
        self.W = W

        # the intrinsic is the same as DeMoN
        K = np.zeros((3,3))
        K[0,0] = 0.89115971 * self.W
        K[0,2] = 0.5 * self.W
        K[1,1] = 1.18821287 * self.H
        K[1, 2] = 0.5 * self.H
        K[2,2] = 1.0
        Ki = np.linalg.inv(K)

        pixel_dir = np.zeros((self.H, self.W, 3))
        for i in range(self.H):
            for j in range(self.W):
                pixel_dir[i, j, :] = np.dot(Ki, np.array([j, i, 1]))
        pixel_dir = pixel_dir.reshape(self.H * self.W, 3, 1).astype(np.float32)

        pixel_loc = np.zeros((self.H, self.W, 2))
        for i in range(self.H):
            for j in range(self.W):
                pixel_loc[i, j, :] = [j,i]
        pixel_loc = pixel_loc.reshape(1, self.H, self.W, 2).astype(np.float32)

        self.K = torch.from_numpy(K.reshape(1,3,3).astype(np.float32)).cuda()
        self.Ki = torch.from_numpy(Ki.reshape(1,3,3).astype(np.float32)).cuda()
        self.pixel_dir = torch.from_numpy(pixel_dir).cuda()
        self.pixel_loc = torch.from_numpy(pixel_loc).cuda()


    def forward(self, R, T, initial_flow):
        B, _, H, W = initial_flow.shape
        first_part = torch.matmul(self.K, R).view(B, 1, 3, 3)
        first_part = torch.matmul(first_part, self.pixel_dir)

        second_part = torch.matmul(self.K, T).view(B, 1, 3, 1)
        second_part = second_part.expand(-1, self.H*self.W, -1, -1)
        
        flow_point = self.pixel_loc + initial_flow.permute(0,2,3,1)
        flow_point = flow_point.view(B, H*W, 2, 1)


        triangle = torch.cat([first_part, second_part, flow_point], dim = 2)

        triangle = triangle.permute(0,2,1,3) #B*8*(H*W)*1
        triangle = triangle.view(B,-1,H,W)

        return triangle