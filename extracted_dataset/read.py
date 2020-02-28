'''
this is python2.7 code
you should have: h5py to read the file
h5py: http://docs.h5py.org/en/stable/
'''
import h5py 
import cv2
import numpy as np

def visualize_depth(depth):
    disp = 1.0 / (depth + 1e-3)
    disp_vis = disp / 0.1 * 255.0
    disp_vis = disp_vis.astype(np.uint8)
    disp_vis = cv2.applyColorMap(disp_vis,cv2.COLORMAP_JET)
    return disp_vis

sample_path = "YourPath/20190124_203632.hdf5"
h5file = h5py.File(sample_path,"r")
img_num = len(h5file.keys()) / 4
print("the sequence has %d images"%img_num)
print("show image, press esc to exit:")
for i in range(img_num):
    img_name = "image_%d"%i
    K_name = "K_%d"%i
    pose_name = "pose_%d"%i
    depth_name = "depth_%d"%i

    img = cv2.imdecode(h5file[img_name][:],cv2.IMREAD_COLOR)
    K = h5file[K_name][:]
    pose = h5file[pose_name][:]
    depth = h5file[depth_name][:]
    disp_vis = visualize_depth(depth)

    cv2.imshow("img",img)
    cv2.imshow("disp",disp_vis)
    if cv2.waitKey(0) == 27:
        break