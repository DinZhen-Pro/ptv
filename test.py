import numpy as np
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from skimage.feature import peak_local_max
import caculate 
import params1

# 读取数据文件
data_left_file = loadmat('D:\matlabdata\StereoEventPTV-master\StereoEventPTV-master\data\events_master_votex_2_5_2s.mat')
data_right_file = loadmat('D:\matlabdata\StereoEventPTV-master\StereoEventPTV-master\data\events_slave_votex_2_5_2s.mat')
cam_resolution = [480, 360]
events_display_time_range = 0.004
iter_num = 5
# framepeak_left = caculate.tracking2d(data_left_file, cam_resolution, events_display_time_range, iter_num)
# framepeak_right = caculate.tracking2d(data_right_file, cam_resolution, events_display_time_range, iter_num)
[a,b,c]=caculate.tracking2d(data_right_file, cam_resolution, events_display_time_range, iter_num)
b=b[0,:]