import numpy as np
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from skimage.feature import peak_local_max
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors

class EM1Params:
    def __init__(self, max_iters=50, sigma=2, min_err=1.0, max_distance=1.0, centered=4):
        self.max_iters = max_iters
        self.sigma = sigma
        self.min_err = min_err
        self.max_distance = max_distance
        self.centered = centered

class params1:
    def __init__(self):
        self.em1_params = EM1Params()  # 嵌套 EM1 参数类
        self.window_size = 15
        self.window_mag = 1.2
        self.min_events_for_em = 20
        self.max_events_per_window = 100000
        self.min_distance = 4

params = params1()

def accumarray(subs, values, shape): #将输入的 (x, y) 坐标及对应的值聚合到一个二维数组（矩阵）中
    rows, cols = subs[:, 0], subs[:, 1]
    return coo_matrix((values, (rows, cols)), shape=shape).toarray()

def pkfnd(im, th, si):

    # 找到所有亮度超过 threshold 的点
    mask = im > th
    
    # 计算局部最大值（峰值点）
    local_max = peak_local_max(im, min_distance=si//2, threshold_abs=th, exclude_border=True)
    
    return local_max

def gaussian_filter1():
    # 定义滤波器大小
    filter_size = 6
    radius = 3  # 过滤半径
    # 创建滤波器
    filter_kernel = np.zeros((filter_size, filter_size))
    # 按照 MATLAB 逻辑填充 1
    for i in range(filter_size):
        for j in range(filter_size):
            if np.sqrt((i - 0.5 - filter_size / 2) ** 2 + (j - 0.5 - filter_size / 2) ** 2) <= radius:
                filter_kernel[i, j] = 1

    return filter_kernel


def rangesearch(X, Y, radius, algorithm='kd_tree', leaf_size=30, sort_indices=True): #在数据集中进行半径范围搜索，找出在给定半径内的点。
    #参考点集，查询点集，搜索半径，近邻搜索算法，树算法的叶子节点大小，是否对结果排序
    #algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}
    nn = NearestNeighbors(radius=radius, 
                          algorithm=algorithm, 
                          leaf_size=leaf_size)
    nn.fit(X)
    
    indices = nn.radius_neighbors(Y, return_distance=True)
    
    if sort_indices:
        sorted_results = []
        for idx, dist in zip(indices[1], indices[0]):
            sorted_idx = np.argsort(dist)
            sorted_results.append((idx[sorted_idx], dist[sorted_idx]))
        return [r[0] for r in sorted_results], [r[1] for r in sorted_results]
    
    return indices[1], indices[0]

def cell2mat(c):#多维单元格数组转换为矩阵的函数,将嵌套的数据结构展平成一个统一的矩阵
    if len(c) == 0:
        return np.array([])
    if len(c) == 1:
        return np.array(c[0])
    first_type = type(c[0])
    if not all(isinstance(item, first_type) for item in c):
        raise TypeError("All elements must be of the same type")
    # 对于二维输入的优化处理
    if np.ndim(c) == 2:
        rows, cols = len(c), len(c[0])
        if rows < cols:
            return np.hstack([np.hstack(row) for row in c])
        else:
            return np.vstack([np.vstack(col) for col in np.array(c).T])
    return np.concatenate(c)


def em_flow(events,feature_pos,flow_init):
    flow=np.array(flow_init)
    flow_out=[np.nan,np.nan]
    time_shifted_points_out=[]
    delta_flows=np.zeros((params.em1_params.max_iters,1))
    num_iter=0
    event_window = np.array([], dtype=bool)
    n_in_window=0
    target_time=events[1,1]
    event_market_out=[]
    centered_events=events
    centered_events[:,1:3]=centered_events[:,1:3]-feature_pos
    prev_flow=flow
    #计算光流
    while True:
        if num_iter>(params.em1_params.max_iters):
            return

        time_shifted_points = centered_events[:, 1:3] + flow * (target_time - centered_events[:, 0])[:, np.newaxis]
        if event_window.size == 0 or not event_window.all():
            half_size = params.window_size / 2
            event_window = np.all([
                time_shifted_points[:, 0] >= -half_size,
                time_shifted_points[:, 1] >= -half_size,
                time_shifted_points[:, 0] <= half_size,
                time_shifted_points[:, 1] <= half_size
            ], axis=0)
            n_in_window = np.count_nonzero(event_window)
            if n_in_window<params.min_events_for_em:
                return

            centered_events1 = centered_events[event_window, :]
            time_shifted_points1 = time_shifted_points[event_window, :]
  
        normalized_events = time_shifted_points1 / (np.sqrt(2) * params.em1_params.sigma)
        print(normalized_events)
    
        kdtree=cKDTree(normalized_events, metric='euclidean')
        [neighbors_1, distance_2]=rangesearch(kdtree,normalized_events,params.em1_params.max_distance)
        # distanccstaked=cell2mat(cellfun())
        print(neighbors_1)

def tracking2d(filename, cam_resolution, events_display_time_range, iter_num):
    data = filename
    curr_events = data[list(data.keys())[-1]]
    curr_events[:, 3] = 1  # 设定极性默认值
    curr_events_pos = curr_events[curr_events[:, 3] > 0]
    curr_events_neg = curr_events[curr_events[:, 3] < 0]
    shape = (cam_resolution[0], cam_resolution[1])
    frameinfo = []
    event_t0 = 0
    start_event_iter = np.searchsorted(curr_events_pos[:, 0], event_t0) #在已排序的数组中找到元素应该插入的索引位置
    new_event_iter = np.searchsorted(curr_events_pos[:, 0], event_t0 + events_display_time_range)
    curr_event_display = curr_events_pos[start_event_iter : new_event_iter -1, :] #当前帧包含的正极性事件
    curr_event_display_pos = curr_event_display[curr_event_display[:,3] > 0, :]
    curr_event_display_neg = curr_event_display[curr_event_display[:,3] < 0, :]
    # print(curr_event_display_pos)
    # 生成事件图像
    indices = np.round(curr_events_pos[start_event_iter:new_event_iter, 1:3]).astype(int)
    curr_events_pos_image = accumarray(indices, np.ones(len(indices)), shape)
    img2 = convolve(curr_events_pos_image, gaussian_filter1(), mode='constant', cval=0.0)
    # 峰值检测
    peak2 = peak_local_max(img2, min_distance=12, threshold_abs=12) #峰值点检测，定义最小峰值间距和绝对强度阈值
    frameinfo.append({
        "peak": peak2,
        "time": events_display_time_range,
    })

    peak_mark = np.full(len(peak2), False, dtype=bool)  
    events_marker = np.full(len(curr_event_display_pos), False, dtype=bool)
    # vol_ini=np.full(len(peak2), dtype=bool)
    flow_nin=[0,0]

    return curr_event_display_pos,peak2,flow_nin

    # for i in range(len(peak2)):
    #     dt=curr_event_display_pos[-1,0] - curr_event_display_pos[1,1]
        

    # for iter in range(iter_num):
    #     new_event_iter = np.searchsorted(curr_events_pos[:, 0], event_t0 + events_display_time_range)
    #     indices = np.round(curr_events_pos[start_event_iter:new_event_iter, 1:3]).astype(int)
    #     curr_events_pos_image = accumarray(indices, np.ones(len(indices)), shape)
    #     img2 = gaussian_filter(curr_events_pos_image, sigma=1)
    #     peak2 = peak_local_max(img2, min_distance=12, threshold_abs=12)

    #     peak_dist = np.linalg.norm(frameinfo[-1]["peak"][:, None] - peak2[None, :], axis=2)#匈牙利算法
    #     row_ind, col_ind = linear_sum_assignment(peak_dist)

    #     frameinfo.append({
    #         "peak": peak2,
    #         "time": events_display_time_range,
    #         "matches": (row_ind, col_ind),
    #     })
    #     event_t0 += events_display_time_range

    # print(frameinfo)
    # return frameinfo

    