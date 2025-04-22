import numpy as np
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter
from scipy.sparse import coo_matrix
from skimage.feature import peak_local_max
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from munkres import Munkres
from numba import njit     
from line_profiler import LineProfiler
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def accumulate(data, shape): #将输入的 (x, y) 坐标及对应的值聚合到一个二维数组（矩阵）中,并进行归一化操作
    # 取出 x 和 y 坐标
    x_indices = np.round(data[:, 1]).astype(int)
    y_indices = np.round(data[:, 2]).astype(int)
    x_indices = np.clip(x_indices - 1, 0, shape[0] - 1)
    y_indices = np.clip(y_indices - 1, 0, shape[1] - 1) 

    # unique_coords = np.unique(np.stack((x_indices, y_indices), axis=1), axis=0)
    # curr_events_image = np.zeros((shape), dtype=int)
    # curr_events_image[unique_coords[:, 0], unique_coords[:, 1]] = 1  # 只设置 1

    curr_events_image = np.zeros(shape, dtype=np.int32)
    curr_events_image[x_indices, y_indices] = 1

    return curr_events_image

def pkfnd(im, th, si):

    # 找到所有亮度超过 threshold 的点
    mask = im > th
    
    # 计算局部最大值（峰值点）
    local_max = peak_local_max(im, min_distance=si//2, threshold_abs=th, exclude_border=True)
    
    return local_max

def gaussian_filter1():
    filter_size = 6
    radius = 3
    filter_kernel = np.zeros((filter_size, filter_size), dtype=int)
    center = filter_size / 2 - 0.5

    for i in range(filter_size):
        for j in range(filter_size):
            distance = np.sqrt((i - center) ** 2 + (j - center) ** 2)
            if distance <= radius:
                filter_kernel[i, j] = 1
    return filter_kernel

@njit
def compute_flow(X, D, weights):
    weighted_D = D * weights
    DDT = (weighted_D * D).sum()
    XDT_x = (weighted_D * X[:, 0]).sum()
    XDT_y = (weighted_D * X[:, 1]).sum()
    return np.array([XDT_x / DDT, XDT_y / DDT])

def rangesearch(X, Y, radius, algorithm='kd_tree', leaf_size=30, sort_indices=True): #在数据集中进行半径范围搜索，找出在给定半径内的点。
    #参考点集，查询点集，搜索半径，近邻搜索算法，树算法的叶子节点大小，是否对结果排序
    #algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}
    nn = NearestNeighbors(radius=radius, 
                          algorithm=algorithm, 
                          leaf_size=leaf_size)
    nn.fit(X.data)
    
    indices = nn.radius_neighbors(Y, return_distance=True)
    
    if sort_indices:
        sorted_results = []
        for idx, dist in zip(indices[1], indices[0]):
            sorted_idx = np.argsort(dist)
            sorted_results.append((idx[sorted_idx], dist[sorted_idx]))
        return [r[0] for r in sorted_results], [r[1] for r in sorted_results]
    
    return indices[1], indices[0]

# def cell2mat(c):#多维单元格数组转换为矩阵的函数,将嵌套的数据结构展平成一个统一的矩阵
#     if len(c.szie) == 0:
#         return np.array([])
#     if len(c) == 1:
#         return np.array(c[0])
#     first_type = type(c[0])
#     if not all(isinstance(item, first_type) for item in c):
#         raise TypeError("All elements must be of the same type")
#     # 对于二维输入的优化处理
#     if np.ndim(c) == 2:
#         rows, cols = len(c), len(c[0])
#         if rows < cols:
#             return np.hstack([np.hstack(row) for row in c])
#         else:
#             return np.vstack([np.vstack(col) for col in np.array(c).T])
#     return np.concatenate(c)

def peak_center_find(peak_in, img2, img_window,peak_center1):
    for i in range(len(peak_in)):
        x, y = peak_in[i]  
        # 获取当前峰值点的坐标
        x_min, x_max = int(x - img_window / 2), int(x + img_window / 2) + 1
        y_min, y_max = int(y - img_window / 2), int(y + img_window / 2) + 1
        peak_window_index = img2[x_min:x_max, y_min:y_max]
        # 根据提供的坐标，提取窗口区域
        peak_win_sum = np.sum(peak_window_index)
        # 计算窗口内所有像素的总和，用于归一化权重
        y_mesh, x_mesh = np.meshgrid(
            np.arange(y_min, y_max), np.arange(x_min, x_max) #生成一个 2D 网格矩阵，对应窗口内每个像素点的坐标
        )
        # 计算加权坐标（加权平均）
        location_x = (peak_window_index * x_mesh) / peak_win_sum
        location_y = (peak_window_index * y_mesh) / peak_win_sum
        # 计算加权中心
        peak_center1[i, :] = [np.sum(location_x), np.sum(location_y)]

    return peak_center1

def peak_link(peaks_p,peaks_t):
    x1 = peaks_p[:, 0].reshape(-1, 1)  # shape: (M, 1)
    x2 = peaks_t[:, 0].reshape(1, -1)       # shape: (1, N)

    y1 = peaks_p[:, 1].reshape(-1, 1)
    y2 = peaks_t[:, 1].reshape(1, -1)

    peak_dist = (x1 - x2) ** 2 + (y1 - y2) ** 2

    return peak_dist

def peak_link1(peaks_p,peaks_t):
    x1 = peaks_p[:, 0].reshape(1, -1)  # shape: (M, 1)
    x2 = peaks_t[:, 0].reshape(-1, 1)       # shape: (1, N)

    y1 = peaks_p[:, 1].reshape(1, -1)
    y2 = peaks_t[:, 1].reshape(-1, 1)

    peak_dist2 = (x1 - x2) ** 2 + (y1 - y2) ** 2

    return peak_dist2


def em_flow_torch(events_np, feature_pos_np, flow_init_np):#并行处理peak[i]
    events = torch.tensor(events_np, dtype=torch.float32, device=device)
    feature_pos = torch.tensor(feature_pos_np, dtype=torch.float32, device=device)
    flow = torch.tensor(flow_init_np, dtype=torch.float32, device=device)

    # 事件中心化
    centered_events = events.clone()
    centered_events[:, 1:3] -= feature_pos

    target_time = centered_events[1, 0]
    dt = target_time - centered_events[:, 0]
    time_shifted = centered_events[:, 1:3] + flow.T * dt[:, None]

    half_size = params.window_size / 2
    window_mask = (time_shifted[:, 0] >= -half_size) & (time_shifted[:, 0] <= half_size) & \
                  (time_shifted[:, 1] >= -half_size) & (time_shifted[:, 1] <= half_size)

    if window_mask.sum() < params.min_events_for_em:
        return [float('nan'), float('nan')], None, None

    centered_events1 = centered_events[window_mask]
    time_shifted1 = time_shifted[window_mask]

    # 归一化
    norm = torch.sqrt(torch.tensor(2.0)) * params.em1_params.sigma
    normalized_events = time_shifted1 / norm

    # 使用广播计算 pairwise 距离矩阵（替代 KDTree）
    diff = normalized_events.unsqueeze(1) - normalized_events.unsqueeze(0)
    dists = torch.norm(diff, dim=2)
    neighbor_mask = dists < params.em1_params.max_distance

    event_inds, neighbor_inds = torch.nonzero(neighbor_mask, as_tuple=True)
    valid = neighbor_inds > event_inds
    event_inds = event_inds[valid]
    neighbor_inds = neighbor_inds[valid]

    # 权重
    dist_vals = dists[event_inds, neighbor_inds]
    weights = torch.exp(-dist_vals)

    ori = centered_events1[event_inds]
    nbr = centered_events1[neighbor_inds]

    X = ori[:, 1:3] - nbr[:, 1:3]
    D = ori[:, 0] - nbr[:, 0]

    weighted_D = D * weights
    DDT = torch.sum(weighted_D * D)
    XDT = torch.sum(weighted_D[:, None] * X, dim=0)

    if DDT == 0:
        return [float('nan'), float('nan')], None, None

    flow = (XDT / DDT).unsqueeze(0)  # 1×2
    flow_np = flow.cpu().numpy()

    # time_shifted_points_out（用于后续窗口筛选）
    total_dt = events[-1, 0] - events[0, 0]
    centered_events[:, 1:3] += flow * total_dt
    time_shifted_points2 = centered_events[:, 1:3]

    window_size2 = round(params.window_size * params.window_mag)
    mask2 = (time_shifted_points2[:, 0] >= -window_size2 / 2) & (time_shifted_points2[:, 0] <= window_size2 / 2) & \
            (time_shifted_points2[:, 1] >= -window_size2 / 2) & (time_shifted_points2[:, 1] <= window_size2 / 2)
    
    return flow_np[0], time_shifted_points2[mask2].cpu().numpy(), mask2.cpu().numpy()





def em_flow(events,feature_pos,flow_init):
    flow=np.array(flow_init)
    flow_out=[np.nan,np.nan]
    time_shifted_points_out=[]
    # delta_flows=np.zeros((params.em1_params.max_iters,1))
    num_iter=0
    event_window = np.array([], dtype=bool)
    n_in_window=0
    target_time=events[1,1]
    # event_market_out=[]
    centered_events=events.copy()
    centered_events[:,1:3]-=feature_pos
    prev_flow=flow
    neighbor_inds = np.array([])
    #计算光流
    while True:
        if num_iter>(params.em1_params.max_iters):
            break

        time_shifted_points = centered_events[:, 1:3] + flow.T * (target_time - centered_events[:, 0])[:, np.newaxis] 
        #计算时间偏移后的坐标,根据当前估计的光流，把每个事件点投影到了同一时刻
        if event_window.size == 0:
            half_size = params.window_size / 2
            event_window = np.all([
                time_shifted_points[:, 0] >= -half_size,
                time_shifted_points[:, 1] >= -half_size,
                time_shifted_points[:, 0] <= half_size,
                time_shifted_points[:, 1] <= half_size #筛选符合窗口事件的点
            ], axis=0)
            n_in_window = np.count_nonzero(event_window)
            if n_in_window<params.min_events_for_em:
                break

            centered_events1 = centered_events[event_window, :]
            
        time_shifted_points1 = time_shifted_points[event_window, :]#每轮都更新对应当前 flow 计算出来的时间偏移
        normalized_events = time_shifted_points1 / (np.sqrt(2) * params.em1_params.sigma) #归一化
    
        kdtree=cKDTree(normalized_events) #建立kd树，来查找邻居点
        [neighbors_1, distance_1]=rangesearch(kdtree,normalized_events,params.em1_params.max_distance)
        distancestacked=np.concatenate([np.array(distance).T for distance in distance_1])
        
        if distancestacked.size == 0: #若没有邻居点直接返回
            break
        
        # num_neighbors_per_event = [len(x) for x in neighbors_1] #每个事件点的邻居数
        # neighbor_inds = np.concatenate([np.array(neighbors).T for neighbors in neighbors_1])#邻居索引
        # event_inds = np.repeat(np.arange(1, n_in_window + 1), num_neighbors_per_event)#事件点索引，最终得到事件点与邻居点的索引关系，使数据扁平化，方便后续处理
        total_neighbors = sum(len(n) for n in neighbors_1)
        neighbor_inds = np.empty(total_neighbors, dtype=np.int32)
        event_inds = np.empty(total_neighbors, dtype=np.int32)
        idx = 0
        for i, neighbors in enumerate(neighbors_1):
            l = len(neighbors)
            if l == 0:
                continue
            neighbor_inds[idx:idx+l] = neighbors
            event_inds[idx:idx+l] = i
            idx += l  #得到事件点与邻居点的索引关系

        valid_correspondences = neighbor_inds > event_inds # 通过布尔索引筛选有效的邻居点、事件点和距离
        neighbor_inds = neighbor_inds[valid_correspondences]
        event_inds = event_inds[valid_correspondences]
        distancestacked = distancestacked[valid_correspondences]

        weights = np.exp(-distancestacked)#高斯权重函数，距离越近，权重越大
        neighbor_events = centered_events1[neighbor_inds, : ] #选取邻居事件点
        original_events = centered_events1[event_inds, :] #选取原始事件点

        X = original_events[:, 1:3] - neighbor_events[:, 1:3] 
        D = original_events[:, 0] - neighbor_events[:, 0]
        weighted_D = D * weights
        # DDT = np.dot(weighted_D.T, D)  
        # XDT = np.dot(weighted_D.T, X)  
        # flow = XDT/DDT # 计算最小二乘解
        # flow = flow.T
        # # delta_flow = []
        flow=compute_flow(X, D, weights)

        flow_change = np.linalg.norm(flow - prev_flow)
        if flow_change < params.em1_params.min_err:
            break  # 提前终止迭代
        # delta_flow.append(flow_change)
        prev_flow = flow.copy() #把当前 flow 存入 prev_flow，用于下次迭代
        num_iter += 1


    dt=events[-1,0]-events[0,0]
    centered_events=events
    centered_events[:,1:3]=centered_events[:,1:3]-feature_pos+flow.T*dt #计算中心化的事件坐标，events(:, 2:3)-feature_pos让事件相对于特征点进行中心化
    time_diff = events[-1, 0] - centered_events[:, 0]  
    flow_scaled = flow.T * time_diff[:, np.newaxis]  #将时间差广播到 flow 形状
    time_shifted_points2 = centered_events[:, 1:3] + flow_scaled #对每一个事件点进行光流偏移

    #筛选较小范围事件
    window_size = round(params.window_size * 1.5)
    x, y = time_shifted_points2[:, 0], time_shifted_points2[:, 1]
    event_window = (x >= -window_size / 2) & (x <= window_size / 2) & \
                (y >= -window_size / 2) & (y <= window_size / 2)
    num_true = np.count_nonzero(event_window)
    time_shifted_points_out = time_shifted_points2[event_window,:]

    #筛选更大范围事件
    window_size2 = round(params.window_size * params.window_mag)
    x, y = time_shifted_points2[:, 0], time_shifted_points2[:, 1]
    event_window = (x >= -window_size2 / 2) & (x <= window_size2 / 2) & \
                (y >= -window_size2 / 2) & (y <= window_size2 / 2)
    event_marker_out=event_window
    flow_out=flow
    return flow_out,time_shifted_points_out, event_marker_out


def tracking2d(filename, cam_resolution, events_display_time_range, iter_num):
    data = filename
    curr_events = data[list(data.keys())[-1]]
    curr_events[:, 3] = 1  # 设定极性默认值
    curr_events_pos = curr_events[curr_events[:, 3] > 0]
    curr_events_neg = curr_events[curr_events[:, 3] < 0]
    curr_events_pos_mask= np.zeros((curr_events_pos.shape[0], 1), dtype=bool)
    curr_events_neg_mask= np.zeros((curr_events_neg.shape[0], 1), dtype=bool)
    shape = (cam_resolution[0], cam_resolution[1])
    frameinfo = []
    event_t0 = 0
    start_event_iter = np.searchsorted(curr_events_pos[:, 0], event_t0) #在已排序的数组中找到元素应该插入的索引位置
    new_event_iter = np.searchsorted(curr_events_pos[:, 0], event_t0 + events_display_time_range) #第一个超出当前时间窗口的事件索引
    event_t0 += events_display_time_range
    old_event_iter = new_event_iter
    curr_event_display = curr_events_pos[start_event_iter : new_event_iter, :] #当前帧包含的正极性事件
    curr_event_display_pos = curr_event_display[curr_event_display[:,3] > 0, :] 
    curr_event_display_neg = curr_event_display[curr_event_display[:,3] < 0, :]

    curr_events_pos_image=accumulate(curr_event_display_pos,shape)
    #开始出现误差

    # 生成事件图像
    # indices = np.round(curr_events_pos[start_event_iter:new_event_iter, 1:3]).astype(int)
    # curr_events_pos_image = accumarray(indices, np.ones(len(indices)), shape)
    
    img2 = convolve(curr_events_pos_image, gaussian_filter1(), mode='constant', cval=0.0)
    # 峰值检测
    peak2 = peak_local_max(img2, min_distance=12, exclude_border=12) #峰值点检测，定义最小峰值间距和绝对强度阈值
    peak_center=np.zeros(peak2.shape)
    img_window=8
    peak2=peak_center_find(peak2, img2, img_window, peak_center) #计算峰值点的加权中心坐标

    # frameinfo.append({
    #     "peak": peak2,
    #     "time": events_display_time_range,
    # })

    peak_mark = np.full(len(peak2), False, dtype=bool)  
    events_marker = np.full((len(curr_event_display_pos),1), False, dtype=bool)
    vol_ini=np.zeros((len(peak2),2))
    flow_ini=[0,0]
    for i in range(len(peak2)):
        curr_event_display_pos_i = curr_event_display[curr_event_display[:,3] > 0, :] 
        dt = curr_event_display_pos_i[-1, 0] - curr_event_display_pos_i[1,0]
        [flow,shifted_points, event_marker] = em_flow_torch(curr_event_display_pos_i, peak2[i], flow_ini)
        events_marker[event_marker] = True #标记已处理点
        if sum(np.isnan(flow))>0:
            peak_mark[i] = False #无法估算速度
        else:
            peak_mark[i] = True #flow为速度dx/dt，dy/dt ，乘以dt得到位移
            vol_ini[i,:]=flow.T*dt

    peak_tmp = peak2[peak_mark, :]
    frameinfo.append({"peak": peak_tmp})  # 保存粒子坐标
    frameinfo.append({"time": dt})# 存储时间间隔
    frameinfo.append({"vol": vol_ini[peak_mark, :]})# 存储有效粒子的运动速度
    vol_before = vol_ini[peak_mark, :]  # 存储当前粒子运动速度
    peaks_before = peak_tmp  # 存储当前坐标
    peaks_predict = peak_tmp + vol_ini[peak_mark, :]  # 计算下一帧的预测位置

    curr_events_pos_mask[start_event_iter:new_event_iter] = events_marker #生成一个索引范围，标记哪些事件被处理
    iter_num_max = np.floor(curr_events_pos[-1, 0] / events_display_time_range) - 1 #时间戳除以时间范围，来确定处理图像的步长，确定最大迭代次数
    if iter_num > iter_num_max:
        print('Frame number exist the frame number in the dataset') #不能超过最大迭代次数
        return
    
    print('2D tracking frame: ') #跳出初始化阶段，从第二帧开始，使用此前传递的参数，继续执行上方相同的命令操作
    for iter in range(iter_num):
        print('%d..',iter)
        int_time=events_display_time_range
        # new_event_iter=np.searchsorted(int_time + event_t0 <= curr_events_pos[:,1],1) #寻找事件边界
        new_event_iter = np.searchsorted(curr_events_pos[:, 0], int_time + event_t0)
        #做到这里
        curr_event_display = curr_events_pos[old_event_iter : new_event_iter - 1,:] #选取范围内事件数据
        event_t0_old = event_t0
        event_t0 = curr_events_pos[new_event_iter-2, 0] #记录新的事件点
        curr_event_display_pos = curr_event_display
        # indices = np.round(curr_event_display_pos[:, 1:3]).astype(int)
        curr_events_pos_image = accumulate(curr_event_display_pos, shape)
        #将事件点的 (x, y) 坐标投影到一个二维图像上，每个像素表示命中该坐标的事件数量
        images_test = curr_events_pos_image.copy()
        images_test = np.clip(images_test, 0, 1) #限制最大值为1
        
        img2 = convolve(images_test, gaussian_filter1(), mode='constant', cval=0.0)
        peak2=peak_local_max(img2, min_distance=12, threshold_abs=12) #峰值点检测，定义最小峰值间距和绝对强度阈值
        peak_center=np.zeros(peak2.shape)
        peak2=peak_center_find(peak2, img2, img_window, peak_center) #计算峰值点的加权中心坐标
        
        peak_mark = np.full((len(peak2),1), False, dtype=bool)
        events_marker = np.full((len(curr_event_display_pos),1), False, dtype=bool)#定义布尔向量，元素初始化为false
        vol_ini=np.zeros((len(peak2),2))
        flow_ini=[0,0]
        for i in range(len(peak2)):
            # curr_event_display_pos_i = curr_event_display[curr_event_display[:,3] > 0, :] 
            # dt = curr_event_display_pos_i[-1, 1] - curr_event_display_pos_i[1,1]
            # [flow,shifted_points, event_marker] = em_flow(curr_event_display_pos_i, peak2[i], flow_ini)
            # events_marker[event_marker] = True #标记已处理点
            # if sum(np.isnan(flow))>0:
            #     peak_mark[i] = False #无法估算速度
            # else:
            #     peak_mark[i] = True #flow为速度dx/dt，dy/dt ，乘以dt得到位移
            #     #print(np.count_nonzero(peak_mark))
            #     vol_ini[i,:]=flow.T*dt

            print(f"开始处理第 {i} 个 peak2，总共 {len(peak2)} 个")
            curr_event_display_pos_i = curr_event_display[curr_event_display[:,3] > 0, :] 
            if curr_event_display_pos_i.shape[0] < 2:
                print(f"[{i}] 事件数太少，跳过")
                continue
            dt = curr_event_display_pos_i[-1, 0] - curr_event_display_pos_i[1,0]
            print(f"[{i}] dt = {dt}")
            print(f"[{i}] 调用 em_flow")

            try:
                flow, shifted_points, event_marker = em_flow_torch(curr_event_display_pos_i, peak2[i], flow_ini)
            except Exception as e:
                print(f"[{i}] em_flow 出错: {e}")
                continue

            print(f"[{i}] flow = {flow}")

            if sum(np.isnan(flow)) > 0:
                print(f"[{i}] flow is NaN")
                peak_mark[i] = False
            else:
                peak_mark[i] = True
                vol_ini[i,:] = flow.T * dt
                print(f"[{i}] 写入 vol_ini 完成")


        curr_events_pos_mask[old_event_iter : new_event_iter-1] = events_marker #生成一个索引范围，标记哪些事件被处理
        old_event_iter = new_event_iter

        print("peak2 shape:", peak2.shape)
        print("peak_mark shape:", peak_mark.shape)

        peak_mark = peak_mark.flatten()
        peak_tmp = peak2[peak_mark, :]
        vol_tmp = vol_ini[peak_mark, :]

        frameinfo[iter + 1] = {
            'peak': peak_tmp, # 保存粒子坐标
            'time': dt,       # 存储时间间隔
            'vol': vol_tmp    # 存储有效粒子的运动速度
            }
        # frameinfo.append({"peak": peak_tmp})  # 保存粒子坐标
        # frameinfo.append({"time": dt})# 存储时间间隔
        # frameinfo.append({"vol": vol_ini[peak_mark, :]})# 存储有效粒子的运动速度

        peak_predict_back = peak2[peak_mark, :] - vol_ini[peak_mark, :] #迭代传递参数

        peak_dist=peak_link(peaks_predict, peak_tmp) #计算当前帧和上一帧的距离矩阵，得到每个点之间的距离
        peak_dist2=peak_link1(peak_predict_back, peaks_before) #计算当前帧和上一帧的距离矩阵，得到每个点之间的距离


        vol_before_max=8*(vol_before[:,0]**2+vol_before[:,1]**2) #计算匹配的最大允许距离
        vol_tmp_max=8*(vol_tmp[:,0]**2+vol_tmp[:,1]**2)

        peak_dist[peak_dist > vol_before_max[:, np.newaxis]] = np.inf
        peak_dist2[peak_dist2 > vol_tmp_max[np.newaxis, :]] = np.inf #过滤掉不符合匹配条件的峰值点，将超出最大允许匹配距离的点设置为inf

        [assignment,cost]=linear_sum_assignment(peak_dist) #匈牙利算法，寻找最小代价匹配，输出的是匹配结果
        [assignment2,cost2]=linear_sum_assignment(peak_dist2)

        assignment_final=assignment
        assignment_index=assignment_final==0
        assignment_final[assignment_index] = assignment2[assignment_index]
        #如果第一种匹配（向前预测）某些点匹配失败了，就尝试用第二种匹配（反向预测）替代

        peak_next=np.zeros((len(peak2,1),2))
        index=assignment_final!=0
        ind = np.arange(1, len(assignment_final) + 1)
        indx = ind[index]
        peak_next[indx, :] = peak_tmp[assignment_final[index], :]#构建新帧位置坐标：根据匹配结果，把当前峰值匹配到下一帧的点上

        vel_rel=peak_tmp[assignment_final[index], :] - peaks_before[index, :] #计算粒子在两帧之间的位移
        vel_comp=vol_before[index,:]
        vel_cor = np.sum(vel_rel * vel_comp, axis=1)
        index[indx(vel_cor<0)]=False #判断相对运动方向和原速度方向是否一致，如果不一致，认为是错误匹配，排除掉

        vel_rel=peak_tmp[assignment_final[index], :] - peaks_before[index, :]
        frameinfo[iter] = {
            'peak_val': peaks_before[index, :], #成功追踪的粒子的坐标
            'vel_est': np.vstack([vel_rel, vol_before[~index]]), #粒子的估算位移
            'volmap': vol_before[index,:], #成功追踪粒子的速度向量
            'peak_current':[peaks_before[index, :], peaks_before[~index, :]],#成功追踪的粒子和追踪失败的粒子的坐标
            'peak_next': peak_next[index, :], # 成功追踪的粒子在当前帧中的预测新位置  
            } #更新结构体

        peaks_predict=peak_tmp+vol_ini[peak_mark, :] #预测下一峰值
        peaks_before=peak_tmp
        vol_before=vol_ini[peak_mark, :]

print('\n')
 

