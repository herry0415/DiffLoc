# -*- coding: utf-8 -*
import os
import os.path as osp

import numpy as np

from utils import *
import open3d as o3d
import scipy

file_list = os.listdir('/ava16t/lw/Data/XMU/XAC/2023120610/velodyne_left')
vo_filename = '/ava16t/lw/Data/XMU/XAC/2023120610/poses.txt'

# time = [i for i in range (5000, 5200)]

ts_raw = []
for file in file_list:
    ts_raw.append(int(file[:-4]))
ts_raw.sort()

# ts = filter_overflow_ts(vo_filename, ts_raw)
# p = np.asarray(interpolate_ins_poses(vo_filename, deepcopy(ts), ts[0]))
ts = filter_overflow_ts(vo_filename, ts_raw)
ground_truth = np.loadtxt(vo_filename)
interp = scipy.interpolate.interp1d(ground_truth[:, 0], ground_truth[:, 1:], kind='nearest', axis=0)
pose_gt = interp(ts)
p = so3_to_euler(pose_gt)

with open('/ava16t/lw/Data/XMU/XAC/ouster64ToGPS.txt') as extrinsics_file:
    extrinsics = next(extrinsics_file)
    G_posesource_laser = build_se3_transform([float(x) for x in extrinsics.split(' ')])

p = np.asarray([np.dot(pose, G_posesource_laser) for pose in p])  # (n, 4, 4)

def calculate_distance(point1, point2):
    point1 = point1.squeeze()
    point2 = point2.squeeze()
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def filter_points(coords, distance_threshold=5.0):
    result = [0]  # 先添加第一个点

    for i in range(1, len(coords)):
        distance = calculate_distance(coords[result[-1], :2, 3:], coords[i, :2, 3:])
        # print(distance)
        if distance >= distance_threshold:
            result.append(i)

    return result

def ground_seg(pcd):
    distance_threshold = 0.3 # 内点到平面模型的最大距离
    ransac_n = 3             # 用于拟合平面的采样点数
    num_iterations = 1000    # 最大迭代次数

    # 返回模型系数plane_model和内点索引inliers，并赋值
    plane_model, inliers = pcd.segment_plane(distance_threshold, ransac_n, num_iterations)

    # 平面外点点云
    outlier_cloud = pcd.select_by_index(inliers, invert=True)

    return outlier_cloud


time = filter_points(p, distance_threshold=1.0)
source = o3d.geometry.PointCloud()

for i in range(len(time) - 1):
    num = 0
    print("*"*20)
    print(i)
    for j in range(time[i], time[i+1]):
        print(j)
        if j == time[i]:
            ttt = p[int((time[i+1] + time[i])/2), :3, 3:]
            rrr = p[int((time[i+1] + time[i])/2), :3, :3]
        points = np.fromfile(osp.join('/ava16t/lw/Data/XMU/XAC/2023120610/velodyne_left', str(ts[j]) + '.bin'), dtype=np.float32).reshape(-1, 4)[:, :3]
        T = np.eye(4) # 对角矩阵，4*4
        T[:3, :3] = rrr.T @ p[j, :3, :3]
        T[:3, 3:] = p[j, :3, 3:] - ttt
        # global_points = (p[j, :3, :3] @ points.transpose()).transpose() + p[j, :3, 3:].squeeze()
        global_points = (T[:3, :3] @ points.transpose()).transpose() + T[:3, 3:].squeeze()
        # source.points = o3d.utility.Vector3dVector(global_points)
        # source = ground_seg(source)
        # global_points = np.array(source.points)
        timemat = j * np.ones((len(global_points), 1))
        global_points = np.concatenate((global_points, timemat), axis=1)

        if num==0:
            all_points = global_points
        else:
            all_points = np.concatenate((all_points, global_points), axis=0)
        num +=1

    # source = o3d.geometry.PointCloud()
    # source.points=o3d.utility.Vector3dVector(all_points)
    # source = ground_seg(source)
    # downpcd = source.voxel_down_sample(voxel_size=0.3)
    map = np.asarray(all_points).astype(np.float32)
    map.tofile('/ava16t/lw/Data/XMU/XAC/2023120610/velodyne_right/' + str(time[i]) + '_' + str(time[i+1]) + '.bin')
    # np.savetxt('/ava16t/lw/Data/XMU/XAC/2023120610/velodyne_right/' + str(time[i]) + '_' + str(time[i+1]) + '.txt', map, fmt='%.4f')
    print("saved")

