# -*- coding: utf-8 -*
import os
from utils import *

# 点云文件夹
file_list = os.listdir('pointclouds')
# GPS位姿文件
gt_filename = 'pose.txt'

# 读入Ouster和GPS之间的坐标变换
with open('ouster64ToGPS.txt') as extrinsics_file:
    extrinsics = next(extrinsics_file)
    G_posesource_laser = build_se3_transform([float(x) for x in extrinsics.split(' ')])

# time = [i for i in range (5000, 5200)]

ts_raw = []
for file in file_list:
    ts_raw.append(int((float(file[:-4]) + 0.) * 1e6))
ts_raw.sort()

ts = filter_overflow_xmu(gt_filename, ts_raw)
p = interpolate_pose_xmu(gt_filename, ts)
p = so3_to_euler(p)
# 生成转化到Ouster坐标系插值后的位姿
p = np.asarray([np.dot(pose, G_posesource_laser) for pose in p])  # (n, 4, 4)
ps = np.reshape(p[:, :3, :], (len(p), -1))  # (n, 12)

# 将旋转转化为四元数，用于计算误差
poses = process_poses(ps)

# 计算误差
# 假设预测的位姿是 pred_pose: [N, 7], 四元数表示形式
# error_t = np.asarray([val_translation(p, q) for p, q in zip(pred_pose[..., :3], poses[..., :3])])
# error_q = np.asarray([val_rotation(p, q) for p, q in zip(pred_pose[, 3:], poses[..., 3:])])

