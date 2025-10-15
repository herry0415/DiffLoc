"""
@author: Wen Li
@file: oxford.py
@time: 2023/9/18 19:27
"""
import os
import h5py
import torch
import numpy as np
import os.path as osp
# import torchvision.transforms as T
from copy import deepcopy
from torch.utils import data
from datasets.projection import RangeProjection
from datasets.augmentor import Augmentor, AugmentParams
from utils.pose_util import process_poses, filter_overflow_ts, poses_foraugmentaion #！ 位姿处理工具
from datasets.robotcar_sdk.python.interpolate_poses import interpolate_ins_poses #！ 对不连续的 INS 位姿数据进行插值，生成连续帧的平滑位姿
from datasets.robotcar_sdk.python.transform import build_se3_transform, euler_to_so3  #！ 构建 SE(3) 变换矩阵 [4x4]，将平移和旋转组合成齐次矩阵
# 转换为 SO(3) 旋转矩阵 [3x3]

BASE_DIR = osp.dirname(osp.abspath(__file__)) #！ 获取 当前 Python 文件所在的目录路径

class Oxford(data.Dataset):
    def __init__(self, config, split='train'):
        # directories
        if split == 'train':
            self.is_train = True
        else:
            self.is_train = False

        lidar = 'velodyne_left'
        data_path = config.train.dataroot

        data_dir = osp.join(data_path, '')
        extrinsics_dir = osp.join(BASE_DIR, 'robotcar_sdk', 'extrinsics')

        # decide which sequences to use
        if split == 'train':
            split_filename = osp.join(data_dir, 'train_split.txt')
        else:
            split_filename = osp.join(data_dir, 'valid_split.txt')
        with open(split_filename, 'r') as f:
            seqs = [l.rstrip() for l in f if not l.startswith('#')]

        ps = {}
        ts = {}
        vo_stats = {}
        self.pcs = []

        # extrinsic reading 外参读取与位姿变换  
        with open(os.path.join(extrinsics_dir, lidar + '.txt')) as extrinsics_file:
            extrinsics = next(extrinsics_file) #！ 就像一个指针，每调用一次就向前移动一步
        G_posesource_laser = build_se3_transform([float(x) for x in extrinsics.split(' ')])  
        #! lidar2ins. 代表LiDAR传感器相对于INS传感器（或INS所在的坐标系）的位姿，即LiDAR的外参。

        with open(os.path.join(extrinsics_dir, 'ins.txt')) as extrinsics_file: #! INS 坐标系相对于世界/车辆系的位姿。
            extrinsics = next(extrinsics_file)
            #! extrinsics. ins2world  INS 坐标系 转换到 世界/车辆坐标系。
            G_posesource_laser = np.linalg.solve(build_se3_transform([float(x) for x in extrinsics.split(' ')]),
                                                 G_posesource_laser) 
            #! 功能： 把 LiDAR 位姿从 INS 坐标系对齐到 全局坐标系
            
        for seq in seqs:
            seq_dir = osp.join(data_dir, seq + '-radar-oxford-10k')
            # read the image timestamps
            h5_path = osp.join(seq_dir, lidar + '_' + 'False.h5')

            if not os.path.isfile(h5_path):
                print('interpolate ' + seq)
                ts_filename = osp.join(seq_dir, lidar + '.timestamps') #todo  时间戳列表也要改
                with open(ts_filename, 'r') as f:
                    ts_raw = [int(l.rstrip().split(' ')[0]) for l in f]
                # GT poses
                ins_filename = osp.join(seq_dir, 'gps', 'ins.csv') #todo 修改真值路径
                ts[seq] = filter_overflow_ts(ins_filename, ts_raw) #! 过滤掉ins时间戳min-max以外的lidar
                p = np.asarray(interpolate_ins_poses(ins_filename, deepcopy(ts[seq]), ts[seq][0]))  #todo 里面的差值函数需要更改。# (n, 4, 4) #！根据过滤后的 LiDAR 时间戳 ts[seq]，从 ins.csv 文件中插值出对应时刻的INS位姿。
                #! ts[seq][0] 是orignal第一个位姿 以这一个为起点
                p = np.asarray([np.dot(pose, G_posesource_laser) for pose in p])  # (n, 4, 4) #! 这里 点是行向量，矩阵右乘，等价于数学上列向量左乘矩阵
                ps[seq] = np.reshape(p[:, :3, :], (len(p), -1))  # (n, 12) #! 为了节省存储空间和简化数据，通常会丢弃它。这个操作后的形状变为 (n, 3, 4)。
                # write to h5 file
                print('write interpolate pose to ' + h5_path)
                h5_file = h5py.File(h5_path, 'w')
                h5_file.create_dataset('valid_timestamps', data=np.asarray(ts[seq], dtype=np.int64))
                h5_file.create_dataset('poses', data=ps[seq])
            else:
                # load h5 file, save pose interpolating time
                print("load " + seq + ' pose from ' + h5_path)
                h5_file = h5py.File(h5_path, 'r')
                ts[seq] = h5_file['valid_timestamps'][...]
                ps[seq] = h5_file['poses'][...]

            vo_stats[seq] = {'R': np.eye(3), 't': np.zeros(3), 's': 1}

            if self.is_train:
                self.pcs.extend(
                [osp.join(seq_dir, 'SPVNAS_velodyne_left_plane_segmented', '{:d}.bin'.format(t)) for t in ts[seq]])
            else:
                self.pcs.extend(
                    [osp.join(seq_dir, 'velodyne_left', '{:d}.bin'.format(t)) for t in ts[seq]]) #! 原始的或未分割处理的点云文件。

        # read / save pose normalization information
        poses = np.empty((0, 12)) #！是一个零行、十二列的二维数组
        for p in ps.values():
            poses = np.vstack((poses, p)) 
        pose_stats_filename = osp.join(data_dir, 'Oxford_pose_stats.txt')

        if split == 'train':
            mean_t = np.mean(poses[:, [3, 7, 11]], axis=0)  # (3,)
            std_t = np.std(poses[:, [3, 7, 11]], axis=0)  # (3,)
            np.savetxt(pose_stats_filename, np.vstack((mean_t, std_t)), fmt='%8.7f') #! 这确保了在验证或测试时，数据使用与训练时相同的归一化参数
        else:
            mean_t, std_t = np.loadtxt(pose_stats_filename)

        self.poses = np.empty((0, 6))
        self.rots = np.empty((0, 3, 3))
        for seq in seqs:
            pss, rotation, pss_max, pss_min = process_poses(poses_in=ps[seq], mean_t=mean_t, std_t=std_t,
                                                            align_R=vo_stats[seq]['R'], align_t=vo_stats[seq]['t'],
                                                            align_s=vo_stats[seq]['s'])
            self.poses = np.vstack((self.poses, pss)) # (37666, 6) 前三列: 归一化和对齐后的平移向量 (x, y, z)。 后三列: 对齐后的旋转分量（四元数的虚部），以三维向量形式表示
            self.rots = np.vstack((self.rots, rotation)) #  每一帧的 3x3 旋转矩阵

        self.proj_img_mean = torch.tensor(config.sensors.image_mean, dtype=torch.float) #todo  sensors.image_mean  sensors.image_stds
        self.proj_img_stds = torch.tensor(config.sensors.image_stds, dtype=torch.float)  

        if split == 'train':
            print("train data num:" + str(len(self.poses)))
        else:
            print("valid data num:" + str(len(self.poses)))

        self.projection = RangeProjection(#todo 修改视场角
            fov_up=config.sensors.fov_up, fov_down=config.sensors.fov_down,
            fov_left=config.sensors.fov_left, fov_right=config.sensors.fov_right,
            proj_h=config.sensors.proj_h, proj_w=config.sensors.proj_w,
        )

        augment_params = AugmentParams()
        augment_config = config.augmentation

        # Point cloud augmentations
        if self.is_train:
            augment_params.setTranslationParams(
                p_transx=augment_config['p_transx'], trans_xmin=augment_config[
                    'trans_xmin'], trans_xmax=augment_config['trans_xmax'],
                p_transy=augment_config['p_transy'], trans_ymin=augment_config[
                    'trans_ymin'], trans_ymax=augment_config['trans_ymax'],
                p_transz=augment_config['p_transz'], trans_zmin=augment_config[
                    'trans_zmin'], trans_zmax=augment_config['trans_zmax'])
            augment_params.setRotationParams(
                p_rot_roll=augment_config['p_rot_roll'], rot_rollmin=augment_config[
                    'rot_rollmin'], rot_rollmax=augment_config['rot_rollmax'],
                p_rot_pitch=augment_config['p_rot_pitch'], rot_pitchmin=augment_config[
                    'rot_pitchmin'], rot_pitchmax=augment_config['rot_pitchmax'],
                p_rot_yaw=augment_config['p_rot_yaw'], rot_yawmin=augment_config[
                    'rot_yawmin'], rot_yawmax=augment_config['rot_yawmax'])
            if 'p_scale' in augment_config:
                augment_params.sefScaleParams(
                    p_scale=augment_config['p_scale'],
                    scale_min=augment_config['scale_min'],
                    scale_max=augment_config['scale_max'])
                print(
                    f'Adding scaling augmentation with range [{augment_params.scale_min}, {augment_params.scale_max}] and probability {augment_params.p_scale}')
            self.augmentor = Augmentor(augment_params)
        else:
            self.augmentor = None

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, idx_N):
        scan_path = self.pcs[idx_N]
        if self.is_train:
            # generate by SPVCNN, (x, y, z, intensity, static objects mask)
            pointcloud = np.fromfile(scan_path, dtype=np.float32).reshape(-1, 5)
        else:
            #fill with zeros
            pointcloud = np.fromfile(scan_path, dtype=np.float32).reshape(4, -1).transpose()
            pointcloud = np.concatenate((pointcloud, np.zeros(len(pointcloud), 1)), axis=1)

        # flip z
        T = euler_to_so3([np.pi, 0, np.pi / 2]) #! 是统一或校正点云数据的坐标系 绕x轴旋转180 y轴不旋转 绕z轴旋转90度
        pointcloud[:, :3] = (T[:3, :3] @ pointcloud[:, :3].transpose()).transpose()

        if self.is_train:
            pointcloud, rotation = self.augmentor.doAugmentation(pointcloud)  # n, 5
            original_rots = self.rots[idx_N]  # [3, 3]
            present_rots = rotation @ original_rots
            poses = poses_foraugmentaion(present_rots, self.poses[idx_N])
        else:
            poses = self.poses[idx_N]
            
        # Generate RangeImage
        proj_pointcloud, proj_range, proj_idx, proj_mask = self.projection.doProjection(pointcloud) # proj_pointcloud [32,512,5] (x, y, z, intensity, static mask)
        proj_mask_tensor = torch.from_numpy(proj_mask) #! mask，标记有效点
        proj_range_tensor = torch.from_numpy(proj_range)  # [32, 512]
        proj_xyz_tensor = torch.from_numpy(proj_pointcloud[..., :3])  # [32, 512, 3]
        proj_intensity_tensor = torch.from_numpy(proj_pointcloud[..., 3]) # [32, 512]
        proj_label_tensor = torch.from_numpy(proj_pointcloud[..., 4]) #! 标记类别
        proj_feature_tensor = torch.cat(
            [proj_range_tensor.unsqueeze(0), proj_xyz_tensor.permute(2, 0, 1), proj_intensity_tensor.unsqueeze(0),
             proj_label_tensor.unsqueeze(0)], 0) #!  # [32, 512, 6]
        pose_tensor = torch.from_numpy(poses.astype(np.float32))
        # normalization
        proj_feature_tensor = (proj_feature_tensor - self.proj_img_mean[:, None, None]) / self.proj_img_stds[:, None
        , None] #! 归一化
        proj_feature_tensor = proj_feature_tensor * proj_mask_tensor.unsqueeze(0).float() #! 
        proj_tensor = torch.cat(
            (proj_feature_tensor,
             proj_mask_tensor.unsqueeze(0)), dim=0)
 
        return proj_tensor[:5], pose_tensor, proj_tensor[5] #! proj_tensor 这通常是深度、x, y, z 坐标和强度，  pose_tensor 过处理的位姿张量，作为模型的另一个输入或标签 
    # proj_tensor 投影掩码，通常用于在计算损失时过滤掉无效的像素点

