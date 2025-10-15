import os
import h5py
import torch
import numpy as np
import os.path as osp
from torch.utils import data
import struct, sys
import open3d as o3d
import MinkowskiEngine as ME
from utils.pose_util import process_poses, poses_to_matrices, poses_foraugmentaion
from datasets.projection import RangeProjection
from datasets.augmentor import Augmentor, AugmentParams
from datasets.robotcar_sdk.python.interpolate_poses import interpolate_ins_poses #！ 对不连续的 INS 位姿数据进行插值，生成连续帧的平滑位姿
from datasets.robotcar_sdk.python.transform import build_se3_transform, euler_to_so3  #！ 构建 SE(3) 变换矩阵 [4x4]，将平移和旋转组合成齐次矩阵

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class Hercules(data.Dataset):
    def __init__(self, config, split='train'):

        if split == 'train':
            self.is_train = True
        else:
            self.is_train = False

        lidar = 'hercules_lidar' #todo 

        data_path =  config.train.dataroot 

        self.sequence_name = config.train.sequence #['Library', 'Mountain', 'Sports']

        self.data_dir = os.path.join(data_path, self.sequence_name)
        
         # 根据 sequence_name 和 train/val 设置序列
        seqs = self._get_sequences(self.sequence_name, self.is_train)
    
        ps = {}
        ts = {}
        vo_stats = {}
        self.pcs = []
        for seq in seqs:
            seq_dir = os.path.join(self.data_dir, seq)

            # h5_path = os.path.join(seq_dir, 'lidar_poses.h5')
         
            # if not os.path.isfile(h5_path):
            print('interpolate ' + seq)
            pose_file_path = os.path.join(seq_dir, 'PR_GT/Aeva_gt.txt')
            ts_raw = np.loadtxt(pose_file_path, dtype=np.int64, usecols=0) # float读取数字丢精度
            ts[seq] = ts_raw
            
            pose_file = np.loadtxt(pose_file_path) #保证pose值不变
            p = poses_to_matrices(pose_file) # (n,4,4) #毫米波雷达坐标系
            ps[seq] = np.reshape(p[:, :3, :], (len(p), -1))     #  (n, 12)
            
            # write to h5 file
            # print('write interpolate pose to ' + h5_path)
            # h5_file = h5py.File(h5_path, 'w') 
            # h5_file.create_dataset('valid_timestamps', data=np.asarray(ts[seq], dtype=np.int64))
            # h5_file.create_dataset('poses', data=ps[seq])
            
            # else:
            #     print("load " + seq + ' pose from ' + h5_path)
            #     h5_file = h5py.File(h5_path, 'r')
            #     ts[seq] = h5_file['valid_timestamps'][...]
            #     ps[seq] = h5_file['poses'][...]
            #     print(f'pose len {len(ts[seq])}')
            if self.is_train:
                self.pcs.extend([osp.join(seq_dir, 'LiDAR/SPVNAS_np8Aeva_plane_segmented', '{:d}.bin'.format(t)) for t in ts[seq]])
                # self.pcs.extend(os.path.join(seq_dir, 'Radar/multi_frame_w7', str(t)+'_multi_w7' + '.bin') for t in ts[seq])
                vo_stats[seq] = {'R': np.eye(3), 't': np.zeros(3), 's': 1}
            else:
                self.pcs.extend(os.path.join(seq_dir, 'LiDAR/np8Aeva', str(t) + '.bin') for t in ts[seq])
                vo_stats[seq] = {'R': np.eye(3), 't': np.zeros(3), 's': 1}

        # read / save pose normalization information
        poses = np.empty((0, 12))
        for p in ps.values():
            poses = np.vstack((poses, p))
        
        pose_stats_filename = os.path.join(self.data_dir, self.sequence_name + '_lidar'+'_pose_stats.txt')
        # pose_max_min_filename = os.path.join(self.data_dir, self.sequence_name + '_lidar'+ '_pose_max_min.txt')
        
        if self.is_train:
            self.mean_t = np.mean(poses[:, [3, 7, 11]], axis=0)  # (3,)
            self.std_t = np.std(poses[:, [3, 7, 11]], axis=0)  # (3,)
            np.savetxt(pose_stats_filename, np.vstack((self.mean_t, self.std_t)), fmt='%8.7f')
            print(f'saving pose stats in {pose_stats_filename}')
        else:
            self.mean_t, self.std_t = np.loadtxt(pose_stats_filename)
        
         # convert the pose to translation + log quaternion, align, normalize
         
        self.poses = np.empty((0, 6))
        self.rots = np.empty((0, 3, 3))
        # self.poses_max  =  np.empty((0, 2))
        # self.poses_min  =  np.empty((0, 2))
        for seq in seqs:
            #! 更改
            pss, rotation, pss_max, pss_min = process_poses(poses_in=ps[seq], mean_t=self.mean_t, std_t=self.std_t,
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

            # if self.is_train:
            #     self.poses_max = np.max(self.poses_max, axis=0)  # (2,)
            #     self.poses_min = np.min(self.poses_min, axis=0)  # (2,)
            #     np.savetxt(pose_max_min_filename, np.vstack((self.poses_max, self.poses_min)), fmt='%8.7f')
            # else:
            #     self.poses_max, self.poses_min = np.loadtxt(pose_max_min_filename)
     
        # self.voxel_size = voxel_size
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
            
    def _get_sequences(self, sequence_name, train):
        mapping = {
            'Library': (['Library_01_Day','Library_02_Night'], ['Library_03_Day']),
            'Mountain': (['Mountain_01_Day','Mountain_02_Night'], ['Mountain_03_Day']),
            'Sports': (['Complex_01_Day','Complex_03_Day'], ['Complex_02_Night'])
        }
        return mapping[sequence_name][0] if train else mapping[sequence_name][1]
    
    def __getitem__(self, idx_N):
        scan_path = self.pcs[idx_N]
        if self.is_train:
            # generate by SPVCNN, (x, y, z, intensity, static objects mask)
            pointcloud = np.fromfile(scan_path, dtype=np.float32).reshape(-1, 5)
        else:
            #fill with zeros
            pointcloud = np.fromfile(scan_path, dtype=np.float32).reshape(-1, 8)   
            pointcloud = np.concatenate((pointcloud[:, :3], pointcloud[:, 7:8]), axis=1) # [N, 4]
            pointcloud = np.concatenate((pointcloud, np.zeros((len(pointcloud), 1))), axis=1) # [N, 5]

            # pointcloud = np.fromfile(scan_path, dtype=np.float32).reshape(-1, 8)
            # pointcloud = np.concatenate((pointcloud, np.zeros((len(pointcloud), 1), dtype=np.float32)), axis=1)
        # flip z
        # T = euler_to_so3([np.pi, 0, np.pi / 2]) #! 是统一或校正点云数据的坐标系。
        # pointcloud[:, :3] = (T[:3, :3] @ pointcloud[:, :3].transpose()).transpose()
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
        # proj_feature_tensor = torch.cat(
        #     [proj_range_tensor.unsqueeze(0), proj_xyz_tensor.permute(2, 0, 1), proj_intensity_tensor.unsqueeze(0),
        #      proj_label_tensor.unsqueeze(0)], 0) #!  # [32, 512, 6]

        proj_feature_tensor = torch.cat(
            [proj_range_tensor.unsqueeze(0), proj_xyz_tensor.permute(2, 0, 1),
             proj_label_tensor.unsqueeze(0)], 0) #!  # [32, 512, 6]
        pose_tensor = torch.from_numpy(poses.astype(np.float32))
        # normalization
        proj_feature_tensor = (proj_feature_tensor - self.proj_img_mean[:, None, None]) / self.proj_img_stds[:, None
        , None] #! 归一化
        proj_feature_tensor = proj_feature_tensor * proj_mask_tensor.unsqueeze(0).float() #! 
        proj_tensor = torch.cat(
            (proj_feature_tensor,
             proj_mask_tensor.unsqueeze(0)), dim=0)
 
        return proj_tensor[:4], pose_tensor, proj_tensor[4] #! proj_tensor 这通常是深度、x, y, z 坐标和强度，  pose_tensor 过处理的位姿张量，作为模型的另一个输入或标签 
    # proj_tensor 投影掩码，通常用于在计算损失时过滤掉无效的像素点

