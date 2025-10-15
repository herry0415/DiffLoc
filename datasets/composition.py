import torch
import numpy as np
import sys
sys.path.insert(0, '../')

from .oxford import Oxford
from .nclt import NCLT
from .hercules_lidar import Hercules
from torch.utils import data
from utils.pose_util import calc_vos_safe_fc


class MF(data.Dataset):
    def __init__(self, dataset, config, split='train', include_vos=False):

        self.steps = config.train.steps # 每个训练样本包含的帧数。
        self.skip = config.train.skip # 帧间隔步长，
        self.train = split

        if dataset == 'Oxford':
            self.dset = Oxford(config, split)
        elif dataset == 'NCLT':
            self.dset = NCLT(config, split)
        elif dataset == 'Hercules':
            self.dset = Hercules(config, split)
        else:
            raise NotImplementedError('{:s} dataset is not implemented!')

        self.L = self.steps * self.skip #! 多帧窗口长度
        # GCS
        self.include_vos = include_vos  # 决定是否使用 VOS 信息
        self.vo_func = calc_vos_safe_fc  # 指定计算 VOS 的函数


    def get_indices(self, index): # 根据中心帧 index 生成多帧窗口的索引。
        skips = self.skip * np.ones(self.steps-1)
        offsets = np.insert(skips, 0, 0).cumsum()  # (self.steps,)
        offsets -= offsets[len(offsets) // 2]
        offsets = offsets.astype(np.int_)
        idx = index + offsets
        idx = np.minimum(np.maximum(idx, 0), len(self.dset)-1)
        assert np.all(idx >= 0), '{:d}'.format(index)
        assert np.all(idx < len(self.dset))
        return idx

    def __getitem__(self, index):  #！ 根据给定索引 index 获取多帧数据
        idx   = self.get_indices(index)
        clip  = [self.dset[i] for i in idx] #! 列表，存储每一帧的数据 (point_cloud, pose, mask)
        pcs   = torch.stack([c[0] for c in clip], dim=0)  # (self.steps, N, 32, 512, 5]) 
        poses = torch.stack([c[1] for c in clip], dim=0)  # (self.steps, N, 32, 512, 6)
        mask = torch.stack([c[2] for c in clip], dim=0)   # (self.steps, N, 32, 512)

        if self.include_vos:
            vos = self.vo_func(poses.unsqueeze(0))[0] #! 增加 batch 维度 [1, steps, 6]
            # 前3维是真值，后3维是相对位姿
            poses = torch.cat((poses, vos), dim=0)

        batch = {
            "image": pcs, # 5纬 特征 range（depth） + x,y,z + intensity
            "pose": poses,  # shape [6] xyzrpy
            "mask": mask # 标签label 分类标签
        }
        return batch

    def __len__(self):
        L = len(self.dset)
        return L