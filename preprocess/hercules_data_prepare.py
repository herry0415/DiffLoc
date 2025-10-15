"""Visualization code for point clouds and 3D bounding boxes with mayavi.

Modified by Charles R. Qi
Date: September 2017
"""

import argparse
import os
from tqdm import tqdm
import numpy as np
import torch

from torchsparse import SparseTensor
# from torchsparse.utils.quantize import sparse_quantize
from torchsparse.utils.quantize import sparse_quantize

from model_zoo import minkunet, spvcnn, spvnas_specialized

import open3d as o3d

cpu_num = 2 # cpu core
os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)


def process_point_cloud(input_point_cloud, input_labels=None, voxel_size=0.15): # 为了把原始点云转换为 稀疏张量
    input_point_cloud[:, 3] = input_point_cloud[:, 3] #！ 写成这种形式只是“自我拷贝”，没有改变数据。
    pc_ = np.round(input_point_cloud[:, :3] / voxel_size)
    pc_ -= pc_.min(0, keepdims=1) #! 将体素坐标平移，使最小值为 0，方便索引


    label_map = create_label_map()
    if input_labels is not None:
        labels_ = label_map[input_labels].astype(np.int64)  # semantic labels
    else:
        labels_ = np.zeros(pc_.shape[0], dtype=np.int64)

    feat_ = input_point_cloud 

    if input_labels is not None: #！ 将标签中最大的类别（通常是 “ignore” 类或无效点） 去掉。
        out_pc = input_point_cloud[labels_ != labels_.max(), :3] #！ 输出 out_pc 只包含有效点的 XYZ
        pc_ = pc_[labels_ != labels_.max()]
        feat_ = feat_[labels_ != labels_.max()]
        labels_ = labels_[labels_ != labels_.max()]
    else:
        out_pc = input_point_cloud #！  输出 out_pc 只包含有效点的 XYZ
        pc_ = pc_ 


    coords_, inds, inverse_map = sparse_quantize(pc_,
                                                 return_index=True,
                                                 return_inverse=True) 
    #!  coords_:  [M, 3] 每个体素的唯一坐标 
    #!  inds：[M,] 原始点云中被选中的索引. 用于从原始点云中选出去重后的点 一个体素里面选一个点类似下采样
    #!  nverse_map： [N,]原始点云到稀疏点云的映射，用于后续把预测结果映射回原始点

    pc = np.zeros((inds.shape[0], 4)) # 稀疏化后的点坐标
    pc[:, :3] = pc_[inds]  

    feat = feat_[inds] # 稀疏化后的特征（包括原始强度）  
    labels = labels_[inds] # 稀疏化后的语义标签
    lidar = SparseTensor(
        torch.from_numpy(feat).float(), #  [M, 4]
        torch.from_numpy(pc).int()) #  [M, 4] 第4纬是0
    return {
        'pc': out_pc, #！ [M, 4]  或者 [M, 3]（如果传入了label）是原始点云 过滤掉“ignore”类点（最大标签值） 后的 XYZ 坐标
        'lidar': lidar, #！ SparseTensor，网络输入
        'targets': labels, #！ [M, 1] 稀疏化后的标签
        'targets_mapped': labels_, #！ [M, 1] 体素里面有的点云标签
        'inverse_map': inverse_map  #！[N, 1] 原始点云 → 稀疏点云索引映射
    }


def create_label_map(num_classes=3):
    name_label_mapping = {
    # 不需要的类
    'unlabeled': 0,
    'outlier': 1,
    'other-ground': 49,
    'other-structure': 52,
    # ground
    'ground': 10,
    # plane
    'plane': 20,
    # other
    'other': 30
    }

    # for k in name_label_mapping:
    #     name_label_mapping[k] = name_label_mapping[k.replace('moving-', '')]
    train_label_name_mapping = {
        0: 'ground',
        1: 'plane',
        2: 'other',
    }

    label_map = np.zeros(260) + num_classes #! 所有都映射到3类
    for i in range(num_classes):
        cls_name = train_label_name_mapping[i]
        # print(cls_name)
        label_map[name_label_mapping[cls_name]] = min(num_classes, i) #! 需要的几类映射为012
    return label_map.astype(np.int64) #！    其他263种都是第3类


cmap = np.array([ #！ 生成颜色映射（colormap），用于给点云或语义类别上色
    [245, 150, 100, 255], 
    [245, 230, 100, 255],
    [150, 60, 30, 255],
    [180, 30, 80, 255],
    [255, 0, 0, 255],
    [30, 30, 255, 255],
    [200, 40, 255, 255],
    [90, 30, 150, 255],
    [255, 0, 255, 255],
    [255, 150, 255, 255],
    [75, 0, 75, 255],
    [75, 0, 175, 255],
    [0, 200, 255, 255],
    [50, 120, 255, 255],
    [0, 175, 0, 255],
    [0, 60, 135, 255],
    [80, 240, 150, 255],
    [150, 240, 255, 255],
    [0, 0, 255, 255],
])
cmap = cmap[:, [2, 1, 0, 3]]  # convert bgra to rgba 这里通过索引 [2, 1, 0, 3] 变换成 RGBA（常用在 Python、OpenGL、Matplotlib 等）


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 数据集位置
    parser.add_argument('--velodyne-dir', type=str, default='/home/data/ldq/HeRCULES/') #todo
    parser.add_argument('--model',
                        type=str,
                        default='SemanticKITTI_val_SPVCNN@119GMACs')
    args = parser.parse_args()
    output_dir = os.path.join(args.velodyne_dir, 'outputs')
    os.makedirs(output_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    if 'MinkUNet' in args.model:
        model = minkunet(args.model, pretrained=True)
    elif 'SPVCNN' in args.model:
        model = spvcnn(args.model, pretrained=True)
    elif 'SPVNAS' in args.model:
        print("use spvnas.....")
        model = spvnas_specialized(args.model, pretrained=True)
    else:
        raise NotImplementedError

    model = model.to(device)

    scene_dict = {
    'Library': (['Library_01_Day', 'Library_02_Night'], ['Library_03_Day']),
    # 将来可以加其他场景：
    # 'City': (['City_01_Day', 'City_02_Night'], ['City_03_Day']),
    }

    # 遍历场景和训练子目录
    for scene_name, (train_subdirs, test_subdirs) in scene_dict.items():
        for subdir in train_subdirs:   # 只计算训练集
            input_path = os.path.join(args.velodyne_dir, scene_name, subdir, 'LiDAR', 'np8Aeva')
            file_list = sorted(os.listdir(input_path))

            # 输出保存目录
            save_dir = os.path.join(args.velodyne_dir, scene_name, subdir, 'LiDAR', 'SPVNAS_np8Aeva_plane_segmented')
            os.makedirs(save_dir, exist_ok=True)

            for file_name in tqdm(file_list, desc="Processing Hercules bin files..."):
                if not file_name.endswith('.bin'):
                    continue

                # Hercules 每点是 [x, y, z, reflectivity, velocity, time-offset, line-index, intensity]
                pc_full = np.fromfile(os.path.join(input_path, file_name), dtype=np.float32).reshape(-1, 8)

                # 只取 x, y, z 和 intensity（第 8 维）
                pc = np.concatenate([pc_full[:, :3], pc_full[:, 7:8]], axis=1)  # shape: [N, 4]

                # （可选）Z 轴是否要翻转，取决于 Hercules 坐标系
                # pc[:, 2] = -pc[:, 2]  

                # Hercules 没有 label，这里直接设 None
                label = None

                # 点云送入预处理 & 模型
                feed_dict = process_point_cloud(pc, label)
                inputs = feed_dict['lidar'].to(device)
                outputs = model(inputs)
                predictions = outputs.argmax(1).cpu().numpy()

                # 还原到原始点云顺序
                predictions = predictions[feed_dict['inverse_map']]
                predictions = predictions.astype(np.int32)

                # 拼接预测结果（只保留 [x, y, z, reflectivity] + label）
                results = np.concatenate((feed_dict['pc'][:, :4], predictions.reshape(-1, 1)), axis=1)

                # 分类 plane / ground / other
                plane_list, ground_list, other_list = [], [], []
                for i in range(results.shape[0]):
                    if results[i, 4] in [12, 13]:  # plane
                        plane_list.append(results[i, :4])
                    elif results[i, 4] in [8, 9, 10, 11]:  # ground
                        ground_list.append(results[i, :4])
                    else:  # other
                        other_list.append(results[i, :4])

                # 拼接 & 标记 (plane=1, ground=1, other=0)
                plane_list  = np.concatenate((np.array(plane_list).reshape(-1, 4), np.ones((len(plane_list), 1))), axis=1)
                ground_list = np.concatenate((np.array(ground_list).reshape(-1, 4), np.ones((len(ground_list), 1))), axis=1)
                other_list  = np.concatenate((np.array(other_list).reshape(-1, 4), np.zeros((len(other_list), 1))), axis=1)

                results = np.concatenate((ground_list, plane_list, other_list), axis=0).astype(np.float32)

                # 保存为 bin 文件
                results.tofile(os.path.join(save_dir, file_name))
