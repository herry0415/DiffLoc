import argparse
import os
from tqdm import tqdm
import numpy as np
import torch
from torchsparse import SparseTensor
from torchsparse.utils.quantize import sparse_quantize
from model_zoo import minkunet, spvcnn, spvnas_specialized
import random

cpu_num = 2
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)


def process_point_cloud(input_point_cloud, input_labels=None, voxel_size=0.15):
    input_point_cloud[:, 3] = input_point_cloud[:, 3]
    pc_ = np.round(input_point_cloud[:, :3] / voxel_size)
    pc_ -= pc_.min(0, keepdims=1)

    label_map = create_label_map()
    if input_labels is not None:
        labels_ = label_map[input_labels].astype(np.int64)
    else:
        labels_ = np.zeros(pc_.shape[0], dtype=np.int64)

    feat_ = input_point_cloud

    if input_labels is not None:
        mask = labels_ != labels_.max()
        out_pc = input_point_cloud[mask, :3]
        pc_ = pc_[mask]
        feat_ = feat_[mask]
        labels_ = labels_[mask]
    else:
        out_pc = input_point_cloud

    coords_, inds, inverse_map = sparse_quantize(pc_, return_index=True, return_inverse=True)

    pc = np.zeros((inds.shape[0], 4))
    pc[:, :3] = pc_[inds]

    feat = feat_[inds]
    labels = labels_[inds]

    lidar = SparseTensor(
        torch.from_numpy(feat).float(),
        torch.from_numpy(pc).int()
    )

    return {
        'pc': out_pc,
        'lidar': lidar,
        'targets': labels,
        'targets_mapped': labels_,
        'inverse_map': inverse_map
    }


def create_label_map(num_classes=3):
    name_label_mapping = {
        'unlabeled': 0, 'outlier': 1, 'other-ground': 49, 'other-structure': 52,
        'ground': 10, 'plane': 20, 'other': 30
    }
    train_label_name_mapping = {0: 'ground', 1: 'plane', 2: 'other'}

    label_map = np.zeros(260) + num_classes
    for i in range(num_classes):
        cls_name = train_label_name_mapping[i]
        label_map[name_label_mapping[cls_name]] = min(num_classes, i)
    return label_map.astype(np.int64)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--velodyne-dir', type=str, default='/home/data/ldq/Oxford Radar Robotcar/')
    parser.add_argument('--model', type=str, default='SemanticKITTI_val_SPVCNN@119GMACs')
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    if 'MinkUNet' in args.model:
        model = minkunet(args.model, pretrained=True)
    elif 'SPVCNN' in args.model:
        model = spvcnn(args.model, pretrained=True)
    elif 'SPVNAS' in args.model:
        model = spvnas_specialized(args.model, pretrained=True)
    else:
        raise NotImplementedError
    model = model.to(device)

    # Oxford 文件夹列表
    file_list = ['2019-01-11-14-02-26-radar-oxford-10k']  # 示例，可修改为实际文件夹

    for file in file_list:
        input_path = os.path.join(args.velodyne_dir, file, 'velodyne_left')
        save_dir = './oxford_segment_txt'
        os.makedirs(save_dir, exist_ok=True)

        point_cloud_files = sorted([f for f in os.listdir(input_path) if f.endswith('.bin')])
        point_cloud_files = random.sample(point_cloud_files, min(3, len(point_cloud_files)))  # 随机选择3帧

        for pc_name in tqdm(point_cloud_files, desc='Processing bin files'):
            pc = np.fromfile(os.path.join(input_path, pc_name), dtype=np.float32).reshape(4, -1).transpose()
            pc[:, 2] = -1 * pc[:, 2]

            feed_dict = process_point_cloud(pc, input_labels=None)
            inputs = feed_dict['lidar'].to(device)
            outputs = model(inputs)
            predictions = outputs.argmax(1).cpu().numpy()
            predictions = predictions[feed_dict['inverse_map']].astype(np.int32)

            results = np.concatenate((feed_dict['pc'][:, :4], predictions.reshape(-1, 1)), axis=1)
            results[:, 2] = -1 * results[:, 2]

            # 分类
            plane_list, ground_list, other_list = [], [], []
            for i in range(results.shape[0]):
                if results[i, 4] in [12, 13]:
                    plane_list.append(results[i, :4])
                elif results[i, 4] in [8, 9, 10, 11]:
                    ground_list.append(results[i, :4])
                else:
                    other_list.append(results[i, :4])

            # 拼接 & 标记
            plane_list = np.concatenate((np.array(plane_list).reshape(-1, 4), np.ones((len(plane_list), 1))), axis=1)
            ground_list = np.concatenate((np.array(ground_list).reshape(-1, 4), np.ones((len(ground_list), 1))), axis=1)
            other_list = np.concatenate((np.array(other_list).reshape(-1, 4), np.zeros((len(other_list), 1))), axis=1)
            results_combined = np.concatenate((ground_list, plane_list, other_list), axis=0).astype(np.float32)

            # 保存 bin
            # results_combined.tofile(os.path.join(save_dir, pc_name))

            # 保存 txt
            timestamp = os.path.splitext(pc_name)[0]
            subfolder = os.path.join(save_dir, timestamp)
            os.makedirs(subfolder, exist_ok=True)
            np.savetxt(os.path.join(subfolder, f"{timestamp}_plane_list.txt"), plane_list, fmt='%.6f')
            np.savetxt(os.path.join(subfolder, f"{timestamp}_ground_list.txt"), ground_list, fmt='%.6f')
            np.savetxt(os.path.join(subfolder, f"{timestamp}_other_list.txt"), other_list, fmt='%.6f')
