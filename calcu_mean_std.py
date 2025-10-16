import os
import numpy as np
from tqdm import tqdm

root_path = '/home/drj/data/HeRCULES/'
output_dir = '/home/drj/project/DiffLoc/stats'

# 定义场景字典
scene_dict = {
    'Library': (['Library_01_Day', 'Library_02_Night'], ['Library_03_Day']),
    # 'Mountain': (['Mountain_01_Day','Mountain_02_Night'], ['Mountain_03_Day']),
    # 'Sports': (['Complex_01_Day','Complex_02_Night'], ['Complex_03_Day'])
}

# 计算函数：通用
def compute_stats(data_files, sensor_type):
    """计算给定文件列表的整体均值与方差"""
    all_mean_xyz, all_std_xyz = [], []
    all_mean_depth, all_std_depth = [], []
    all_mean_intensity, all_std_intensity = [], []

    for file_name in tqdm(data_files, desc=f"Calculating {sensor_type} stats..."):
        data = np.fromfile(file_name, dtype=np.float32)

        # 根据传感器类型 reshape & 选择强度列
        if sensor_type == "LiDAR":
            data = data.reshape(-1, 8)
            xyz = data[:, :3]
            intensity = data[:, 7]  # LiDAR: 第7列 (索引7)
        elif sensor_type == "Radar":
            data = data.reshape(-1, 8)
            xyz = data[:, :3]
            intensity = data[:, 5]  # Radar: 第6列 (索引5)
        else:
            raise ValueError(f"Unknown sensor type: {sensor_type}")

        # 计算单帧统计
        all_mean_xyz.append(np.mean(xyz, axis=0))
        all_std_xyz.append(np.std(xyz, axis=0))

        depth = np.linalg.norm(xyz, 2, axis=1)
        all_mean_depth.append(np.mean(depth))
        all_std_depth.append(np.std(depth))

        all_mean_intensity.append(np.mean(intensity))
        all_std_intensity.append(np.std(intensity))

    # 汇总为整体场景的平均值
    stats = {
        "mean_xyz": np.mean(all_mean_xyz, axis=0),
        "std_xyz": np.mean(all_std_xyz, axis=0),
        "mean_depth": np.mean(all_mean_depth),
        "std_depth": np.mean(all_std_depth),
        "mean_intensity": np.mean(all_mean_intensity),
        "std_intensity": np.mean(all_std_intensity),
    }
    return stats


# ===== 主流程 =====
for scene_name, (train_subdirs, test_subdirs) in scene_dict.items():
    lidar_files, radar_files = [], []

    for subdir in train_subdirs:
        lidar_path = os.path.join(root_path, scene_name, subdir, 'LiDAR', 'np8Aeva')
        radar_path = os.path.join(root_path, scene_name, subdir, 'Radar', 'multi_frame_w7')

        if os.path.exists(lidar_path):
            lidar_files += [os.path.join(lidar_path, f) for f in os.listdir(lidar_path) if f.endswith('.bin')]
        if os.path.exists(radar_path):
            radar_files += [os.path.join(radar_path, f) for f in os.listdir(radar_path) if f.endswith('_multi_w7.bin')]

    # ===== 计算整体 LiDAR 统计 =====
    if len(lidar_files) > 0:
        lidar_stats = compute_stats(lidar_files, "LiDAR")
        np.savetxt(os.path.join(output_dir, f'overall_lidar_{scene_name}_stats.txt'),
                   np.array([
                       np.concatenate([[lidar_stats["mean_depth"]], lidar_stats["mean_xyz"], [lidar_stats["mean_intensity"]]]),
                       np.concatenate([[lidar_stats["std_depth"]], lidar_stats["std_xyz"], [lidar_stats["std_intensity"]]])
                   ]),
                   fmt='%.6f',
                   header='mean_depth mean_xyz(3) mean_intensity | std_depth std_xyz(3) std_intensity')
        print(f"\n===== Overall LiDAR Stats ({scene_name}) =====")
        for k, v in lidar_stats.items():
            print(f"{k}: {v}")
        print("*" * 60)

    # ===== 计算整体 Radar 统计 =====
    if len(radar_files) > 0:
        radar_stats = compute_stats(radar_files, "Radar")
        np.savetxt(os.path.join(output_dir, f'overall_radar_{scene_name}_stats.txt'),
                   np.array([
                       np.concatenate([[radar_stats["mean_depth"]], radar_stats["mean_xyz"], [radar_stats["mean_intensity"]]]),
                       np.concatenate([[radar_stats["std_depth"]], radar_stats["std_xyz"], [radar_stats["std_intensity"]]])
                   ]),
                   fmt='%.6f',
                   header='mean_depth mean_xyz(3) mean_intensity | std_depth std_xyz(3) std_intensity')
        print(f"\n===== Overall Radar Stats ({scene_name}) =====")
        for k, v in radar_stats.items():
            print(f"{k}: {v}")
        print("=" * 60)
