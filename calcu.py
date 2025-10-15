import os
import numpy as np
from datasets.robotcar_sdk.python.transform import euler_to_so3
from utils.bin2pcd import bin_to_pcd
from tqdm import tqdm

root_path = '/home/data/ldq/HeRCULES'
output_dir = '/home/ldq/code/DiffLoc-main/stats'

# 定义场景字典：训练子目录、测试子目录
scene_dict = {
    'Library': (['Library_01_Day', 'Library_02_Night'], ['Library_03_Day']),
    # 将来可以加其他场景：
    # 'City': (['City_01_Day', 'City_02_Night'], ['City_03_Day']),
}

num = 0
# 预分配数组（最大 50 万帧，够大）
mean_xyz = np.zeros((10000, 3))
std_xyz = np.zeros((10000, 3))
mean_depth = np.zeros((10000, 1))
std_depth = np.zeros((10000, 1))
mean_intensity = np.zeros((10000, 1))
std_intensity = np.zeros((10000, 1))

# 遍历场景和训练子目录
for scene_name, (train_subdirs, test_subdirs) in scene_dict.items():
    for subdir in train_subdirs:   # 只计算训练集
        path = os.path.join(root_path, scene_name, subdir, 'LiDAR', 'np8Aeva')
        file_list = os.listdir(path)
        for file in tqdm(file_list, desc=f"Calculating {subdir} point cloud stats..."):
            file_name = os.path.join(path, file)
            # print(f"Processing: {file_name}")
            # print(f"Frame idx: {num}")
            
            # 读取点云 (XYZ + intensity)
            # 这里的 sensor_type 固定为 "Aeva"
            # points, extra, _ = bin_to_pcd(file_name, "Aeva")

            data = np.fromfile(file_name, dtype=np.float32).reshape(-1, 8)

            # 提取数据
            xyz = data[:, :3]
            # 根据 bin_to_pcd 函数的 Aeva 模式，强度信息在 extra 数组的最后一列 (第4列，索引为3)
            intensity = data[:, 7]

            # mean xyz
            m_xyz = np.mean(xyz, axis=0)
            mean_xyz[num, :] = m_xyz
            # std xyz
            s_xyz = np.std(xyz, axis=0)
            std_xyz[num, :] = s_xyz

            # mean depth
            depth = np.linalg.norm(xyz, 2, axis=1)
            m_depth = np.mean(depth)
            mean_depth[num, :] = m_depth
            # std depth
            s_depth = np.std(depth)
            std_depth[num, :] = s_depth

            # mean intensity
            m_i = np.mean(intensity)
            mean_intensity[num, :] = m_i
            # std intensity
            s_i = np.std(intensity)
            std_intensity[num, :] = s_i

            num += 1

# ===== 保存统计结果 =====
np.savetxt(os.path.join(output_dir, 'mean_xyz_library.txt'), mean_xyz[:num, :], fmt='%.4f')
print("mean_xyz")
print(np.mean(mean_xyz[:num, :], 0))
print("*" * 20)

np.savetxt(os.path.join(output_dir, 'std_xyz_library.txt'), std_xyz[:num, :], fmt='%.4f')
print("std_xyz")
print(np.mean(std_xyz[:num, :], 0))
print("*" * 20)

np.savetxt(os.path.join(output_dir, 'mean_depth_library.txt'), mean_depth[:num, :], fmt='%.4f')
print("mean_depth")
print(np.mean(mean_depth[:num, :]))
print("*" * 20)

np.savetxt(os.path.join(output_dir, 'std_depth_library.txt'), std_depth[:num, :], fmt='%.4f')
print("std_depth")
print(np.mean(std_depth[:num, :]))
print("*" * 20)

np.savetxt(os.path.join(output_dir, 'mean_intensity_library.txt'), mean_intensity[:num, :], fmt='%.4f')
print("mean_intensity")
print(np.mean(mean_intensity[:num, :]))
print("*" * 20)

np.savetxt(os.path.join(output_dir, 'std_intensity_library.txt'), std_intensity[:num, :], fmt='%.4f')
print("std_intensity")
print(np.mean(std_intensity[:num, :]))
print("*" * 20)