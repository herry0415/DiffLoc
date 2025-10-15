import os
import numpy as np

# 输入路径（和之前保持一致）
output_dir = '/home/ldq/code/DiffLoc-main/stats'

# 1. 读取之前保存的逐帧均值和方差文件
mean_xyz = np.loadtxt(os.path.join(output_dir, 'mean_xyz_library.txt'))  # (N,3)
std_xyz = np.loadtxt(os.path.join(output_dir, 'std_xyz_library.txt'))    # (N,3)
mean_depth = np.loadtxt(os.path.join(output_dir, 'mean_depth_library.txt'))  # (N,1)
std_depth = np.loadtxt(os.path.join(output_dir, 'std_depth_library.txt'))    # (N,1)

# 2. intensity 没有存文件，只是 print 出来过 -> 我们需要重新存 or 改为加载（建议保存）
# 这里假设之前也保存了 mean_intensity / std_intensity
mean_intensity = np.loadtxt(os.path.join(output_dir, 'mean_intensity_library.txt'))  # (N,1)
std_intensity = np.loadtxt(os.path.join(output_dir, 'std_intensity_library.txt'))    # (N,1)

# 3. 计算整体场景均值
# 按顺序：[depth, x, y, z, intensity, proj_label_tensor]
global_mean = np.zeros(6)
global_std = np.zeros(6)

global_mean[0] = np.mean(mean_depth)        # depth
global_mean[1:4] = np.mean(mean_xyz, axis=0)  # xyz
global_mean[4] = np.mean(mean_intensity)    # intensity
global_mean[5] = 0                          # proj_label_tensor

global_std[0] = np.mean(std_depth)          # depth
global_std[1:4] = np.mean(std_xyz, axis=0)    # xyz
global_std[4] = np.mean(std_intensity)      # intensity
global_std[5] = 1                           # proj_label_tensor

# 4. 保存结果
np.savetxt(os.path.join(output_dir, 'hercules_global_mean.txt'), global_mean[None, :], fmt='%.6f')
np.savetxt(os.path.join(output_dir, 'hercules_global_std.txt'), global_std[None, :], fmt='%.6f')

# 5. 打印结果
print("Global Mean (depth, x, y, z, intensity, proj_label_tensor):")
print(global_mean)
print("*" * 40)

print("Global Std (depth, x, y, z, intensity, proj_label_tensor):")
print(global_std)
print("*" * 40)
