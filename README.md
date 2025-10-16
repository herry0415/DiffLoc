# DiffLoc
Run the contrastive experiment for Hercules using this model

`此代码修改为只能训radar 对mask进行了调整` 在模型和损失函数里面

# Environment setup
- 按照给定的install.sh环境安装一下torch和其他的库。下面几个库如果遇到问题就手动安装：
- 安装torchsparse 的时候要去官方github下载一下 然后放进去 运行`pip install -e .` 来安装
- 安装torch_scatter的时候
    - 需要去一个官网https://data.pyg.org/whl/下载一个包  手动安装一下
    - 找到一个对应版本包进行下载 `torch_scatter-2.0.9-cp38-linux_x86`
- pytorch3d 也是需要去官网下载git 文件。然后  `pip install -e .`
- spvnas要放在diffloc主文件夹下、其他几个包放项目文件夹下也可以
- 最后要注意有些包需要加载到PYTHON环境变量中去PYTHONPATH  `export PYTHONPATH=/home/ldq/code/DiffLoc-main/spvnas:/home/ldq/code/DiffLoc-main`

# Data prepare
- lidar 需要运行`hercules_prepare.py` 对lidar进行处理
- radar 不需要进行处理 需要在代码中进行修改

# config 文件
在`train/test` 之前要修改`hercules_bev.yaml/hercules_radar_bev.yaml`的参数
- hercules.yaml   lidar 的train/test 文件
- hercules_radar.yaml   radar 的train/test 文件

# Train

## 1. 更改配置文件 

选择`lidar/radar` 对应  `hercules_bev.yaml/hercules_radar_bev.yaml`
  - 数据集路径 `train.dataroot`
  - 序列名 `train.sequence`
  - 权重路径 `exp_name 和 ckpt `
  - 数据集类别 `train.dataset`  `'Hercules_radar'`加载radar数据集类  `'Hercules'` 加载lidar数据集类
  - 深度投影图像对应的均值和方差 `sensors.image_mean` 和 `sensors.image_stds`

## 2. 修改train.py加载配置文件部分`
```
conf = OmegaConf.load('cfgs/hercules.yaml')`
```


## 3. 终端进行分布式训练
- 一定用`accelerate`指令且**半精度训练**

```python
accelerate launch --num_processes 3 --mixed_precision fp16 train_bev.py
```
- num_processes 可以指定为多个 具体使用哪个GPU 在代码里面用`os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2, 3"` 去进行调整
- 如果出现多进程训练端口被占用的情况 可以用 `--main_process_port 29501` 或任何一个大于 1024 且小于 65535 的端口

# Test
"基本流程同训练
## 1. 更改配置文件
## 2. 修改test.py加载配置文件部分
```
conf = OmegaConf.load('cfgs/hercules_bev.yaml')
```

## 3. 运行代码进行测试

```python
python test.py
```






