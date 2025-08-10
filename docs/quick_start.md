# Quick Start

### Set up a new virtual environment
```bash
conda create -n sparsedrive python=3.8 -y
conda activate sparsedrive
```

### Install dependency packpages
```bash
sparsedrive_path="path/to/sparsedrive"
cd ${sparsedrive_path}
pip3 install --upgrade pip
pip3 install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
pip3 install -r requirement.txt
```

### Compile the deformable_aggregation CUDA op
```bash
cd projects/mmdet3d_plugin/ops
python3 setup.py develop
cd ../../../
```

### Prepare the data
Download the [NuScenes dataset](https://www.nuscenes.org/nuscenes#download) and CAN bus expansion, put CAN bus expansion in /path/to/nuscenes, create symbolic links.
```bash
cd ${sparsedrive_path}
mkdir data
ln -s path/to/nuscenes ./data/nuscenes
```

Pack the meta-information and labels of the dataset, and generate the required pkl files to data/infos. Note that we also generate map_annos in data_converter, with a roi_size of (30, 60) as default, if you want a different range, you can modify roi_size in tools/data_converter/nuscenes_converter.py.
```bash
sh scripts/create_data.sh
```

### Generate anchors by K-means
Gnerated anchors are saved to data/kmeans and can be visualized in vis/kmeans.
```bash
sh scripts/kmeans.sh
```


### Download pre-trained weights
Download the required backbone [pre-trained weights](https://download.pytorch.org/models/resnet50-19c8e357.pth).
```bash
mkdir ckpt
wget https://download.pytorch.org/models/resnet50-19c8e357.pth -O ckpt/resnet50-19c8e357.pth
```

### Commence training and testing
```bash
# train
sh scripts/train.sh

# test
sh scripts/test.sh
```

### Visualization
```
sh scripts/visualize.sh
```
不需要跑满 4000 iters。按你这个 GUI 面板，给你两套专业可落地的配置，用于抓“稳定阶段”的短窗时间线。照着勾选、填参数即可开跑。

### 方案A：轻量时间线（推荐先跑，几十秒足够）
- 勾选
  - Collect CUDA trace
  - Collect NVTX trace
- 不勾选
  - Collect CPU IP/backtrace samples
  - Collect CPU context switch trace
  - Collect OS runtime libraries trace
  - Collect GPU context switch trace
  - Collect GPU Metrics
  - 其他 OpenMP/OpenGL/Vulkan/Video
- Python profiling options
  - 全部不勾（默认关）。如果只想感知 Python 开销，可勾 Collect Python backtrace samples，并把 Sampling rate 拉到 100–200 Hz（不要 1 kHz）。
- 右侧 Start 面板
  - Start profiling after: 15.0 seconds（避开冷启动/数据加载预热）
  - Limit profiling to: 45.0 seconds（抓稳定期一小段）
  - 其他保持未勾
- 点击 Start 后“要运行的命令”
  - Application: /bin/bash
  - Arguments:
    -lc 'cd /workspace/SparseDrive && PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=0 bash ./tools/dist_train.sh projects/configs/sparsedrive_small_stage1.py 1 --deterministic'
  - Working directory: /workspace/SparseDrive
  - Environment（如有单独栏位，可填写，没的话已在 Arguments 中设置了）:
    - PYTHONPATH=/workspace/SparseDrive
    - CUDA_VISIBLE_DEVICES=0
    - PORT=28651
- 预期
  - 开销低、文件体积可控（通常 < 500MB），时间线上能看到我们加的 NVTX 区段：
    - Main Function / Model Initialization / Dataset Preparation / Training Process
    - DataLoader… / Model: To CUDA + Wrap… / Runner… / Eval… / Runner: run() Iteration Loop
    - train_iter N（每迭代标记）

### 方案B：深度时间线（短窗且机器空闲时使用）
- 在方案A基础上，额外勾选（开销显著增加）：
  - Collect OS runtime libraries trace（osrt）
  - 如果需要再勾 Collect GPU Metrics（更重，建议只采样 10–20 秒）
- 右侧 Start 面板
  - Start profiling after: 10.0 seconds
  - Limit profiling to: 20.0–30.0 seconds（尽量短）
- 其他与方案A一致

### 什么时候需要 Python 采样
- 仅当你怀疑 Python 调度或 Dataloader 开销异常时再开：
  - 勾 Collect Python backtrace samples，Sampling rate 设置 100–200 Hz
  - 不建议同时开 Python Functions trace（非常重），除非只采 5–10 秒做定位

### 文件体积与可视化建议
- 只抓稳定窗口（几十秒），不要全程；NVTX 区段已帮助你在时间线上快速定位。
- 若要多维度对比（开/关 AMP、不同 batch、不同 GPU），按上述短窗配置重复多次，文件依然可控。
- GUI 里打开 .qdrep，展开 NVTX 轨道对照 CUDA Kernel 轨道看：
  - 查看每个 train_iter 的 Kernel 密度、重叠度、memcpy 比例
  - 需要算子级性能用 Nsight Compute（非 Nsight Systems）

这样设置就能在你的场景下跑出“专业的”时间线结果，且体积、开销可控。