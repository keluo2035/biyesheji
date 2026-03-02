# 基于多序列特征融合的无对比剂肝脏 MRI 自动分割方法研究

本项目针对无对比剂肝脏 MRI 图像自动分割问题，研究多序列特征融合的分割方法。通过利用 T1 DUAL（InPhase / OutPhase）和 T2 SPIR 不同序列的互补信息，在 U-Net 结构基础上构建多分支编码器和特征融合模块，包含简单拼接融合与注意力引导融合两类策略，验证多序列融合在无对比剂条件下提升肝脏分割精度的作用。

## 项目结构

```
├── config.py              # 全局配置：数据路径、超参数、实验定义
├── dataset.py             # 数据预处理：DICOM 读取、序列对齐、缓存、PyTorch Dataset
├── models.py              # 模型定义：U-Net、多分支注意力融合及消融变体
├── train.py               # 训练流程：损失函数、5 折交叉验证、早停、结果保存
├── evaluate.py            # 评估指标：Dice、IoU、Precision、Recall
├── visualize.py           # 可视化：分割对比图、训练曲线
├── stats.py               # 统计分析：显著性检验、箱线图、LaTeX 表格
├── run_experiments.py     # 主入口：通过命令行控制运行哪个实验
├── README.md
└── outputs/               # 运行后自动生成，存放预处理缓存与实验结果
```

## 实验设计

| 编号 | 实验名称 | 模型 | 输入序列 | 目的 |
|------|---------|------|---------|------|
| 1 | 单序列基线 | U-Net (1ch) | T2 | 建立性能基准 |
| 2 | 早期拼接融合 | U-Net (3ch) | T1-In + T1-Out + T2 | 最简单的多序列融合对照 |
| 3 | 注意力多分支融合 | AttentionMultiBranchUNet | T1-In + T1-Out + T2 | **核心创新**：独立编码器 + CBAM 注意力融合 |
| 4 | 消融：无注意力 | MultiBranchNoAttentionUNet | T1-In + T1-Out + T2 | 证明注意力机制的必要性 |
| 5 | 消融：共享编码器 | SharedEncoderAttentionUNet | T1-In + T1-Out + T2 | 证明独立编码器的必要性 |
| 6 | 可视化分析 | — | — | 分割结果定性对比 |
| 7 | 统计分析 | — | — | Wilcoxon / t-test 显著性检验 |

## 环境要求

### 硬件

| 组件 | 经过验证的配置 |
|------|--------------|
| GPU | NVIDIA RTX 4090 (24 GB) |
| CPU | AMD EPYC 7T83 (22 核) |
| 内存 | 90 GB |
| 磁盘 | 50 GB 以上可用空间 |

> 最低要求：任意支持 CUDA 的 NVIDIA GPU（显存 >= 6 GB）。纯 CPU 也可运行，但单个实验需要数天。

### 软件

| 组件 | 经过验证的版本 |
|------|--------------|
| 操作系统 | Ubuntu 22.04 / Windows 10+ |
| Python | 3.12.3 |
| NVIDIA 驱动 | 550.100 |
| CUDA | 12.4 |

### 依赖安装

推荐使用 AutoDL 等云平台选择 **PyTorch 2.5.1 / Python 3.12 / CUDA 12.4 (Ubuntu 22.04)** 镜像，镜像已预装 PyTorch 和基础科学计算库，只需补装少量依赖：

```bash
pip install SimpleITK==2.4.1 monai opencv-python-headless scikit-image scikit-learn nibabel
```

如果从零搭建环境，完整安装命令如下：

```bash
# 创建 conda 环境
conda create -n bishe python=3.12 -y
conda activate bishe

# 安装 PyTorch（CUDA 12.4）
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

# 安装全部依赖
pip install SimpleITK==2.4.1 numpy scipy scikit-learn scikit-image \
    opencv-python-headless pillow matplotlib tqdm nibabel pandas monai
```

> **注意：** 只安装 `SimpleITK`，不要单独安装 `itk` 包，否则会因 ITK 二进制版本冲突导致崩溃。

经过验证的完整依赖版本：

| 包 | 版本 |
|---|------|
| torch | 2.5.1+cu124 |
| torchvision | 0.20.1+cu124 |
| numpy | 2.1.3 |
| SimpleITK | 2.4.1 |
| scikit-learn | 1.8.0 |
| scikit-image | 0.26.0 |
| scipy | 1.17.1 |
| matplotlib | 3.9.2 |
| opencv-python-headless | 4.13.0.92 |
| monai | 1.5.2 |
| pillow | 11.0.0 |
| tqdm | 4.66.2 |
| nibabel | 5.3.3 |
| pandas | 3.0.1 |

## 数据准备

本项目使用 [CHAOS Challenge](https://chaos.grand-challenge.org/) 的腹部 MRI 数据集。仅需其中的 **MR 训练集**部分。

1. 从 [CHAOS 官方下载页面](https://chaos.grand-challenge.org/Download/) 下载 `Train_Sets`
2. 解压后确保目录结构如下（只需 `MR` 文件夹）：

```
<数据根目录>/
└── Train_Sets/
    └── MR/
        └── [case_id]/          # 共 20 个 case: 1,2,3,5,8,10,13,15,19-22,31-34,36-39
            ├── T1DUAL/
            │   ├── DICOM_anon/
            │   │   ├── InPhase/    # T1 同相位 DICOM 文件
            │   │   └── OutPhase/   # T1 反相位 DICOM 文件
            │   └── Ground/         # T1 标注（本项目未使用）
            └── T2SPIR/
                ├── DICOM_anon/     # T2 DICOM 文件
                └── Ground/         # T2 标注 PNG（肝脏像素值 55-70）
```

3. 打开 `config.py`，将 `DATA_ROOT` 修改为你的实际数据路径：

```python
# Linux 示例
DATA_ROOT = "/root/autodl-tmp/data"

# Windows 示例
DATA_ROOT = r"C:\paperdate\dates"
```

4. 如使用 Linux 服务器，建议同时修改 `config.py` 中的 `NUM_WORKERS`：

```python
NUM_WORKERS = 4    # Linux 下可设为 2-4 加速数据加载；Windows 保持 0
```

## 使用方法

所有操作通过 `run_experiments.py` 统一入口执行。

### 第 0 步：数据预处理（首次必须执行）

读取 DICOM 文件，将 T1 InPhase / OutPhase 重采样到 T2 空间，归一化后缓存为 `.npz` 文件，后续训练直接加载。

```bash
python run_experiments.py --preprocess
```

### 第 1 步：训练实验 1-5

逐个运行：

```bash
python run_experiments.py --experiment 1   # 单序列 T2 基线
python run_experiments.py --experiment 2   # 早期拼接融合
python run_experiments.py --experiment 3   # 注意力多分支融合（核心）
python run_experiments.py --experiment 4   # 消融：无注意力
python run_experiments.py --experiment 5   # 消融：共享编码器
```

或一条命令顺序执行全部实验：

```bash
python run_experiments.py --experiment 1 && \
python run_experiments.py --experiment 2 && \
python run_experiments.py --experiment 3 && \
python run_experiments.py --experiment 4 && \
python run_experiments.py --experiment 5 && \
python run_experiments.py --experiment 6 && \
python run_experiments.py --experiment 7
```

可选参数：

```bash
# 只跑单个 fold（调试用）
python run_experiments.py --experiment 3 --fold 0

# 自定义超参数
python run_experiments.py --experiment 3 --epochs 150 --lr 5e-5 --batch_size 8

# 指定 GPU
python run_experiments.py --experiment 3 --gpu 0
```

### 第 2 步：可视化（实验六）

需在实验 1-5 全部训练完成后运行：

```bash
python run_experiments.py --experiment 6
```

生成内容：各方法分割结果对比图、训练曲线，保存至 `outputs/visualizations/`。

### 第 3 步：统计分析（实验七）

```bash
python run_experiments.py --experiment 7
```

生成内容：Dice/IoU 箱线图、Wilcoxon 显著性检验结果、LaTeX 格式结果表，保存至 `outputs/statistics/`。

### 参考训练时间（RTX 4090, batch_size=4）

| 实验 | 每 epoch | 5 fold 总计 |
|------|---------|------------|
| 1 (T2 UNet) | ~5 s | ~45 min |
| 2 (早期拼接) | ~6 s | ~50 min |
| 3 (注意力多分支) | ~15 s | ~2 h |
| 4 (无注意力多分支) | ~14 s | ~2 h |
| 5 (共享编码器) | ~12 s | ~1.5 h |
| 6 (可视化) | — | ~5 min |
| 7 (统计分析) | — | ~1 min |
| **合计** | | **约 4-7 小时** |

## 输出目录结构

```
outputs/
├── preprocessed/           # 预处理缓存（.npz）
├── exp1_t2_unet/           # 实验 1 结果
│   ├── fold_0/
│   │   ├── best_model.pth          # 最佳模型权重
│   │   ├── best_val_metrics.json   # 该折验证集 per-case 指标
│   │   └── history.json            # 逐 epoch 训练 loss / 验证 Dice
│   ├── fold_1/ ... fold_4/
│   ├── all_case_metrics.json       # 全部 20 case 汇总指标
│   └── summary.json                # 5 折平均 Dice ± 标准差
├── exp2_early_fusion/
├── exp3_attn_multibranch/
├── exp4_no_attention/
├── exp5_shared_encoder/
├── visualizations/                 # 可视化图片
│   ├── training_curves.png         # 训练曲线对比
│   └── fold*_case*.png             # 分割轮廓对比图
└── statistics/                     # 统计分析结果
    ├── boxplot_dice.png            # Dice 箱线图
    ├── boxplot_iou.png             # IoU 箱线图
    ├── significance_tests.json     # Wilcoxon / t-test p 值
    └── results_table.tex           # LaTeX 结果表（可直接插入论文）
```

## 评估指标

- **Dice Similarity Coefficient (DSC)**：衡量预测与标注的重叠程度
- **Intersection over Union (IoU)**：交并比
- **Precision**：预测为肝脏的区域中实际为肝脏的比例
- **Recall**：实际肝脏区域被正确检出的比例

所有指标基于 3D 体积计算（per-case），采用 5 折交叉验证，20 个 case 均有独立评估结果。

## 注意事项

- 首次运行 `--preprocess` 需要约 5-15 分钟（取决于磁盘 I/O 速度）
- **强烈建议使用 GPU 训练**，CPU 训练单个实验可能需要数天
- **不要安装 `itk` 包**，SimpleITK 已内置匹配的 ITK 二进制，混装会导致版本冲突崩溃
- Windows 下 `config.py` 中 `NUM_WORKERS` 应保持为 `0`，Linux 下可设为 `2-4`
- 训练中途系统休眠不影响结果正确性，仅导致该 epoch 计时异常
