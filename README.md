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

- **Python** 3.8+
- **操作系统** Windows / Linux / macOS

### 依赖安装

```bash
# 建议使用 conda 创建独立环境
conda create -n bishe python=3.8 -y
conda activate bishe

# 安装 PyTorch（根据你的 CUDA 版本选择，以下示例为 CUDA 11.8）
# 如无 GPU 可安装 CPU 版本，但训练速度会很慢
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install numpy scipy scikit-learn scikit-image SimpleITK opencv-python pillow matplotlib tqdm nibabel pandas monai
```

核心依赖版本参考：

| 包 | 版本 |
|---|------|
| torch | >= 2.0 |
| numpy | >= 1.24 |
| SimpleITK | >= 2.2 |
| scikit-learn | >= 1.3 |
| scikit-image | >= 0.21 |
| scipy | >= 1.10 |
| matplotlib | >= 3.7 |
| opencv-python | >= 4.8 |
| tqdm | >= 4.60 |

## 数据准备

本项目使用 [CHAOS Challenge](https://chaos.grand-challenge.org/) 的腹部 MRI 数据集。

1. 从 [CHAOS 官方下载页面](https://chaos.grand-challenge.org/Download/) 下载数据
2. 解压后确保目录结构如下：

```
<数据根目录>/
├── Train_Sets/
│   └── MR/
│       └── [case_id]/
│           ├── T1DUAL/
│           │   ├── DICOM_anon/
│           │   │   ├── InPhase/
│           │   │   └── OutPhase/
│           │   └── Ground/
│           └── T2SPIR/
│               ├── DICOM_anon/
│               └── Ground/
└── Test_Sets/
    └── MR/
        └── ...
```

3. 打开 `config.py`，将 `DATA_ROOT` 修改为你的实际数据路径：

```python
DATA_ROOT = r"你的数据存放路径"
```

## 使用方法

所有操作通过 `run_experiments.py` 统一入口执行：

### 第 0 步：数据预处理（首次必须执行）

将 DICOM 数据读取、T1→T2 空间重采样、归一化后缓存为 `.npz` 文件，后续训练直接加载。

```bash
python run_experiments.py --preprocess
```

### 第 1 步：训练实验 1-5

```bash
python run_experiments.py --experiment 1   # 单序列 T2 基线
python run_experiments.py --experiment 2   # 早期拼接融合
python run_experiments.py --experiment 3   # 注意力多分支融合（核心）
python run_experiments.py --experiment 4   # 消融：无注意力
python run_experiments.py --experiment 5   # 消融：共享编码器
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

## 输出目录结构

```
outputs/
├── preprocessed/           # 预处理缓存（.npz）
├── exp1_t2_unet/           # 实验 1 结果
│   ├── fold_0/
│   │   ├── best_model.pth
│   │   ├── best_val_metrics.json
│   │   └── history.json
│   ├── ...
│   ├── all_case_metrics.json
│   └── summary.json
├── exp2_early_fusion/
├── exp3_attn_multibranch/
├── exp4_no_attention/
├── exp5_shared_encoder/
├── visualizations/         # 可视化图片
└── statistics/             # 统计分析结果
    ├── boxplot_dice.png
    ├── boxplot_iou.png
    ├── significance_tests.json
    └── results_table.tex
```

## 评估指标

- **Dice Similarity Coefficient (DSC)**：衡量预测与标注的重叠程度
- **Intersection over Union (IoU)**：交并比
- **Precision**：预测为肝脏的区域中实际为肝脏的比例
- **Recall**：实际肝脏区域被正确检出的比例

所有指标基于 3D 体积计算（per-case），采用 5 折交叉验证，20 个 case 均有独立评估结果。

## 注意事项

- 首次运行 `--preprocess` 需要较长时间（依据磁盘速度约 5-15 分钟）
- **强烈建议使用 GPU 训练**，CPU 训练单个 fold 可能需要数小时
- Windows 下 `NUM_WORKERS` 默认为 0，如使用 Linux 可在 `config.py` 中调大以加速数据加载
- 训练中途电脑休眠不会影响结果正确性，但会导致 epoch 耗时异常（计时包含休眠时间）
"# biyesheji" 
"# biyesheji" 
"# biyesheji" 
