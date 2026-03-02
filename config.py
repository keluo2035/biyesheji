"""
全局配置：数据路径、训练超参数、实验定义。
"""
import os
import argparse

# ======================== 路径 ========================
DATA_ROOT = r"C:\paperdate\dates"
TRAIN_ROOT = os.path.join(DATA_ROOT, "Train_Sets", "MR")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
CACHE_DIR = os.path.join(OUTPUT_DIR, "preprocessed")

# ======================== 数据 ========================
CASE_IDS = [1, 2, 3, 5, 8, 10, 13, 15, 19, 20,
            21, 22, 31, 32, 33, 34, 36, 37, 38, 39]
IMAGE_SIZE = 256
LIVER_LO, LIVER_HI = 55, 70  # Ground truth 中肝脏像素值范围

# ======================== 训练 ========================
NUM_FOLDS = 5
BATCH_SIZE = 4
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
PATIENCE = 15          # 早停耐心
NUM_WORKERS = 0        # Windows 建议为 0；Linux/Mac 可设为 2-4
SEED = 42

# U-Net 各层通道数
FEATURES = [64, 128, 256, 512]

# ======================== 实验定义 ========================
EXPERIMENTS = {
    1: dict(name="exp1_t2_unet",
            desc="实验一：单序列 T2 U-Net 基线",
            model="unet",
            sequences=["t2"]),
    2: dict(name="exp2_early_fusion",
            desc="实验二：早期拼接融合（简单多序列基线）",
            model="unet",
            sequences=["t1in", "t1out", "t2"]),
    3: dict(name="exp3_attn_multibranch",
            desc="实验三：注意力多分支融合（核心创新）",
            model="attention_multibranch",
            sequences=["t1in", "t1out", "t2"]),
    4: dict(name="exp4_no_attention",
            desc="实验四：消融——无注意力多分支",
            model="multibranch_no_attention",
            sequences=["t1in", "t1out", "t2"]),
    5: dict(name="exp5_shared_encoder",
            desc="实验五：消融——共享编码器+注意力",
            model="shared_encoder_attention",
            sequences=["t1in", "t1out", "t2"]),
}


def exp_dir(exp_id):
    return os.path.join(OUTPUT_DIR, EXPERIMENTS[exp_id]["name"])


def parse_args():
    p = argparse.ArgumentParser(description="多序列 MRI 肝脏分割实验")
    p.add_argument("--preprocess", action="store_true",
                   help="预处理 CHAOS 数据（首次运行需执行一次）")
    p.add_argument("--experiment", type=int, choices=range(1, 8),
                   metavar="N", help="实验编号 1-7")
    p.add_argument("--fold", type=int, default=None,
                   help="指定单个 fold（0-4），默认跑全部")
    p.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=LEARNING_RATE)
    p.add_argument("--gpu", type=int, default=0, help="GPU 编号")
    return p.parse_args()
