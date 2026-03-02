"""
训练 / 验证流程：
  - Dice + BCE 混合损失
  - 5 折交叉验证
  - 早停 + 模型保存
  - Per-case 评估并输出 JSON
"""
import os
import json
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_torch
from torch.utils.data import DataLoader
from collections import defaultdict
from tqdm import tqdm

import config
from dataset import LiverDataset, get_fold_splits
from models import create_model, prepare_input
from evaluate import compute_case_metrics


# ======================== 损失函数 ========================

class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets):
        bce = F_torch.binary_cross_entropy_with_logits(logits, targets)
        probs = torch.sigmoid(logits)
        smooth = 1e-5
        inter = (probs * targets).sum()
        dice_loss = 1.0 - (2.0 * inter + smooth) / (
            probs.sum() + targets.sum() + smooth)
        return bce + dice_loss


# ======================== 种子 ========================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ======================== 单 epoch 训练 ========================

def train_one_epoch(model, loader, criterion, optimizer, device, exp_id):
    model.train()
    running_loss = 0.0
    for batch in loader:
        inputs = prepare_input(batch, exp_id, device)
        masks = batch["mask"].unsqueeze(1).to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * masks.size(0)
    return running_loss / len(loader.dataset)


# ======================== 验证（返回 loss + per-case 指标）========================

@torch.no_grad()
def validate(model, loader, criterion, device, exp_id):
    model.eval()
    running_loss = 0.0
    case_preds = defaultdict(list)
    case_masks = defaultdict(list)
    case_slices = defaultdict(list)

    for batch in loader:
        inputs = prepare_input(batch, exp_id, device)
        masks_gpu = batch["mask"].unsqueeze(1).to(device)

        outputs = model(inputs)
        loss = criterion(outputs, masks_gpu)
        running_loss += loss.item() * masks_gpu.size(0)

        preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()[:, 0]
        masks_np = batch["mask"].numpy()
        cids = batch["case_id"]
        sidxs = batch["slice_idx"]

        if isinstance(cids, torch.Tensor):
            cids = cids.tolist()
        if isinstance(sidxs, torch.Tensor):
            sidxs = sidxs.tolist()

        for i in range(len(cids)):
            c = int(cids[i])
            case_preds[c].append(preds[i])
            case_masks[c].append(masks_np[i])
            case_slices[c].append(int(sidxs[i]))

    avg_loss = running_loss / len(loader.dataset)

    metrics = {}
    for c in case_preds:
        order = np.argsort(case_slices[c])
        pred_vol = np.stack([case_preds[c][i] for i in order])
        mask_vol = np.stack([case_masks[c][i] for i in order])
        metrics[c] = compute_case_metrics(pred_vol, mask_vol)

    return avg_loss, metrics


# ======================== 单折训练 ========================

def train_fold(exp_id, fold_idx, train_ids, val_ids, args):
    set_seed(config.SEED + fold_idx)
    device = torch.device(f"cuda:{args.gpu}"
                          if torch.cuda.is_available() else "cpu")

    # 数据
    train_ds = LiverDataset(train_ids, train=True)
    val_ds = LiverDataset(val_ids, train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=config.NUM_WORKERS,
                              pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=config.NUM_WORKERS,
                            pin_memory=True)

    # 模型 / 优化器 / 损失
    model = create_model(exp_id).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=7, verbose=True)
    criterion = DiceBCELoss()

    # 保存路径
    fold_dir = os.path.join(config.exp_dir(exp_id), f"fold_{fold_idx}")
    os.makedirs(fold_dir, exist_ok=True)

    best_dice = 0.0
    patience_cnt = 0
    history = {"train_loss": [], "val_loss": [], "val_dice": []}

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*60}")
    print(f"实验 {exp_id} | Fold {fold_idx} | "
          f"训练 {len(train_ids)} cases ({len(train_ds)} slices) | "
          f"验证 {len(val_ids)} cases ({len(val_ds)} slices)")
    print(f"模型参数量: {n_params:,}")
    print(f"{'='*60}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, exp_id)
        val_loss, val_metrics = validate(
            model, val_loader, criterion, device, exp_id)

        mean_dice = np.mean([m["dice"] for m in val_metrics.values()])
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(mean_dice)

        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
              f"val_dice={mean_dice:.4f} | {elapsed:.1f}s")

        if mean_dice > best_dice:
            best_dice = mean_dice
            patience_cnt = 0
            torch.save(model.state_dict(),
                       os.path.join(fold_dir, "best_model.pth"))
            with open(os.path.join(fold_dir, "best_val_metrics.json"), "w") as f:
                json.dump({str(k): v for k, v in val_metrics.items()}, f, indent=2)
        else:
            patience_cnt += 1

        if patience_cnt >= config.PATIENCE:
            print(f"早停于 epoch {epoch}，最佳 val_dice={best_dice:.4f}")
            break

    with open(os.path.join(fold_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"Fold {fold_idx} 完成，最佳 Dice = {best_dice:.4f}")
    return best_dice


# ======================== 运行完整实验（全部 fold）========================

def run_training_experiment(exp_id, args):
    cfg = config.EXPERIMENTS[exp_id]
    print(f"\n{'#'*60}")
    print(f"# {cfg['desc']}")
    print(f"{'#'*60}")

    splits = get_fold_splits()
    fold_dices = []

    folds_to_run = ([args.fold] if args.fold is not None
                    else list(range(config.NUM_FOLDS)))

    for fi in folds_to_run:
        train_ids, val_ids = splits[fi]
        d = train_fold(exp_id, fi, train_ids, val_ids, args)
        fold_dices.append(d)

    if len(fold_dices) == config.NUM_FOLDS:
        summary = {
            "experiment": cfg["name"],
            "fold_best_dice": fold_dices,
            "mean_dice": float(np.mean(fold_dices)),
            "std_dice": float(np.std(fold_dices)),
        }
        out_path = os.path.join(config.exp_dir(exp_id), "summary.json")
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n汇总: mean Dice = {summary['mean_dice']:.4f} "
              f"± {summary['std_dice']:.4f}")

    return fold_dices


# ======================== 聚合全部 fold 的 per-case 结果 ========================

def aggregate_results(exp_id):
    """读取所有 fold 的 best_val_metrics，合并为完整的 per-case 结果。"""
    all_metrics = {}
    for fi in range(config.NUM_FOLDS):
        path = os.path.join(config.exp_dir(exp_id),
                            f"fold_{fi}", "best_val_metrics.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            fold_m = json.load(f)
        for cid, m in fold_m.items():
            all_metrics[int(cid)] = m

    out_path = os.path.join(config.exp_dir(exp_id), "all_case_metrics.json")
    with open(out_path, "w") as f:
        json.dump({str(k): v for k, v in all_metrics.items()}, f, indent=2)
    return all_metrics
