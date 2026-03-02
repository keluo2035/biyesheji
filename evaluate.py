"""
评估指标：Dice、IoU，以及基于体积的 per-case 评估。
"""
import numpy as np


def dice_coeff(pred, target, smooth=1e-5):
    """计算 Dice 系数，输入为二值 numpy 数组。"""
    inter = (pred * target).sum()
    return (2.0 * inter + smooth) / (pred.sum() + target.sum() + smooth)


def iou_score(pred, target, smooth=1e-5):
    """计算 IoU，输入为二值 numpy 数组。"""
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter
    return (inter + smooth) / (union + smooth)


def precision_score(pred, target, smooth=1e-5):
    tp = (pred * target).sum()
    return (tp + smooth) / (pred.sum() + smooth)


def recall_score(pred, target, smooth=1e-5):
    tp = (pred * target).sum()
    return (tp + smooth) / (target.sum() + smooth)


def compute_case_metrics(pred_volume, gt_volume):
    """
    对一个 case 的完整 3-D 预测和标注计算指标。
    pred_volume, gt_volume: (D, H, W) 二值 ndarray
    """
    p = pred_volume.flatten().astype(np.float64)
    g = gt_volume.flatten().astype(np.float64)
    return {
        "dice": float(dice_coeff(p, g)),
        "iou": float(iou_score(p, g)),
        "precision": float(precision_score(p, g)),
        "recall": float(recall_score(p, g)),
    }
