"""
实验六：可视化——分割结果展示、多方法对比、失败案例分析。
"""
import os
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from skimage import measure

import config
from dataset import LiverDataset, get_fold_splits
from models import create_model, prepare_input

plt.rcParams["font.family"] = "SimHei"
plt.rcParams["axes.unicode_minus"] = False


# ======================== 工具函数 ========================

def load_best_model(exp_id, fold_idx, device):
    model = create_model(exp_id).to(device)
    path = os.path.join(config.exp_dir(exp_id),
                        f"fold_{fold_idx}", "best_model.pth")
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def predict_dataset(model, dataset, exp_id, device):
    """对 dataset 做推理，返回 {case_id: pred_volume} 和 {case_id: gt_volume}。"""
    from torch.utils.data import DataLoader
    from collections import defaultdict

    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE,
                        shuffle=False, num_workers=0)
    case_preds = defaultdict(list)
    case_masks = defaultdict(list)
    case_slices = defaultdict(list)

    with torch.no_grad():
        for batch in loader:
            inputs = prepare_input(batch, exp_id, device)
            outputs = model(inputs)
            preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()[:, 0]
            masks = batch["mask"].numpy()

            cids = batch["case_id"]
            sidxs = batch["slice_idx"]
            if isinstance(cids, torch.Tensor):
                cids = cids.tolist()
            if isinstance(sidxs, torch.Tensor):
                sidxs = sidxs.tolist()

            for i in range(len(cids)):
                c = int(cids[i])
                case_preds[c].append(preds[i])
                case_masks[c].append(masks[i])
                case_slices[c].append(int(sidxs[i]))

    pred_vols, gt_vols = {}, {}
    for c in case_preds:
        order = np.argsort(case_slices[c])
        pred_vols[c] = np.stack([case_preds[c][i] for i in order])
        gt_vols[c] = np.stack([case_masks[c][i] for i in order])
    return pred_vols, gt_vols


def get_raw_slices(dataset, case_id, slice_idx):
    """从 dataset 中提取某 case 某 slice 的原始图像（未增强）。"""
    vol = dataset.volumes[case_id]
    return {
        "t1in": vol["t1in"][slice_idx],
        "t1out": vol["t1out"][slice_idx],
        "t2": vol["t2"][slice_idx],
    }


def draw_contour(ax, mask, color="lime", lw=1.5):
    """在 ax 上绘制 mask 轮廓。"""
    contours = measure.find_contours(mask, 0.5)
    for c in contours:
        ax.plot(c[:, 1], c[:, 0], color=color, linewidth=lw)


# ======================== 可视化主函数 ========================

def visualize_comparison(exp_ids=(1, 2, 3, 4, 5),
                         num_cases=4, slices_per_case=3):
    """
    对每个 fold 的验证集选取代表性 case，
    展示各实验方法的分割结果对比。
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    save_dir = os.path.join(config.OUTPUT_DIR, "visualizations")
    os.makedirs(save_dir, exist_ok=True)

    splits = get_fold_splits()
    exp_names = {eid: config.EXPERIMENTS[eid]["name"] for eid in exp_ids}
    short_labels = {
        1: "T2 UNet",
        2: "Early Fusion",
        3: "Attn MBranch",
        4: "No Attn",
        5: "Shared Enc",
    }

    for fi in range(config.NUM_FOLDS):
        _, val_ids = splits[fi]
        val_ds = LiverDataset(val_ids, train=False)

        models = {}
        preds_all = {}
        for eid in exp_ids:
            model_path = os.path.join(config.exp_dir(eid),
                                      f"fold_{fi}", "best_model.pth")
            if not os.path.exists(model_path):
                continue
            m = load_best_model(eid, fi, device)
            pv, gv = predict_dataset(m, val_ds, eid, device)
            models[eid] = m
            preds_all[eid] = pv

        if not models:
            continue

        first_eid = list(preds_all.keys())[0]
        gt_vols = predict_dataset(
            models[first_eid], val_ds, first_eid, device)[1]

        selected_cases = val_ids[:num_cases]
        for cid in selected_cases:
            if cid not in gt_vols:
                continue
            n_slices = gt_vols[cid].shape[0]
            liver_slices = [s for s in range(n_slices)
                           if gt_vols[cid][s].sum() > 0]
            if not liver_slices:
                continue
            indices = np.linspace(0, len(liver_slices) - 1,
                                  min(slices_per_case, len(liver_slices)),
                                  dtype=int)
            chosen = [liver_slices[i] for i in indices]

            n_methods = len(models)
            n_cols = n_methods + 2
            fig, axes = plt.subplots(len(chosen), n_cols,
                                     figsize=(3 * n_cols, 3 * len(chosen)))
            if len(chosen) == 1:
                axes = axes[np.newaxis, :]

            for row, si in enumerate(chosen):
                raw = get_raw_slices(val_ds, cid, si)
                gt_s = gt_vols[cid][si]

                axes[row, 0].imshow(raw["t2"], cmap="gray")
                axes[row, 0].set_title("T2 SPIR" if row == 0 else "")
                axes[row, 0].axis("off")

                axes[row, 1].imshow(raw["t2"], cmap="gray")
                draw_contour(axes[row, 1], gt_s, color="lime", lw=2)
                axes[row, 1].set_title("Ground Truth" if row == 0 else "")
                axes[row, 1].axis("off")

                for col, eid in enumerate(models.keys()):
                    ax = axes[row, col + 2]
                    pred_s = preds_all[eid][cid][si]
                    ax.imshow(raw["t2"], cmap="gray")
                    draw_contour(ax, gt_s, color="lime", lw=1.5)
                    draw_contour(ax, pred_s, color="red", lw=1.5)
                    label = short_labels.get(eid, str(eid))
                    ax.set_title(label if row == 0 else "")
                    ax.axis("off")

            fig.suptitle(f"Case {cid} (Fold {fi})", fontsize=14)
            plt.tight_layout()
            fig.savefig(os.path.join(save_dir,
                                     f"fold{fi}_case{cid}.png"),
                        dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  已保存 fold{fi}_case{cid}.png")


def visualize_training_curves(exp_ids=(1, 2, 3, 4, 5)):
    """绘制各实验的训练损失和验证 Dice 曲线。"""
    save_dir = os.path.join(config.OUTPUT_DIR, "visualizations")
    os.makedirs(save_dir, exist_ok=True)

    short_labels = {
        1: "T2 UNet", 2: "Early Fusion", 3: "Attn MBranch",
        4: "No Attn", 5: "Shared Enc",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for eid in exp_ids:
        all_train, all_val_dice = [], []
        for fi in range(config.NUM_FOLDS):
            hpath = os.path.join(config.exp_dir(eid),
                                 f"fold_{fi}", "history.json")
            if not os.path.exists(hpath):
                continue
            with open(hpath) as f:
                h = json.load(f)
            all_train.append(h["train_loss"])
            all_val_dice.append(h["val_dice"])

        if not all_train:
            continue

        min_len = min(len(x) for x in all_train)
        train_arr = np.array([x[:min_len] for x in all_train])
        dice_arr = np.array([x[:min_len] for x in all_val_dice])

        label = short_labels.get(eid, str(eid))
        epochs = np.arange(1, min_len + 1)
        ax1.plot(epochs, train_arr.mean(0), label=label)
        ax2.plot(epochs, dice_arr.mean(0), label=label)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss")
    ax1.set_title("训练损失曲线")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Val Dice")
    ax2.set_title("验证 Dice 曲线")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "training_curves.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  已保存 training_curves.png")


def run_visualization():
    """实验六入口。"""
    print("\n" + "=" * 60)
    print("实验六：可视化分析")
    print("=" * 60)
    visualize_training_curves()
    visualize_comparison()
    print(f"\n可视化结果已保存至: {os.path.join(config.OUTPUT_DIR, 'visualizations')}")
