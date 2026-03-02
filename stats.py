"""
实验七：统计分析——显著性检验、汇总表格、箱线图。
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, ttest_rel

import config
from train import aggregate_results

plt.rcParams["font.family"] = "SimHei"
plt.rcParams["axes.unicode_minus"] = False

SHORT = {
    1: "T2 UNet",
    2: "Early Fusion",
    3: "Attn MBranch",
    4: "No Attn",
    5: "Shared Enc",
}


def load_all_experiment_metrics(exp_ids=(1, 2, 3, 4, 5)):
    """加载所有实验的 per-case 指标。"""
    results = {}
    for eid in exp_ids:
        path = os.path.join(config.exp_dir(eid), "all_case_metrics.json")
        if not os.path.exists(path):
            m = aggregate_results(eid)
        else:
            with open(path) as f:
                m = {int(k): v for k, v in json.load(f).items()}
        if m:
            results[eid] = m
    return results


def _extract_metric(metrics_dict, case_ids, metric="dice"):
    return [metrics_dict[c][metric] for c in case_ids if c in metrics_dict]


# ======================== 显著性检验 ========================

def pairwise_tests(results, metric="dice"):
    """
    对实验 3（核心方法）与其余实验做配对 Wilcoxon 和 t-test。
    """
    if 3 not in results:
        print("[WARN] 实验 3 结果不存在，无法做对比")
        return {}

    ref = results[3]
    common_cases = sorted(ref.keys())
    ref_vals = _extract_metric(ref, common_cases, metric)

    comparisons = {}
    pairs = [
        (1, "核心方法 vs T2 基线"),
        (2, "核心方法 vs 早期拼接"),
        (4, "核心方法 vs 无注意力（消融）"),
        (5, "核心方法 vs 共享编码器（消融）"),
    ]

    for eid, desc in pairs:
        if eid not in results:
            continue
        cases_both = [c for c in common_cases if c in results[eid]]
        a = [ref[c][metric] for c in cases_both]
        b = [results[eid][c][metric] for c in cases_both]

        if len(a) < 5:
            print(f"  {desc}: 样本不足 ({len(a)})，跳过")
            continue

        diff = np.array(a) - np.array(b)
        if np.all(diff == 0):
            print(f"  {desc}: 两组完全相同，跳过检验")
            continue

        try:
            w_stat, w_p = wilcoxon(a, b)
        except ValueError:
            w_stat, w_p = float("nan"), float("nan")
        t_stat, t_p = ttest_rel(a, b)

        comp = {
            "description": desc,
            "n": len(a),
            "exp3_mean": float(np.mean(a)),
            "other_mean": float(np.mean(b)),
            "mean_diff": float(np.mean(diff)),
            "wilcoxon_stat": float(w_stat),
            "wilcoxon_p": float(w_p),
            "ttest_stat": float(t_stat),
            "ttest_p": float(t_p),
        }
        comparisons[eid] = comp
        sig = "***" if w_p < 0.001 else ("**" if w_p < 0.01 else
              ("*" if w_p < 0.05 else "n.s."))
        print(f"  {desc}: diff={comp['mean_diff']:+.4f}, "
              f"Wilcoxon p={w_p:.4f} {sig}, t-test p={t_p:.4f}")

    return comparisons


# ======================== 汇总表格 ========================

def print_summary_table(results, metric="dice"):
    """打印所有实验的汇总指标表格。"""
    print(f"\n{'实验':<20s} {'Mean':>8s} {'Std':>8s} "
          f"{'Min':>8s} {'Max':>8s} {'N':>5s}")
    print("-" * 55)
    for eid in sorted(results.keys()):
        vals = list(m[metric] for m in results[eid].values())
        name = SHORT.get(eid, str(eid))
        print(f"{name:<20s} {np.mean(vals):8.4f} {np.std(vals):8.4f} "
              f"{np.min(vals):8.4f} {np.max(vals):8.4f} {len(vals):5d}")


# ======================== 箱线图 ========================

def plot_boxplots(results, save_dir=None):
    """为 Dice 和 IoU 生成箱线图。"""
    if save_dir is None:
        save_dir = os.path.join(config.OUTPUT_DIR, "statistics")
    os.makedirs(save_dir, exist_ok=True)

    for metric in ("dice", "iou"):
        fig, ax = plt.subplots(figsize=(10, 6))
        data, labels = [], []
        for eid in sorted(results.keys()):
            vals = [m[metric] for m in results[eid].values()]
            data.append(vals)
            labels.append(SHORT.get(eid, str(eid)))

        bp = ax.boxplot(data, labels=labels, patch_artist=True, showmeans=True)
        colors = ["#8dd3c7", "#ffffb3", "#fb8072", "#bebada", "#80b1d3"]
        for patch, color in zip(bp["boxes"], colors[:len(data)]):
            patch.set_facecolor(color)

        ax.set_ylabel(metric.upper())
        ax.set_title(f"各实验 {metric.upper()} 对比")
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, f"boxplot_{metric}.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  已保存 boxplot_{metric}.png")


# ======================== 生成 LaTeX 表格 ========================

def generate_latex_table(results, comparisons, save_dir=None):
    if save_dir is None:
        save_dir = os.path.join(config.OUTPUT_DIR, "statistics")
    os.makedirs(save_dir, exist_ok=True)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{各方法在 CHAOS MRI 数据集上的肝脏分割结果}",
        r"\label{tab:results}",
        r"\begin{tabular}{lcccc}",
        r"\hline",
        r"方法 & Dice(\%) & IoU(\%) & Precision(\%) & Recall(\%) \\",
        r"\hline",
    ]
    for eid in sorted(results.keys()):
        metrics = list(results[eid].values())
        name = SHORT.get(eid, str(eid))
        d = np.mean([m["dice"] for m in metrics]) * 100
        iou = np.mean([m["iou"] for m in metrics]) * 100
        p = np.mean([m["precision"] for m in metrics]) * 100
        r = np.mean([m["recall"] for m in metrics]) * 100
        d_s = np.std([m["dice"] for m in metrics]) * 100
        iou_s = np.std([m["iou"] for m in metrics]) * 100
        p_s = np.std([m["precision"] for m in metrics]) * 100
        r_s = np.std([m["recall"] for m in metrics]) * 100
        bold = eid == 3
        fmt = r"\textbf{%.2f$\pm$%.2f}" if bold else r"%.2f$\pm$%.2f"
        cells = [fmt % (v, s) for v, s in [(d, d_s), (iou, iou_s),
                                            (p, p_s), (r, r_s)]]
        lines.append(f"{name} & {' & '.join(cells)} \\\\")

    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    text = "\n".join(lines)

    out = os.path.join(save_dir, "results_table.tex")
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  已保存 results_table.tex")
    return text


# ======================== 入口 ========================

def run_statistics():
    print("\n" + "=" * 60)
    print("实验七：统计分析")
    print("=" * 60)

    results = load_all_experiment_metrics()
    if not results:
        print("未找到任何实验结果，请先运行实验 1-5。")
        return

    save_dir = os.path.join(config.OUTPUT_DIR, "statistics")
    os.makedirs(save_dir, exist_ok=True)

    print("\n--- Dice 汇总 ---")
    print_summary_table(results, "dice")
    print("\n--- IoU 汇总 ---")
    print_summary_table(results, "iou")

    print("\n--- 配对显著性检验 (Dice) ---")
    comp = pairwise_tests(results, "dice")
    with open(os.path.join(save_dir, "significance_tests.json"), "w") as f:
        json.dump({str(k): v for k, v in comp.items()}, f, indent=2)

    print("\n--- 箱线图 ---")
    plot_boxplots(results, save_dir)

    print("\n--- LaTeX 表格 ---")
    generate_latex_table(results, comp, save_dir)

    print(f"\n统计结果已保存至: {save_dir}")
