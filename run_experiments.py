"""
主入口脚本——通过命令行参数控制执行哪个实验。

用法：
  # 第 0 步：预处理数据（首次运行必须执行一次）
  python run_experiments.py --preprocess

  # 第 1 步：训练实验 1-5（可并行或依次执行）
  python run_experiments.py --experiment 1
  python run_experiments.py --experiment 2
  python run_experiments.py --experiment 3
  python run_experiments.py --experiment 4
  python run_experiments.py --experiment 5

  # 只跑某一折
  python run_experiments.py --experiment 3 --fold 0

  # 调整超参数
  python run_experiments.py --experiment 3 --epochs 150 --lr 5e-5 --batch_size 8

  # 第 2 步：可视化（实验六）
  python run_experiments.py --experiment 6

  # 第 3 步：统计分析（实验七）
  python run_experiments.py --experiment 7
"""
import config


def main():
    args = config.parse_args()

    if args.preprocess:
        print("开始预处理 CHAOS MRI 数据...")
        from dataset import preprocess_all
        preprocess_all()
        print("预处理完成！")
        return

    if args.experiment is None:
        print("请指定 --preprocess 或 --experiment N（N=1-7）")
        print("示例: python run_experiments.py --experiment 1")
        return

    exp = args.experiment

    if 1 <= exp <= 5:
        from train import run_training_experiment
        run_training_experiment(exp, args)

    elif exp == 6:
        from visualize import run_visualization
        run_visualization()

    elif exp == 7:
        from stats import run_statistics
        run_statistics()

    else:
        print(f"未知实验编号: {exp}")


if __name__ == "__main__":
    main()
