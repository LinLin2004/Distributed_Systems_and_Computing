import torch
import os
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from optimizers import optimize
import numpy as np
import matplotlib.ticker as ticker 
def load_data():
    dataset_dir = '/home/smbu/junior/Distributed_Systems_and_Computing/EX2' # 数据集存放目录
    data = fetch_california_housing(data_home=dataset_dir, download_if_missing=True)
    X = torch.tensor(data.data, dtype=torch.float32)
    y = torch.tensor(data.target, dtype=torch.float32).view(-1, 1)

    # 归一化
    X = (X - X.mean(dim=0)) / X.std(dim=0)

    return train_test_split(X, y, test_size=0.2, random_state=42)

def evaluate(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        preds = model(X_test)
        loss = torch.nn.functional.mse_loss(preds, y_test)
    return loss.item()

def search_learning_rate(args, model_fn, X_train, y_train, X_test, y_test, method, lr_list, num_iters=200):
    """
    自动搜索给定算法的最优学习率。
    返回：最优 lr，所有训练损失曲线，测试集误差。
    """
    losses = {}
    test_losses = {}

    for lr in lr_list:
        model = model_fn()
        train_losses = optimize(model, X_train, y_train, method, lr, radius=args.radius, l1_lambda=args.l1_lambda, gamma=args.gamma, num_iters=num_iters)
        final_loss = evaluate(model, X_test, y_test)
        losses[lr] = train_losses
        test_losses[lr] = round(final_loss, 8)  # 保留更高精度
    best_lr = min(test_losses, key=test_losses.get)
    return best_lr, losses, test_losses



def plot_lr_search(losses, test_losses, method, outdir="plots", show=False, title_prefix=""):
    """
    绘制不同学习率下的训练过程曲线图和测试误差散点图。
    """
    os.makedirs(outdir, exist_ok=True)
    sorted_lrs = sorted(losses.keys())

    # --- 1. 收敛曲线 ---
    plt.figure(figsize=(8, 5))
    for lr in sorted_lrs:
        loss_list = losses[lr]
        label = f"lr={lr:.3g} (Test={test_losses[lr]:.6f})"
        plt.plot(loss_list, label=label)

    plt.yscale("log")
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:.2e}"))

    plt.xlabel("Iterations")
    plt.ylabel("Training Loss (log scale)")
    plt.title(f"{title_prefix} LR Ablation: {method}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    filename1 = f"{outdir}/{method}_lr_ablation.png"
    plt.savefig(filename1)
    if show: plt.show()
    plt.close()

    # --- 2. 搜索散点图 ---
    plt.figure(figsize=(6, 5))
    lr_vals = np.array(sorted_lrs)
    mse_vals = np.array([test_losses[lr] for lr in lr_vals])
    plt.scatter(lr_vals, mse_vals, c='blue', edgecolors='k')

    best_lr = lr_vals[np.argmin(mse_vals)]
    best_mse = mse_vals.min()
    plt.scatter(best_lr, best_mse, c='red', label=f"Best LR = {best_lr:.3g}", zorder=5)

    # plt.xscale("log")
    plt.xlabel("Learning Rate")
    plt.ylabel("Final Test MSE")
    plt.title(f"{title_prefix} Search Result: {method}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    filename2 = f"{outdir}/{method}_scatter.png"
    plt.savefig(filename2)
    if show: plt.show()
    plt.close()


def plot_alg_comparison(results, test_losses, lr, outdir="plots", show=False):
    """
    比较不同算法在相同学习率下的收敛速度与最终测试性能。
    """
    os.makedirs(outdir, exist_ok=True)

    plt.figure(figsize=(8, 5))
    for method, loss_list in results.items():
        test_mse = test_losses[method]
        label = f"{method} (Test MSE={test_mse:.6f})"
        plt.plot(loss_list, label=label)

    plt.yscale('log')
    min_loss = min(min(lst) for lst in results.values())
    max_loss = max(max(lst) for lst in results.values())
    plt.ylim(bottom=min_loss * 0.8, top=max_loss * 1.2)

    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:.2e}"))

    plt.xlabel("Iterations")
    plt.ylabel("Training Loss (log scale)")
    # plt.title(f"Method Comparison @ lr={lr}")
    plt.title(f"Method Comparison with best par and lr")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # fname = f"{outdir}/method_comparison_lr{lr}.png"
    fname = f"{outdir}/method_comparison_best_par_and_lr.png"
    plt.savefig(fname)
    if show: plt.show()
    plt.close()


def plot_losses(results, test_losses, outdir="plots", show=False, tag=None):
    os.makedirs(outdir, exist_ok=True)
    plt.figure(figsize=(8, 5))
    for label, loss_list in results.items():
        test_mse = test_losses[label]
        plt.plot(loss_list, label=f"{label} (Test MSE={test_mse:.6f})")

    plt.yscale("log")
    plt.xlabel("Iterations")
    plt.ylabel("Training Loss")
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:.2e}"))
    plt.title("Training Curves with Test MSE")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    suffix = f"_{tag}" if tag else ""
    plt.savefig(f"{outdir}/training_curves{suffix}.png")
    if show: plt.show()
    plt.close()


def plot_hyperparam_ablation(losses_dict, test_losses_dict, method, lr, title_prefix="", outdir="plots"):
    os.makedirs(outdir, exist_ok=True)
    plt.figure(figsize=(8, 5))

    for param_val, loss_list in losses_dict.items():
        label = f"{title_prefix}{param_val} (Test: {test_losses_dict[param_val]:.4f})"
        plt.plot(loss_list, label=label)

    plt.xlabel("Iterations")
    plt.ylabel("Train Loss")
    plt.title(f"{method.upper()} Ablation (lr={lr})")
    plt.legend()
    plt.grid(True)
    fname = os.path.join(outdir, f"{method}_ablation_lr{lr}.png")
    plt.savefig(fname)
    plt.close()