import torch
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from optimizers import optimize
import numpy as np
import matplotlib.ticker as ticker 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torchvision

def load_data(batch_size=1024):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    full_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    # train_size = int(0.8 * len(full_train))
    # val_size = len(full_train) - train_size
    # train_dataset, val_dataset = random_split(full_train, [train_size, val_size])
    train_dataset = full_train
    
    X_train = torch.stack([img for img, _ in train_dataset])
    Y_train = torch.tensor([label for _, label in train_dataset])
    
    X_test = torch.stack([img for img, _ in test_dataset])
    Y_test = torch.tensor([label for _, label in test_dataset])
    
    return X_train.view(-1, 784), Y_train, X_test.view(-1, 784), Y_test


def evaluate(model, X_test, y_test):
    correct = 0
    total = 0
    with torch.no_grad():
        # for X_batch, y_batch in test_loader:
        outputs = model(X_test.cuda())
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted.cpu() == y_test).sum().item()
        total += y_test.size(0)
    return correct / total

def search_learning_rate(args, model_fn, X_train, y_train, X_test, y_test, method, lr_list, num_epochs=200):
    """
    自动搜索最优学习率
    返回: (最佳学习率, 训练损失曲线字典, 测试准确率字典)
    """
    losses = {}
    test_accuracies = {}

    for lr in lr_list:
        model = model_fn()
        train_losses, _ = optimize(model, X_train, y_train, method, lr, num_epochs=num_epochs, args=args)
        final_acc = evaluate(model, X_test, y_test)
        losses[lr] = train_losses
        test_accuracies[lr] = round(final_acc, 4)  # 保留四位小数
        
    best_lr = max(test_accuracies, key=test_accuracies.get)
    print(args)  
    return best_lr, losses, test_accuracies

def plot_lr_search(losses, test_accuracies, method, outdir="plots_search", title_prefix=""):
    """绘制学习率搜索曲线"""
    os.makedirs(outdir, exist_ok=True)
    sorted_lrs = sorted(losses.keys())

    # 训练损失曲线
    plt.figure(figsize=(8, 5))
    for lr in sorted_lrs:
        plt.plot(losses[lr], label=f"lr={lr:.3g} (Acc={test_accuracies[lr]:.2%})")
        
    plt.yscale("log")
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:.2e}"))
    plt.xlabel("Epochs")
    plt.ylabel("Training Loss (log scale)")
    plt.title(f"{title_prefix}Learning Rate Search: {method.upper()}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{outdir}/{method}_lr_curve.png")
    plt.show()
    plt.close()

    # 准确率散点图
    plt.figure(figsize=(6, 5))
    lr_vals = np.array(sorted_lrs)
    acc_vals = np.array([test_accuracies[lr] for lr in lr_vals])
    
    plt.scatter(lr_vals, acc_vals, c='blue', edgecolors='k')
    best_idx = np.argmax(acc_vals)
    plt.scatter(lr_vals[best_idx], acc_vals[best_idx], c='red', 
                label=f"Best: lr={lr_vals[best_idx]:.3g}\nAcc={acc_vals[best_idx]:.2%}")
    
    plt.xlabel("Learning Rate")
    plt.ylabel("Test Accuracy")
    plt.ylim(0.5, 1.0)  # 限制Y轴范围
    plt.title(f"{title_prefix}Best Learning Rate: {method.upper()}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{outdir}/{method}_lr_scatter.png")
    plt.show()
    plt.close()

def plot_alg_comparison(results, test_accuracies, outdir="plots", lr = 0.01):
    """算法比较可视化"""
    os.makedirs(outdir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    
    for method, loss_list in results.items():
        plt.plot(loss_list, label=f"{method.upper()} (Acc={test_accuracies[method]:.2%})")
    
    plt.yscale('log')
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:.2e}"))
    plt.xlabel("epochs")
    plt.ylabel("Training Loss (log scale)")
    plt.title(f"Optimizer Comparison @ lr={lr}")
    # plt.title(f"Method Comparison with best par and lr")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{outdir}/method_comparison_lr{lr}.png")
    # plt.savefig(f"{outdir}/method_comparison_best_par_and_lr.png")
    plt.show()
    plt.close()

def plot_hyperparam_ablation(losses_dict, test_acc_dict, method, lr, 
                            title_prefix="", outdir="plots"):
    """超参数消融实验可视化"""
    os.makedirs(outdir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    
    for param_val, loss_list in losses_dict.items():
        plt.plot(loss_list, 
                label=f"{title_prefix}{param_val} (Acc={test_acc_dict[param_val]:.2%})")
    
    plt.xlabel("Epochs")
    plt.ylabel("Training Loss")
    plt.title(f"{method.upper()} Hyperparameter Ablation (lr={lr})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    fname = os.path.join(outdir, f"{method}_ablation_lr{lr}.png")
    plt.savefig(fname)
    plt.close()

def plot_losses(results, test_accuracies, outdir="plots", show=False, tag=None):
    """训练过程可视化"""
    os.makedirs(outdir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    
    for label, loss_list in results.items():
        plt.plot(loss_list, label=f"{label} (Acc={test_accuracies[label]:.2%})")
    
    plt.yscale("log")
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:.2e}"))
    plt.xlabel("Epochs")
    plt.ylabel("Training Loss")
    plt.title("Training Curves with Final Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    suffix = f"_{tag}" if tag else ""
    plt.savefig(f"{outdir}/training_curves{suffix}.png")
    plt.show()
    plt.close()