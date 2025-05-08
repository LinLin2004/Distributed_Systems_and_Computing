import argparse
import os
import datetime
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from model import MLP
from optimizers import optimize
from utils import plot_hyperparam_ablation,plot_losses, load_data, evaluate, search_learning_rate, plot_lr_search, plot_alg_comparison

def run_training(args, X_train, y_train, X_test, y_test, input_dim):
    lr_list = [args.lr]
    if args.ablation:
        lr_list = [0.001, 0.01, 0.1]

    results = {}
    test_losses = {}
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    for lr in lr_list:
        model = MLP(input_dim, args.hidden, args.dropout, use_bn=args.use_bn)

        log_dir = os.path.join("runs", f"{args.method}_lr{lr}_{timestamp}")
        writer = SummaryWriter(log_dir)

        losses = optimize(model, X_train, y_train, method=args.method, lr=lr,
                          num_iters=args.num_iters, writer=writer,
                          radius=args.radius, l1_lambda=args.l1_lambda, gamma=args.gamma)

        writer.close()

        test_loss = evaluate(model, X_test, y_test)
        results[f'{args.method}_lr={lr}'] = losses
        test_losses[f'{args.method}_lr={lr}'] = test_loss

    plot_losses(results, test_losses)


def run_lr_search(args, X_train, y_train, X_test, y_test, input_dim):
    lr_list = np.linspace(1e-4, 0.1, 10)
    methods = ['gd', 'nesterov', 'fw', 'proj', 'prox']
    for method in methods:
        def model_fn():
            return MLP(input_dim, args.hidden, args.dropout, use_bn=args.use_bn)

        best_lr, losses, test_losses = search_learning_rate(args,
            model_fn, X_train, y_train, X_test, y_test,
            method, lr_list, num_iters=args.num_iters
        )
        print(f"[{method}] Best LR = {best_lr:.4g} | Test MSE = {test_losses[best_lr]:.4f}")
        plot_lr_search(losses, test_losses, method)


def run_alg_comparison(args, X_train, y_train, X_test, y_test, input_dim):
    methods = ['gd', 'nesterov', 'fw', 'proj', 'prox']
    results = {}
    test_losses = {}

    for method in methods:
        model = MLP(input_dim, args.hidden, args.dropout, use_bn=args.use_bn)
        losses = optimize(model, X_train, y_train,
                          method=method,
                          lr=args.lr,
                          num_iters=args.num_iters,
                          radius=args.radius,
                          l1_lambda=args.l1_lambda,
                          gamma=args.gamma)
        test_loss = evaluate(model, X_test, y_test)
        results[method] = losses
        test_losses[method] = test_loss

    plot_alg_comparison(results, test_losses, lr=args.lr)


def run_hyperparam_ablation(args, X_train, y_train, X_test, y_test, input_dim):
    os.makedirs("plots", exist_ok=True)

    if args.method == 'nesterov':
        gamma_list = [0.3, 0.5, 0.7, 0.9, 0.99]
        losses = {}
        test_losses = {}
        for gamma in gamma_list:
            model = MLP(input_dim, args.hidden, args.dropout, use_bn=args.use_bn)
            loss_list = optimize(model, X_train, y_train,
                                 method='nesterov',
                                 lr=args.lr,
                                 gamma=gamma,
                                 num_iters=args.num_iters)
            mse = evaluate(model, X_test, y_test)
            losses[gamma] = loss_list
            test_losses[gamma] = mse

        plot_hyperparam_ablation(losses, test_losses, method='nesterov',
                                 lr=args.lr, title_prefix="gamma=", outdir="plots")

        best_gamma = min(test_losses, key=test_losses.get)
        best_mse = test_losses[best_gamma]
        print(f"Best gamma for nesterov: {best_gamma} (Test MSE = {best_mse:.4f})")
        with open("plots/nesterov_best_gamma.txt", "w") as f:
            f.write(f"Best gamma for nesterov: {best_gamma}\n")
            f.write(f"Test MSE: {best_mse:.6f}\n")

    elif args.method == 'prox':
        lambda_list = [0.0, 0.001, 0.01, 0.1, 1.0]
        losses = {}
        test_losses = {}
        for l1_lambda in lambda_list:
            model = MLP(input_dim, args.hidden, args.dropout, use_bn=args.use_bn)
            loss_list = optimize(model, X_train, y_train,
                                 method='prox',
                                 lr=args.lr,
                                 l1_lambda=l1_lambda,
                                 num_iters=args.num_iters)
            mse = evaluate(model, X_test, y_test)
            losses[l1_lambda] = loss_list
            test_losses[l1_lambda] = mse

        plot_hyperparam_ablation(losses, test_losses, method='prox',
                                 lr=args.lr, title_prefix="l1_lambda=", outdir="plots")

        # 自动选择最佳 lambda
        best_lambda = min(test_losses, key=test_losses.get)
        best_mse = test_losses[best_lambda]
        print(f"Best λ for prox: {best_lambda} (Test MSE = {best_mse:.4f})")
        with open("plots/prox_best_lambda.txt", "w") as f:
            f.write(f"Best λ for prox: {best_lambda}\n")
            f.write(f"Test MSE: {best_mse:.6f}\n")

    elif args.method == 'proj':
        radius_list = [0.01, 0.1, 0.5, 1.0, 5.0]
        losses = {}
        test_losses = {}
        for radius in radius_list:
            model = MLP(input_dim, args.hidden, args.dropout, use_bn=args.use_bn)
            loss_list = optimize(model, X_train, y_train,
                                 method='proj',
                                 lr=args.lr,
                                 radius=radius,
                                 num_iters=args.num_iters)
            mse = evaluate(model, X_test, y_test)
            losses[radius] = loss_list
            test_losses[radius] = mse

        plot_hyperparam_ablation(losses, test_losses, method='proj',
                                 lr=args.lr, title_prefix="radius=", outdir="plots")

        # 自动选择最佳 radius
        best_radius = min(test_losses, key=test_losses.get)
        best_mse = test_losses[best_radius]
        print(f"Best radius for proj: {best_radius} (Test MSE = {best_mse:.4f})")
        with open("plots/proj_best_radius.txt", "w") as f:
            f.write(f"Best radius for proj: {best_radius}\n")
            f.write(f"Test MSE: {best_mse:.6f}\n")


def main(args):
    X_train, X_test, y_train, y_test = load_data()
    input_dim = X_train.shape[1]

    if args.search:
        run_lr_search(args, X_train, y_train, X_test, y_test, input_dim)
    elif args.compare:
        run_alg_comparison(args, X_train, y_train, X_test, y_test, input_dim)
    elif args.ablation:
        if args.method in ['nesterov', 'prox', 'proj']:
            run_hyperparam_ablation(args, X_train, y_train, X_test, y_test, input_dim)
        else:
            run_training(args, X_train, y_train, X_test, y_test, input_dim)
    else:
        run_training(args, X_train, y_train, X_test, y_test, input_dim)


if __name__ == "__main__":
    import torch
    import random
    import numpy as np
    torch.random.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    parser = argparse.ArgumentParser()
    # 各方法专用参数
    parser.add_argument('--radius', type=float, default=1.0,
                        help="Radius for projection in projected gradient descent (PGD)")
    parser.add_argument('--l1_lambda', type=float, default=0.01,
                        help="L1 regularization lambda for proximal gradient descent")
    parser.add_argument('--gamma', type=float, default=0.9,
                        help="Momentum term for Nesterov accelerated gradient")
    # 通用参数
    parser.add_argument('-m', '--method', type=str, default='gd',
                        choices=['gd', 'nesterov', 'fw', 'proj', 'prox'],
                        help="Optimization algorithm to use")
    parser.add_argument('-r', '--lr', type=float, default=None,
                    help="Learning rate (default: use recommended value for each method)")
    parser.add_argument('-i', '--num_iters', type=int, default=200,
                        help="Number of optimization iterations")
    parser.add_argument('-b', '--use_bn', action='store_true',
                        help='Use BatchNorm in hidden layers')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate for MLP')
    parser.add_argument('--hidden', type=int, nargs='+', default=[256, 128, 128, 64],
                        help='Hidden layer sizes')
    # 实验模式
    parser.add_argument('--ablation', action='store_true',
                        help="Run with multiple learning rates for ablation")
    parser.add_argument('--compare', action='store_true',
                        help="Compare methods")
    parser.add_argument('--search', action='store_true',
                        help="Perform auto learning rate search across methods")

    args = parser.parse_args()
    main(args)
# [警告] 同时启用 --search 和 --ablation，仅执行 search。