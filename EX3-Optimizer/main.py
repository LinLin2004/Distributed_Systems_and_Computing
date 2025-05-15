import argparse
import os
import datetime
import numpy as np
# from torch.utils.tensorboard import SummaryWriter
from model import MLP
from optimizers import optimize
from utils import plot_hyperparam_ablation, plot_losses, load_data, evaluate, search_learning_rate, plot_lr_search, plot_alg_comparison


def run_training(args, X_train, y_train, X_test, y_test, input_dim):
    lr_list = [args.lr]
    if args.ablation:
        lr_list = [0.001, 0.01, 0.1]

    results = {}
    test_accuracies = {}  
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    for lr in lr_list:
        model = MLP(input_dim, args.hidden, args.dropout, use_bn=args.use_bn).cuda()
        log_dir = os.path.join("runs", f"{args.method}_lr{lr}_{timestamp}")
        # writer = SummaryWriter(log_dir)
        losses, _ = optimize(model, X_train, y_train, 
                           method=args.method, 
                           lr=lr,
                           num_epochs=args.num_epochs, 
                           args=args)
        # writer.close()
        acc = evaluate(model, X_test, y_test)
        results[f'{args.method}_lr={lr}'] = losses
        test_accuracies[f'{args.method}_lr={lr}'] = acc  

    plot_losses(results, test_accuracies) 

def run_lr_search(args, X_train, y_train, X_test, y_test, input_dim):
    lr_list = np.linspace(1e-4, 0.1, 10)
    methods = ['gd','adam', 'adagrad', 'rmsprop', 'adadelta', 'nesterov']
    
    for method in methods:
        def model_fn():
            return MLP(input_dim, args.hidden, args.dropout, use_bn=args.use_bn).cuda()

        best_lr, losses, test_accuracies = search_learning_rate(
            args, model_fn, X_train, y_train, X_test, y_test,
            method, lr_list, num_epochs=args.num_epochs
        )
        print(f"[{method}] Best LR = {best_lr:.4g} | Test Acc = {test_accuracies[best_lr]:.2%}")
        plot_lr_search(losses, test_accuracies, method)  # 传递准确率数据

def run_alg_comparison(args, X_train, y_train, X_test, y_test, input_dim):
    methods = ['adam', 'adagrad', 'rmsprop', 'adadelta', 'nesterov']
    results = {}
    test_accuracies = {}  

    for method in methods:
        model = MLP(input_dim, args.hidden, args.dropout, use_bn=args.use_bn).cuda()
        losses, _ = optimize(model, X_train, y_train,
                           method=method,
                           lr=args.lr,
                           num_epochs=args.num_epochs,
                           args=args)
        acc = evaluate(model, X_test, y_test)
        results[method] = losses
        test_accuracies[method] = acc
    print(args)
    plot_alg_comparison(results, test_accuracies,lr = args.lr)  # 传递准确率

def run_hyperparam_ablation(args, X_train, y_train, X_test, y_test, input_dim):
    method = args.method
    lr = args.lr 
    losses_dict = {}
    test_acc_dict = {}  

    def model_fn():
        return MLP(input_dim, args.hidden, args.dropout, use_bn=args.use_bn).cuda()

    if method == 'adam':
        beta1_list = [0.5, 0.7, 0.9, 0.99]
        for beta1 in beta1_list:
            model = model_fn()
            args.beta1 = beta1
            losses, _ = optimize(model, X_train, y_train, method='adam', lr=lr,
                                num_epochs=args.num_epochs, args=args)
            test_acc_dict[beta1] = evaluate(model, X_test, y_test)
            losses_dict[beta1] = losses

    elif method == 'adagrad':
        eps_list = [1e-6, 1e-5, 1e-4, 1e-3]
        for eps in eps_list:
            model = model_fn()
            args.eps = eps
            losses, _ = optimize(model, X_train, y_train, method='adagrad', lr=lr,
                                num_epochs=args.num_epochs, args=args)
            test_acc_dict[eps] = evaluate(model, X_test, y_test)
            losses_dict[eps] = losses

    elif method == 'rmsprop':
        rhor_list = [0.8, 0.9, 0.95, 0.99]
        for rhor in rhor_list:
            model = model_fn()
            args.rhor = rhor
            losses, _ = optimize(model, X_train, y_train, method='rmsprop', lr=lr,
                                num_epochs=args.num_epochs, args=args)
            test_acc_dict[rhor] = evaluate(model, X_test, y_test)
            losses_dict[rhor] = losses

    elif method == 'adadelta':
        rho_list = [0.8, 0.9, 0.95, 0.99]
        for rho in rho_list:
            model = model_fn()
            args.rho = rho
            
            losses, _ = optimize(model, X_train, y_train, method='adadelta', lr=lr,
                                num_epochs=args.num_epochs, args=args)
            test_acc_dict[rho] = evaluate(model, X_test, y_test)
            losses_dict[rho] = losses

    elif method == 'nesterov':
        gamma_list = [0.5, 0.7, 0.9, 0.99]
        for gamma in gamma_list:
            model = model_fn()
            args.gamma = gamma
            losses, _ = optimize(model, X_train, y_train, method='nesterov', lr=lr,
                                num_epochs=args.num_epochs,args=args)
            test_acc_dict[gamma] = evaluate(model, X_test, y_test)
            losses_dict[gamma] = losses
    else:
        print(f"[Error] Unsupported method: {method}")
        return

    plot_hyperparam_ablation(losses_dict, test_acc_dict, method, lr)
    best_param = max(test_acc_dict, key=test_acc_dict.get)
    best_acc = test_acc_dict[best_param]
    param_name = {
        "adam": "beta1",
        "adagrad": "eps",
        "rmsprop": "rhor",
        "adadelta": "rho",
        "nesterov": "gamma"
    }[method]

    print(f"Best {param_name} for {method}: {best_param} (Accuracy: {best_acc:.2%})")
    
def main(args):
    X_train, y_train, X_test, y_test = load_data()  # 注意load_data返回4个值
    input_dim = X_train.shape[1]

    if args.search:
        run_lr_search(args, X_train, y_train, X_test, y_test, input_dim)
    elif args.compare:
        run_alg_comparison(args, X_train, y_train, X_test, y_test, input_dim)
    elif args.ablation:
        if args.method in ['adam', 'adagrad', 'rmsprop', 'adadelta', 'nesterov']:
            run_hyperparam_ablation(args, X_train, y_train, X_test, y_test, input_dim)
        else:
            run_training(args, X_train, y_train, X_test, y_test, input_dim)
    else:
        run_training(args, X_train, y_train, X_test, y_test, input_dim)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 各方法专用参数
    parser.add_argument('--gamma', type=float, default=0.9, help="Momentum for Nesterov")
    parser.add_argument('--rhor', type=float, default=0.9, help="Momentum for RMSProp")
    parser.add_argument('--beta1', type=float, default=0.9, help="Beta1 for Adam")
    parser.add_argument('--beta2', type=float, default=0.999, help="Beta2 for Adam")
    parser.add_argument('--eps', type=float, default=1e-8, help="eps for Adagrad")
    parser.add_argument('--rho', type=float, default=0.95, help="Rho for AdaDelta")
    # 通用参数
    parser.add_argument('-m', '--method', type=str, default='nesterov',
            choices=[ 'gd','adam', 'adagrad', 'rmsprop', 'adadelta','nesterov'],
            help="Optimization algorithm to use")
    parser.add_argument('-r', '--lr', type=float, default=None,
                    help="Learning rate (default: use recommended value for each method)")
    parser.add_argument('-b', '--batch-size', type=int, default=2048)
    parser.add_argument('-e', '--num-epochs', type=int, default=500,
                        help="Number of optimization epochs")
    parser.add_argument('-bn', '--use_bn', action='store_true',
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