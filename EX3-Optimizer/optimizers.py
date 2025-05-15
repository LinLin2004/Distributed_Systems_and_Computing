import torch
import torch.nn as nn
import math
from tqdm import tqdm

def optimize(model, X_train, y_train, method='adam', lr=0.01, num_epochs=100, args=None):

    """
    支持方法:  nesterov, adagrad, rmsprop, adadelta, adam
    """
    losses = []
    accuracies = []
    W = list(model.parameters()) 
    
    # 初始化优化器状态变量
    v = [torch.zeros_like(w) for w in W]  # 动量（Nesterov）
    m = [torch.zeros_like(w) for w in W]  # 一阶矩估计（Adam）
    u = [torch.zeros_like(w) for w in W]  # 二阶矩估计（AdaGrad/RMSProp/Adam）
    delta = [torch.zeros_like(w) for w in W]  # AdaDelta状态变量
    
    # 超参数设置
    gamma = args.gamma
    rhor = args.rhor
    epsilon = 1e-8  # 避免除零错误
    eps = args.eps      # AdaGra小常数
    rho = args.rho        # RMSProp/AdaDelta衰减率
    beta1 = args.beta1    # Adam一阶矩衰减率
    beta2 = args.beta2    # Adam二阶矩衰减率
    criterion = nn.CrossEntropyLoss()

    # 默认学习率设置
    if lr is None:
        lr = {
            # 'gd': 0.01,
            'nesterov':  0.0667,
            'adagrad': 0.0112,
            'rmsprop': 0.0001,
            'adadelta': 0.0001,
            'adam': 0.0001 
        }.get(method, 0.01)
    
    losses = []
    for e in tqdm(list(range(1, num_epochs + 1))):
        batch_size = args.batch_size
        num_iters = math.ceil(len(X_train) / batch_size)
        e_losses = 0
        for t in range(num_iters):
            start_idx = t * batch_size
            b_X_train = X_train[start_idx:start_idx + batch_size].cuda()
            b_y_train = y_train[start_idx:start_idx + batch_size].cuda()
            
            outputs = model(b_X_train)
            loss = criterion(outputs, b_y_train)

            _, predicted = torch.max(outputs.data, 1)
            acc = (predicted == b_y_train).float().mean()
            accuracies.append(acc.item())

            model.zero_grad() 
            loss.backward()
            
            # 参数更新逻辑
            with torch.no_grad():
                for i, w in enumerate(W):
                    g = w.grad  # 当前参数梯度
                    if method == 'gd':
                        # 标准梯度下降
                        w -= lr * g
                    
                    elif method == 'nesterov':
                        # Nesterov动量
                        # # # 计算预测点梯度（实际在w_k处计算）
                        v_prev = v[i].clone()
                        v[i] = w - lr * g  # v_{k+1} = w_k - η∇f(w_k)
                        w.data = v[i] + gamma * (v[i] - v_prev)  # w_{k+1} = v_{k+1} + γ(v_{k+1} - v_k)
                
                    elif method == 'adagrad':
                        # AdaGrad自适应学习率
                        u[i] += g ** 2
                        w -= lr * g / (torch.sqrt(u[i]) + eps)
                    
                    elif method == 'rmsprop':
                        # RMSProp自适应学习率
                        u[i] = rhor * u[i] + (1 - rhor) * g ** 2
                        w -= lr * g / (torch.sqrt(u[i]) + epsilon)
                    
                    elif method == 'adadelta':
                        # AdaDelta自适应学习率
                        u[i] = rho * u[i] + (1 - rho) * g ** 2
                        update = (torch.sqrt(delta[i] + epsilon) / torch.sqrt(u[i] + epsilon)) * g
                        w -= update
                        delta[i] = rho * delta[i] + (1 - rho) * update ** 2
                    
                    elif method == 'adam':
                        # Adam自适应学习率
                        m[i] = beta1 * m[i] + (1 - beta1) * g
                        u[i] = beta2 * u[i] + (1 - beta2) * g ** 2
                        m_hat = m[i] / (1 - beta1 ** (t + 1))  # 为当前迭代次数t
                        u_hat = u[i] / (1 - beta2 ** (t + 1))
                        w -= lr * m_hat / (torch.sqrt(u_hat) + epsilon)
            # 记录训练指标
            e_losses += loss.item()
        losses.append(e_losses / num_iters)
    print("gamma =", gamma, "eps =", eps,"epsilon =", epsilon, "rhor =", rhor,"rho =", rho, "beta1 =", beta1)
        
    return losses, accuracies