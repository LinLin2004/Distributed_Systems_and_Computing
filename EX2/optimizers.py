import torch

def soft_thresholding(w, lam):
    return torch.sign(w) * torch.clamp(torch.abs(w) - lam, min=0.0)

def project_onto_l2_ball(w, radius=1.0):
    norm = torch.norm(w)
    if norm > radius:
        return w / norm * radius
    return w

def optimize(model, X, y, method='gd', lr=None, num_iters=200, writer=None,
             radius=1.0, l1_lambda=0.01, gamma=0.9):
    losses = []
    W = list(model.parameters())
    v_prev = [torch.zeros_like(w) for w in W]  # for Nesterov

    
    # 根据消融实验结果手动设置最佳学习率
    recommended_lrs = {
        'gd': 0.0223,
        'nesterov': 0.0334,
        'fw': 0.0112,
        'proj': 0.0334,
        'prox': 0.0223
    }
    if lr is None:
        lr = recommended_lrs.get(method, 0.01)

    # print(method, lr, radius, l1_lambda, gamma)
    for i in range(num_iters):
        model.zero_grad()
        y_pred = model(X)
        loss = torch.nn.functional.mse_loss(y_pred, y)

        if method == 'prox':
            l1_penalty = sum(torch.norm(w, 1) for w in W)
            loss += l1_lambda * l1_penalty

        if method == 'fw':
            grad = torch.autograd.grad(loss, W, retain_graph=True, create_graph=False)
        else:
            loss.backward()

        with torch.no_grad():
            if method == 'gd':
                for w in W:
                    w -= lr * w.grad
            elif method == 'nesterov':
                for j, w in enumerate(W):
                    w -= lr * (w.grad + gamma * v_prev[j])
                    v_prev[j] = -lr * w.grad.clone()
            elif method == 'fw':
                for j, w in enumerate(W):
                    s = -grad[j]
                    w += lr * (s - w)
            elif method == 'proj':
                for w in W:
                    w -= lr * w.grad
                    w.copy_(project_onto_l2_ball(w, radius))
            elif method == 'prox':
                for w in W:
                    w -= lr * w.grad
                    w.copy_(soft_thresholding(w, lr * l1_lambda))

        if writer:
            writer.add_scalar('Train/Loss', loss.item(), i)
        losses.append(loss.item())

    return losses
