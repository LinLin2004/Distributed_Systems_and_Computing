
# 手动实现多优化算法（不使用 PyTorch 内建 Optimizer） MLP 分类实验框架
本项目为基于 PyTorch 实现的 手动优化器实验框架，支持在 MNIST 数据集上测试多个优化算法，包括学习率搜索、算法比较、超参数消融等实验。

---

## 🚀 使用方式

在终端中运行：

```bash
python main.py [OPTIONS]
```

所有参数均可通过命令行指定。

---

## 📌 命令行参数说明

### 通用参数
| 参数名                   | 说明                            | 默认值                   |
| --------------------- | ----------------------------- | --------------------- |
| `--method` 或 `-m`     | 选择优化器（如 adam、rmsprop）         | `nesterov`            |
| `--lr` 或 `-r`         | 学习率（未设置则部分方法采用推荐值）            | `None`                |
| `--num-epochs` 或 `-e` | 训练轮数                          | `500`                 |
| `--batch-size` 或 `-b` | 批大小                           | `2048`                |
| `--dropout`           | MLP Dropout 概率                | `0.0`                 |
| `--use_bn`            | 是否使用 BatchNorm                | `False`               |
| `--hidden`            | 隐藏层结构（如 `--hidden 128 64 32`） | `[256, 128, 128, 64]` |

### 各方法专用参数
| 方法       | 可调参数      |
| -------- | --------- |
| Adam     | `--beta1` |
| Adagrad  | `--eps`   |
| RMSProp  | `--rhor`  |
| AdaDelta | `--rho`   |
| Nesterov | `--gamma` |


### 实验模式参数（互斥使用）

| 参数 | 类型 | 说明 |
|------|------|------|
| `--ablation` | flag | 对当前方法进行超参数消融实验 |
| `--compare` | flag | 比较所有优化方法表现 |
| `--search` | flag | 对所有方法执行学习率搜索 |

> 默认情况下（不设置 ablation / compare / search），将使用指定算法和学习率进行单次训练。

---

## 🧪 示例命令（参考但不限于）

### 普通训练（gd +学习率）
```bash
python main.py -m gd -r 0.01 --num-epochs 200
```

### 学习率搜索（所有方法）
```bash
python main.py --search
```

### 方法对比（固定学习率）
```bash
python main.py --compare -r 0.01
```

### adam 消融实验
```bash
python main.py --ablation --method adam --lr 0.1
```
### adagrad 消融实验
```bash
python main.py -m adagrad --ablation -r 0.1
```
### rmsprop 消融实验
```bash
python main.py -m rmsprop --ablation -r 0.01
```
### adadelta 消融实验
```bash
python main.py -m adadelta --ablation -r 0.01
```
### nesterov 消融实验 
```bash
python main.py -m nesterov --ablation -r 0.1
```
---

## 📂 输出结果

- 所有图像保存在 `plots/` 目录下。（也可在相应函数中修改保存路径）
- 可视化包括：收敛曲线、学习率搜索图、算法对比图、参数消融图等。
- 使用 `--use_bn` 和 `--dropout` 可观察正则化对结果的影响。

---

## 📎 推荐

- 配合 TensorBoard 可进一步可视化训练过程（日志在 `runs/`）。

---
