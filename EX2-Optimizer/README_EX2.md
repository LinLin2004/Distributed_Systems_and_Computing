
# 多优化算法 MLP 回归实验框架

本项目使用多种优化算法在 MLP 模型上拟合数据（如 California Housing），并支持不同运行模式如学习率搜索、算法对比、超参数消融实验等。所有优化算法均为手动实现，不依赖 PyTorch 内建优化器。

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

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `-m`, `--method` | str | `'gd'` | 选择优化算法（gd, nesterov, fw, proj, prox） |
| `-r`, `--lr` | float | None | 学习率，不设置时使用推荐值 |
| `-i`, `--num_iters` | int | 200 | 迭代次数 |
| `-b`, `--use_bn` | flag | False | 是否在隐藏层中使用 BatchNorm |
| `--dropout` | float | 0.0 | Dropout 比率 |
| `--hidden` | int list | `[256, 128, 128, 64]` | MLP 隐藏层结构 |

### 各方法专用参数

| 参数 | 类型 | 默认值 | 适用算法 | 说明 |
|------|------|--------|----------|------|
| `--radius` | float | 1.0 | proj | 投影半径（PGD） |
| `--l1_lambda` | float | 0.01 | prox | L1 正则系数（Prox） |
| `--gamma` | float | 0.9 | nesterov | 动量系数（NAG） |

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
python main.py -m gd -r 0.01
```

### 学习率搜索（所有方法）
```bash
python main.py --search
```

### 方法对比（固定学习率）
```bash
python main.py --compare -r 0.01
```

### Prox 消融实验（不同 l1_lambda）
```bash
python main.py -m prox --ablation -r 0.001
```

### PGD 半径消融实验
```bash
python main.py -m proj --ablation -r 0.01
```

### nesterov 消融实验 （不同 gamma）
```bash
python main.py -m nesterov --ablation -r 0.1
```
---

## 📂 输出结果

- 所有图像保存在 `plots/` 目录下。
- 可视化包括：收敛曲线、学习率搜索图、算法对比图、参数消融图等。
- 使用 `--use_bn` 和 `--dropout` 可观察正则化对结果的影响。

---

## 📎 推荐

- 配合 TensorBoard 可进一步可视化训练过程（日志在 `runs/`）。

---
