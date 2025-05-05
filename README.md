
# 实验框架命令行参数说明

## 使用方式

```bash
python main.py [OPTIONS]
```

---

## 📌 通用参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `-m`, `--method` | str | `'gd'` | 选择优化算法：`gd`、`nesterov`、`fw`、`proj`、`prox` |
| `-r`, `--lr` | float | `None` | 学习率。若不指定，则使用每种算法推荐值 |
| `-i`, `--num_iters` | int | `200` | 最大迭代次数 |
| `-b`, `--use_bn` | flag | False | 是否启用 Batch Normalization |
| `--dropout` | float | `0.0` | Dropout 比例 |
| `--hidden` | int list | `[256, 128, 128, 64]` | 隐藏层结构（可变层数） |

---

## 🔧 各方法专用参数

| 参数 | 类型 | 默认值 | 适用方法 | 说明 |
|------|------|--------|-----------|------|
| `--gamma` | float | `0.9` | `nesterov` | Nesterov 动量项 |
| `--radius` | float | `1.0` | `proj` | 投影半径（L2 Ball） |
| `--l1_lambda` | float | `0.01` | `prox` | L1 正则化系数 |

---

## 🎯 实验模式（互斥使用）

| 参数 | 类型 | 默认值 | 功能说明 |
|------|------|--------|----------|
| `--ablation` | flag | False | 对**当前方法**进行参数消融实验（如 gamma、lambda、radius） |
| `--compare` | flag | False | 使用固定学习率，对比所有优化算法表现 |
| `--search` | flag | False | 对所有方法执行自动学习率搜索（log space） |

> **备注：** 未设置上述三项时默认运行普通训练流程（单一算法、指定学习率）。

---

## 🧪 示例命令

### 1. 普通训练（使用 Nesterov，lr=0.01）
```bash
python main.py -m nesterov -r 0.01
```

### 2. 自动学习率搜索
```bash
python main.py --search
```

### 3. 方法对比（固定学习率）
```bash
python main.py --compare -r 0.01
```

### 4. Proximal 方法的 λ 消融实验
```bash
python main.py -m prox --ablation -r 0.01
```

### 5. PGD 投影半径消融实验
```bash
python main.py -m proj --ablation -r 0.001
```
