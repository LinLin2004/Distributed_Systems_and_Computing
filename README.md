
# 📘 Distributed Systems and Computing Experiments

本仓库包含三个实验任务，分别聚焦于**网络通信编程（Socket）**、**确定性优化算法（PyTorch 实现）**、**随机性优化算法（PyTorch 实现）**，用于课程《分布式系统与计算》相关实验内容。

---

## 🧪 实验一：基于 Python 的网络通信编程

### 🎯 实验目标

在Python编程环境中完成基于 Socket 的通信协议及服务：

- UDP 单向通信
- UDP 双向通信
- UDP 模拟 Daytime 协议服务
- TCP 双向通信
- TCP 单向通信
- TCP 实现 Daytime 协议

### 📂 目录结构（EX1-network_lab/）

```

EX1-network_lab/
├── tcp_daytime.py              # TCP实现的Daytime协议
├── tcp_one-way_commun.py       # TCP单向通信
├── tcp_two-way_server.py       # TCP双向通信服务端
├── tcp_two-way_client.py       # TCP双向通信客户端
├── udp_daytime.py              # UDP实现的Daytime协议
├── udp_one-way_commun.py       # UDP单向通信
└── udp_two-way_commun.py       # UDP双向通信
```

---

## 🧪 实验二：确定性优化算法实现与对比（EX2-Optimizer）

### 🎯 实验目标及要求

使用手写方式实现以下优化算法并完成训练过程可视化与超参数对比：

- 梯度下降法 (GD)
- Nesterov 加速法 (NAG)
- Frank-Wolfe 算法 (FW)
- 投影次梯度下降法 (PGD)
- 近端梯度下降法 (Proximal GD)

### 

- 使用 California Housing 数据
- 手动实现优化器，无 PyTorch 内建 optimizer
- 日志支持 TensorBoard
- 输出训练曲线图（支持对数坐标）
- 学习率搜索、算法对比、超参数消融实验一体化

### 📂 目录结构（EX2-Optimizer/）

```
EX2-Optimizer/
├── main.py          # 实验入口，支持不同运行模式
├── model.py         # MLP 模型
├── optimizers.py    # 所有优化算法实现
├── utils.py         # 数据预处理、绘图、评估
├── runs/            # TensorBoard 日志目录
├── plots/           # 输出图像目录
├── .gitignore       # 忽略日志和临时文件
```
---

## 🧪 实验二：单机随机优化算法实现与对比（E3-Optimizer）

### 🎯 实验目标及要求

使用手写方式实现以下优化算法并完成训练过程可视化与超参数对比：

- Nesterov Momentum算法
- AdaGrad算法
- RMSProp算法
- AdaDelta算法
- Adam算法

### 

- 使用 MNIST数据集 数据
- 手动实现优化器，无 PyTorch 内建 optimizer
- 日志支持 TensorBoard
- 输出训练曲线图（支持对数坐标）
- 学习率搜索、算法对比、超参数消融实验一体化

### 📂 目录结构（EX3-Optimizer/）

```
EX3-Optimizer/
├── main.py          # 实验入口，支持不同运行模式
├── model.py         # MLP 模型
├── optimizers.py    # 所有优化算法实现
├── utils.py         # 数据预处理、绘图、评估
├── runs/            # TensorBoard 日志目录
├── plots/           # 输出图像目录
├── .gitignore       # 忽略日志和临时文件
```
---

## 🤝 作者说明

- 该仓库为课程实验使用。
- 代码可能存在错误，欢迎指正。
- e-mail：1071753343@qq.com
