# 量子-经典混合模型用于医学影像分析

## 项目概述

本项目实现了一个量子-经典混合模型，用于处理和分析医学影像（胸部X光片）及其相关报告文本。该模型结合了量子计算的优势和传统深度学习技术，旨在提高医学影像分析的准确性和效率。

## 特点

- **量子-经典混合架构**：结合量子计算电路和经典神经网络的优势
- **多模态数据处理**：同时处理图像和文本数据
- **可视化工具**：提供量子电路和训练过程的可视化
- **灵活的预处理管道**：支持各种医学影像和报告格式

## 目录结构

```
.
├── preprocessing/       # 数据预处理模块
├── quantum/            # 量子计算模块
├── training/           # 模型训练模块
├── data/               # 数据存储目录
│   ├── images/         # 医学影像数据
│   └── reports/        # 医学报告文本
├── NLMCXR_reports/     # NLMCXR数据集报告
├── results/            # 结果输出目录
│   ├── accuracy_plots/ # 准确度曲线图
│   └── quantum_circuits/ # 量子电路图
├── create_dirs.py      # 目录结构创建脚本
└── requirements.txt    # 项目依赖
```

## 安装指南

1. 克隆仓库：
   ```
   git clone https://github.com/pocari834/quantum-medical-imaging.git
   cd quantum-medical-imaging
   ```

2. 创建并激活虚拟环境：
   ```
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   .venv\Scripts\activate     # Windows
   ```

3. 安装依赖：
   ```
   pip install -r requirements.txt
   ```

4. 创建必要的目录结构：
   ```
   python create_dirs.py
   ```

## 使用方法

### 数据准备

1. 将医学影像文件放入 `data/images/` 目录
2. 将对应的报告文本放入 `data/reports/` 目录
3. 运行预处理脚本：
   ```
   python preprocessing/preprocess.py
   ```

### 模型训练

```
python training/train.py
```

训练过程中的指标将被记录到 `results/metrics.csv`，准确度曲线将保存到 `results/accuracy_plots/` 目录。

### 量子电路可视化

```
python quantum/visualize_circuit.py
```

量子电路图将保存到 `results/quantum_circuits/` 目录。

## 技术细节

本项目使用 PennyLane 进行量子计算模拟，结合 TensorFlow/PyTorch 进行经典深度学习。量子部分实现了参数化量子电路，用于特征提取和变换，而经典部分则负责高级特征处理和最终预测。


