# Quantum-Enhanced Medical Image Analysis

## 项目概述
这个项目结合了量子计算和深度学习技术，用于医学影像分析，特别是肺炎诊断。系统使用胸部X光图像和放射学报告进行分析，通过混合量子-经典模型提高诊断准确性。

## 项目结构
- **preprocessing/**: 数据预处理模块
  - `data_loader.py`: 负责加载图像和报告数据
  - `generate_metadata.py`: 生成元数据CSV文件，基于报告中"pneumonia"关键词分配标签
  - `xml_parser.py`: 从XML文件中提取诊断文本

- **quantum/**: 量子计算模块
  - `hybrid_model.py`: 定义混合量子-经典模型，结合CNN、量子电路和BERT模型
  - `quantum_circuit.py`: 实现量子电路

- **training/**: 模型训练模块
  - `train.py`: 实现模型训练流程
  - `visualize.py`: 比较不同模型性能

- **data/**: 数据目录
  - `images/`: 存储医学影像
  - `reports/`: 存储处理后的医学报告

- **NLMCXR_reports/**: 存储原始医学报告XML文件

## 安装与使用
```bash
# 克隆仓库
git clone https://git.woa.com/sihuuhuang/projectfour.git
cd projectfour

# 安装依赖
pip install -r requirements.txt

# 运行预处理
python -m preprocessing.generate_metadata

# 训练模型
python -m training.train
```

## 技术栈
- Python
- PyTorch
- Qiskit (量子计算)
- BERT (自然语言处理)
- Matplotlib (可视化) 
