# AI - C++ 大模型推理与训练框架

这是一个基于 C++ 实现的轻量级 AI 大模型（Transformer 架构）项目，包含：

- **张量系统**：多维张量管理与基础运算
- **数学库**：矩阵运算、激活函数（GELU/ReLU）、Softmax、LayerNorm 等
- **神经网络层**：Linear, Embedding, Multi-Head Attention, FeedForward
- **模型系统**：完整的 Transformer Encoder 架构（GPT 风格解码器）
- **训练引擎**：前向传播、反向传播、AdamW 优化器、余弦学习率调度
- **推理引擎**：贪婪解码、温度采样、Top-k 过滤
- **模型 IO**：二进制格式的模型保存与加载

## 项目结构

```
AI/
├── CMakeLists.txt           # 根构建配置
├── include/ai/               # 头文件
│   ├── tensor.hpp            # 张量
│   ├── math.hpp              # 数学函数
│   ├── utils.hpp             # 工具（随机初始化、分词器、Timer）
│   ├── layers.hpp            # 神经网络层
│   ├── transformer.hpp       # Transformer 块
│   ├── model.hpp             # 完整模型
│   └── engine.hpp            # 训练与推理引擎
├── src/                      # 源文件
│   ├── tensor.cpp
│   ├── math.cpp
│   ├── utils.cpp
│   ├── layers.cpp
│   ├── transformer.cpp
│   ├── model.cpp
│   ├── engine.cpp
│   └── main.cpp
└── build/                    # 构建输出（CMake 生成）
```

## 构建要求

- C++17 或更高版本
- CMake 3.15 或更高版本

## 构建步骤

```bash
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

## 运行

```bash
./AI
```

## 测试输出示例

程序运行后会依次执行以下测试：

1. **张量系统测试** — 验证加减乘除、reshape、transpose 等操作
2. **神经网络层测试** — 验证 Linear、Embedding、Multi-Head Attention、FeedForward
3. **Transformer 块测试** — 验证 Encoder Block 的前向传播
4. **完整模型测试** — 构建 112K 参数的小模型，验证推理和初始损失
5. **训练演示** — 在简单序列数据上训练 5 个 epoch，观察损失下降
6. **保存/加载演示** — 验证模型二进制序列化

## 设计特点

- **纯 C++ 实现，零外部依赖**（仅标准库）
- **模块化架构**，便于扩展新层和新模型
- **支持 CPU 多线程**（可通过 OpenMP 扩展）
- **完整的训练管线**：前向传播 → 损失计算 → 反向传播 → 参数更新
- **自回归文本生成**：支持贪婪解码和采样解码

> ⚠️ **注意**：这是一个**教育/演示性质的轻量级框架**。使用纯 C++ 实现，不依赖外部深度学习库（如 PyTorch、TensorFlow）。对于生产环境，建议使用成熟的框架。

## 架构说明

本模型采用 **纯解码器 Transformer（GPT 风格）** 架构：

```
Input Tokens
    ↓
Embedding + Positional Encoding
    ↓
[Transformer Encoder Block] × N
    ├─ LayerNorm → Multi-Head Attention → Residual
    └─ LayerNorm → FeedForward → Residual
    ↓
LayerNorm
    ↓
Linear (输出投影到 vocab_size)
    ↓
Softmax → Logits / 概率分布
```

## 许可证

MIT License
