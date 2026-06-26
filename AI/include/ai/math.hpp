#pragma once

#include "tensor.hpp"
#include <cmath>

namespace ai {

/**
 * @brief 数学运算与激活函数
 */
namespace math {

// 基本统计
float mean(const Tensor& t);
float variance(const Tensor& t, float mean_val);
float max_val(const Tensor& t);
float min_val(const Tensor& t);
float sum(const Tensor& t);

// 逐元素操作
Tensor relu(const Tensor& x);
Tensor relu_derivative(const Tensor& x);
Tensor gelu(const Tensor& x);        // GELU激活函数（Transformer常用）
Tensor gelu_derivative(const Tensor& x);
Tensor sigmoid(const Tensor& x);
Tensor tanh(const Tensor& x);
Tensor softmax(const Tensor& x);      // 对最后一个维度做softmax
Tensor log_softmax(const Tensor& x);

// 矩阵运算
/**
 * @brief 矩阵乘法 C = A * B^T
 *        A: [M, K], B: [N, K] -> C: [M, N]
 */
Tensor matmul(const Tensor& a, const Tensor& b, bool transpose_b = false);

/**
 * @brief 批量矩阵乘法
 *        A: [B, M, K], B: [B, N, K] -> C: [B, M, N]
 */
Tensor bmm(const Tensor& a, const Tensor& b, bool transpose_b = false);

/**
 * @brief 矩阵与向量乘法
 */
Tensor matmul_vec(const Tensor& a, const Tensor& b);

// Layer Normalization
/**
 * @brief Layer Normalization
 *        对最后一个维度做归一化
 *        x: [..., features]
 *        返回: [..., features]
 */
Tensor layer_norm(const Tensor& x, const Tensor& gamma, const Tensor& beta, float eps = 1e-6f);

/**
 * @brief Layer Normalization 反向传播
 */
Tensor layer_norm_backward(
    const Tensor& grad_out,
    const Tensor& x,
    const Tensor& gamma,
    float eps = 1e-6f
);

// Dropout（训练时）
Tensor dropout(const Tensor& x, float p, bool training = true);

// 缩放与偏移
Tensor scale_add(const Tensor& x, float scale, float shift);

// 位置编码
Tensor sinusoidal_positional_encoding(size_t seq_len, size_t d_model);
Tensor learned_positional_encoding(size_t seq_len, size_t d_model);

} // namespace math

} // namespace ai
