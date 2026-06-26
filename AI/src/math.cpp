#include "ai/math.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace ai {
namespace math {

// --- 基本统计 ---

float mean(const Tensor& t) {
    if (t.size() == 0) return 0.0f;
    float sum = 0.0f;
    for (size_t i = 0; i < t.size(); ++i) {
        sum += t.data()[i];
    }
    return sum / static_cast<float>(t.size());
}

float variance(const Tensor& t, float mean_val) {
    if (t.size() <= 1) return 0.0f;
    float var = 0.0f;
    for (size_t i = 0; i < t.size(); ++i) {
        float diff = t.data()[i] - mean_val;
        var += diff * diff;
    }
    return var / static_cast<float>(t.size());
}

float max_val(const Tensor& t) {
    if (t.size() == 0) return 0.0f;
    float mx = t.data()[0];
    for (size_t i = 1; i < t.size(); ++i) {
        if (t.data()[i] > mx) mx = t.data()[i];
    }
    return mx;
}

float min_val(const Tensor& t) {
    if (t.size() == 0) return 0.0f;
    float mn = t.data()[0];
    for (size_t i = 1; i < t.size(); ++i) {
        if (t.data()[i] < mn) mn = t.data()[i];
    }
    return mn;
}

float sum(const Tensor& t) {
    float s = 0.0f;
    for (size_t i = 0; i < t.size(); ++i) {
        s += t.data()[i];
    }
    return s;
}

// --- 激活函数 ---

Tensor relu(const Tensor& x) {
    Tensor result(x.shape());
    for (size_t i = 0; i < x.size(); ++i) {
        result.data()[i] = std::max(0.0f, x.data()[i]);
    }
    return result;
}

Tensor relu_derivative(const Tensor& x) {
    Tensor result(x.shape());
    for (size_t i = 0; i < x.size(); ++i) {
        result.data()[i] = x.data()[i] > 0.0f ? 1.0f : 0.0f;
    }
    return result;
}

Tensor gelu(const Tensor& x) {
    Tensor result(x.shape());
    for (size_t i = 0; i < x.size(); ++i) {
        float v = x.data()[i];
        // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        float cdf = 0.5f * (1.0f + std::tanh(
            0.7978845608f * (v + 0.044715f * v * v * v)));
        result.data()[i] = v * cdf;
    }
    return result;
}

Tensor gelu_derivative(const Tensor& x) {
    Tensor result(x.shape());
    for (size_t i = 0; i < x.size(); ++i) {
        float v = x.data()[i];
        float cdf = 0.5f * (1.0f + std::tanh(
            0.7978845608f * (v + 0.044715f * v * v * v)));
        float pdf = 0.5f * 0.7978845608f * (1.0f + 3.0f * 0.044715f * v * v)
            / std::cosh(0.7978845608f * (v + 0.044715f * v * v * v));
        pdf = pdf * pdf; // sech^2 approximation
        result.data()[i] = cdf + v * pdf;
    }
    return result;
}

Tensor sigmoid(const Tensor& x) {
    Tensor result(x.shape());
    for (size_t i = 0; i < x.size(); ++i) {
        result.data()[i] = 1.0f / (1.0f + std::exp(-x.data()[i]));
    }
    return result;
}

Tensor tanh(const Tensor& x) {
    Tensor result(x.shape());
    for (size_t i = 0; i < x.size(); ++i) {
        result.data()[i] = std::tanh(x.data()[i]);
    }
    return result;
}

Tensor softmax(const Tensor& x) {
    // x 的最后一个维度是特征维度
    if (x.ndim() == 0) return Tensor();
    
    std::vector<size_t> shape = x.shape();
    Tensor result(shape);
    
    size_t last_dim = shape.back();
    size_t outer = 1;
    for (size_t i = 0; i < shape.size() - 1; ++i) {
        outer *= shape[i];
    }
    
    for (size_t o = 0; o < outer; ++o) {
        // 找到最大值（数值稳定性）
        float max_val = x.data()[o * last_dim];
        for (size_t i = 1; i < last_dim; ++i) {
            if (x.data()[o * last_dim + i] > max_val)
                max_val = x.data()[o * last_dim + i];
        }
        
        // 计算exp和sum
        float sum = 0.0f;
        for (size_t i = 0; i < last_dim; ++i) {
            float exp_val = std::exp(x.data()[o * last_dim + i] - max_val);
            result.data()[o * last_dim + i] = exp_val;
            sum += exp_val;
        }
        
        // 归一化
        for (size_t i = 0; i < last_dim; ++i) {
            result.data()[o * last_dim + i] /= sum;
        }
    }
    
    return result;
}

Tensor log_softmax(const Tensor& x) {
    Tensor sm = softmax(x);
    Tensor result(x.shape());
    for (size_t i = 0; i < x.size(); ++i) {
        result.data()[i] = std::log(std::max(sm.data()[i], 1e-10f));
    }
    return result;
}

// --- 矩阵运算 ---

Tensor matmul(const Tensor& a, const Tensor& b, bool transpose_b) {
    if (a.ndim() != 2 || b.ndim() != 2)
        throw std::runtime_error("matmul: both tensors must be 2D");
    
    size_t M = a.shape()[0];
    size_t K = a.shape()[1];
    size_t N = transpose_b ? b.shape()[0] : b.shape()[1];
    size_t K2 = transpose_b ? b.shape()[1] : b.shape()[0];
    
    if (K != K2)
        throw std::runtime_error("matmul: inner dimension mismatch");
    
    Tensor result({M, N});
    result.fill(0.0f);
    
    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            float a_ik = a(i, k);
            for (size_t j = 0; j < N; ++j) {
                float b_kj = transpose_b ? b(j, k) : b(k, j);
                result(i, j) += a_ik * b_kj;
            }
        }
    }
    
    return result;
}

Tensor bmm(const Tensor& a, const Tensor& b, bool transpose_b) {
    if (a.ndim() != 3 || b.ndim() != 3)
        throw std::runtime_error("bmm: both tensors must be 3D");
    if (a.shape()[0] != b.shape()[0])
        throw std::runtime_error("bmm: batch size mismatch");
    
    size_t B = a.shape()[0];
    size_t M = a.shape()[1];
    size_t K = a.shape()[2];
    size_t N = transpose_b ? b.shape()[1] : b.shape()[2];
    size_t K2 = transpose_b ? b.shape()[2] : b.shape()[1];
    
    if (K != K2) {
        throw std::runtime_error("bmm: inner dimension mismatch");
    }
    
    Tensor result({B, M, N});
    result.fill(0.0f);
    
    for (size_t batch = 0; batch < B; ++batch) {
        for (size_t i = 0; i < M; ++i) {
            for (size_t k = 0; k < K; ++k) {
                float a_ik = a(batch, i, k);
                for (size_t j = 0; j < N; ++j) {
                    float b_kj = transpose_b ? b(batch, j, k) : b(batch, k, j);
                    result(batch, i, j) += a_ik * b_kj;
                }
            }
        }
    }
    
    return result;
}

Tensor matmul_vec(const Tensor& a, const Tensor& b) {
    if (a.ndim() != 2 || b.ndim() != 1)
        throw std::runtime_error("matmul_vec: a must be 2D, b must be 1D");
    if (a.shape()[1] != b.shape()[0])
        throw std::runtime_error("matmul_vec: dimension mismatch");
    
    size_t M = a.shape()[0];
    size_t K = a.shape()[1];
    Tensor result({M});
    
    for (size_t i = 0; i < M; ++i) {
        float sum = 0.0f;
        for (size_t k = 0; k < K; ++k) {
            sum += a(i, k) * b(k);
        }
        result(i) = sum;
    }
    
    return result;
}

// --- Layer Normalization ---

Tensor layer_norm(const Tensor& x, const Tensor& gamma, const Tensor& beta, float eps) {
    if (x.ndim() == 0) return Tensor();
    
    std::vector<size_t> shape = x.shape();
    size_t last_dim = shape.back();
    size_t outer = 1;
    for (size_t i = 0; i < shape.size() - 1; ++i) {
        outer *= shape[i];
    }
    
    Tensor result(shape);
    
    for (size_t o = 0; o < outer; ++o) {
        // 计算均值
        float mean = 0.0f;
        for (size_t i = 0; i < last_dim; ++i) {
            mean += x.data()[o * last_dim + i];
        }
        mean /= static_cast<float>(last_dim);
        
        // 计算方差
        float var = 0.0f;
        for (size_t i = 0; i < last_dim; ++i) {
            float diff = x.data()[o * last_dim + i] - mean;
            var += diff * diff;
        }
        var /= static_cast<float>(last_dim);
        
        // 归一化
        float std_inv = 1.0f / std::sqrt(var + eps);
        for (size_t i = 0; i < last_dim; ++i) {
            float normalized = (x.data()[o * last_dim + i] - mean) * std_inv;
            result.data()[o * last_dim + i] = normalized * gamma.data()[i] + beta.data()[i];
        }
    }
    
    return result;
}

Tensor layer_norm_backward(const Tensor& grad_out, const Tensor& x, const Tensor& gamma, float eps) {
    if (x.ndim() == 0) return Tensor();
    
    std::vector<size_t> shape = x.shape();
    size_t last_dim = shape.back();
    size_t outer = 1;
    for (size_t i = 0; i < shape.size() - 1; ++i) {
        outer *= shape[i];
    }
    
    Tensor result(shape);
    
    for (size_t o = 0; o < outer; ++o) {
        float mean = 0.0f;
        for (size_t i = 0; i < last_dim; ++i) {
            mean += x.data()[o * last_dim + i];
        }
        mean /= static_cast<float>(last_dim);
        
        float var = 0.0f;
        for (size_t i = 0; i < last_dim; ++i) {
            float diff = x.data()[o * last_dim + i] - mean;
            var += diff * diff;
        }
        var /= static_cast<float>(last_dim);
        
        float std_inv = 1.0f / std::sqrt(var + eps);
        
        // 简化的反向传播（对于教育目的）
        for (size_t i = 0; i < last_dim; ++i) {
            result.data()[o * last_dim + i] = grad_out.data()[o * last_dim + i] * gamma.data()[i] * std_inv;
        }
    }
    
    return result;
}

// --- Dropout ---

Tensor dropout(const Tensor& x, float p, bool training) {
    if (!training || p <= 0.0f) return x;
    if (p >= 1.0f) {
        Tensor result(x.shape());
        result.fill(0.0f);
        return result;
    }
    
    Tensor result(x.shape());
    float scale = 1.0f / (1.0f - p);
    
    // 简单的伪随机（非加密级）
    static thread_local unsigned seed = 0;
    seed = seed * 1103515245 + 12345;
    
    for (size_t i = 0; i < x.size(); ++i) {
        seed = seed * 1103515245 + 12345;
        float rand_val = static_cast<float>(seed & 0x7FFFFFFF) / 0x7FFFFFFF;
        result.data()[i] = (rand_val > p) ? x.data()[i] * scale : 0.0f;
    }
    
    return result;
}

// --- 缩放与偏移 ---

Tensor scale_add(const Tensor& x, float scale, float shift) {
    Tensor result(x.shape());
    for (size_t i = 0; i < x.size(); ++i) {
        result.data()[i] = x.data()[i] * scale + shift;
    }
    return result;
}

// --- 位置编码 ---

Tensor sinusoidal_positional_encoding(size_t seq_len, size_t d_model) {
    Tensor result({seq_len, d_model});
    
    for (size_t pos = 0; pos < seq_len; ++pos) {
        for (size_t i = 0; i < d_model; ++i) {
            float angle = static_cast<float>(pos) / std::pow(10000.0f, 
                static_cast<float>(i) / static_cast<float>(d_model));
            if (i % 2 == 0) {
                result(pos, i) = std::sin(angle);
            } else {
                result(pos, i) = std::cos(angle);
            }
        }
    }
    
    return result;
}

Tensor learned_positional_encoding(size_t seq_len, size_t d_model) {
    // 返回可学习的随机初始化位置编码
    Tensor result({seq_len, d_model});
    // 这里用简单随机初始化，实际应由模型参数管理
    for (size_t i = 0; i < result.size(); ++i) {
        result.data()[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.02f;
    }
    return result;
}

} // namespace math
} // namespace ai
