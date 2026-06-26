#pragma once

#include "tensor.hpp"
#include "math.hpp"
#include <memory>

namespace ai {

// 前向声明
class Optimizer;

/**
 * @brief 神经网络层基类
 */
class Layer {
public:
    virtual ~Layer() = default;
    
    // 前向传播
    virtual Tensor forward(const Tensor& input) = 0;
    
    // 反向传播（返回对输入的梯度）
    virtual Tensor backward(const Tensor& grad_output) = 0;
    
    // 获取所有可训练参数
    virtual std::vector<Tensor*> parameters() = 0;
    
    // 参数梯度（与parameters一一对应）
    virtual std::vector<Tensor*> gradients() = 0;
    
    // 训练/评估模式
    void set_training(bool training) { is_training_ = training; }
    bool is_training() const { return is_training_; }
    
    // 参数计数
    virtual size_t param_count() const = 0;
    
    // 保存/加载
    virtual void save(std::ofstream& ofs) const;
    virtual void load(std::ifstream& ifs);
    
    // 名称
    virtual std::string name() const = 0;
    
protected:
    bool is_training_ = true;
};

/**
 * @brief 全连接层 (Linear / Dense)
 *        output = input * W^T + b
 *        input: [batch, in_features]
 *        output: [batch, out_features]
 */
class Linear : public Layer {
public:
    Linear(size_t in_features, size_t out_features, bool use_bias = true);
    
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    
    size_t param_count() const override;
    std::string name() const override { return "Linear"; }
    
    void save(std::ofstream& ofs) const override;
    void load(std::ifstream& ifs) override;

private:
    size_t in_features_;
    size_t out_features_;
    bool use_bias_;
    
    Tensor weight_;   // [out_features, in_features]
    Tensor bias_;     // [out_features]
    
    Tensor grad_weight_; // [out_features, in_features]
    Tensor grad_bias_;   // [out_features]
    
    Tensor input_cache_; // 缓存输入用于反向传播
};

/**
 * @brief 嵌入层 (Embedding)
 *        input: [batch, seq_len] (token ids)
 *        output: [batch, seq_len, embedding_dim]
 */
class Embedding : public Layer {
public:
    Embedding(size_t vocab_size, size_t embedding_dim);
    
    Tensor forward(const Tensor& input) override; // 注意：input应为int类型，这里用float存储id
    Tensor backward(const Tensor& grad_output) override;
    
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    
    size_t param_count() const override;
    std::string name() const override { return "Embedding"; }
    
    void save(std::ofstream& ofs) const override;
    void load(std::ifstream& ifs) override;

private:
    size_t vocab_size_;
    size_t embedding_dim_;
    
    Tensor weight_; // [vocab_size, embedding_dim]
    Tensor grad_weight_;
    
    Tensor input_cache_; // 保存token ids
};

/**
 * @brief 多头注意力层 (Multi-Head Attention)
 *        实现缩放点积注意力
 */
class MultiHeadAttention : public Layer {
public:
    MultiHeadAttention(size_t d_model, size_t num_heads, float dropout = 0.1f);
    
    Tensor forward(const Tensor& input) override; // self-attention
    Tensor forward(const Tensor& query, const Tensor& key, const Tensor& value);
    Tensor backward(const Tensor& grad_output) override;
    
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    
    size_t param_count() const override;
    std::string name() const override { return "MultiHeadAttention"; }
    
    void save(std::ofstream& ofs) const override;
    void load(std::ifstream& ifs) override;

private:
    size_t d_model_;
    size_t num_heads_;
    size_t head_dim_;
    float dropout_p_;
    float scale_;
    
    // Q, K, V 投影
    Linear wq_; 
    Linear wk_;
    Linear wv_;
    Linear wo_; // 输出投影
    
    // 缓存
    Tensor q_cache_, k_cache_, v_cache_;
    Tensor attn_scores_cache_; // attention weights
    Tensor input_cache_;
};

/**
 * @brief 前馈网络层 (Feed-Forward Network)
 *        两个Linear层中间夹ReLU/GELU
 */
class FeedForward : public Layer {
public:
    FeedForward(size_t d_model, size_t d_ff, const std::string& activation = "gelu");
    
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    
    size_t param_count() const override;
    std::string name() const override { return "FeedForward"; }
    
    void save(std::ofstream& ofs) const override;
    void load(std::ifstream& ifs) override;

private:
    Linear fc1_; // d_model -> d_ff
    Linear fc2_; // d_ff -> d_model
    
    Tensor activation_cache_; // 缓存fc1的输出
    Tensor input_cache_;
    
    std::string activation_;
};

} // namespace ai
