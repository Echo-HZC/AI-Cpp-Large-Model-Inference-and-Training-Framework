#include "ai/layers.hpp"
#include "ai/math.hpp"
#include "ai/utils.hpp"

namespace ai {

// --- Layer 基类 ---

void Layer::save(std::ofstream& ofs) const {
    // 基类无参数，子类覆盖
}

void Layer::load(std::ifstream& ifs) {
    // 基类无参数，子类覆盖
}

// --- Linear ---

Linear::Linear(size_t in_features, size_t out_features, bool use_bias)
    : in_features_(in_features), out_features_(out_features), use_bias_(use_bias),
      weight_({out_features, in_features}),
      bias_(use_bias ? std::vector<size_t>{out_features} : std::vector<size_t>{0}),
      grad_weight_({out_features, in_features}),
      grad_bias_(use_bias ? std::vector<size_t>{out_features} : std::vector<size_t>{0}),
      input_cache_({0}) {
    utils::xavier_init(weight_, in_features, out_features);
    if (use_bias_) {
        bias_.fill(0.0f);
    }
    grad_weight_.fill(0.0f);
    if (use_bias_) grad_bias_.fill(0.0f);
}

Tensor Linear::forward(const Tensor& input) {
    if (input.ndim() != 2) {
        throw std::runtime_error("Linear::forward: input must be 2D [batch, features]");
    }
    if (input.shape()[1] != in_features_) {
        throw std::runtime_error("Linear::forward: input features mismatch");
    }
    
    input_cache_ = Tensor({input.shape()[0], input.shape()[1]});
    std::memcpy(input_cache_.data(), input.data(), input.size() * sizeof(float));
    
    // output = input * W^T
    Tensor output = math::matmul(input, weight_, true);
    
    // + bias
    if (use_bias_) {
        size_t batch = output.shape()[0];
        for (size_t b = 0; b < batch; ++b) {
            for (size_t j = 0; j < out_features_; ++j) {
                output(b, j) += bias_(j);
            }
        }
    }
    
    return output;
}

Tensor Linear::backward(const Tensor& grad_output) {
    if (grad_output.ndim() != 2 || grad_output.shape()[1] != out_features_) {
        throw std::runtime_error("Linear::backward: grad_output shape mismatch");
    }
    
    size_t batch = grad_output.shape()[0];
    
    // grad_weight = grad_output^T * input_cache
    for (size_t b = 0; b < batch; ++b) {
        for (size_t i = 0; i < out_features_; ++i) {
            for (size_t j = 0; j < in_features_; ++j) {
                grad_weight_(i, j) += grad_output(b, i) * input_cache_(b, j);
            }
        }
    }
    
    // grad_bias = sum(grad_output, axis=0)
    if (use_bias_) {
        for (size_t b = 0; b < batch; ++b) {
            for (size_t i = 0; i < out_features_; ++i) {
                grad_bias_(i) += grad_output(b, i);
            }
        }
    }
    
    // grad_input = grad_output * W
    Tensor grad_input = math::matmul(grad_output, weight_, false);
    
    return grad_input;
}

std::vector<Tensor*> Linear::parameters() {
    std::vector<Tensor*> params;
    params.push_back(&weight_);
    if (use_bias_) params.push_back(&bias_);
    return params;
}

std::vector<Tensor*> Linear::gradients() {
    std::vector<Tensor*> grads;
    grads.push_back(&grad_weight_);
    if (use_bias_) grads.push_back(&grad_bias_);
    return grads;
}

size_t Linear::param_count() const {
    size_t count = weight_.size();
    if (use_bias_) count += bias_.size();
    return count;
}

void Linear::save(std::ofstream& ofs) const {
    weight_.save(ofs);
    if (use_bias_) bias_.save(ofs);
}

void Linear::load(std::ifstream& ifs) {
    Tensor w = Tensor::load(ifs);
    std::memcpy(weight_.data(), w.data(), weight_.size() * sizeof(float));
    if (use_bias_) {
        Tensor b = Tensor::load(ifs);
        std::memcpy(bias_.data(), b.data(), bias_.size() * sizeof(float));
    }
}

// --- Embedding ---

Embedding::Embedding(size_t vocab_size, size_t embedding_dim)
    : vocab_size_(vocab_size), embedding_dim_(embedding_dim),
      weight_({vocab_size, embedding_dim}),
      grad_weight_({vocab_size, embedding_dim}),
      input_cache_({0}) {
    utils::xavier_init(weight_, vocab_size, embedding_dim);
    grad_weight_.fill(0.0f);
}

Tensor Embedding::forward(const Tensor& input) {
    if (input.ndim() != 2) {
        throw std::runtime_error("Embedding::forward: input must be 2D [batch, seq_len]");
    }
    
    size_t batch = input.shape()[0];
    size_t seq_len = input.shape()[1];
    
    Tensor output({batch, seq_len, embedding_dim_});
    
    // 保存token ids用于反向传播
    input_cache_ = Tensor({batch, seq_len});
    std::memcpy(input_cache_.data(), input.data(), input.size() * sizeof(float));
    
    for (size_t b = 0; b < batch; ++b) {
        for (size_t s = 0; s < seq_len; ++s) {
            int token_id = static_cast<int>(input(b, s));
            if (token_id < 0) token_id = 0;
            if (token_id >= static_cast<int>(vocab_size_)) token_id = vocab_size_ - 1;
            
            for (size_t d = 0; d < embedding_dim_; ++d) {
                output(b, s, d) = weight_(token_id, d);
            }
        }
    }
    
    return output;
}

Tensor Embedding::backward(const Tensor& grad_output) {
    if (grad_output.ndim() != 3 || grad_output.shape()[2] != embedding_dim_) {
        throw std::runtime_error("Embedding::backward: grad_output shape mismatch");
    }
    
    size_t batch = grad_output.shape()[0];
    size_t seq_len = grad_output.shape()[1];
    
    for (size_t b = 0; b < batch; ++b) {
        for (size_t s = 0; s < seq_len; ++s) {
            int token_id = static_cast<int>(input_cache_(b, s));
            if (token_id < 0) token_id = 0;
            if (token_id >= static_cast<int>(vocab_size_)) token_id = vocab_size_ - 1;
            
            for (size_t d = 0; d < embedding_dim_; ++d) {
                grad_weight_(token_id, d) += grad_output(b, s, d);
            }
        }
    }
    
    return Tensor({0}); // 嵌入层没有输入梯度
}

std::vector<Tensor*> Embedding::parameters() {
    return {&weight_};
}

std::vector<Tensor*> Embedding::gradients() {
    return {&grad_weight_};
}

size_t Embedding::param_count() const {
    return weight_.size();
}

void Embedding::save(std::ofstream& ofs) const {
    weight_.save(ofs);
}

void Embedding::load(std::ifstream& ifs) {
    Tensor w = Tensor::load(ifs);
    std::memcpy(weight_.data(), w.data(), weight_.size() * sizeof(float));
}

// --- MultiHeadAttention ---

MultiHeadAttention::MultiHeadAttention(size_t d_model, size_t num_heads, float dropout)
    : d_model_(d_model), num_heads_(num_heads), dropout_p_(dropout),
      wq_(d_model, d_model, false),
      wk_(d_model, d_model, false),
      wv_(d_model, d_model, false),
      wo_(d_model, d_model, false) {
    if (d_model % num_heads != 0) {
        throw std::runtime_error("d_model must be divisible by num_heads");
    }
    head_dim_ = d_model / num_heads;
    scale_ = 1.0f / std::sqrt(static_cast<float>(head_dim_));
}

Tensor MultiHeadAttention::forward(const Tensor& input) {
    return forward(input, input, input); // self-attention
}

Tensor MultiHeadAttention::forward(const Tensor& query, const Tensor& key, const Tensor& value) {
    if (query.ndim() != 3) {
        throw std::runtime_error("MultiHeadAttention: query must be 3D [batch, seq_len, d_model]");
    }
    
    size_t batch = query.shape()[0];
    size_t seq_len = query.shape()[1];
    
    input_cache_ = Tensor({batch, seq_len, d_model_});
    std::memcpy(input_cache_.data(), query.data(), query.size() * sizeof(float));
    
    // Q, K, V 投影
    // query: [batch, seq_len, d_model] -> [batch*seq_len, d_model]
    Tensor q_flat = query.reshape({batch * seq_len, d_model_});
    Tensor k_flat = key.reshape({batch * seq_len, d_model_});
    Tensor v_flat = value.reshape({batch * seq_len, d_model_});
    
    Tensor q_proj = wq_.forward(q_flat); // [batch*seq_len, d_model]
    Tensor k_proj = wk_.forward(k_flat);
    Tensor v_proj = wv_.forward(v_flat);
    
    // reshape to [batch, seq_len, num_heads, head_dim]
    q_proj = q_proj.reshape({batch, seq_len, num_heads_, head_dim_});
    k_proj = k_proj.reshape({batch, seq_len, num_heads_, head_dim_});
    v_proj = v_proj.reshape({batch, seq_len, num_heads_, head_dim_});
    
    // transpose to [batch, num_heads, seq_len, head_dim]
    q_proj = q_proj.transpose(1, 2);
    k_proj = k_proj.transpose(1, 2);
    v_proj = v_proj.transpose(1, 2);
    
    q_cache_ = q_proj;
    k_cache_ = k_proj;
    v_cache_ = v_proj;
    
    // reshape to 3D for bmm: [batch * num_heads, seq_len, head_dim]
    Tensor q_3d = q_proj.reshape({batch * num_heads_, seq_len, head_dim_});
    Tensor k_3d = k_proj.reshape({batch * num_heads_, seq_len, head_dim_});
    Tensor v_3d = v_proj.reshape({batch * num_heads_, seq_len, head_dim_});
    
    // attention scores: [batch * num_heads, seq_len, seq_len]
    Tensor scores = math::bmm(q_3d, k_3d, true); // Q * K^T
    scores = scores * scale_;
    
    // softmax (对最后一个维度)
    Tensor attn_weights = math::softmax(scores);
    attn_weights = math::dropout(attn_weights, dropout_p_, is_training_);
    
    // reshape back to 4D for caching: [batch, num_heads, seq_len, seq_len]
    attn_scores_cache_ = attn_weights.reshape({batch, num_heads_, seq_len, seq_len});
    
    // attention output: [batch * num_heads, seq_len, head_dim]
    Tensor attn_out = math::bmm(attn_weights, v_3d, false);
    
    // reshape back: [batch, num_heads, seq_len, head_dim]
    attn_out = attn_out.reshape({batch, num_heads_, seq_len, head_dim_});
    
    // reshape: [batch, seq_len, d_model]
    attn_out = attn_out.reshape({batch, seq_len, d_model_});
    
    // flatten for output projection: [batch*seq_len, d_model]
    Tensor attn_flat = attn_out.reshape({batch * seq_len, d_model_});
    Tensor output = wo_.forward(attn_flat);
    
    return output.reshape({batch, seq_len, d_model_});
}

Tensor MultiHeadAttention::backward(const Tensor& grad_output) {
    size_t batch = grad_output.shape()[0];
    size_t seq_len = grad_output.shape()[1];
    
    // flatten grad_output
    Tensor grad_flat = grad_output.reshape({batch * seq_len, d_model_});
    
    // backward through output projection
    Tensor grad_attn_flat = wo_.backward(grad_flat);
    
    // reshape: [batch, seq_len, num_heads, head_dim]
    Tensor grad_attn = grad_attn_flat.reshape({batch, seq_len, num_heads_, head_dim_});
    grad_attn = grad_attn.transpose(1, 2); // [batch, num_heads, seq_len, head_dim]
    
    // reshape all 4D tensors to 3D for bmm: [batch*num_heads, seq_len, head_dim]
    Tensor grad_attn_3d = grad_attn.reshape({batch * num_heads_, seq_len, head_dim_});
    Tensor v_cache_3d = v_cache_.reshape({batch * num_heads_, seq_len, head_dim_});
    Tensor k_cache_3d = k_cache_.reshape({batch * num_heads_, seq_len, head_dim_});
    Tensor q_cache_3d = q_cache_.reshape({batch * num_heads_, seq_len, head_dim_});
    Tensor attn_scores_3d = attn_scores_cache_.reshape({batch * num_heads_, seq_len, seq_len});
    
    // backward through attention: grad_v = attn_weights^T * grad_attn
    Tensor attn_weights_t = attn_scores_3d.transpose(1, 2); // [B*H, S, S] transpose last two dims
    Tensor grad_v = math::bmm(attn_weights_t, grad_attn_3d, false); // [B*H, S, D]
    
    // backward through softmax: grad_scores = grad_attn * v^T
    Tensor grad_scores = math::bmm(grad_attn_3d, v_cache_3d, true); // [batch*num_heads, seq_len, seq_len]
    
    // 应用scale
    grad_scores = grad_scores * scale_;
    
    // backward through Q/K projections
    Tensor grad_scores_t = grad_scores.transpose(1, 2); // transpose last two dims [B*H, S, S]
    Tensor grad_q = math::bmm(grad_scores, k_cache_3d, false); // [B*H, S, D]
    Tensor grad_k = math::bmm(grad_scores_t, q_cache_3d, false); // [B*H, S, D]
    
    // transpose back and reshape to [batch*seq_len, d_model]
    grad_q = grad_q.reshape({batch, num_heads_, seq_len, head_dim_});
    grad_k = grad_k.reshape({batch, num_heads_, seq_len, head_dim_});
    grad_v = grad_v.reshape({batch, num_heads_, seq_len, head_dim_});
    
    grad_q = grad_q.transpose(1, 2); // [batch, seq_len, num_heads, head_dim]
    grad_k = grad_k.transpose(1, 2);
    grad_v = grad_v.transpose(1, 2);
    
    grad_q = grad_q.reshape({batch * seq_len, d_model_});
    grad_k = grad_k.reshape({batch * seq_len, d_model_});
    grad_v = grad_v.reshape({batch * seq_len, d_model_});
    
    // backward through Q, K, V projections
    wq_.backward(grad_q);
    wk_.backward(grad_k);
    wv_.backward(grad_v);
    
    // 返回对输入的梯度（简化：只返回query的梯度）
    Tensor grad_input = wq_.backward(grad_q); // 这里需要重新计算，简化处理
    
    return grad_input.reshape({batch, seq_len, d_model_});
}

std::vector<Tensor*> MultiHeadAttention::parameters() {
    auto params = wq_.parameters();
    auto k_params = wk_.parameters();
    auto v_params = wv_.parameters();
    auto o_params = wo_.parameters();
    params.insert(params.end(), k_params.begin(), k_params.end());
    params.insert(params.end(), v_params.begin(), v_params.end());
    params.insert(params.end(), o_params.begin(), o_params.end());
    return params;
}

std::vector<Tensor*> MultiHeadAttention::gradients() {
    auto grads = wq_.gradients();
    auto k_grads = wk_.gradients();
    auto v_grads = wv_.gradients();
    auto o_grads = wo_.gradients();
    grads.insert(grads.end(), k_grads.begin(), k_grads.end());
    grads.insert(grads.end(), v_grads.begin(), v_grads.end());
    grads.insert(grads.end(), o_grads.begin(), o_grads.end());
    return grads;
}

size_t MultiHeadAttention::param_count() const {
    return wq_.param_count() + wk_.param_count() + wv_.param_count() + wo_.param_count();
}

void MultiHeadAttention::save(std::ofstream& ofs) const {
    wq_.save(ofs); wk_.save(ofs); wv_.save(ofs); wo_.save(ofs);
}

void MultiHeadAttention::load(std::ifstream& ifs) {
    wq_.load(ifs); wk_.load(ifs); wv_.load(ifs); wo_.load(ifs);
}

// --- FeedForward ---

FeedForward::FeedForward(size_t d_model, size_t d_ff, const std::string& activation)
    : fc1_(d_model, d_ff, true),
      fc2_(d_ff, d_model, true),
      activation_(activation) {}

Tensor FeedForward::forward(const Tensor& input) {
    input_cache_ = Tensor(input.shape());
    std::memcpy(input_cache_.data(), input.data(), input.size() * sizeof(float));
    
    Tensor h = fc1_.forward(input);
    
    if (activation_ == "gelu") {
        activation_cache_ = math::gelu(h);
    } else if (activation_ == "relu") {
        activation_cache_ = math::relu(h);
    } else {
        activation_cache_ = h; // linear
    }
    
    Tensor output = fc2_.forward(activation_cache_);
    return output;
}

Tensor FeedForward::backward(const Tensor& grad_output) {
    Tensor grad_h = fc2_.backward(grad_output);
    
    // 通过激活函数的反向传播
    if (activation_ == "gelu") {
        grad_h = grad_h * math::gelu_derivative(activation_cache_);
    } else if (activation_ == "relu") {
        grad_h = grad_h * math::relu_derivative(activation_cache_);
    }
    
    Tensor grad_input = fc1_.backward(grad_h);
    return grad_input;
}

std::vector<Tensor*> FeedForward::parameters() {
    auto params = fc1_.parameters();
    auto p2 = fc2_.parameters();
    params.insert(params.end(), p2.begin(), p2.end());
    return params;
}

std::vector<Tensor*> FeedForward::gradients() {
    auto grads = fc1_.gradients();
    auto g2 = fc2_.gradients();
    grads.insert(grads.end(), g2.begin(), g2.end());
    return grads;
}

size_t FeedForward::param_count() const {
    return fc1_.param_count() + fc2_.param_count();
}

void FeedForward::save(std::ofstream& ofs) const {
    fc1_.save(ofs); fc2_.save(ofs);
}

void FeedForward::load(std::ifstream& ifs) {
    fc1_.load(ifs); fc2_.load(ifs);
}

} // namespace ai
