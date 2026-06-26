#include "ai/transformer.hpp"
#include "ai/math.hpp"
#include "ai/utils.hpp"

namespace ai {

// --- TransformerEncoderBlock ---

TransformerEncoderBlock::TransformerEncoderBlock(size_t d_model, size_t num_heads, size_t d_ff, float dropout)
    : attn_(d_model, num_heads, dropout),
      ffn_(d_model, d_ff, "gelu"),
      ln1_gamma_({d_model}), ln1_beta_({d_model}),
      ln2_gamma_({d_model}), ln2_beta_({d_model}),
      grad_ln1_gamma_({d_model}), grad_ln1_beta_({d_model}),
      grad_ln2_gamma_({d_model}), grad_ln2_beta_({d_model}),
      input_cache_({0}) {
    
    ln1_gamma_.fill(1.0f); ln1_beta_.fill(0.0f);
    ln2_gamma_.fill(1.0f); ln2_beta_.fill(0.0f);
    grad_ln1_gamma_.fill(0.0f); grad_ln1_beta_.fill(0.0f);
    grad_ln2_gamma_.fill(0.0f); grad_ln2_beta_.fill(0.0f);
}

Tensor TransformerEncoderBlock::forward(const Tensor& input) {
    size_t batch = input.shape()[0];
    size_t seq_len = input.shape()[1];
    size_t d_model = input.shape()[2];
    
    input_cache_ = Tensor(input.shape());
    std::memcpy(input_cache_.data(), input.data(), input.size() * sizeof(float));
    
    // 1. Multi-Head Attention (with pre-norm)
    pre_ln1_ = math::layer_norm(input, ln1_gamma_, ln1_beta_);
    Tensor attn_out = attn_.forward(pre_ln1_);
    
    // 残差连接
    Tensor post_attn = input + attn_out;
    
    // 2. FeedForward (with pre-norm)
    pre_ln2_ = math::layer_norm(post_attn, ln2_gamma_, ln2_beta_);
    Tensor pre_ln2_2d = pre_ln2_.reshape({batch * seq_len, d_model});
    Tensor ff_out_2d = ffn_.forward(pre_ln2_2d);
    Tensor ff_out = ff_out_2d.reshape({batch, seq_len, d_model});
    
    // 残差连接
    Tensor output = post_attn + ff_out;
    
    post_attn_ = post_attn;
    post_ffn_ = output;
    
    return output;
}

Tensor TransformerEncoderBlock::backward(const Tensor& grad_output) {
    size_t batch = grad_output.shape()[0];
    size_t seq_len = grad_output.shape()[1];
    size_t d_model = grad_output.shape()[2];
    
    // 1. 反向传播通过 FeedForward (reshape to 2D)
    Tensor grad_output_2d = grad_output.reshape({batch * seq_len, d_model});
    Tensor grad_ffn_2d = ffn_.backward(grad_output_2d);
    Tensor grad_ffn = grad_ffn_2d.reshape({batch, seq_len, d_model});
    
    // 2. 残差连接
    Tensor grad_post_attn = grad_output + grad_ffn;
    
    // 3. 反向传播通过 Attention
    Tensor grad_attn = attn_.backward(grad_post_attn);
    
    // 4. 残差连接
    Tensor grad_input = grad_post_attn + grad_attn;
    
    return grad_input;
}

std::vector<Tensor*> TransformerEncoderBlock::parameters() {
    auto params = attn_.parameters();
    auto ffn_params = ffn_.parameters();
    params.insert(params.end(), ffn_params.begin(), ffn_params.end());
    params.push_back(&ln1_gamma_); params.push_back(&ln1_beta_);
    params.push_back(&ln2_gamma_); params.push_back(&ln2_beta_);
    return params;
}

std::vector<Tensor*> TransformerEncoderBlock::gradients() {
    auto grads = attn_.gradients();
    auto ffn_grads = ffn_.gradients();
    grads.insert(grads.end(), ffn_grads.begin(), ffn_grads.end());
    grads.push_back(&grad_ln1_gamma_); grads.push_back(&grad_ln1_beta_);
    grads.push_back(&grad_ln2_gamma_); grads.push_back(&grad_ln2_beta_);
    return grads;
}

size_t TransformerEncoderBlock::param_count() const {
    return attn_.param_count() + ffn_.param_count() + 4 * ln1_gamma_.size();
}

void TransformerEncoderBlock::save(std::ofstream& ofs) const {
    attn_.save(ofs); ffn_.save(ofs);
    ln1_gamma_.save(ofs); ln1_beta_.save(ofs);
    ln2_gamma_.save(ofs); ln2_beta_.save(ofs);
}

void TransformerEncoderBlock::load(std::ifstream& ifs) {
    attn_.load(ifs); ffn_.load(ifs);
    Tensor g1 = Tensor::load(ifs); std::memcpy(ln1_gamma_.data(), g1.data(), ln1_gamma_.size() * sizeof(float));
    Tensor b1 = Tensor::load(ifs); std::memcpy(ln1_beta_.data(), b1.data(), ln1_beta_.size() * sizeof(float));
    Tensor g2 = Tensor::load(ifs); std::memcpy(ln2_gamma_.data(), g2.data(), ln2_gamma_.size() * sizeof(float));
    Tensor b2 = Tensor::load(ifs); std::memcpy(ln2_beta_.data(), b2.data(), ln2_beta_.size() * sizeof(float));
}

// --- TransformerDecoderBlock ---

TransformerDecoderBlock::TransformerDecoderBlock(size_t d_model, size_t num_heads, size_t d_ff, float dropout)
    : self_attn_(d_model, num_heads, dropout),
      cross_attn_(d_model, num_heads, dropout),
      ffn_(d_model, d_ff, "gelu"),
      ln1_gamma_({d_model}), ln1_beta_({d_model}),
      ln2_gamma_({d_model}), ln2_beta_({d_model}),
      ln3_gamma_({d_model}), ln3_beta_({d_model}),
      grad_ln1_gamma_({d_model}), grad_ln1_beta_({d_model}),
      grad_ln2_gamma_({d_model}), grad_ln2_beta_({d_model}),
      grad_ln3_gamma_({d_model}), grad_ln3_beta_({d_model}) {
    
    ln1_gamma_.fill(1.0f); ln1_beta_.fill(0.0f);
    ln2_gamma_.fill(1.0f); ln2_beta_.fill(0.0f);
    ln3_gamma_.fill(1.0f); ln3_beta_.fill(0.0f);
    grad_ln1_gamma_.fill(0.0f); grad_ln1_beta_.fill(0.0f);
    grad_ln2_gamma_.fill(0.0f); grad_ln2_beta_.fill(0.0f);
    grad_ln3_gamma_.fill(0.0f); grad_ln3_beta_.fill(0.0f);
}

Tensor TransformerDecoderBlock::forward(const Tensor& input, const Tensor& encoder_output) {
    encoder_output_cache_ = Tensor(encoder_output.shape());
    std::memcpy(encoder_output_cache_.data(), encoder_output.data(), encoder_output.size() * sizeof(float));
    
    // 1. Masked Self-Attention
    Tensor ln1 = math::layer_norm(input, ln1_gamma_, ln1_beta_);
    Tensor self_attn_out = self_attn_.forward(ln1);
    Tensor post_self = input + self_attn_out;
    
    // 2. Cross-Attention
    Tensor ln2 = math::layer_norm(post_self, ln2_gamma_, ln2_beta_);
    Tensor cross_attn_out = cross_attn_.forward(ln2, encoder_output, encoder_output);
    Tensor post_cross = post_self + cross_attn_out;
    
    // 3. FeedForward
    Tensor ln3 = math::layer_norm(post_cross, ln3_gamma_, ln3_beta_);
    Tensor ff_out = ffn_.forward(ln3);
    Tensor output = post_cross + ff_out;
    
    return output;
}

Tensor TransformerDecoderBlock::forward(const Tensor& input) {
    // 仅自注意力模式（GPT风格）
    Tensor ln1 = math::layer_norm(input, ln1_gamma_, ln1_beta_);
    Tensor self_attn_out = self_attn_.forward(ln1);
    Tensor post_self = input + self_attn_out;
    
    Tensor ln2 = math::layer_norm(post_self, ln2_gamma_, ln2_beta_);
    Tensor ff_out = ffn_.forward(ln2);
    Tensor output = post_self + ff_out;
    
    return output;
}

Tensor TransformerDecoderBlock::backward(const Tensor& grad_output) {
    // 简化版反向传播
    Tensor grad_ffn = ffn_.backward(grad_output);
    Tensor grad_post_self = grad_output + grad_ffn;
    Tensor grad_self = self_attn_.backward(grad_post_self);
    Tensor grad_input = grad_post_self + grad_self;
    return grad_input;
}

std::vector<Tensor*> TransformerDecoderBlock::parameters() {
    auto params = self_attn_.parameters();
    auto cross_params = cross_attn_.parameters();
    auto ffn_params = ffn_.parameters();
    params.insert(params.end(), cross_params.begin(), cross_params.end());
    params.insert(params.end(), ffn_params.begin(), ffn_params.end());
    params.push_back(&ln1_gamma_); params.push_back(&ln1_beta_);
    params.push_back(&ln2_gamma_); params.push_back(&ln2_beta_);
    params.push_back(&ln3_gamma_); params.push_back(&ln3_beta_);
    return params;
}

std::vector<Tensor*> TransformerDecoderBlock::gradients() {
    auto grads = self_attn_.gradients();
    auto cross_grads = cross_attn_.gradients();
    auto ffn_grads = ffn_.gradients();
    grads.insert(grads.end(), cross_grads.begin(), cross_grads.end());
    grads.insert(grads.end(), ffn_grads.begin(), ffn_grads.end());
    grads.push_back(&grad_ln1_gamma_); grads.push_back(&grad_ln1_beta_);
    grads.push_back(&grad_ln2_gamma_); grads.push_back(&grad_ln2_beta_);
    grads.push_back(&grad_ln3_gamma_); grads.push_back(&grad_ln3_beta_);
    return grads;
}

size_t TransformerDecoderBlock::param_count() const {
    return self_attn_.param_count() + cross_attn_.param_count() + ffn_.param_count() + 6 * ln1_gamma_.size();
}

void TransformerDecoderBlock::save(std::ofstream& ofs) const {
    self_attn_.save(ofs); cross_attn_.save(ofs); ffn_.save(ofs);
    ln1_gamma_.save(ofs); ln1_beta_.save(ofs);
    ln2_gamma_.save(ofs); ln2_beta_.save(ofs);
    ln3_gamma_.save(ofs); ln3_beta_.save(ofs);
}

void TransformerDecoderBlock::load(std::ifstream& ifs) {
    self_attn_.load(ifs); cross_attn_.load(ifs); ffn_.load(ifs);
    Tensor g1 = Tensor::load(ifs); std::memcpy(ln1_gamma_.data(), g1.data(), ln1_gamma_.size() * sizeof(float));
    Tensor b1 = Tensor::load(ifs); std::memcpy(ln1_beta_.data(), b1.data(), ln1_beta_.size() * sizeof(float));
    Tensor g2 = Tensor::load(ifs); std::memcpy(ln2_gamma_.data(), g2.data(), ln2_gamma_.size() * sizeof(float));
    Tensor b2 = Tensor::load(ifs); std::memcpy(ln2_beta_.data(), b2.data(), ln2_beta_.size() * sizeof(float));
    Tensor g3 = Tensor::load(ifs); std::memcpy(ln3_gamma_.data(), g3.data(), ln3_gamma_.size() * sizeof(float));
    Tensor b3 = Tensor::load(ifs); std::memcpy(ln3_beta_.data(), b3.data(), ln3_beta_.size() * sizeof(float));
}

} // namespace ai
