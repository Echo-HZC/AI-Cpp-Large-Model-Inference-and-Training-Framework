#include "ai/model.hpp"
#include "ai/math.hpp"
#include "ai/utils.hpp"

namespace ai {

// --- Config ---

void TransformerModel::Config::save(std::ofstream& ofs) const {
    ofs.write(reinterpret_cast<const char*>(&vocab_size), sizeof(vocab_size));
    ofs.write(reinterpret_cast<const char*>(&d_model), sizeof(d_model));
    ofs.write(reinterpret_cast<const char*>(&num_heads), sizeof(num_heads));
    ofs.write(reinterpret_cast<const char*>(&num_layers), sizeof(num_layers));
    ofs.write(reinterpret_cast<const char*>(&d_ff), sizeof(d_ff));
    ofs.write(reinterpret_cast<const char*>(&max_seq_len), sizeof(max_seq_len));
    ofs.write(reinterpret_cast<const char*>(&dropout), sizeof(dropout));
}

void TransformerModel::Config::load(std::ifstream& ifs) {
    ifs.read(reinterpret_cast<char*>(&vocab_size), sizeof(vocab_size));
    ifs.read(reinterpret_cast<char*>(&d_model), sizeof(d_model));
    ifs.read(reinterpret_cast<char*>(&num_heads), sizeof(num_heads));
    ifs.read(reinterpret_cast<char*>(&num_layers), sizeof(num_layers));
    ifs.read(reinterpret_cast<char*>(&d_ff), sizeof(d_ff));
    ifs.read(reinterpret_cast<char*>(&max_seq_len), sizeof(max_seq_len));
    ifs.read(reinterpret_cast<char*>(&dropout), sizeof(dropout));
}

// --- TransformerModel ---

TransformerModel::TransformerModel(const Config& config)
    : config_(config),
      embedding_(config.vocab_size, config.d_model),
      final_ln_gamma_({config.d_model}), final_ln_beta_({config.d_model}),
      grad_final_ln_gamma_({config.d_model}), grad_final_ln_beta_({config.d_model}),
      output_proj_(config.d_model, config.vocab_size, false) {
    
    for (size_t i = 0; i < config.num_layers; ++i) {
        layers_.emplace_back(config.d_model, config.num_heads, config.d_ff, config.dropout);
    }
    
    final_ln_gamma_.fill(1.0f);
    final_ln_beta_.fill(0.0f);
    grad_final_ln_gamma_.fill(0.0f);
    grad_final_ln_beta_.fill(0.0f);
    
    // 预计算位置编码
    pos_encoding_ = math::sinusoidal_positional_encoding(config.max_seq_len, config.d_model);
}

Tensor TransformerModel::token_ids_to_tensor(const std::vector<std::vector<int>>& ids) const {
    if (ids.empty()) return Tensor({0, 0});
    
    size_t batch = ids.size();
    size_t seq_len = ids[0].size();
    
    Tensor result({batch, seq_len});
    for (size_t b = 0; b < batch; ++b) {
        for (size_t s = 0; s < seq_len; ++s) {
            result(b, s) = static_cast<float>(ids[b][s]);
        }
    }
    return result;
}

Tensor TransformerModel::token_ids_to_tensor(const std::vector<int>& ids) const {
    Tensor result({1, ids.size()});
    for (size_t s = 0; s < ids.size(); ++s) {
        result(0, s) = static_cast<float>(ids[s]);
    }
    return result;
}

Tensor TransformerModel::add_positional_encoding(const Tensor& x) const {
    size_t batch = x.shape()[0];
    size_t seq_len = x.shape()[1];
    size_t d_model = x.shape()[2];
    
    Tensor result(x.shape());
    for (size_t b = 0; b < batch; ++b) {
        for (size_t s = 0; s < seq_len; ++s) {
            for (size_t d = 0; d < d_model; ++d) {
                result(b, s, d) = x(b, s, d) + pos_encoding_(s, d);
            }
        }
    }
    return result;
}

Tensor TransformerModel::forward(const std::vector<std::vector<int>>& input_ids) {
    Tensor input = token_ids_to_tensor(input_ids);
    
    // Embedding: [batch, seq_len] -> [batch, seq_len, d_model]
    Tensor x = embedding_.forward(input);
    
    // 添加位置编码
    x = add_positional_encoding(x);
    
    embedding_cache_ = x;
    
    // 通过各层
    for (auto& layer : layers_) {
        x = layer.forward(x);
    }
    
    // 最终 LayerNorm
    x = math::layer_norm(x, final_ln_gamma_, final_ln_beta_);
    
    // 输出投影: [batch, seq_len, d_model] -> [batch, seq_len, vocab_size]
    size_t batch = x.shape()[0];
    size_t seq_len = x.shape()[1];
    
    Tensor x_flat = x.reshape({batch * seq_len, config_.d_model});
    Tensor logits_flat = output_proj_.forward(x_flat);
    Tensor logits = logits_flat.reshape({batch, seq_len, config_.vocab_size});
    
    return logits;
}

float TransformerModel::compute_loss(const Tensor& logits, const std::vector<std::vector<int>>& labels) {
    size_t batch = logits.shape()[0];
    size_t seq_len = logits.shape()[1];
    size_t vocab_size = logits.shape()[2];
    
    float total_loss = 0.0f;
    size_t total_tokens = 0;
    
    for (size_t b = 0; b < batch; ++b) {
        for (size_t s = 0; s < seq_len; ++s) {
            int label = labels[b][s];
            if (label < 0 || label >= static_cast<int>(vocab_size)) continue;
            
            // 计算 softmax 概率
            Tensor logit_slice({vocab_size});
            for (size_t v = 0; v < vocab_size; ++v) {
                logit_slice(v) = logits(b, s, v);
            }
            
            Tensor probs = math::softmax(logit_slice);
            float prob = probs(static_cast<size_t>(label));
            
            total_loss += -std::log(std::max(prob, 1e-10f));
            total_tokens++;
        }
    }
    
    return total_tokens > 0 ? total_loss / static_cast<float>(total_tokens) : 0.0f;
}

void TransformerModel::backward(const Tensor& grad_logits) {
    // 反向传播通过输出投影
    size_t batch = grad_logits.shape()[0];
    size_t seq_len = grad_logits.shape()[1];
    
    Tensor grad_flat = grad_logits.reshape({batch * seq_len, config_.vocab_size});
    Tensor grad_x_flat = output_proj_.backward(grad_flat);
    
    Tensor grad_x = grad_x_flat.reshape({batch, seq_len, config_.d_model});
    
    // 反向传播通过各层
    for (int i = static_cast<int>(layers_.size()) - 1; i >= 0; --i) {
        grad_x = layers_[i].backward(grad_x);
    }
    
    // 嵌入层反向传播
    embedding_.backward(grad_x);
}

Tensor TransformerModel::forward_step(const std::vector<int>& input_ids) {
    Tensor input = token_ids_to_tensor(input_ids);
    
    Tensor x = embedding_.forward(input);
    x = add_positional_encoding(x);
    
    for (auto& layer : layers_) {
        x = layer.forward(x);
    }
    
    x = math::layer_norm(x, final_ln_gamma_, final_ln_beta_);
    
    // 取最后一个位置
    size_t seq_len = x.shape()[1];
    Tensor last = x.slice(1, seq_len - 1, seq_len); // [batch, 1, d_model]
    Tensor last_flat = last.reshape({1, config_.d_model});
    
    Tensor logits = output_proj_.forward(last_flat);
    return logits; // [1, vocab_size]
}

std::vector<Tensor*> TransformerModel::parameters() {
    std::vector<Tensor*> params;
    
    auto embed_params = embedding_.parameters();
    params.insert(params.end(), embed_params.begin(), embed_params.end());
    
    for (auto& layer : layers_) {
        auto layer_params = layer.parameters();
        params.insert(params.end(), layer_params.begin(), layer_params.end());
    }
    
    params.push_back(&final_ln_gamma_);
    params.push_back(&final_ln_beta_);
    
    auto out_params = output_proj_.parameters();
    params.insert(params.end(), out_params.begin(), out_params.end());
    
    return params;
}

std::vector<Tensor*> TransformerModel::gradients() {
    std::vector<Tensor*> grads;
    
    auto embed_grads = embedding_.gradients();
    grads.insert(grads.end(), embed_grads.begin(), embed_grads.end());
    
    for (auto& layer : layers_) {
        auto layer_grads = layer.gradients();
        grads.insert(grads.end(), layer_grads.begin(), layer_grads.end());
    }
    
    grads.push_back(&grad_final_ln_gamma_);
    grads.push_back(&grad_final_ln_beta_);
    
    auto out_grads = output_proj_.gradients();
    grads.insert(grads.end(), out_grads.begin(), out_grads.end());
    
    return grads;
}

size_t TransformerModel::param_count() const {
    size_t count = embedding_.param_count();
    for (const auto& layer : layers_) {
        count += layer.param_count();
    }
    count += final_ln_gamma_.size() + final_ln_beta_.size();
    count += output_proj_.param_count();
    return count;
}

void TransformerModel::save(const std::string& path) const {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) return;
    
    config_.save(ofs);
    embedding_.save(ofs);
    for (const auto& layer : layers_) {
        layer.save(ofs);
    }
    final_ln_gamma_.save(ofs); final_ln_beta_.save(ofs);
    output_proj_.save(ofs);
}

void TransformerModel::load(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) return;
    
    Config config;
    config.load(ifs);
    if (config.vocab_size != config_.vocab_size ||
        config.d_model != config_.d_model ||
        config.num_layers != config_.num_layers) {
        // 配置不匹配，可能需要重新初始化
        return;
    }
    
    embedding_.load(ifs);
    for (auto& layer : layers_) {
        layer.load(ifs);
    }
    Tensor g = Tensor::load(ifs); std::memcpy(final_ln_gamma_.data(), g.data(), final_ln_gamma_.size() * sizeof(float));
    Tensor b = Tensor::load(ifs); std::memcpy(final_ln_beta_.data(), b.data(), final_ln_beta_.size() * sizeof(float));
    output_proj_.load(ifs);
}

void TransformerModel::set_training(bool training) {
    for (auto& layer : layers_) {
        layer.set_training(training);
    }
    embedding_.set_training(training);
    output_proj_.set_training(training);
}

} // namespace ai
