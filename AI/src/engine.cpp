#include "ai/engine.hpp"
#include "ai/math.hpp"
#include "ai/utils.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>

namespace ai {

// --- AdamW ---

AdamW::AdamW(std::vector<Tensor*> params, std::vector<Tensor*> grads,
             float lr, float beta1, float beta2, float eps, float weight_decay)
    : params_(std::move(params)), grads_(std::move(grads)),
      lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps), weight_decay_(weight_decay),
      timestep_(0) {
    
    for (auto* p : params_) {
        m_.emplace_back(p->shape());
        v_.emplace_back(p->shape());
        m_.back().fill(0.0f);
        v_.back().fill(0.0f);
    }
}

void AdamW::step() {
    timestep_++;
    
    float lr_t = lr_ * std::sqrt(1.0f - std::pow(beta2_, timestep_)) 
                 / (1.0f - std::pow(beta1_, timestep_));
    
    for (size_t i = 0; i < params_.size(); ++i) {
        Tensor* param = params_[i];
        Tensor* grad = grads_[i];
        Tensor& m = m_[i];
        Tensor& v = v_[i];
        
        for (size_t j = 0; j < param->size(); ++j) {
            float g = grad->data()[j];
            
            // 更新一阶和二阶矩估计
            m.data()[j] = beta1_ * m.data()[j] + (1.0f - beta1_) * g;
            v.data()[j] = beta2_ * v.data()[j] + (1.0f - beta2_) * g * g;
            
            // 更新参数（AdamW: weight decay 直接作用于参数，不在梯度上）
            param->data()[j] -= lr_t * m.data()[j] / (std::sqrt(v.data()[j]) + eps_)
                                + lr_ * weight_decay_ * param->data()[j];
        }
    }
}

void AdamW::zero_grad() {
    for (auto* grad : grads_) {
        grad->fill(0.0f);
    }
}

// --- CosineScheduler ---

CosineScheduler::CosineScheduler(float max_lr, float min_lr, int warmup_steps, int total_steps)
    : max_lr_(max_lr), min_lr_(min_lr), warmup_steps_(warmup_steps), total_steps_(total_steps) {}

float CosineScheduler::get_lr(int step) const {
    if (step < warmup_steps_) {
        return max_lr_ * static_cast<float>(step) / static_cast<float>(warmup_steps_);
    }
    
    if (step >= total_steps_) {
        return min_lr_;
    }
    
    float progress = static_cast<float>(step - warmup_steps_) 
                     / static_cast<float>(total_steps_ - warmup_steps_);
    float cosine = 0.5f * (1.0f + std::cos(progress * 3.14159265358979323846f));
    return min_lr_ + (max_lr_ - min_lr_) * cosine;
}

// --- Trainer ---

Trainer::Trainer(TransformerModel& model, std::unique_ptr<Optimizer> optimizer,
                 std::unique_ptr<LRScheduler> scheduler)
    : model_(model), optimizer_(std::move(optimizer)), 
      scheduler_(std::move(scheduler)), step_count_(0) {}

float Trainer::train_step(const std::vector<std::vector<int>>& input_ids,
                          const std::vector<std::vector<int>>& labels) {
    model_.set_training(true);
    
    // 前向传播
    Tensor logits = model_.forward(input_ids);
    
    // 计算损失
    float loss = model_.compute_loss(logits, labels);
    
    // 计算损失梯度（简化版：softmax交叉熵的梯度）
    size_t batch = logits.shape()[0];
    size_t seq_len = logits.shape()[1];
    size_t vocab_size = logits.shape()[2];
    
    Tensor grad_logits({batch, seq_len, vocab_size});
    grad_logits.fill(0.0f);
    
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
            
            // 交叉熵梯度: prob - one_hot(label)
            for (size_t v = 0; v < vocab_size; ++v) {
                float target = (v == static_cast<size_t>(label)) ? 1.0f : 0.0f;
                grad_logits(b, s, v) = (probs(v) - target) / static_cast<float>(batch * seq_len);
            }
        }
    }
    
    // 反向传播
    model_.backward(grad_logits);
    
    // 更新参数
    if (scheduler_) {
        optimizer_->set_lr(scheduler_->get_lr(step_count_));
    }
    optimizer_->step();
    optimizer_->zero_grad();
    
    step_count_++;
    return loss;
}

float Trainer::train_epoch(const std::vector<std::vector<int>>& data, size_t batch_size) {
    if (data.empty() || batch_size == 0) return 0.0f;
    
    size_t num_batches = (data.size() + batch_size - 1) / batch_size;
    float total_loss = 0.0f;
    
    for (size_t b = 0; b < num_batches; ++b) {
        size_t start = b * batch_size;
        size_t end = std::min(start + batch_size, data.size());
        
        std::vector<std::vector<int>> batch_data;
        for (size_t i = start; i < end; ++i) {
            batch_data.push_back(data[i]);
        }
        
        // 使用相同数据作为输入和标签（自回归）
        float loss = train_step(batch_data, batch_data);
        total_loss += loss;
    }
    
    return total_loss / static_cast<float>(num_batches);
}

float Trainer::evaluate(const std::vector<std::vector<int>>& data, size_t batch_size) {
    if (data.empty() || batch_size == 0) return 0.0f;
    
    model_.set_training(false);
    
    size_t num_batches = (data.size() + batch_size - 1) / batch_size;
    float total_loss = 0.0f;
    
    for (size_t b = 0; b < num_batches; ++b) {
        size_t start = b * batch_size;
        size_t end = std::min(start + batch_size, data.size());
        
        std::vector<std::vector<int>> batch_data;
        for (size_t i = start; i < end; ++i) {
            batch_data.push_back(data[i]);
        }
        
        Tensor logits = model_.forward(batch_data);
        float loss = model_.compute_loss(logits, batch_data);
        total_loss += loss;
    }
    
    return total_loss / static_cast<float>(num_batches);
}

// --- InferenceEngine ---

InferenceEngine::InferenceEngine(TransformerModel& model) : model_(model) {}

std::vector<int> InferenceEngine::generate(const std::vector<int>& prompt, 
                                           size_t max_length, int eos_token) {
    model_.set_training(false);
    
    std::vector<int> generated = prompt;
    
    for (size_t i = 0; i < max_length; ++i) {
        Tensor logits = model_.forward_step(generated);
        
        int next_token = argmax(std::vector<float>(logits.data(), logits.data() + logits.size()));
        
        generated.push_back(next_token);
        
        if (next_token == eos_token) {
            break;
        }
    }
    
    return generated;
}

std::vector<int> InferenceEngine::generate_sample(const std::vector<int>& prompt,
                                                  size_t max_length, float temperature,
                                                  int top_k, int eos_token) {
    model_.set_training(false);
    
    std::vector<int> generated = prompt;
    
    for (size_t i = 0; i < max_length; ++i) {
        Tensor logits = model_.forward_step(generated);
        
        std::vector<float> logits_vec(logits.data(), logits.data() + logits.size());
        
        // 应用温度缩放
        logits_vec = apply_temperature(logits_vec, temperature);
        
        // Top-k 过滤
        if (top_k > 0) {
            logits_vec = top_k_filter(logits_vec, top_k);
        }
        
        int next_token = sample_from_probs(logits_vec, temperature);
        
        generated.push_back(next_token);
        
        if (next_token == eos_token) {
            break;
        }
    }
    
    return generated;
}

std::vector<float> InferenceEngine::get_probs(const std::vector<int>& tokens) {
    Tensor logits = model_.forward_step(tokens);
    
    Tensor logit_t({logits.size()});
    std::memcpy(logit_t.data(), logits.data(), logits.size() * sizeof(float));
    
    Tensor probs = math::softmax(logit_t);
    
    return std::vector<float>(probs.data(), probs.data() + probs.size());
}

int InferenceEngine::sample_from_probs(const std::vector<float>& logits, float temperature) {
    // 将 logits 转换为概率
    std::vector<float> probs = logits;
    
    // 减去最大值以数值稳定性
    float max_logit = *std::max_element(probs.begin(), probs.end());
    for (auto& p : probs) {
        p = std::exp((p - max_logit) / temperature);
    }
    
    // 归一化
    float sum = 0.0f;
    for (auto p : probs) sum += p;
    for (auto& p : probs) p /= sum;
    
    // 采样
    static thread_local unsigned seed = 0;
    seed = seed * 1103515245 + 12345;
    float rand_val = static_cast<float>(seed & 0x7FFFFFFF) / 0x7FFFFFFF;
    
    float cumsum = 0.0f;
    for (size_t i = 0; i < probs.size(); ++i) {
        cumsum += probs[i];
        if (rand_val <= cumsum) {
            return static_cast<int>(i);
        }
    }
    
    return static_cast<int>(probs.size() - 1);
}

int InferenceEngine::argmax(const std::vector<float>& logits) {
    int max_idx = 0;
    float max_val = logits[0];
    for (size_t i = 1; i < logits.size(); ++i) {
        if (logits[i] > max_val) {
            max_val = logits[i];
            max_idx = static_cast<int>(i);
        }
    }
    return max_idx;
}

std::vector<float> InferenceEngine::apply_temperature(const std::vector<float>& logits, float temp) {
    if (temp <= 0.0f) return logits;
    
    std::vector<float> result = logits;
    for (auto& v : result) {
        v /= temp;
    }
    return result;
}

std::vector<float> InferenceEngine::top_k_filter(const std::vector<float>& logits, int k) {
    if (k <= 0 || k >= static_cast<int>(logits.size())) return logits;
    
    std::vector<float> sorted = logits;
    std::sort(sorted.begin(), sorted.end(), std::greater<float>());
    float threshold = sorted[k - 1];
    
    std::vector<float> result = logits;
    for (auto& v : result) {
        if (v < threshold) {
            v = -1e9f; //  mask out
        }
    }
    return result;
}

// --- DialogueEngine ---

DialogueEngine::DialogueEngine(TransformerModel& model, utils::Tokenizer& tokenizer)
    : model_(model), tokenizer_(tokenizer) {}

void DialogueEngine::set_system_prompt(const std::string& prompt) {
    system_prompt_ = prompt;
}

void DialogueEngine::add_user_message(const std::string& message) {
    turns_.push_back({"user", message});
}

void DialogueEngine::add_assistant_message(const std::string& message) {
    turns_.push_back({"assistant", message});
}

void DialogueEngine::reset() {
    system_prompt_.clear();
    turns_.clear();
}

std::string DialogueEngine::get_context() const {
    std::string context;
    if (!system_prompt_.empty()) {
        context += "<|system|>" + system_prompt_;
    }
    for (const auto& turn : turns_) {
        if (turn.role == "user") {
            context += "<|user|>" + turn.content;
        } else if (turn.role == "assistant") {
            context += "<|assistant|>" + turn.content;
        }
    }
    return context;
}

std::vector<int> DialogueEngine::build_context_tokens() const {
    std::string text;
    if (!system_prompt_.empty()) {
        text += tokenizer_.system_token() + system_prompt_;
    }
    for (const auto& turn : turns_) {
        if (turn.role == "user") {
            text += tokenizer_.user_token() + turn.content;
        } else if (turn.role == "assistant") {
            text += tokenizer_.assistant_token() + turn.content;
        }
    }
    // 添加 assistant 标记，让模型开始生成回复
    text += tokenizer_.assistant_token();
    
    return tokenizer_.encode(text);
}

std::string DialogueEngine::extract_reply(const std::vector<int>& tokens) const {
    std::string decoded = tokenizer_.decode(tokens);
    
    // 找到最后一个 <|assistant|> 标记后的内容
    std::string assistant_token = tokenizer_.assistant_token();
    size_t pos = decoded.rfind(assistant_token);
    if (pos != std::string::npos) {
        std::string reply = decoded.substr(pos + assistant_token.length());
        
        // 去掉 <|turn_end|> 及之后的内容
        size_t end_pos = reply.find(tokenizer_.turn_end_token());
        if (end_pos != std::string::npos) {
            reply = reply.substr(0, end_pos);
        }
        
        // 去掉 <|user|> 及之后的内容
        size_t user_pos = reply.find(tokenizer_.user_token());
        if (user_pos != std::string::npos) {
            reply = reply.substr(0, user_pos);
        }
        
        // 去掉特殊标记
        std::vector<std::string> special_tokens = {
            "<BOS>", "<EOS>", "<PAD>", "<UNK>",
            "<|system|>", "<|user|>", "<|assistant|>", "<|turn_end|>"
        };
        for (const auto& token : special_tokens) {
            size_t p = 0;
            while ((p = reply.find(token, p)) != std::string::npos) {
                reply.erase(p, token.length());
            }
        }
        
        return reply;
    }
    
    return decoded;
}

std::string DialogueEngine::respond(const std::string& user_input, 
                                     size_t max_length,
                                     float temperature,
                                     int top_k) {
    // 添加用户输入到上下文
    add_user_message(user_input);
    
    // 构建上下文token
    std::vector<int> prompt_tokens = build_context_tokens();
    
    // 截断到最大长度
    if (prompt_tokens.size() > model_.config().max_seq_len - max_length) {
        prompt_tokens.erase(prompt_tokens.begin(), 
            prompt_tokens.begin() + (prompt_tokens.size() - (model_.config().max_seq_len - max_length)));
    }
    
    // 推理生成
    InferenceEngine inference(model_);
    std::vector<int> generated = inference.generate_sample(prompt_tokens, max_length, temperature, top_k, tokenizer_.turn_end_id());
    
    // 提取回复
    std::string reply = extract_reply(generated);
    
    // 添加助手回复到上下文
    add_assistant_message(reply);
    
    return reply;
}

} // namespace ai
