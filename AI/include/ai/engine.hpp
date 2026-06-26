#pragma once

#include "model.hpp"
#include "utils.hpp"

namespace ai {

/**
 * @brief 优化器基类
 */
class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual void step() = 0;
    virtual void zero_grad() = 0;
    virtual void set_lr(float lr) {};
    virtual float lr() const { return 0.0f; };
};

/**
 * @brief AdamW 优化器
 */
class AdamW : public Optimizer {
public:
    AdamW(std::vector<Tensor*> params, std::vector<Tensor*> grads,
          float lr = 1e-4f, float beta1 = 0.9f, float beta2 = 0.999f,
          float eps = 1e-8f, float weight_decay = 0.01f);
    
    void step() override;
    void zero_grad() override;
    
    void set_lr(float lr) { lr_ = lr; }
    float lr() const { return lr_; }

private:
    std::vector<Tensor*> params_;
    std::vector<Tensor*> grads_;
    
    float lr_;
    float beta1_, beta2_;
    float eps_;
    float weight_decay_;
    
    int timestep_;
    
    // 一阶和二阶矩估计
    std::vector<Tensor> m_;
    std::vector<Tensor> v_;
};

/**
 * @brief 学习率调度器
 */
class LRScheduler {
public:
    virtual ~LRScheduler() = default;
    virtual float get_lr(int step) const = 0;
};

class CosineScheduler : public LRScheduler {
public:
    CosineScheduler(float max_lr, float min_lr, int warmup_steps, int total_steps);
    float get_lr(int step) const override;

private:
    float max_lr_, min_lr_;
    int warmup_steps_;
    int total_steps_;
};

/**
 * @brief 训练引擎
 */
class Trainer {
public:
    Trainer(TransformerModel& model, std::unique_ptr<Optimizer> optimizer,
            std::unique_ptr<LRScheduler> scheduler = nullptr);
    
    // 训练一个batch
    float train_step(const std::vector<std::vector<int>>& input_ids,
                     const std::vector<std::vector<int>>& labels);
    
    // 完整训练epoch
    float train_epoch(const std::vector<std::vector<int>>& data, size_t batch_size);
    
    // 验证
    float evaluate(const std::vector<std::vector<int>>& data, size_t batch_size);
    
    int step_count() const { return step_count_; }

private:
    TransformerModel& model_;
    std::unique_ptr<Optimizer> optimizer_;
    std::unique_ptr<LRScheduler> scheduler_;
    int step_count_;
};

/**
 * @brief 推理引擎
 */
class InferenceEngine {
public:
    explicit InferenceEngine(TransformerModel& model);
    
    // 贪婪解码
    std::vector<int> generate(const std::vector<int>& prompt, 
                               size_t max_length = 50,
                               int eos_token = 1);
    
    // 采样解码
    std::vector<int> generate_sample(const std::vector<int>& prompt,
                                      size_t max_length = 50,
                                      float temperature = 1.0f,
                                      int top_k = 50,
                                      int eos_token = 1);
    
    // 计算概率分布
    std::vector<float> get_probs(const std::vector<int>& tokens);

private:
    TransformerModel& model_;
    
    // 采样辅助函数
    int sample_from_probs(const std::vector<float>& probs, float temperature);
    int argmax(const std::vector<float>& logits);
    
    // 温度缩放与top-k
    std::vector<float> apply_temperature(const std::vector<float>& logits, float temp);
    std::vector<float> top_k_filter(const std::vector<float>& logits, int k);
};

/**
 * @brief 对话引擎
 *        支持多轮对话的上下文管理和回复生成
 */
class DialogueEngine {
public:
    DialogueEngine(TransformerModel& model, utils::Tokenizer& tokenizer);
    
    // 设置系统提示词
    void set_system_prompt(const std::string& prompt);
    
    // 添加用户消息到上下文
    void add_user_message(const std::string& message);
    
    // 添加助手消息到上下文
    void add_assistant_message(const std::string& message);
    
    // 生成回复（基于当前对话上下文）
    std::string respond(const std::string& user_input, 
                         size_t max_length = 128,
                         float temperature = 0.8f,
                         int top_k = 20);
    
    // 重置对话上下文
    void reset();
    
    // 获取当前对话历史（用于调试）
    std::string get_context() const;

private:
    TransformerModel& model_;
    utils::Tokenizer& tokenizer_;
    
    std::string system_prompt_;
    std::vector<utils::DialogueDataset::Turn> turns_;
    
    // 构建对话上下文为token序列
    std::vector<int> build_context_tokens() const;
    
    // 从生成的token中提取助手回复（去除角色标记）
    std::string extract_reply(const std::vector<int>& tokens) const;
};

} // namespace ai
