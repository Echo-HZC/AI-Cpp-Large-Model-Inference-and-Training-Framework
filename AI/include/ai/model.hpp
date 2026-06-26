#pragma once

#include "transformer.hpp"
#include "utils.hpp"

namespace ai {

/**
 * @brief 完整的 Transformer 语言模型
 *        纯解码器架构（GPT风格），支持自回归文本生成
 */
class TransformerModel {
public:
    struct Config {
        size_t vocab_size = 5000;
        size_t d_model = 512;
        size_t num_heads = 8;
        size_t num_layers = 6;
        size_t d_ff = 2048;
        size_t max_seq_len = 512;
        float dropout = 0.1f;
        
        void save(std::ofstream& ofs) const;
        void load(std::ifstream& ifs);
    };
    
    TransformerModel() = default;
    explicit TransformerModel(const Config& config);
    
    // 前向传播（训练模式）
    // input_ids: [batch, seq_len] - token indices
    // labels: [batch, seq_len] - 目标token indices (用于计算loss)
    // 返回 logits: [batch, seq_len, vocab_size]
    Tensor forward(const std::vector<std::vector<int>>& input_ids);
    
    // 计算交叉熵损失
    float compute_loss(const Tensor& logits, const std::vector<std::vector<int>>& labels);
    
    // 反向传播
    void backward(const Tensor& grad_logits);
    
    // 推理：生成下一个token
    // 返回 logits [batch, vocab_size]
    Tensor forward_step(const std::vector<int>& input_ids);
    
    // 获取所有参数
    std::vector<Tensor*> parameters();
    std::vector<Tensor*> gradients();
    
    // 参数统计
    size_t param_count() const;
    
    // 保存/加载
    void save(const std::string& path) const;
    void load(const std::string& path);
    
    // 训练模式开关
    void set_training(bool training);
    
    const Config& config() const { return config_; }

private:
    Config config_;
    
    Embedding embedding_;
    std::vector<TransformerEncoderBlock> layers_; // 使用EncoderBlock作为DecoderBlock的简化版
    
    // 最终LayerNorm
    Tensor final_ln_gamma_, final_ln_beta_;
    Tensor grad_final_ln_gamma_, grad_final_ln_beta_;
    
    // 输出投影（共享embedding权重或独立）
    Linear output_proj_;
    
    // 位置编码
    Tensor pos_encoding_;
    
    // 缓存
    Tensor embedding_cache_;
    
    // 辅助：将token ids转为Tensor
    Tensor token_ids_to_tensor(const std::vector<std::vector<int>>& ids) const;
    Tensor token_ids_to_tensor(const std::vector<int>& ids) const;
    
    // 添加位置编码
    Tensor add_positional_encoding(const Tensor& x) const;
};

} // namespace ai
