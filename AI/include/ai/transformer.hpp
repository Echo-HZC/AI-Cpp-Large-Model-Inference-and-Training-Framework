#pragma once

#include "layers.hpp"

namespace ai {

/**
 * @brief Transformer Encoder Block
 *        包含：Multi-Head Attention + Add&Norm + FeedForward + Add&Norm
 */
class TransformerEncoderBlock : public Layer {
public:
    TransformerEncoderBlock(size_t d_model, size_t num_heads, size_t d_ff, 
                             float dropout = 0.1f);
    
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    
    size_t param_count() const override;
    std::string name() const override { return "TransformerEncoderBlock"; }
    
    void save(std::ofstream& ofs) const override;
    void load(std::ifstream& ifs) override;

private:
    MultiHeadAttention attn_;
    FeedForward ffn_;
    
    // LayerNorm 参数
    Tensor ln1_gamma_, ln1_beta_;
    Tensor ln2_gamma_, ln2_beta_;
    
    Tensor grad_ln1_gamma_, grad_ln1_beta_;
    Tensor grad_ln2_gamma_, grad_ln2_beta_;
    
    // 缓存
    Tensor pre_ln1_, post_attn_, pre_ln2_, post_ffn_;
    Tensor input_cache_;
};

/**
 * @brief Transformer Decoder Block
 *        包含：Masked Self-Attention + Cross-Attention + FeedForward
 */
class TransformerDecoderBlock : public Layer {
public:
    TransformerDecoderBlock(size_t d_model, size_t num_heads, size_t d_ff,
                             float dropout = 0.1f);
    
    // 自注意力 + 交叉注意力版本
    Tensor forward(const Tensor& input, const Tensor& encoder_output);
    
    // 仅自注意力（用于纯解码器模型如GPT）
    Tensor forward(const Tensor& input) override;
    
    Tensor backward(const Tensor& grad_output) override;
    
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    
    size_t param_count() const override;
    std::string name() const override { return "TransformerDecoderBlock"; }
    
    void save(std::ofstream& ofs) const override;
    void load(std::ifstream& ifs) override;

private:
    MultiHeadAttention self_attn_;   // masked self-attention
    MultiHeadAttention cross_attn_;  // cross-attention (decoder -> encoder)
    FeedForward ffn_;
    
    // LayerNorm 参数
    Tensor ln1_gamma_, ln1_beta_;
    Tensor ln2_gamma_, ln2_beta_;
    Tensor ln3_gamma_, ln3_beta_;
    
    Tensor grad_ln1_gamma_, grad_ln1_beta_;
    Tensor grad_ln2_gamma_, grad_ln2_beta_;
    Tensor grad_ln3_gamma_, grad_ln3_beta_;
    
    Tensor encoder_output_cache_;
};

} // namespace ai
