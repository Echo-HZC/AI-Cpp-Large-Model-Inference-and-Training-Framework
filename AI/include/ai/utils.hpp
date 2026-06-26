#pragma once

#include "tensor.hpp"
#include <random>
#include <chrono>
#include <string>
#include <sstream>
#include <iomanip>
#include <unordered_map>
#include <vector>

namespace ai {

/**
 * @brief 工具函数
 */
namespace utils {

// 随机初始化
class Random {
public:
    static Random& instance();
    
    // Xavier/Glorot 初始化
    void xavier_init(Tensor& t, size_t fan_in, size_t fan_out);
    // He 初始化
    void he_init(Tensor& t, size_t fan_in);
    // 均匀分布初始化
    void uniform_init(Tensor& t, float min_val, float max_val);
    // 正态分布初始化
    void normal_init(Tensor& t, float mean, float std);
    // 标准正态分布
    void std_normal_init(Tensor& t);
    
private:
    Random();
    std::mt19937 gen_;
};

// 初始化辅助函数
void xavier_init(Tensor& t, size_t fan_in, size_t fan_out);
void he_init(Tensor& t, size_t fan_in);
void uniform_init(Tensor& t, float min_val, float max_val);
void normal_init(Tensor& t, float mean, float std);
void std_normal_init(Tensor& t);

// 分词器（支持对话角色标记）
class Tokenizer {
public:
    Tokenizer();
    
    // 构建词汇表
    void build_vocab(const std::string& text);
    void build_vocab(const std::vector<std::string>& texts);
    
    // 编码/解码
    std::vector<int> encode(const std::string& text) const;
    std::string decode(const std::vector<int>& tokens) const;
    
    // 特殊token
    int bos_id() const { return bos_id_; }
    int eos_id() const { return eos_id_; }
    int pad_id() const { return pad_id_; }
    int unk_id() const { return unk_id_; }
    
    // 对话角色标记
    int system_id() const { return system_id_; }
    int user_id() const { return user_id_; }
    int assistant_id() const { return assistant_id_; }
    int turn_end_id() const { return turn_end_id_; }
    
    // 构建对话角色标记的字符串
    std::string system_token() const { return "<|system|>"; }
    std::string user_token() const { return "<|user|>"; }
    std::string assistant_token() const { return "<|assistant|>"; }
    std::string turn_end_token() const { return "<|turn_end|>"; }
    
    size_t vocab_size() const { return vocab_.size(); }
    
    void save(const std::string& path) const;
    void load(const std::string& path);
    
private:
    std::vector<std::string> vocab_;       // id -> token
    std::unordered_map<std::string, int> token_to_id_;
    int bos_id_ = 0;
    int eos_id_ = 1;
    int pad_id_ = 2;
    int unk_id_ = 3;
    int system_id_ = 4;
    int user_id_ = 5;
    int assistant_id_ = 6;
    int turn_end_id_ = 7;
};

// 对话数据集
class DialogueDataset {
public:
    struct Turn {
        std::string role;       // "user" 或 "assistant"
        std::string content;
    };
    
    struct Dialogue {
        std::string system_prompt;
        std::vector<Turn> turns;
    };
    
    DialogueDataset() = default;
    
    // 添加对话
    void add_dialogue(const Dialogue& dialogue);
    
    // 获取对话（编码为token序列）
    // 格式: [<|system|> system_prompt <|user|> user_msg <|assistant|> assistant_msg <|turn_end|> ...]
    std::vector<int> encode_dialogue(const Dialogue& dialogue, const Tokenizer& tokenizer) const;
    
    // 将对话编码为 (input, label) 对，用于自回归训练
    // input:  对话前缀（不含最后一个token）
    // label:  对话（每个位置预测下一个token）
    std::vector<std::pair<std::vector<int>, std::vector<int>>> get_training_pairs(
        const Tokenizer& tokenizer, size_t max_length = 256) const;
    
    size_t size() const { return dialogues_.size(); }
    const Dialogue& get(size_t idx) const { return dialogues_[idx]; }
    
    void shuffle(unsigned seed = 0);
    
    // 内置预设对话数据（用于演示）
    void load_preset_dialogues();
    
private:
    std::vector<Dialogue> dialogues_;
};

// 模型保存/加载
bool save_model(const std::string& path, const std::vector<Tensor*>& params);
bool load_model(const std::string& path, const std::vector<Tensor*>& params);

// 时间统计
class Timer {
public:
    Timer();
    void start();
    double elapsed_ms() const;
    double elapsed_s() const;
    void print(const std::string& label = "") const;
private:
    std::chrono::high_resolution_clock::time_point start_;
};

// 字符串工具
std::vector<std::string> split(const std::string& s, char delimiter);
std::string trim(const std::string& s);

// 对话格式化工具
std::string format_dialogue_turn(const std::string& role, const std::string& content);

} // namespace utils

} // namespace ai
