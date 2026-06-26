#include "ai/utils.hpp"
#include <fstream>
#include <algorithm>
#include <random>
#include <chrono>
#include <sstream>
#include <iomanip>

namespace ai {
namespace utils {

// --- Random ---

Random::Random() : gen_(std::random_device{}()) {}

Random& Random::instance() {
    static Random inst;
    return inst;
}

void Random::xavier_init(Tensor& t, size_t fan_in, size_t fan_out) {
    float limit = std::sqrt(6.0f / static_cast<float>(fan_in + fan_out));
    std::uniform_real_distribution<float> dist(-limit, limit);
    for (size_t i = 0; i < t.size(); ++i) {
        t.data()[i] = dist(gen_);
    }
}

void Random::he_init(Tensor& t, size_t fan_in) {
    float std = std::sqrt(2.0f / static_cast<float>(fan_in));
    std::normal_distribution<float> dist(0.0f, std);
    for (size_t i = 0; i < t.size(); ++i) {
        t.data()[i] = dist(gen_);
    }
}

void Random::uniform_init(Tensor& t, float min_val, float max_val) {
    std::uniform_real_distribution<float> dist(min_val, max_val);
    for (size_t i = 0; i < t.size(); ++i) {
        t.data()[i] = dist(gen_);
    }
}

void Random::normal_init(Tensor& t, float mean, float std) {
    std::normal_distribution<float> dist(mean, std);
    for (size_t i = 0; i < t.size(); ++i) {
        t.data()[i] = dist(gen_);
    }
}

void Random::std_normal_init(Tensor& t) {
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < t.size(); ++i) {
        t.data()[i] = dist(gen_);
    }
}

// 便利函数
void xavier_init(Tensor& t, size_t fan_in, size_t fan_out) {
    Random::instance().xavier_init(t, fan_in, fan_out);
}
void he_init(Tensor& t, size_t fan_in) {
    Random::instance().he_init(t, fan_in);
}
void uniform_init(Tensor& t, float min_val, float max_val) {
    Random::instance().uniform_init(t, min_val, max_val);
}
void normal_init(Tensor& t, float mean, float std) {
    Random::instance().normal_init(t, mean, std);
}
void std_normal_init(Tensor& t) {
    Random::instance().std_normal_init(t);
}

// --- Tokenizer ---

Tokenizer::Tokenizer() {
    // 预留特殊token位置
    vocab_.reserve(256);
    vocab_.push_back("<BOS>");     // 0
    vocab_.push_back("<EOS>");     // 1
    vocab_.push_back("<PAD>");     // 2
    vocab_.push_back("<UNK>");     // 3
    vocab_.push_back("<|system|>"); // 4
    vocab_.push_back("<|user|>");  // 5
    vocab_.push_back("<|assistant|>"); // 6
    vocab_.push_back("<|turn_end|>");   // 7
    token_to_id_["<BOS>"] = 0;
    token_to_id_["<EOS>"] = 1;
    token_to_id_["<PAD>"] = 2;
    token_to_id_["<UNK>"] = 3;
    token_to_id_["<|system|>"] = 4;
    token_to_id_["<|user|>"] = 5;
    token_to_id_["<|assistant|>"] = 6;
    token_to_id_["<|turn_end|>"] = 7;
}

void Tokenizer::build_vocab(const std::string& text) {
    for (char c : text) {
        std::string token(1, c);
        if (token_to_id_.find(token) == token_to_id_.end()) {
            int id = static_cast<int>(vocab_.size());
            vocab_.push_back(token);
            token_to_id_[token] = id;
        }
    }
}

void Tokenizer::build_vocab(const std::vector<std::string>& texts) {
    for (const auto& text : texts) {
        build_vocab(text);
    }
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
    std::vector<int> tokens;
    for (char c : text) {
        std::string token(1, c);
        auto it = token_to_id_.find(token);
        if (it != token_to_id_.end()) {
            tokens.push_back(it->second);
        } else {
            tokens.push_back(unk_id_);
        }
    }
    return tokens;
}

std::string Tokenizer::decode(const std::vector<int>& tokens) const {
    std::string text;
    for (int id : tokens) {
        if (id >= 0 && id < static_cast<int>(vocab_.size())) {
            text += vocab_[id];
        }
    }
    return text;
}

void Tokenizer::save(const std::string& path) const {
    std::ofstream ofs(path, std::ios::binary);
    size_t vocab_size = vocab_.size();
    ofs.write(reinterpret_cast<const char*>(&vocab_size), sizeof(vocab_size));
    for (const auto& token : vocab_) {
        size_t len = token.size();
        ofs.write(reinterpret_cast<const char*>(&len), sizeof(len));
        ofs.write(token.data(), len);
    }
}

void Tokenizer::load(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    size_t vocab_size;
    ifs.read(reinterpret_cast<char*>(&vocab_size), sizeof(vocab_size));
    vocab_.clear();
    token_to_id_.clear();
    for (size_t i = 0; i < vocab_size; ++i) {
        size_t len;
        ifs.read(reinterpret_cast<char*>(&len), sizeof(len));
        std::string token(len, '\0');
        ifs.read(&token[0], len);
        vocab_.push_back(token);
        token_to_id_[token] = static_cast<int>(i);
    }
}

// --- DialogueDataset ---

void DialogueDataset::add_dialogue(const Dialogue& dialogue) {
    dialogues_.push_back(dialogue);
}

std::vector<int> DialogueDataset::encode_dialogue(const Dialogue& dialogue, const Tokenizer& tokenizer) const {
    std::string text;
    
    // System prompt
    if (!dialogue.system_prompt.empty()) {
        text += tokenizer.system_token() + dialogue.system_prompt;
    }
    
    // Turns
    for (const auto& turn : dialogue.turns) {
        if (turn.role == "user") {
            text += tokenizer.user_token() + turn.content;
        } else if (turn.role == "assistant") {
            text += tokenizer.assistant_token() + turn.content;
        }
    }
    text += tokenizer.turn_end_token();
    
    return tokenizer.encode(text);
}

std::vector<std::pair<std::vector<int>, std::vector<int>>> DialogueDataset::get_training_pairs(
    const Tokenizer& tokenizer, size_t max_length) const {
    
    std::vector<std::pair<std::vector<int>, std::vector<int>>> pairs;
    
    for (const auto& dialogue : dialogues_) {
        auto tokens = encode_dialogue(dialogue, tokenizer);
        
        if (tokens.empty()) continue;
        if (tokens.size() > max_length) {
            tokens.resize(max_length);
        }
        
        // 自回归训练：input = tokens[:-1], label = tokens[1:]
        std::vector<int> input(tokens.begin(), tokens.end() - 1);
        std::vector<int> label(tokens.begin() + 1, tokens.end());
        
        pairs.push_back({input, label});
    }
    
    return pairs;
}

void DialogueDataset::shuffle(unsigned seed) {
    if (seed == 0) {
        seed = std::random_device{}();
    }
    std::mt19937 g(seed);
    std::shuffle(dialogues_.begin(), dialogues_.end(), g);
}

void DialogueDataset::load_preset_dialogues() {
    // 预设中文对话数据（用于演示训练）
    {
        Dialogue d;
        d.system_prompt = "你是一个友好的AI助手。";
        d.turns.push_back({"user", "你好"});
        d.turns.push_back({"assistant", "你好！很高兴见到你。有什么我可以帮你的吗？"});
        add_dialogue(d);
    }
    {
        Dialogue d;
        d.system_prompt = "你是一个友好的AI助手。";
        d.turns.push_back({"user", "今天天气怎么样？"});
        d.turns.push_back({"assistant", "我没有实时天气数据，但我可以建议你查看天气预报应用。一般来说，出门前记得带伞是个好主意！"});
        add_dialogue(d);
    }
    {
        Dialogue d;
        d.system_prompt = "你是一个友好的AI助手。";
        d.turns.push_back({"user", "你会做什么？"});
        d.turns.push_back({"assistant", "我可以回答你的问题、陪你聊天、帮你写代码、解答数学题，或者只是陪你聊聊天。你想做什么呢？"});
        add_dialogue(d);
    }
    {
        Dialogue d;
        d.system_prompt = "你是一个友好的AI助手。";
        d.turns.push_back({"user", "讲个笑话"});
        d.turns.push_back({"assistant", "为什么程序员总是分不清圣诞节和万圣节？因为Oct 31 == Dec 25！"});
        add_dialogue(d);
    }
    {
        Dialogue d;
        d.system_prompt = "你是一个友好的AI助手。";
        d.turns.push_back({"user", "再见"});
        d.turns.push_back({"assistant", "再见！祝你有美好的一天。如果还有问题，随时找我！"});
        add_dialogue(d);
    }
    {
        Dialogue d;
        d.system_prompt = "你是一个友好的AI助手。";
        d.turns.push_back({"user", "帮我算一下2+3等于多少"});
        d.turns.push_back({"assistant", "2+3=5。这是一个简单的加法！"});
        add_dialogue(d);
    }
    {
        Dialogue d;
        d.system_prompt = "你是一个友好的AI助手。";
        d.turns.push_back({"user", "推荐一本书"});
        d.turns.push_back({"assistant", "我推荐《三体》，这是刘慈欣写的科幻小说，非常精彩。如果你喜欢其他类型，请告诉我！"});
        add_dialogue(d);
    }
    {
        Dialogue d;
        d.system_prompt = "你是一个友好的AI助手。";
        d.turns.push_back({"user", "谢谢"});
        d.turns.push_back({"assistant", "不客气！很高兴能帮到你。还有什么需要帮忙的吗？"});
        add_dialogue(d);
    }
    {
        Dialogue d;
        d.system_prompt = "你是一个友好的AI助手。";
        d.turns.push_back({"user", "你是谁"});
        d.turns.push_back({"assistant", "我是一个AI助手，由C++代码构建而成。我可以回答问题、陪你聊天、帮你解决问题。"});
        add_dialogue(d);
    }
    {
        Dialogue d;
        d.system_prompt = "你是一个友好的AI助手。";
        d.turns.push_back({"user", "你会写代码吗"});
        d.turns.push_back({"assistant", "当然！我可以帮你写C++、Python、JavaScript等代码。请告诉我你的需求。"});
        add_dialogue(d);
    }
}

// --- 模型保存/加载 ---

bool save_model(const std::string& path, const std::vector<Tensor*>& params) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) return false;
    
    size_t count = params.size();
    ofs.write(reinterpret_cast<const char*>(&count), sizeof(count));
    for (auto* p : params) {
        p->save(ofs);
    }
    return ofs.good();
}

bool load_model(const std::string& path, const std::vector<Tensor*>& params) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) return false;
    
    size_t count;
    ifs.read(reinterpret_cast<char*>(&count), sizeof(count));
    if (count != params.size()) {
        return false;
    }
    for (auto* p : params) {
        Tensor loaded = Tensor::load(ifs);
        if (!loaded.same_shape(*p)) {
            return false;
        }
        std::memcpy(p->data(), loaded.data(), p->size() * sizeof(float));
    }
    return ifs.good();
}

// --- Timer ---

Timer::Timer() : start_(std::chrono::high_resolution_clock::now()) {}

void Timer::start() {
    start_ = std::chrono::high_resolution_clock::now();
}

double Timer::elapsed_ms() const {
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start_).count();
}

double Timer::elapsed_s() const {
    return elapsed_ms() / 1000.0;
}

void Timer::print(const std::string& label) const {
    if (!label.empty()) std::cout << "[" << label << "] ";
    std::cout << "Elapsed: " << std::fixed << std::setprecision(2) 
              << elapsed_ms() << " ms" << std::endl;
}

// --- 字符串工具 ---

std::vector<std::string> split(const std::string& s, char delimiter) {
    std::vector<std::string> tokens;
    std::stringstream ss(s);
    std::string token;
    while (std::getline(ss, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

std::string trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\n\r");
    if (start == std::string::npos) return "";
    size_t end = s.find_last_not_of(" \t\n\r");
    return s.substr(start, end - start + 1);
}

// --- 对话格式化工具 ---

std::string format_dialogue_turn(const std::string& role, const std::string& content) {
    if (role == "user") {
        return "<|user|>" + content;
    } else if (role == "assistant") {
        return "<|assistant|>" + content;
    } else if (role == "system") {
        return "<|system|>" + content;
    }
    return content;
}

} // namespace utils
} // namespace ai
