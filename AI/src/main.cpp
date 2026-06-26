#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <random>
#include <fstream>

#include "ai/tensor.hpp"
#include "ai/math.hpp"
#include "ai/utils.hpp"
#include "ai/layers.hpp"
#include "ai/transformer.hpp"
#include "ai/model.hpp"
#include "ai/engine.hpp"

using namespace ai;

// 打印欢迎信息
void print_banner() {
    std::cout << R"(
    _    ____  
   / \  |_  /  
  / _ \_ / /   
 /_/ \_\___|   
               
   C++ AI 大模型框架 v1.0
   纯C++实现的Transformer架构
)" << std::endl;
    std::cout << "======================================" << std::endl;
}

// 测试张量基本功能
void test_tensor() {
    std::cout << "\n>>> 测试张量系统 <<<" << std::endl;
    
    Tensor a({2, 3}, 1.0f);
    Tensor b({2, 3}, 2.0f);
    
    std::cout << "张量 a:" << std::endl;
    a.print_info("a");
    
    std::cout << "张量 b:" << std::endl;
    b.print_info("b");
    
    Tensor c = a + b;
    std::cout << "a + b:" << std::endl;
    c.print_info("c");
    
    Tensor d = a * 3.0f;
    std::cout << "a * 3.0:" << std::endl;
    d.print_info("d");
    
    Tensor e = c.reshape({3, 2});
    std::cout << "reshape c to [3, 2]:" << std::endl;
    e.print_info("e");
    
    std::cout << "张量测试通过!" << std::endl;
}

// 测试神经网络层
void test_layers() {
    std::cout << "\n>>> 测试神经网络层 <<<" << std::endl;
    
    // 测试 Linear
    std::cout << "\n--- Linear 层 ---" << std::endl;
    Linear linear(4, 8);
    Tensor input({2, 4});
    utils::uniform_init(input, -1.0f, 1.0f);
    input.print_info("Linear input");
    
    Tensor output = linear.forward(input);
    output.print_info("Linear output");
    std::cout << "Linear 参数数: " << linear.param_count() << std::endl;
    
    // 测试 Embedding
    std::cout << "\n--- Embedding 层 ---" << std::endl;
    Embedding emb(10, 16);
    Tensor ids({2, 3});
    ids(0, 0) = 1; ids(0, 1) = 2; ids(0, 2) = 3;
    ids(1, 0) = 4; ids(1, 1) = 5; ids(1, 2) = 6;
    
    Tensor emb_out = emb.forward(ids);
    emb_out.print_info("Embedding output");
    
    // 测试 MultiHeadAttention
    std::cout << "\n--- Multi-Head Attention ---" << std::endl;
    MultiHeadAttention attn(16, 4, 0.0f);
    Tensor attn_input({1, 4, 16}); // [batch, seq_len, d_model]
    utils::uniform_init(attn_input, -0.5f, 0.5f);
    
    Tensor attn_output = attn.forward(attn_input);
    attn_output.print_info("Attention output");
    std::cout << "Attention 参数数: " << attn.param_count() << std::endl;
    
    // 测试 FeedForward
    std::cout << "\n--- FeedForward ---" << std::endl;
    FeedForward ff(16, 64, "gelu");
    Tensor ff_input({1, 4, 16});
    // 需要先reshape为2D
    Tensor ff_input_2d = ff_input.reshape({4, 16});
    Tensor ff_output_2d = ff.forward(ff_input_2d);
    Tensor ff_output = ff_output_2d.reshape({1, 4, 16});
    ff_output.print_info("FeedForward output");
    
    std::cout << "神经网络层测试通过!" << std::endl;
}

// 测试 Transformer 块
void test_transformer_block() {
    std::cout << "\n>>> 测试 Transformer 块 <<<" << std::endl;
    
    TransformerEncoderBlock block(64, 8, 256, 0.0f);
    Tensor input({1, 5, 64}); // [batch, seq_len, d_model]
    utils::uniform_init(input, -0.5f, 0.5f);
    
    Tensor output = block.forward(input);
    output.print_info("Transformer block output");
    std::cout << "Transformer block 参数数: " << block.param_count() << std::endl;
    
    std::cout << "Transformer 块测试通过!" << std::endl;
}

// 测试完整模型
void test_model() {
    std::cout << "\n>>> 测试完整模型 <<<" << std::endl;
    
    TransformerModel::Config config;
    config.vocab_size = 100;
    config.d_model = 64;
    config.num_heads = 4;
    config.num_layers = 2;
    config.d_ff = 256;
    config.max_seq_len = 32;
    config.dropout = 0.0f;
    
    TransformerModel model(config);
    
    std::cout << "\n模型配置:" << std::endl;
    std::cout << "  vocab_size: " << config.vocab_size << std::endl;
    std::cout << "  d_model: " << config.d_model << std::endl;
    std::cout << "  num_heads: " << config.num_heads << std::endl;
    std::cout << "  num_layers: " << config.num_layers << std::endl;
    std::cout << "  d_ff: " << config.d_ff << std::endl;
    std::cout << "  总参数数: " << model.param_count() << std::endl;
    
    // 创建输入数据
    std::vector<std::vector<int>> input_ids = {
        {1, 2, 3, 4, 5},
        {6, 7, 8, 9, 10}
    };
    
    Tensor logits = model.forward(input_ids);
    logits.print_info("Model logits");
    
    float loss = model.compute_loss(logits, input_ids);
    std::cout << "初始损失: " << loss << std::endl;
    
    // 推理测试
    std::cout << "\n--- 推理测试 ---" << std::endl;
    InferenceEngine inference(model);
    
    std::vector<int> prompt = {1, 2, 3};
    std::vector<int> generated = inference.generate(prompt, 10);
    std::cout << "Prompt: [1, 2, 3]" << std::endl;
    std::cout << "生成结果: [";
    for (size_t i = 0; i < generated.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << generated[i];
    }
    std::cout << "]" << std::endl;
    
    std::cout << "模型测试通过!" << std::endl;
}

// 训练演示
void demo_training() {
    std::cout << "\n>>> 训练演示 <<<" << std::endl;
    
    TransformerModel::Config config;
    config.vocab_size = 50;
    config.d_model = 32;
    config.num_heads = 4;
    config.num_layers = 2;
    config.d_ff = 128;
    config.max_seq_len = 16;
    config.dropout = 0.0f;
    
    TransformerModel model(config);
    
    std::cout << "模型总参数数: " << model.param_count() << std::endl;
    
    // 创建简单训练数据（自回归任务）
    std::vector<std::vector<int>> train_data;
    for (int i = 0; i < 20; ++i) {
        std::vector<int> seq;
        for (int j = 0; j < 8; ++j) {
            seq.push_back((i + j) % 40 + 4); // 避免使用特殊token
        }
        train_data.push_back(seq);
    }
    
    // 创建优化器
    auto params = model.parameters();
    auto grads = model.gradients();
    auto optimizer = std::make_unique<AdamW>(params, grads, 0.01f, 0.9f, 0.999f, 1e-8f, 0.01f);
    
    Trainer trainer(model, std::move(optimizer));
    
    std::cout << "\n开始训练..." << std::endl;
    
    for (int epoch = 0; epoch < 5; ++epoch) {
        float loss = trainer.train_epoch(train_data, 4);
        std::cout << "Epoch " << (epoch + 1) << ", Loss: " << loss << std::endl;
    }
    
    // 训练后推理
    std::cout << "\n--- 训练后推理 ---" << std::endl;
    InferenceEngine inference(model);
    std::vector<int> prompt = {4, 5, 6};
    std::vector<int> generated = inference.generate(prompt, 8);
    std::cout << "Prompt: [4, 5, 6]" << std::endl;
    std::cout << "生成: [";
    for (size_t i = 0; i < generated.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << generated[i];
    }
    std::cout << "]" << std::endl;
    
    std::cout << "训练演示完成!" << std::endl;
}

// 模型保存与加载演示
void demo_save_load() {
    std::cout << "\n>>> 保存/加载演示 <<<" << std::endl;
    
    TransformerModel::Config config;
    config.vocab_size = 30;
    config.d_model = 16;
    config.num_heads = 2;
    config.num_layers = 1;
    config.d_ff = 64;
    config.max_seq_len = 16;
    config.dropout = 0.0f;
    
    TransformerModel model1(config);
    
    // 保存前参数
    float param_before = model1.parameters()[0]->data()[0];
    std::cout << "保存前参数[0][0]: " << param_before << std::endl;
    
    // 保存模型
    std::string path = "model_test.bin";
    model1.save(path);
    std::cout << "模型已保存到: " << path << std::endl;
    
    // 加载到新模型
    TransformerModel model2(config);
    model2.load(path);
    
    float param_after = model2.parameters()[0]->data()[0];
    std::cout << "加载后参数[0][0]: " << param_after << std::endl;
    
    if (std::abs(param_before - param_after) < 1e-5) {
        std::cout << "保存/加载成功! 参数一致。" << std::endl;
    } else {
        std::cout << "保存/加载可能有问题。" << std::endl;
    }
    
    std::cout << "保存/加载演示完成!" << std::endl;
}

// --- 对话功能 ---

// 构建对话词汇表
void build_dialogue_vocab(utils::Tokenizer& tokenizer) {
    utils::DialogueDataset dataset;
    dataset.load_preset_dialogues();
    
    // 收集所有文本用于构建词汇表
    std::vector<std::string> all_texts;
    for (size_t i = 0; i < dataset.size(); ++i) {
        const auto& d = dataset.get(i);
        all_texts.push_back(d.system_prompt);
        for (const auto& turn : d.turns) {
            all_texts.push_back(turn.content);
        }
    }
    
    // 添加角色标记的字符
    all_texts.push_back(tokenizer.system_token());
    all_texts.push_back(tokenizer.user_token());
    all_texts.push_back(tokenizer.assistant_token());
    all_texts.push_back(tokenizer.turn_end_token());
    
    tokenizer.build_vocab(all_texts);
    std::cout << "词汇表大小: " << tokenizer.vocab_size() << std::endl;
}

// 对话训练演示
void demo_dialogue_training() {
    std::cout << "\n>>> 对话训练演示 <<<" << std::endl;
    
    // 创建分词器并构建词汇表
    utils::Tokenizer tokenizer;
    build_dialogue_vocab(tokenizer);
    
    // 加载预设对话数据
    utils::DialogueDataset dataset;
    dataset.load_preset_dialogues();
    
    // 编码为训练对
    auto pairs = dataset.get_training_pairs(tokenizer, 128);
    std::cout << "训练样本数: " << pairs.size() << std::endl;
    
    // 展示一个样本
    if (!pairs.empty()) {
        std::cout << "示例输入 (token): [";
        for (size_t i = 0; i < pairs[0].first.size() && i < 20; ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << pairs[0].first[i];
        }
        std::cout << "]" << std::endl;
        std::cout << "示例解码: " << tokenizer.decode(pairs[0].first).substr(0, 100) << std::endl;
    }
    
    // 模型配置（根据词汇表大小调整）
    TransformerModel::Config config;
    config.vocab_size = tokenizer.vocab_size();
    config.d_model = 64;
    config.num_heads = 4;
    config.num_layers = 3;
    config.d_ff = 256;
    config.max_seq_len = 128;
    config.dropout = 0.0f;
    
    TransformerModel model(config);
    std::cout << "模型总参数数: " << model.param_count() << std::endl;
    
    // 准备训练数据
    std::vector<std::vector<int>> train_inputs;
    std::vector<std::vector<int>> train_labels;
    for (const auto& pair : pairs) {
        train_inputs.push_back(pair.first);
        train_labels.push_back(pair.second);
    }
    
    // 创建优化器
    auto params = model.parameters();
    auto grads = model.gradients();
    auto optimizer = std::make_unique<AdamW>(params, grads, 0.005f, 0.9f, 0.999f, 1e-8f, 0.01f);
    
    Trainer trainer(model, std::move(optimizer));
    
    std::cout << "\n开始对话训练..." << std::endl;
    
    for (int epoch = 0; epoch < 20; ++epoch) {
        float loss = 0.0f;
        for (size_t i = 0; i < train_inputs.size(); ++i) {
            std::vector<std::vector<int>> batch_input = {train_inputs[i]};
            std::vector<std::vector<int>> batch_label = {train_labels[i]};
            loss += trainer.train_step(batch_input, batch_label);
        }
        loss /= static_cast<float>(train_inputs.size());
        
        if ((epoch + 1) % 5 == 0) {
            std::cout << "Epoch " << (epoch + 1) << ", Loss: " << loss << std::endl;
        }
    }
    
    // 保存模型和分词器
    model.save("dialogue_model.bin");
    tokenizer.save("dialogue_tokenizer.bin");
    std::cout << "模型和分词器已保存。" << std::endl;
    
    std::cout << "对话训练演示完成!" << std::endl;
}

// 对话界面
void chat_interface() {
    std::cout << "\n>>> 对话模式 <<<" << std::endl;
    
    // 创建分词器并构建词汇表
    utils::Tokenizer tokenizer;
    build_dialogue_vocab(tokenizer);
    
    // 加载预设对话数据
    utils::DialogueDataset dataset;
    dataset.load_preset_dialogues();
    
    // 模型配置
    TransformerModel::Config config;
    config.vocab_size = tokenizer.vocab_size();
    config.d_model = 64;
    config.num_heads = 4;
    config.num_layers = 3;
    config.d_ff = 256;
    config.max_seq_len = 128;
    config.dropout = 0.0f;
    
    TransformerModel model(config);
    
    // 尝试加载已有模型，否则从头训练
    bool loaded = false;
    std::ifstream check_model("dialogue_model.bin");
    if (check_model.good()) {
        check_model.close();
        model.load("dialogue_model.bin");
        tokenizer.load("dialogue_tokenizer.bin");
        loaded = true;
        std::cout << "已加载预训练对话模型。" << std::endl;
    }
    
    if (!loaded) {
        std::cout << "未找到预训练模型，开始快速训练..." << std::endl;
        
        // 快速训练
        auto pairs = dataset.get_training_pairs(tokenizer, 128);
        
        std::vector<std::vector<int>> train_inputs;
        std::vector<std::vector<int>> train_labels;
        for (const auto& pair : pairs) {
            train_inputs.push_back(pair.first);
            train_labels.push_back(pair.second);
        }
        
        auto params = model.parameters();
        auto grads = model.gradients();
        auto optimizer = std::make_unique<AdamW>(params, grads, 0.005f, 0.9f, 0.999f, 1e-8f, 0.01f);
        Trainer trainer(model, std::move(optimizer));
        
        std::cout << "训练中..." << std::endl;
        for (int epoch = 0; epoch < 30; ++epoch) {
            float loss = 0.0f;
            for (size_t i = 0; i < train_inputs.size(); ++i) {
                std::vector<std::vector<int>> batch_input = {train_inputs[i]};
                std::vector<std::vector<int>> batch_label = {train_labels[i]};
                loss += trainer.train_step(batch_input, batch_label);
            }
            loss /= static_cast<float>(train_inputs.size());
            if ((epoch + 1) % 10 == 0) {
                std::cout << "  Epoch " << (epoch + 1) << ", Loss: " << loss << std::endl;
            }
        }
        
        model.save("dialogue_model.bin");
        tokenizer.save("dialogue_tokenizer.bin");
        std::cout << "训练完成，模型已保存。" << std::endl;
    }
    
    // 创建对话引擎
    DialogueEngine dialogue(model, tokenizer);
    dialogue.set_system_prompt("你是一个友好的AI助手。");
    
    std::cout << "\n=====================================" << std::endl;
    std::cout << "  AI 对话助手已就绪！" << std::endl;
    std::cout << "  输入 'quit' 或 'exit' 退出对话" << std::endl;
    std::cout << "  输入 'reset' 重置对话上下文" << std::endl;
    std::cout << "=====================================" << std::endl;
    
    std::string user_input;
    while (true) {
        std::cout << "\n你: ";
        std::getline(std::cin, user_input);
        
        if (user_input == "quit" || user_input == "exit") {
            std::cout << "再见！" << std::endl;
            break;
        }
        
        if (user_input == "reset") {
            dialogue.reset();
            dialogue.set_system_prompt("你是一个友好的AI助手。");
            std::cout << "对话已重置。" << std::endl;
            continue;
        }
        
        if (user_input.empty()) {
            continue;
        }
        
        // 生成回复
        std::string reply = dialogue.respond(user_input, 64, 0.8f, 10);
        
        std::cout << "AI: " << reply << std::endl;
    }
}

// 显示菜单
void show_menu() {
    std::cout << "\n========== 菜单 ==========" << std::endl;
    std::cout << "1. 运行全部测试" << std::endl;
    std::cout << "2. 对话训练演示" << std::endl;
    std::cout << "3. 进入对话模式" << std::endl;
    std::cout << "0. 退出" << std::endl;
    std::cout << "=========================" << std::endl;
    std::cout << "请选择: ";
}

int main() {
    print_banner();
    
    try {
        int choice = -1;
        while (choice != 0) {
            show_menu();
            std::cin >> choice;
            std::cin.ignore(); // 忽略换行符
            
            switch (choice) {
                case 1:
                    test_tensor();
                    test_layers();
                    test_transformer_block();
                    test_model();
                    demo_training();
                    demo_save_load();
                    std::cout << "\n======================================" << std::endl;
                    std::cout << "所有测试通过！" << std::endl;
                    std::cout << "======================================" << std::endl;
                    break;
                    
                case 2:
                    demo_dialogue_training();
                    break;
                    
                case 3:
                    chat_interface();
                    break;
                    
                case 0:
                    std::cout << "再见！" << std::endl;
                    break;
                    
                default:
                    std::cout << "无效选择，请重试。" << std::endl;
                    break;
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "\n错误: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
