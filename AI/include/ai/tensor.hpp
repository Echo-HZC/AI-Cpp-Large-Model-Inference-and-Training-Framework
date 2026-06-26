#pragma once

#include <vector>
#include <cstddef>
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <fstream>

namespace ai {

/**
 * @brief 多维张量类，支持1D到4D
 *        采用行优先(Row-Major)存储
 */
class Tensor {
public:
    Tensor(); // 空张量
    explicit Tensor(const std::vector<size_t>& shape);
    Tensor(const std::vector<size_t>& shape, float init_val);
    
    // 拷贝构造
    Tensor(const Tensor& other) = default;
    Tensor& operator=(const Tensor& other) = default;
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;
    
    ~Tensor() = default;

    // 形状操作
    const std::vector<size_t>& shape() const { return shape_; }
    size_t ndim() const { return shape_.size(); }
    size_t size() const { return data_.size(); }
    
    // 计算元素总数和步长
    size_t total_size() const;
    std::vector<size_t> strides() const;
    
    // 维度便捷访问
    size_t batch() const;
    size_t seq_len() const;
    size_t features() const;
    size_t heads() const;
    
    // 数据访问
    float* data() { return data_.data(); }
    const float* data() const { return data_.data(); }
    
    float& operator()(size_t i);
    float operator()(size_t i) const;
    float& operator()(size_t i, size_t j);
    float operator()(size_t i, size_t j) const;
    float& operator()(size_t i, size_t j, size_t k);
    float operator()(size_t i, size_t j, size_t k) const;
    float& operator()(size_t i, size_t j, size_t k, size_t l);
    float operator()(size_t i, size_t j, size_t k, size_t l) const;
    
    // 索引计算（扁平化索引）
    size_t index(size_t i) const;
    size_t index(size_t i, size_t j) const;
    size_t index(size_t i, size_t j, size_t k) const;
    size_t index(size_t i, size_t j, size_t k, size_t l) const;
    
    // 切片操作：返回某个维度的子视图（复制数据）
    Tensor slice(size_t dim, size_t start, size_t end) const;
    
    // 重塑形状
    Tensor reshape(const std::vector<size_t>& new_shape) const;
    
    // 转置（交换两个维度）
    Tensor transpose(size_t dim1, size_t dim2) const;
    
    // 填充
    void fill(float value);
    
    // 打印信息
    void print_info(const std::string& name = "") const;
    
    // 保存/加载
    void save(std::ofstream& ofs) const;
    static Tensor load(std::ifstream& ifs);
    
    // 检查形状是否匹配
    bool same_shape(const Tensor& other) const;
    
    // 重置为新的形状
    void reset(const std::vector<size_t>& shape);

private:
    std::vector<size_t> shape_;
    std::vector<float> data_;
};

// 全局张量操作
Tensor operator+(const Tensor& a, const Tensor& b);
Tensor operator-(const Tensor& a, const Tensor& b);
Tensor operator*(const Tensor& a, const Tensor& b); // 逐元素乘
Tensor operator/(const Tensor& a, const Tensor& b); // 逐元素除
Tensor operator+(const Tensor& a, float scalar);
Tensor operator-(const Tensor& a, float scalar);
Tensor operator*(const Tensor& a, float scalar);
Tensor operator/(const Tensor& a, float scalar);

Tensor& operator+=(Tensor& a, const Tensor& b);
Tensor& operator-=(Tensor& a, const Tensor& b);
Tensor& operator*=(Tensor& a, float scalar);
Tensor& operator/=(Tensor& a, float scalar);

} // namespace ai
