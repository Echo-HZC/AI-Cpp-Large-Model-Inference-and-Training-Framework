#include "ai/tensor.hpp"
#include <iostream>
#include <iomanip>
#include <functional>

namespace ai {

// --- 构造与析构 ---

Tensor::Tensor() : shape_({0}), data_() {}

Tensor::Tensor(const std::vector<size_t>& shape) : shape_(shape) {
    size_t total = total_size();
    data_.resize(total, 0.0f);
}

Tensor::Tensor(const std::vector<size_t>& shape, float init_val) : shape_(shape) {
    size_t total = total_size();
    data_.resize(total, init_val);
}

Tensor::Tensor(Tensor&& other) noexcept
    : shape_(std::move(other.shape_)), data_(std::move(other.data_)) {
    other.shape_ = {0};
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        shape_ = std::move(other.shape_);
        data_ = std::move(other.data_);
        other.shape_ = {0};
    }
    return *this;
}

// --- 形状操作 ---

size_t Tensor::total_size() const {
    if (shape_.empty() || (shape_.size() == 1 && shape_[0] == 0)) return 0;
    size_t total = 1;
    for (auto s : shape_) total *= s;
    return total;
}

std::vector<size_t> Tensor::strides() const {
    std::vector<size_t> s(shape_.size());
    size_t stride = 1;
    for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; --i) {
        s[i] = stride;
        stride *= shape_[i];
    }
    return s;
}

size_t Tensor::batch() const { return shape_.size() > 0 ? shape_[0] : 0; }
size_t Tensor::seq_len() const { return shape_.size() > 1 ? shape_[1] : 0; }
size_t Tensor::features() const { return shape_.size() > 2 ? shape_[2] : (shape_.size() > 1 ? shape_[1] : 0); }
size_t Tensor::heads() const { return shape_.size() > 2 ? shape_[2] : 0; }

// --- 索引访问 ---

size_t Tensor::index(size_t i) const {
    if (shape_.size() != 1) throw std::runtime_error("Tensor::index(1D): wrong ndim");
    if (i >= shape_[0]) throw std::out_of_range("Tensor 1D index out of range");
    return i;
}

size_t Tensor::index(size_t i, size_t j) const {
    if (shape_.size() != 2) throw std::runtime_error("Tensor::index(2D): wrong ndim");
    if (i >= shape_[0] || j >= shape_[1]) throw std::out_of_range("Tensor 2D index out of range");
    return i * shape_[1] + j;
}

size_t Tensor::index(size_t i, size_t j, size_t k) const {
    if (shape_.size() != 3) throw std::runtime_error("Tensor::index(3D): wrong ndim");
    if (i >= shape_[0] || j >= shape_[1] || k >= shape_[2])
        throw std::out_of_range("Tensor 3D index out of range");
    return (i * shape_[1] + j) * shape_[2] + k;
}

size_t Tensor::index(size_t i, size_t j, size_t k, size_t l) const {
    if (shape_.size() != 4) throw std::runtime_error("Tensor::index(4D): wrong ndim");
    if (i >= shape_[0] || j >= shape_[1] || k >= shape_[2] || l >= shape_[3])
        throw std::out_of_range("Tensor 4D index out of range");
    return ((i * shape_[1] + j) * shape_[2] + k) * shape_[3] + l;
}

float& Tensor::operator()(size_t i) { return data_[index(i)]; }
float Tensor::operator()(size_t i) const { return data_[index(i)]; }
float& Tensor::operator()(size_t i, size_t j) { return data_[index(i, j)]; }
float Tensor::operator()(size_t i, size_t j) const { return data_[index(i, j)]; }
float& Tensor::operator()(size_t i, size_t j, size_t k) { return data_[index(i, j, k)]; }
float Tensor::operator()(size_t i, size_t j, size_t k) const { return data_[index(i, j, k)]; }
float& Tensor::operator()(size_t i, size_t j, size_t k, size_t l) { return data_[index(i, j, k, l)]; }
float Tensor::operator()(size_t i, size_t j, size_t k, size_t l) const { return data_[index(i, j, k, l)]; }

// --- 切片 ---

Tensor Tensor::slice(size_t dim, size_t start, size_t end) const {
    if (dim >= shape_.size()) throw std::runtime_error("slice: dim out of range");
    if (end > shape_[dim]) end = shape_[dim];
    if (start >= end) throw std::runtime_error("slice: empty slice");
    
    std::vector<size_t> new_shape = shape_;
    new_shape[dim] = end - start;
    
    Tensor result(new_shape);
    // 简单实现：遍历所有元素，复制对应范围
    // 对于高维张量，这可以通过更高效的stride copy实现
    
    std::vector<size_t> indices(shape_.size(), 0);
    size_t src_idx = 0;
    size_t dst_idx = 0;
    
    // 扁平化复制（简化实现）
    std::function<void(size_t)> copy_dim = [&](size_t d) {
        if (d == shape_.size()) {
            if (indices[dim] >= start && indices[dim] < end) {
                result.data_[dst_idx++] = data_[src_idx];
            }
            src_idx++;
            return;
        }
        for (size_t i = 0; i < shape_[d]; ++i) {
            indices[d] = i;
            copy_dim(d + 1);
        }
    };
    
    copy_dim(0);
    return result;
}

// --- 重塑 ---

Tensor Tensor::reshape(const std::vector<size_t>& new_shape) const {
    size_t new_total = 1;
    for (auto s : new_shape) new_total *= s;
    if (new_total != total_size())
        throw std::runtime_error("reshape: total size mismatch");
    
    Tensor result(new_shape);
    result.data_ = data_; // 复制数据
    return result;
}

// --- 转置 ---

Tensor Tensor::transpose(size_t dim1, size_t dim2) const {
    if (dim1 >= shape_.size() || dim2 >= shape_.size())
        throw std::runtime_error("transpose: dim out of range");
    
    std::vector<size_t> new_shape = shape_;
    std::swap(new_shape[dim1], new_shape[dim2]);
    
    Tensor result(new_shape);
    std::vector<size_t> indices(shape_.size(), 0);
    
    std::function<void(size_t)> transpose_dim = [&](size_t d) {
        if (d == shape_.size()) {
            std::vector<size_t> dst_indices = indices;
            std::swap(dst_indices[dim1], dst_indices[dim2]);
            size_t src_idx = 0;
            size_t dst_idx = 0;
            for (size_t i = 0; i < shape_.size(); ++i) {
                src_idx = src_idx * shape_[i] + indices[i];
            }
            for (size_t i = 0; i < new_shape.size(); ++i) {
                dst_idx = dst_idx * new_shape[i] + dst_indices[i];
            }
            result.data_[dst_idx] = data_[src_idx];
            return;
        }
        for (size_t i = 0; i < shape_[d]; ++i) {
            indices[d] = i;
            transpose_dim(d + 1);
        }
    };
    
    transpose_dim(0);
    return result;
}

// --- 填充 ---

void Tensor::fill(float value) {
    std::fill(data_.begin(), data_.end(), value);
}

// --- 打印 ---

void Tensor::print_info(const std::string& name) const {
    std::cout << "Tensor";
    if (!name.empty()) std::cout << " [" << name << "]";
    std::cout << ": shape=[";
    for (size_t i = 0; i < shape_.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << shape_[i];
    }
    std::cout << "], size=" << size() << std::endl;
    
    if (size() <= 20) {
        std::cout << "  data=[";
        for (size_t i = 0; i < data_.size() && i < 20; ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << std::fixed << std::setprecision(4) << data_[i];
        }
        if (data_.size() > 20) std::cout << ", ...";
        std::cout << "]" << std::endl;
    }
}

// --- 保存/加载 ---

void Tensor::save(std::ofstream& ofs) const {
    size_t ndim = shape_.size();
    ofs.write(reinterpret_cast<const char*>(&ndim), sizeof(ndim));
    for (auto s : shape_) {
        ofs.write(reinterpret_cast<const char*>(&s), sizeof(s));
    }
    size_t sz = data_.size();
    ofs.write(reinterpret_cast<const char*>(&sz), sizeof(sz));
    ofs.write(reinterpret_cast<const char*>(data_.data()), sz * sizeof(float));
}

Tensor Tensor::load(std::ifstream& ifs) {
    size_t ndim;
    ifs.read(reinterpret_cast<char*>(&ndim), sizeof(ndim));
    std::vector<size_t> shape(ndim);
    for (size_t i = 0; i < ndim; ++i) {
        ifs.read(reinterpret_cast<char*>(&shape[i]), sizeof(shape[i]));
    }
    size_t sz;
    ifs.read(reinterpret_cast<char*>(&sz), sizeof(sz));
    Tensor result(shape);
    ifs.read(reinterpret_cast<char*>(result.data_.data()), sz * sizeof(float));
    return result;
}

// --- 其他 ---

bool Tensor::same_shape(const Tensor& other) const {
    return shape_ == other.shape_;
}

void Tensor::reset(const std::vector<size_t>& shape) {
    shape_ = shape;
    data_.resize(total_size(), 0.0f);
}

// --- 全局操作 ---

Tensor operator+(const Tensor& a, const Tensor& b) {
    if (!a.same_shape(b)) throw std::runtime_error("Tensor add: shape mismatch");
    Tensor result(a.shape());
    for (size_t i = 0; i < a.size(); ++i) {
        result.data()[i] = a.data()[i] + b.data()[i];
    }
    return result;
}

Tensor operator-(const Tensor& a, const Tensor& b) {
    if (!a.same_shape(b)) throw std::runtime_error("Tensor sub: shape mismatch");
    Tensor result(a.shape());
    for (size_t i = 0; i < a.size(); ++i) {
        result.data()[i] = a.data()[i] - b.data()[i];
    }
    return result;
}

Tensor operator*(const Tensor& a, const Tensor& b) {
    if (!a.same_shape(b)) throw std::runtime_error("Tensor mul: shape mismatch");
    Tensor result(a.shape());
    for (size_t i = 0; i < a.size(); ++i) {
        result.data()[i] = a.data()[i] * b.data()[i];
    }
    return result;
}

Tensor operator/(const Tensor& a, const Tensor& b) {
    if (!a.same_shape(b)) throw std::runtime_error("Tensor div: shape mismatch");
    Tensor result(a.shape());
    for (size_t i = 0; i < a.size(); ++i) {
        result.data()[i] = a.data()[i] / b.data()[i];
    }
    return result;
}

Tensor operator+(const Tensor& a, float scalar) {
    Tensor result(a.shape());
    for (size_t i = 0; i < a.size(); ++i) {
        result.data()[i] = a.data()[i] + scalar;
    }
    return result;
}

Tensor operator-(const Tensor& a, float scalar) {
    Tensor result(a.shape());
    for (size_t i = 0; i < a.size(); ++i) {
        result.data()[i] = a.data()[i] - scalar;
    }
    return result;
}

Tensor operator*(const Tensor& a, float scalar) {
    Tensor result(a.shape());
    for (size_t i = 0; i < a.size(); ++i) {
        result.data()[i] = a.data()[i] * scalar;
    }
    return result;
}

Tensor operator/(const Tensor& a, float scalar) {
    Tensor result(a.shape());
    for (size_t i = 0; i < a.size(); ++i) {
        result.data()[i] = a.data()[i] / scalar;
    }
    return result;
}

Tensor& operator+=(Tensor& a, const Tensor& b) {
    if (!a.same_shape(b)) throw std::runtime_error("Tensor +=: shape mismatch");
    for (size_t i = 0; i < a.size(); ++i) {
        a.data()[i] += b.data()[i];
    }
    return a;
}

Tensor& operator-=(Tensor& a, const Tensor& b) {
    if (!a.same_shape(b)) throw std::runtime_error("Tensor -=: shape mismatch");
    for (size_t i = 0; i < a.size(); ++i) {
        a.data()[i] -= b.data()[i];
    }
    return a;
}

Tensor& operator*=(Tensor& a, float scalar) {
    for (size_t i = 0; i < a.size(); ++i) {
        a.data()[i] *= scalar;
    }
    return a;
}

Tensor& operator/=(Tensor& a, float scalar) {
    for (size_t i = 0; i < a.size(); ++i) {
        a.data()[i] /= scalar;
    }
    return a;
}

} // namespace ai
