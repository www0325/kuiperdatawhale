//
// Created by fss on 22-11-12.
//

#include "data/tensor.hpp"
#include <glog/logging.h>
#include <memory>
#include <numeric>

namespace kuiper_infer {
Tensor<float>::Tensor(uint32_t channels, uint32_t rows, uint32_t cols) {
  data_ = arma::fcube(rows, cols, channels);
  if (channels == 1 && rows == 1) {
    this->raw_shapes_ = std::vector<uint32_t>{cols};
  } else if (channels == 1) {
    this->raw_shapes_ = std::vector<uint32_t>{rows, cols};
  } else {
    this->raw_shapes_ = std::vector<uint32_t>{channels, rows, cols};
  }
}

Tensor<float>::Tensor(uint32_t size) {
  data_ = arma::fcube(1, size, 1);
  this->raw_shapes_ = std::vector<uint32_t>{size};
}

Tensor<float>::Tensor(uint32_t rows, uint32_t cols) {
  data_ = arma::fcube(rows, cols, 1);
  this->raw_shapes_ = std::vector<uint32_t>{rows, cols};
}

Tensor<float>::Tensor(const std::vector<uint32_t>& shapes) {
  CHECK(!shapes.empty() && shapes.size() <= 3);

  uint32_t remaining = 3 - shapes.size();
  std::vector<uint32_t> shapes_(3, 1);
  std::copy(shapes.begin(), shapes.end(), shapes_.begin() + remaining);

  uint32_t channels = shapes_.at(0);
  uint32_t rows = shapes_.at(1);
  uint32_t cols = shapes_.at(2);

  data_ = arma::fcube(rows, cols, channels);
  if (channels == 1 && rows == 1) {
    this->raw_shapes_ = std::vector<uint32_t>{cols};
  } else if (channels == 1) {
    this->raw_shapes_ = std::vector<uint32_t>{rows, cols};
  } else {
    this->raw_shapes_ = std::vector<uint32_t>{channels, rows, cols};
  }
}

Tensor<float>::Tensor(const Tensor& tensor) {
  if (this != &tensor) {
    this->data_ = tensor.data_;
    this->raw_shapes_ = tensor.raw_shapes_;
  }
}

Tensor<float>::Tensor(Tensor<float>&& tensor) noexcept {
  if (this != &tensor) {
    this->data_ = std::move(tensor.data_);
    this->raw_shapes_ = tensor.raw_shapes_;
  }
}

Tensor<float>& Tensor<float>::operator=(Tensor<float>&& tensor) noexcept {
  if (this != &tensor) {
    this->data_ = std::move(tensor.data_);
    this->raw_shapes_ = tensor.raw_shapes_;
  }
  return *this;
}

Tensor<float>& Tensor<float>::operator=(const Tensor& tensor) {
  if (this != &tensor) {
    this->data_ = tensor.data_;
    this->raw_shapes_ = tensor.raw_shapes_;
  }
  return *this;
}

uint32_t Tensor<float>::rows() const {
  CHECK(!this->data_.empty());
  return this->data_.n_rows;
}

uint32_t Tensor<float>::cols() const {
  CHECK(!this->data_.empty());
  return this->data_.n_cols;
}

uint32_t Tensor<float>::channels() const {
  CHECK(!this->data_.empty());
  return this->data_.n_slices;
}

uint32_t Tensor<float>::size() const {
  CHECK(!this->data_.empty());
  return this->data_.size();
}

void Tensor<float>::set_data(const arma::fcube& data) {
  CHECK(data.n_rows == this->data_.n_rows)
      << data.n_rows << " != " << this->data_.n_rows;
  CHECK(data.n_cols == this->data_.n_cols)
      << data.n_cols << " != " << this->data_.n_cols;
  CHECK(data.n_slices == this->data_.n_slices)
      << data.n_slices << " != " << this->data_.n_slices;
  this->data_ = data;
}

bool Tensor<float>::empty() const { return this->data_.empty(); }

float Tensor<float>::index(uint32_t offset) const {
  CHECK(offset < this->data_.size()) << "Tensor index out of bound!";
  return this->data_.at(offset);
}

float& Tensor<float>::index(uint32_t offset) {
  CHECK(offset < this->data_.size()) << "Tensor index out of bound!";
  return this->data_.at(offset);
}

std::vector<uint32_t> Tensor<float>::shapes() const {
  CHECK(!this->data_.empty());
  return {this->channels(), this->rows(), this->cols()};
}

arma::fcube& Tensor<float>::data() { return this->data_; }

const arma::fcube& Tensor<float>::data() const { return this->data_; }

arma::fmat& Tensor<float>::slice(uint32_t channel) {
  CHECK_LT(channel, this->channels());
  return this->data_.slice(channel);
}

const arma::fmat& Tensor<float>::slice(uint32_t channel) const {
  CHECK_LT(channel, this->channels());
  return this->data_.slice(channel);
}

float Tensor<float>::at(uint32_t channel, uint32_t row, uint32_t col) const {
  CHECK_LT(row, this->rows());
  CHECK_LT(col, this->cols());
  CHECK_LT(channel, this->channels());
  return this->data_.at(row, col, channel);
}

float& Tensor<float>::at(uint32_t channel, uint32_t row, uint32_t col) {
  CHECK_LT(row, this->rows());
  CHECK_LT(col, this->cols());
  CHECK_LT(channel, this->channels());
  return this->data_.at(row, col, channel);
}

void Tensor<float>::Padding(const std::vector<uint32_t>& pads,
                            float padding_value) {
  CHECK(!this->data_.empty());
  CHECK_EQ(pads.size(), 4);
  // 四周填充的维度
  uint32_t pad_rows1 = pads.at(0);  // up
  uint32_t pad_rows2 = pads.at(1);  // bottom
  uint32_t pad_cols1 = pads.at(2);  // left
  uint32_t pad_cols2 = pads.at(3);  // right

  // 请补充代码
  uint32_t rows = this->rows();
  uint32_t cols = this->cols();
  uint32_t channels = this->channels();
  uint32_t new_rows = rows + pad_rows1 + pad_rows2;
  uint32_t new_cols = cols + pad_cols1 + pad_cols2;
  // 思路：新建fmat再移动赋值会报错，因为改变了size，因而只能重建整个fcube即data

  //优化1：构造的同时fill
  /*
  arma::fcube new_data(new_rows,new_cols,channels);
  new_data.fill(padding_value);
  */
  arma::fcube new_data(new_rows,new_cols,channels,arma::fill::value(padding_value));

  //优化2：调用subcube(start_row, start_col, start_slice, end_row, end_col, end_slice)
  /*
  for(uint32_t i=0 ; i<channels ; i++){
    for(uint32_t j=0 ; j<rows ; j++){
      for(uint32_t k=0 ; k<cols ; k++){
        new_data.at(pad_rows1+j,pad_cols1+k,i) = this->at(i,j,k);
      }
    }
  }*/
  new_data.subcube(pad_rows1,pad_cols1,0,pad_rows1+rows-1,pad_cols1+cols-1,channels-1) = this->data_;
  //this->set_data不可用，因为其检查了长度均相等
  this->data_ = std::move(new_data);

  //调整：应当遵循构造函数中的raw_shapes分维度讨论
  //this->raw_shapes_ = std::vector<uint32_t>{channels,new_rows,new_cols};
  if (channels == 1 && new_rows == 1) {
    this->raw_shapes_ = {new_cols};
  } else if (channels == 1) {
    this->raw_shapes_ = {new_rows, new_cols};
  } else {
    this->raw_shapes_ = {channels, new_rows, new_cols};
  }
}

void Tensor<float>::Fill(float value) {
  CHECK(!this->data_.empty());
  this->data_.fill(value);
}

void Tensor<float>::Fill(const std::vector<float>& values, bool row_major) {
  CHECK(!this->data_.empty());
  const uint32_t total_elems = this->data_.size();
  CHECK_EQ(values.size(), total_elems);
  if (row_major) {
    const uint32_t rows = this->rows();
    const uint32_t cols = this->cols();
    const uint32_t planes = rows * cols;
    const uint32_t channels = this->data_.n_slices;

    for (uint32_t i = 0; i < channels; ++i) {
      auto& channel_data = this->data_.slice(i);      //第i个通道的数据
      // 通过构造函数将values中的数据转换为arma::fmat类型，因为arma::fmat是列主序的，
      // 所以填充时把cols赋给行数，rows赋给列数
      const arma::fmat& channel_data_t =
          arma::fmat(values.data() + i * planes, this->cols(), this->rows());
      channel_data = channel_data_t.t();
    }
  } else {
    // 列主序则直接填充
    std::copy(values.begin(), values.end(), this->data_.memptr());
  }
}

void Tensor<float>::Show() {
  for (uint32_t i = 0; i < this->channels(); ++i) {
    LOG(INFO) << "Channel: " << i;
    LOG(INFO) << "\n" << this->data_.slice(i);
  }
}

void Tensor<float>::Flatten(bool row_major) {
  CHECK(!this->data_.empty());
  // 请补充代码
  // 优化：1维则提前返回
  if (this->raw_shapes_.size() == 1) {
    return;
  }
  unsigned int sz = this->size();
  this->Reshape({sz},row_major);   //注意Reshape会判断维度
  
  /* 
  system("pause")在linux下无效，需要采取缓冲区读取的方式
  printf("!!!!!!!!!!!!!!!!!!!!!!!!!");
  std::cout << "按 Enter 键继续..." << std::endl;
  std::cin.get();
  */
}

void Tensor<float>::Rand() {
  CHECK(!this->data_.empty());
  this->data_.randn();
}

void Tensor<float>::Ones() {
  CHECK(!this->data_.empty());
  this->Fill(1.f);
}

void Tensor<float>::Transform(const std::function<float(float)>& filter) {
  CHECK(!this->data_.empty());
  this->data_.transform(filter);
}

const std::vector<uint32_t>& Tensor<float>::raw_shapes() const {
  CHECK(!this->raw_shapes_.empty());
  CHECK_LE(this->raw_shapes_.size(), 3);
  CHECK_GE(this->raw_shapes_.size(), 1);
  return this->raw_shapes_;
}

void Tensor<float>::Reshape(const std::vector<uint32_t>& shapes,
                            bool row_major) {
  CHECK(!this->data_.empty());
  CHECK(!shapes.empty());
  const uint32_t origin_size = this->size();
  const uint32_t current_size =
      std::accumulate(shapes.begin(), shapes.end(), 1, std::multiplies());
  CHECK(shapes.size() <= 3);
  CHECK(current_size == origin_size);

  // 只有来的数据是行主序的，才需要先把数据取出再填充；列主序则数据存储方式不变直接reshape即可
  std::vector<float> values;
  if (row_major) {
    values = this->values(true);
  }
  if (shapes.size() == 3) {
    this->data_.reshape(shapes.at(1), shapes.at(2), shapes.at(0));
    this->raw_shapes_ = {shapes.at(0), shapes.at(1), shapes.at(2)};
  } else if (shapes.size() == 2) {
    this->data_.reshape(shapes.at(0), shapes.at(1), 1);
    this->raw_shapes_ = {shapes.at(0), shapes.at(1)};
  } else {
    this->data_.reshape(1, shapes.at(0), 1);
    this->raw_shapes_ = {shapes.at(0)};
  }

  if (row_major) {
    this->Fill(values, true);
  }
}

float* Tensor<float>::raw_ptr() {
  CHECK(!this->data_.empty());
  return this->data_.memptr();
}

float* Tensor<float>::raw_ptr(uint32_t offset) {
  const uint32_t size = this->size();
  CHECK(!this->data_.empty());
  CHECK_LT(offset, size);
  return this->data_.memptr() + offset;
}

std::vector<float> Tensor<float>::values(bool row_major) {
  CHECK_EQ(this->data_.empty(), false);
  std::vector<float> values(this->data_.size());

  if (!row_major) {
    std::copy(this->data_.mem, this->data_.mem + this->data_.size(),
              values.begin());
  } else {
    uint32_t index = 0;
    for (uint32_t c = 0; c < this->data_.n_slices; ++c) {
      const arma::fmat& channel = this->data_.slice(c).t();
      std::copy(channel.begin(), channel.end(), values.begin() + index);
      index += channel.size();
    }
    CHECK_EQ(index, values.size());
  }
  return values;
}

float* Tensor<float>::matrix_raw_ptr(uint32_t index) {
  CHECK_LT(index, this->channels());
  uint32_t offset = index * this->rows() * this->cols();
  CHECK_LE(offset, this->size());
  float* mem_ptr = this->raw_ptr() + offset;
  return mem_ptr;
}
}  // namespace kuiper_infer
