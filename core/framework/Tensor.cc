#include "Tensor.hpp"

namespace MyNet {

template<typename Dtype>
Tensor<Dtype>::Tensor(uint32_t num, uint32_t channels, uint32_t height, uint32_t width) : capacity_(0ul) {
  Reshape({num, channels, height, width});
}

template<typename Dtype>
Tensor<Dtype>::Tensor(const std::vector<uint32_t>&shape):capacity_(0ul) {
  Reshape(shape);
}

template<typename Dtype>
void Tensor<Dtype>::Reshape(uint32_t num, uint32_t channels, uint32_t height, uint32_t width) {
  Reshape({num, channels, height, width});
}

template<typename Dtype>
void Tensor<Dtype>::Reshape(const std::vector<uint32_t> &shape) {
  DCHECK_LE(shape.size, kMaxTensorAxes);
  count_ = 1ul;
  shape_.resize(shape.size());
  if (!shape_data_ || shape_data_.size() < shape.size() * sizeof(uint32_t)) {
    shape_data_.reset(new SyncedMemory(shape.size() * sizeof(uint32_t)))
  }

  uint32_t *shape_data = static_cast<uint32_t *>(shape_data_->mutable_cpu_data());
  for (uint32_t i = 0; i < shape.size(); i++) {
    DCHECK_LE(shape[i], UINT32_MAX / count_) << "Tensor size exceeds UINT32_MAX";
    count_ *= shape[i];
    shape_[i] = shape[i];
    shape_data[i] = shape[i];
  }

  if (count_ > capacity_) {
    capacity_ = count_;
    data_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));
    diff_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));
  }
}


} // namespace MyNet