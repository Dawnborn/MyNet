// Copyright 2021 coordinate
// Author: Dawn

#ifndef CORE_FRAME_TENSOR_HPP_
#define CORE_FRAME_TENSOR_HPP_

#include <vector>
#include <cstdint>
#include <string>
#include "syncedmem.hpp"
#include "MyNet_generated.h"
#include "common.hpp"

const uint32_t kMaxTensorAxes = 32;

namespace MyNet {

template<typename Dtype>
class Tensor {
 public:
  Tensor() : data_(), diff_(), count_(0ul) {}
  explicit Tensor(uint32_t num, uint32_t channels, uint32_t height, uint32_t width);
  explicit Tensor(const std::vector<uint32_t> &shape);

  void Reshape(uint32_t num, uint32_t channels, uint32_t height, uint32_t width);
  void Reshape(const std::vector<uint32_t> &shape);
  void Reshape(const TensorShapeT *shape);
  void ReshapeLike(const Tensor& other);

  inline std::string shape_string() const {
    std::ostringstream stm;
    for (uint32_t i = 0; i < shape_.size(); i++) {
      stm << shape_[i] << " ";
    }
    stm << "(" << count_ << ")" << std::endl;
    return stm.str();
  }

  inline const std::vector<uint32_t> &shape() const { return shape_; }

  inline uint32_t shape(int32_t index) const {
    /*
     * return the shape in given dimension
     */
    return shape_[CanonicalAxisIndex(index)];
  }

  inline uint32_t num_axes() const {
    return shape_.size();
  }

  inline uint32_t count() const { return count_; }

  inline uint32_t count(uint32_t start_axis, uint32_t end_axis) const {
    DCHECK_LE(start_axis, end_axis);
    uint32_t num_axes_t = num_axes();
    DCHECK_LE(start_axix, num_axes_t);
    DCHECK_LE(end_axis, num_axes_t);
    uint32_t count = 1ul;
    for (uint32_t i = start_axis; i < end_axis; i++) {
      count *= shape(i);
    }
    return count;
  }

  inline uint32_t count(uint32_t start_axis) const {
    return count(start_axis, num_axes());
  }

  inline uint32_t CanonicalAxisIndex(int32_t axis_index) const {
    /*
     * to make use of negative index input
     * */
    int32_t num_axes_t = static_cast<int32_t>(num_axes());
    DCHECK_GE(axis_index, -num_axes_t)
      << "axis:" << axis_index << " out of range for " << num_axes_t
      << "-D Tensor with shape " << shape_string();
    DCHECK_LT(axis_index, num_axes_t)
      << "axis:" << axis_index << " out of range for " << num_axes_t
      << "-D Tensor with shape " << shape_string();
    if (axis_index < 0) {
      return axis_index + num_axes_t;
    }
    return axis_index;
  }

  inline uint32_t LegacyShape(int32_t index) const {
    DCHECK_LE(num_axes(), 4ul) << "Cannot use legacy accessors on Tensors with not 4 axes!" << std::endl;
    DCHECK_LE(index, 4);
    DCHECK_GE(index, -4);
    int32_t num_axis_t = static_cast<int32_t>(num_axes());
    if (index >= num_axes_t || index < -num_axis_t) {
      return 1;
    }
    return shape(index);
  }

  inline uint32_t num() const { return LegacyShape(0); }
  inline uint32_t channels() const { return LegacyShape(1); }
  inline uint32_t heights() const { return LegacyShape(2); }
  inline uint32_t width() const { return LegacyShape(3); }

  inline uint32_t offset(uint32_t n, uint32_t c = 0, uint32_t h = 0,
                         uint32_t w = 0) const {
    DCHECK_LE(n, num());
    DCHECK_LE(c, channels());
    DCHECK_LE(h, heights());
    DCHECK_LE(w, width());
    return ((n*channels()+c)*heights()+h)*width()+w;
  };

  inline uint32_t offset(const std::vector<uint32_t>& indices ) const {
    /*
     * return the position in a 1-D vector
     * the smallest dimensions can be omitted,
     * the count start with the largest dimension
     */
    DCHECK_LE(indices.size(), num_axes());
    uint32_t offset=0;
    for(uint32_t i = 0; i < num_axes(); i++){
      offset *= shape(i);
      if (indices.size()>i){
        DCHECK_LE(indices[i],shape(i));
        offset += indices[i];
      }
    }
    return offset;
  }

  void CopyFrom(const Tensor<Dtype>& src, bool copy_diff = false,
                bool reshape = false);

  inline Dtype data_at(uint32_t n, uint32_t c = 0, uint32_t h = 0,
                       uint32_t w = 0) const {
    return cpu_data()[offset(n,c,h,w)];
  }

  inline Dtype data_at(const std::vector<uint32_t>& index) const {
    return cpu_data()[offset(index)];
  }

  inline Dtype diff_at(uint32_t n, uint32_t c = 0, uint32_t h = 0,
                       uint32_t w = 0) const {
    return cpu_diff()[offset(n,c,h,w)];
  }

  inline Dtype diff_at(const std::vector<uint32_t>& index) const {
    return cpu_diff()[offset(index)];
  }

  inline const std::shared_ptr<SyncedMemory>& data() const {
    DCHECK(data_);
    return data_;
  }

  inline const std::shared_ptr<SyncedMemory>& diff() const {
    DCHECK(diff_);
    return diff_;
  }

  const Dtype* cpu_data() const;
  void set_cpu_data(Dtype* data);

  const Dtype* cpu_diff() const;

  Dtype* mutable_cpu_data() const;
  Dtype* mutable_cpu_diff() const;

  void Update();

  void FromFlat(const TensorFlatT* flat, bool reshape=true);
  flatbuffers::DetachedBuffer ToFlat(bool write_diff = false) const;

  Dtype asum_data() const;
  Dtype asum_diff() const;
  Dtype sqsum_data() const;
  Dtype sqsum_diff() const;

  void scale_data(Dtype scale_factor);
  void scale_diff(Dtype scale_factor);

  void ShareData(const Tnesor& other);
  void ShareDiff(const Tensor& other);
  bool ShapeEquals(const TensorFlatT* other);

 protected:
  std::shared_ptr<SyncedMemory> data_;
  std::shared_ptr<SyncedMemory> diff_;
  std::vector<uint32_t> shape_;
  std::shared_ptr<SyncedMemory> shape_data_;
  uint32_t count_;
  uint32_t capacity_; // size of SyncedMemory,

  DISABLE_COPY_AND_ASSIGN(Tensor);
};
}

#endif  // CORE_FRAME_TENSOR_HPP_