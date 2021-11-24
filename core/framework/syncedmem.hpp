//
// Created by hjp on 2021/11/4.
//
#ifndef MYNET_CORE_FRAMEWORK_SYNCEDMEM_HPP_
#define MYNET_CORE_FRAMEWORK_SYNCEDMEM_HPP_

#include "common.hpp"
#include <cstdlib>

namespace MyNet {

inline void MyNetMallocHost(void **ptr, uint32_t size) {
  *ptr = malloc(size);
  DCHECK(*ptr) << "host allocation of "<< size << " failed! ";
}

inline void MyNetFreeHost(void *ptr) {

}

class SyncedMemory {
 public:
  SyncedMemory();
  explicit SyncedMemory(uint32_t size);
  ~SyncedMemory();
  const void *cpu_data();
  void set_cpu_data();
  void set_gpu_data();
  void *mutable_cpu_data();
  void *mutable_gpu_data();
  enum SyncedHead { UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED };
  SyncedHead head() const { return head_; };
  uint32_t size() const { return size_; }
 private:
  void to_cpu_();
  void to_gpu_();
  void *cpu_ptr_;
  SyncedHead head_;
  uint32_t size_;
  bool own_cpu_data_;
  bool own_gpu_data_;

 DISABLE_COPY_AND_ASSIGN(SyncedMemory); //
};
} // namespace MyNet

#endif //MYNET_CORE_FRAMEWORK_SYNCEDMEM_HPP_
