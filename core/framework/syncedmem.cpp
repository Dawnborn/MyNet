//
// Created by hjp on 2021/11/5.
//

#include "syncedmem.hpp"

namespace MyNet {
SyncedMemory::SyncedMemory()
    : cpu_ptr_(nullptr),
      head_(UNINITIALIZED),
      size_(0ul),
      own_cpu_data_(false) {}

SyncedMemory::SyncedMemory(uint32_t size)
    : cpu_ptr_(nullptr),
      head_(UNINITIALIZED),
      size_(size),
      own_cpu_data_(false) {}

SyncedMemory::~SyncedMemory() {
  if (cpu_ptr_) {
    MyNetFreeHost(cpu_ptr_);
  }
}

inline void SyncedMemory::to_cpu_() {
  switch (head_) {
    case UNINITIALIZED:MyNetMallocHost(&cpu_ptr_, size_);
      std::memset(cpu_ptr_, 0, size_);
      head_ = HEAD_AT_CPU;
      own_cpu_data_ = true;
      break;
    case HEAD_AT_GPU:
    case HEAD_AT_CPU:
    case SYNCED:break;
  }
}

const void *SyncedMemory::cpu_data() {
  to_cpu_();
  return const_cast<const void *>(cpu_ptr_); // return const, no change
}

void SyncedMemory::set_cpu_data(void *data) {
  DCHECK(data);
  if (own_cpu_data_) {
    MyNetFreeHost(cpu_ptr_);
  }
  cpu_ptr_ = data;
  head_ = HEAD_AT_CPU;
  own_cpu_data_ = false;
}

void *SyncedMemory::mutable_cpu_data() {
  /*
   * copy data from gpu to cpu
   * */
  to_cpu_();
  head_ = HEAD_AT_CPU;
  return cpu_ptr_;
}

}// namespace MyNet