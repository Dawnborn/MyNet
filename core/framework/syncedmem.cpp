//
// Created by hjp on 2021/11/5.
//

#include "syncedmem.hpp"

namespace MyNet {
SyncedMemory::SyncedMemory()
    : cpu_ptr_(nullptr),
      size_(0ul),
      head_(UNINITIALIZED),
      own_cpu_data_(false) {}

SyncedMemory::SyncedMemory(uint32_t size)
    : cpu_ptr_(nullptr),
      size_(size),
      head_(UNINITIALIZED),
      own_cpu_data_(false) {}

SyncedMemory::~
}