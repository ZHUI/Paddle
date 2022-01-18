// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <memory>
#include <mutex>  // NOLINT
#include <string>
#include <unordered_set>
#include <utility>

#include "paddle/fluid/memory/allocation/allocator.h"
#include "paddle/fluid/memory/allocation/mmap_allocator.h"
#include "paddle/fluid/platform/cuda_device_guard.h"
#include "paddle/fluid/platform/device/gpu/gpu_info.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/stream/cuda_stream.h"

namespace paddle {
namespace memory {
namespace allocation {

std::shared_ptr<void> GetIpcBasePtr(std::string handle);

bool CudaIPCCollect();

struct CudaIPCReceivedData final {
  CudaIPCReceivedData() = default;
  explicit CudaIPCReceivedData(std::shared_ptr<void> shared_ptr)
      : shared_ptr_(std::move(shared_ptr)) {}
  std::shared_ptr<void> shared_ptr_;
};

struct IpcDeleterContext {
  std::string ref_counter_handle;
  ptrdiff_t ref_counter_offset;
  int64_t device_id;
};

struct CudaIPCSentData final {
  std::string handle_;
  int64_t offset_;
  int64_t* counter_ptr_;  // Reference counter shared memory block
  std::shared_ptr<Allocation> original_ptr_;  // Original mem allocation
  int device_id_;

  CudaIPCSentData(std::string handle, int64_t offset, int64_t* counter_ptr,
                  int device_id);
  ~CudaIPCSentData();

  int64_t counter_value();
  std::string handle() { return handle_; }
  int64_t offset() { return offset_; }
  void set_original_ptr(std::shared_ptr<Allocation> data_ptr) {
    VLOG(6) << "set_original_ptr";
    original_ptr_ = data_ptr;
  }
};

class CudaIpcRecvAllocation : public Allocation {
 public:
  // explicit CudaIpcRecvAllocation(void *ptr, size_t size, int device_id,
  //                            std::shared_ptr<void> shared_ptr)
  //     : Allocation(ptr, size, platform::CUDAPlace(device_id)),
  //       device_id_(std::move(device_id)),
  //       shared_ptr_(std::move(shared_ptr)) {}

  explicit CudaIpcRecvAllocation(void* ptr, size_t size, int device_id,
                                 std::shared_ptr<void> shared_ptr,
                                 IpcDeleterContext* ctx)
      : Allocation(ptr, size, platform::CUDAPlace(device_id)),
        device_id_(std::move(device_id)),
        shared_ptr_(std::move(shared_ptr)),
        ctx_(std::move(ctx)) {}

  inline const int& device_id() const { return device_id_; }

  ~CudaIpcRecvAllocation() override;

 private:
  int device_id_;
  // the base ptr
  std::shared_ptr<void> shared_ptr_;
  IpcDeleterContext* ctx_ = nullptr;
};

class CudaIpcSendAllocation : public Allocation {
 public:
  explicit CudaIpcSendAllocation(void* ptr, size_t size, int device_id,
                                 CudaIPCSentData* sent_data)
      : Allocation(ptr, size, platform::CUDAPlace(device_id)),
        device_id_(std::move(device_id)),
        sent_data_(sent_data) {
    VLOG(6) << "handle" << sent_data_->handle();
    VLOG(6) << "offset" << sent_data_->offset();
  }

  inline const int& device_id() const { return device_id_; }
  CudaIPCSentData* sent_data() { return sent_data_; }

  ~CudaIpcSendAllocation() override;

 private:
  int device_id_;
  CudaIPCSentData* sent_data_;
};

std::shared_ptr<CudaIpcSendAllocation> GetNewRefCountedSentData(void* data,
                                                                size_t size,
                                                                int device_id);

namespace {  // NOLINT

constexpr int64_t CUDA_IPC_REF_COUNTER_FILE_SIZE = 10000;
constexpr int64_t CUDA_IPC_WARN_AFTER_X_BLOCKS_IN_LIMBO = 1000;

struct CudaIPCSentDataLimbo final {
  ~CudaIPCSentDataLimbo();
  bool collect();
  void add(std::unique_ptr<CudaIPCSentData> shared_block);
  uint64_t size();

 private:
  std::vector<std::unique_ptr<CudaIPCSentData>> shared_blocks_;
  std::mutex limbo_mutex_;
};

struct CudaIPCRefCountersFile final {
  CudaIPCRefCountersFile(std::string handle, uint64_t size,
                         std::shared_ptr<Allocation> data_ptr)
      : next_offset_(0),
        size_(size),
        used_slots_(0),
        handle_(std::move(handle)),
        refcounted_shared_mem_(std::move(data_ptr)) {}

  int64_t* counter_ptr() {
    return static_cast<int64_t*>(refcounted_shared_mem_->ptr()) + next_offset_;
  }

  void set_counter(uint64_t value) { *counter_ptr() = value; }

  bool have_offsets() { return next_offset_ < size_; }

  bool offsets_in_use() { return used_slots_; }

  int64_t get_offset() { return next_offset_; }

  void rotate_offset() {
    next_offset_++;
    used_slots_++;
  }

  void return_offset(uint64_t offset /* unused */) { used_slots_--; }

  std::string handle() { return handle_; }

 private:
  uint64_t next_offset_;
  uint64_t size_;
  uint64_t used_slots_;
  std::string handle_;
  std::shared_ptr<Allocation> refcounted_shared_mem_;
};

}  // namespace

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
