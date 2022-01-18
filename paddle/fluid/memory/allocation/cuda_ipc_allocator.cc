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

#include "paddle/fluid/memory/allocation/cuda_ipc_allocator.h"
#include "paddle/fluid/platform/cuda_device_guard.h"

#include <fcntl.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <random>
#include <string>

#include "glog/logging.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace memory {
namespace allocation {

namespace {
std::mutex ipc_mutex_;
std::unordered_map<std::string, std::weak_ptr<void>> ipc_handle_to_baseptr_;

struct CudaIPCGlobalEntities {
  static bool alive;

  std::mutex ref_counters_mutex_;
  std::map<std::string, std::shared_ptr<CudaIPCRefCountersFile>>
      ref_counters_files_;
  std::shared_ptr<CudaIPCRefCountersFile> next_available_ref_counters_file_;
  CudaIPCSentDataLimbo CudaIPCSentDataLimbo_;
  CudaIPCGlobalEntities() { alive = true; }
  ~CudaIPCGlobalEntities() {
    CudaIPCSentDataLimbo_.collect();
    safe_clean_current_file();
    if (next_available_ref_counters_file_) {
      VLOG(6) << "warnProducerTerminatedBeforeSharedTensorsReleased";
    }
    alive = false;
  }
  void safe_clean_current_file() {
    std::lock_guard<std::mutex> lock(ref_counters_mutex_);
    if (next_available_ref_counters_file_ &&
        next_available_ref_counters_file_->offsets_in_use() == 0) {
      ref_counters_files_.erase(next_available_ref_counters_file_->handle());
      next_available_ref_counters_file_.reset();
    }
  }
};

bool CudaIPCGlobalEntities::alive = false;
CudaIPCGlobalEntities cuda_ipc_global_entities;

CudaIPCSentDataLimbo::~CudaIPCSentDataLimbo() {
  collect();
  if (size() > 0) {
    VLOG(6) << "warnProducerTerminatedBeforeSharedTensorsReleased";
  }
}

bool CudaIPCSentDataLimbo::collect() {
  bool freed_memory = false;
  std::vector<std::unique_ptr<CudaIPCSentData>> reset_blocks;
  {  // Begin critical section to modify shared blocks
    std::lock_guard<std::mutex> lock(limbo_mutex_);
    std::vector<std::unique_ptr<CudaIPCSentData>> kept_blocks;
    // Delete the pointer of no referenced tensor
    // So, there must be some place increase the counter_value
    for (auto& sd : shared_blocks_) {
      if (sd->counter_value() > 0) {
        kept_blocks.push_back(std::move(sd));
      } else {
        freed_memory = true;
        reset_blocks.push_back(std::move(sd));
      }
    }
    shared_blocks_ = std::move(kept_blocks);
  }
  // Need to reset blocks out of the critical section here, otherwise it
  // deadlocks.
  for (auto& sd : reset_blocks) {
    sd.reset();
  }
  return freed_memory;
}

void CudaIPCSentDataLimbo::add(std::unique_ptr<CudaIPCSentData> shared_block) {
  std::lock_guard<std::mutex> lock(limbo_mutex_);
  static bool warned = false;
  if (shared_blocks_.size() > CUDA_IPC_WARN_AFTER_X_BLOCKS_IN_LIMBO &&
      !warned) {
    LOG(WARNING) << "Producer process tried to deallocate over "
                 << CUDA_IPC_WARN_AFTER_X_BLOCKS_IN_LIMBO
                 << " memory blocks referred by consumer processes. "
                    "Deallocation might be significantly slowed down. "
                 << "We assume it will never going to be the case, but if it "
                    "is, please file but to https://github.com/pytorch/pytorch";
    warned = true;
  }
  shared_blocks_.push_back(std::move(shared_block));
}

uint64_t CudaIPCSentDataLimbo::size() {
  std::lock_guard<std::mutex> lock(limbo_mutex_);
  return shared_blocks_.size();
}

void CudaIPCSentDataDelete(void* ptr) {
  std::unique_ptr<CudaIPCSentData> sent_data(
      static_cast<CudaIPCSentData*>(ptr));
  if (!CudaIPCGlobalEntities::alive) {
    return;
  }
  if (sent_data->counter_value() > 0) {
    cuda_ipc_global_entities.CudaIPCSentDataLimbo_.add(std::move(sent_data));
  }
  cuda_ipc_global_entities.CudaIPCSentDataLimbo_.collect();
}

void ReturnRefCounter(const std::string& handle, uint64_t offset /* unused */) {
  if (!CudaIPCGlobalEntities::alive) {
    return;
  }
  std::lock_guard<std::mutex> lock(
      cuda_ipc_global_entities.ref_counters_mutex_);
  auto& map = cuda_ipc_global_entities.ref_counters_files_;
  auto it = map.find(handle);
  if (it != map.end()) {
    it->second->return_offset(offset);
    // all slot is used, and no tensor in use
    if (it->second->offsets_in_use() == 0 && !it->second->have_offsets()) {
      map.erase(handle);
    }
  }
}
}  // namespace

std::shared_ptr<void> GetIpcBasePtr(std::string handle) {
  std::lock_guard<std::mutex> lock(ipc_mutex_);

  auto iter = ipc_handle_to_baseptr_.find(handle);
  if (iter != ipc_handle_to_baseptr_.end()) {
    auto baseptr = iter->second.lock();
    if (baseptr) return baseptr;
  }
  // The IpcMemHandle can only open once for the same handle,
  // so here we cache it here.
  void* baseptr = nullptr;
  auto ipc_handle = reinterpret_cast<const cudaIpcMemHandle_t*>(handle.c_str());
  VLOG(6) << "calling cudaIpcOpenMemHandle" PADDLE_ENFORCE_GPU_SUCCESS(
      cudaIpcOpenMemHandle(&baseptr, *ipc_handle,
                           cudaIpcMemLazyEnablePeerAccess));
  // Close ipc handle on the same device.
  int device_id = platform::GetCurrentDeviceId();
  // Add deleter to close ipc handle.
  auto sp = std::shared_ptr<void>(baseptr, [handle, device_id](void* ptr) {
    platform::CUDADeviceGuard guard(device_id);
    std::lock_guard<std::mutex> lock(ipc_mutex_);
    PADDLE_ENFORCE_GPU_SUCCESS(cudaIpcCloseMemHandle(ptr));
    ipc_handle_to_baseptr_.erase(handle);
    VLOG(6) << "cudaIpcCloseMemHandle for ptr:"
            << "\t" << ptr;
  });
  std::weak_ptr<void> wp = sp;
  ipc_handle_to_baseptr_.insert(iter, {handle, wp});

  return sp;
}

CudaIpcRecvAllocation::~CudaIpcRecvAllocation() {
  shared_ptr_.reset();
  VLOG(6) << "CudaIpcRecvAllocation deleted cudaIpcCloseMemHandle for ptr:"
          << "\t" << this->ptr();

  auto stream = paddle::platform::stream::get_current_stream(device_id_);
  stream->Synchronize();

  int flags = MAPPED_SHAREDMEM | MAPPED_NOCREATE;
  auto sptr = AllocateRefcountedMemoryMapAllocation(
      ctx_->ref_counter_handle.c_str(), flags,
      sizeof(int64_t) * CUDA_IPC_REF_COUNTER_FILE_SIZE);
  *(static_cast<int64_t*>(sptr->ptr()) + ctx_->ref_counter_offset) -= 1;
  VLOG(6) << "gpu tensor ref count: "
          << *(static_cast<int64_t*>(sptr->ptr()) + ctx_->ref_counter_offset);
}

CudaIpcSendAllocation::~CudaIpcSendAllocation() {
  VLOG(6) << "delete CudaIpcSendAllocation";
  CudaIPCSentDataDelete(std::move(sent_data_));
  VLOG(6) << "tensor send for ptr:"
          << "\t" << this->ptr();
}

// maybe new namespace

CudaIPCSentData::CudaIPCSentData(std::string handle, int64_t offset,
                                 int64_t* counter_ptr, int device_id)
    : handle_(std::move(handle)),
      offset_(offset),
      counter_ptr_(counter_ptr),
      original_ptr_(),
      device_id_(device_id) {
  auto stream = paddle::platform::stream::get_current_stream(device_id);
  stream->Synchronize();
}

CudaIPCSentData::~CudaIPCSentData() { ReturnRefCounter(handle_, offset_); }

int64_t CudaIPCSentData::counter_value() { return *counter_ptr_; }

std::shared_ptr<CudaIpcSendAllocation> GetNewRefCountedSentData(void* data,
                                                                size_t size,
                                                                int device_id) {
  {
    std::lock_guard<std::mutex> lock(
        cuda_ipc_global_entities.ref_counters_mutex_);
    if (!cuda_ipc_global_entities.next_available_ref_counters_file_) {
      std::string ref_counter_handle = GetIPCName();

      int flags = MAPPED_SHAREDMEM | MAPPED_EXCLUSIVE;
      // Size 10000 int64 share vector
      std::shared_ptr<RefcountedMemoryMapAllocation> sptr =
          AllocateRefcountedMemoryMapAllocation(
              ref_counter_handle.c_str(), flags,
              sizeof(int64_t) * CUDA_IPC_REF_COUNTER_FILE_SIZE);
      // Set to CudaIPCRefCountersFile
      auto rc = std::make_shared<CudaIPCRefCountersFile>(
          ref_counter_handle, CUDA_IPC_REF_COUNTER_FILE_SIZE, std::move(sptr));
      // Map? strings handle -> CudaIPCRefCountersFile
      cuda_ipc_global_entities.ref_counters_files_[ref_counter_handle] = rc;
      cuda_ipc_global_entities.next_available_ref_counters_file_ = rc;
    }
  }
  // ref_counter -> set 1
  cuda_ipc_global_entities.next_available_ref_counters_file_->set_counter(1);
  // cuda_ipc_global_entities is a global storage space of RC.
  auto sent_data = new CudaIPCSentData(
      cuda_ipc_global_entities.next_available_ref_counters_file_->handle(),
      cuda_ipc_global_entities.next_available_ref_counters_file_->get_offset(),
      cuda_ipc_global_entities.next_available_ref_counters_file_->counter_ptr(),
      device_id);
  VLOG(6) << "handle: " << sent_data->handle();
  VLOG(6) << "offse: " << sent_data->offset();

  // ref_counter -> pos + 1, next value 0
  cuda_ipc_global_entities.next_available_ref_counters_file_->rotate_offset();
  // Is 10000 int64 vector is enough
  if (!cuda_ipc_global_entities.next_available_ref_counters_file_
           ->have_offsets()) {
    // Release the space of the pointer
    cuda_ipc_global_entities.next_available_ref_counters_file_.reset();
  }
  return std::make_shared<CudaIpcSendAllocation>(data, size, device_id,
                                                 sent_data);
}

bool CudaIPCCollect() {
  if (!CudaIPCGlobalEntities::alive) {
    return true;
  }
  bool freed_memory = cuda_ipc_global_entities.CudaIPCSentDataLimbo_.collect();
  if (cuda_ipc_global_entities.CudaIPCSentDataLimbo_.size() == 0) {
    cuda_ipc_global_entities.safe_clean_current_file();
  }
  return freed_memory;
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
