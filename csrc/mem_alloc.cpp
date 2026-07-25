#include <cuda_runtime.h>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <errno.h>
#include <fcntl.h>
#include <cstring>            // for strerror
#include <linux/mempolicy.h>
#include <vector>
#include "mem_alloc.h"

uintptr_t alloc_pinned_ptr(size_t size, unsigned int flags) {
  void* ptr = nullptr;
  cudaError_t err = cudaHostAlloc(&ptr, size, flags);
  if (err != cudaSuccess) {
    throw std::runtime_error("cudaHostAlloc failed: " + std::to_string(err));
  }
  return reinterpret_cast<uintptr_t>(ptr);
}

void free_pinned_ptr(uintptr_t ptr) {
  cudaError_t err = cudaFreeHost(reinterpret_cast<void*>(ptr));
  if (err != cudaSuccess) {
    throw std::runtime_error("cudaFreeHost failed: " + std::to_string(err));
  }
}

static void first_touch(void* p, size_t size) {
  const long ps = sysconf(_SC_PAGESIZE);
  for (size_t off = 0; off < size; off += ps) {
    volatile char* c = (volatile char*)p + off;
    *c = 0;
  }
}

static inline int mbind_sys(void* addr, unsigned long len, int mode,
                            const unsigned long* nodemask,
                            unsigned long maxnode, unsigned int flags) {
  long rc = syscall(SYS_mbind, addr, len, mode, nodemask, maxnode, flags);
  return (rc == -1) ? -errno : 0;
}

static inline int get_mempolicy_sys(int* mode) {
  long rc = syscall(SYS_get_mempolicy, mode, nullptr, 0, nullptr, 0);
  return (rc == -1) ? -errno : 0;
}

static inline int set_mempolicy_sys(int mode, const unsigned long* nodemask,
                                    unsigned long maxnode) {
  long rc = syscall(SYS_set_mempolicy, mode, nodemask, maxnode);
  return (rc == -1) ? -errno : 0;
}

namespace {

class ScopedInterleavePolicy {
 public:
  explicit ScopedInterleavePolicy(const std::vector<int>& nodes) {
    if (nodes.empty()) return;

    int mode;
    int rc = get_mempolicy_sys(&mode);
    if (rc != 0)
      throw std::runtime_error(std::string("get_mempolicy failed: ") +
                               strerror(-rc));
    if (mode != MPOL_DEFAULT)
      throw std::runtime_error(
          "shared CPU cache NUMA interleave requires the default thread "
          "memory policy; remove the external numactl policy");

    int max_node = *std::max_element(nodes.begin(), nodes.end());
    if (max_node < 0)
      throw std::runtime_error("NUMA interleave nodes must be non-negative");
    constexpr size_t bits_per_word = sizeof(unsigned long) * 8;
    std::vector<unsigned long> mask(max_node / bits_per_word + 1);
    for (int node : nodes) {
      if (node < 0)
        throw std::runtime_error("NUMA interleave nodes must be non-negative");
      mask[node / bits_per_word] |= 1UL << (node % bits_per_word);
    }
    rc = set_mempolicy_sys(MPOL_INTERLEAVE, mask.data(), max_node + 1);
    if (rc != 0)
      throw std::runtime_error(std::string("set_mempolicy failed: ") +
                               strerror(-rc));
    active_ = true;
  }

  ~ScopedInterleavePolicy() {
    if (active_) set_mempolicy_sys(MPOL_DEFAULT, nullptr, 0);
  }

  void restore() {
    if (!active_) return;
    int rc = set_mempolicy_sys(MPOL_DEFAULT, nullptr, 0);
    if (rc != 0)
      throw std::runtime_error(std::string("restore mempolicy failed: ") +
                               strerror(-rc));
    active_ = false;
  }

 private:
  bool active_ = false;
};

}  // namespace

uintptr_t alloc_numa_ptr(size_t size, int node) {
  void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (ptr == MAP_FAILED)
    throw std::runtime_error(std::string("mmap failed: ") + strerror(errno));

  // Maximum of 64 numa nodes
  unsigned long mask = 1UL << node;
  long maxnode = 8 * sizeof(mask);
  if (mbind_sys(ptr, size, MPOL_BIND, &mask, maxnode,
                MPOL_MF_MOVE | MPOL_MF_STRICT) != 0) {
    int err = errno;
    munmap(ptr, size);
    throw std::runtime_error(std::string("mbind failed: ") + strerror(err));
  }

  first_touch(ptr, size);

  return reinterpret_cast<uintptr_t>(ptr);
}

void free_numa_ptr(uintptr_t ptr, size_t size) {
  void* p = reinterpret_cast<void*>(ptr);
  if (munmap(p, size) != 0) {
    throw std::runtime_error(std::string("munmap failed: ") + strerror(errno));
  }
}

uintptr_t alloc_pinned_numa_ptr(size_t size, int node) {
  void* ptr = reinterpret_cast<void*>(alloc_numa_ptr(size, node));

  cudaError_t st = cudaHostRegister(ptr, size, 0);
  if (st != cudaSuccess) {
    munmap(ptr, size);
    throw std::runtime_error(std::string("cudaHostRegister failed: ") +
                             cudaGetErrorString(st));
  }

  return reinterpret_cast<uintptr_t>(ptr);
}

void free_pinned_numa_ptr(uintptr_t ptr, size_t size) {
  void* p = reinterpret_cast<void*>(ptr);
  // Unpin first, then unmap.
  cudaError_t st = cudaHostUnregister(p);
  if (st != cudaSuccess) {
    munmap(p, size);
    throw std::runtime_error(std::string("cudaHostUnregister failed: ") +
                             cudaGetErrorString(st));
  }
  if (munmap(p, size) != 0) {
    throw std::runtime_error(std::string("munmap failed: ") + strerror(errno));
  }
}

uintptr_t alloc_shm_pinned_ptr(
    size_t size, const std::string& shm_name,
    const std::vector<int>& interleave_nodes) {
  if (size == 0)
    throw std::runtime_error("alloc_shm_pinned_ptr requires size > 0 for " +
                             shm_name);
  if (shm_name.empty())
    throw std::runtime_error("alloc_shm_pinned_ptr requires a shm_name");

  ScopedInterleavePolicy numa_policy(interleave_nodes);
  int fd = shm_open(shm_name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
  if (fd < 0)
    throw std::runtime_error(std::string("shm_open create failed for ") +
                             shm_name +
                             " (shared CPU cache segment already exists or "
                             "cannot be created; this usually means a live "
                             "name collision or stale segment from an unclean "
                             "shutdown, so choose a unique "
                             "shared_cpu_cache_name or unlink the stale "
                             "segment before restart): " +
                             strerror(errno));

  if (ftruncate(fd, size) != 0) {
    int err = errno;
    close(fd);
    shm_unlink(shm_name.c_str());
    throw std::runtime_error(std::string("ftruncate failed: ") + strerror(err));
  }

  void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  close(fd);
  if (ptr == MAP_FAILED) {
    shm_unlink(shm_name.c_str());
    throw std::runtime_error(std::string("mmap failed: ") + strerror(errno));
  }

  try {
    first_touch(ptr, size);
    numa_policy.restore();
  } catch (...) {
    munmap(ptr, size);
    shm_unlink(shm_name.c_str());
    throw;
  }

  cudaError_t st = cudaHostRegister(ptr, size, 0);
  if (st != cudaSuccess) {
    munmap(ptr, size);
    shm_unlink(shm_name.c_str());
    throw std::runtime_error(std::string("cudaHostRegister failed: ") +
                             cudaGetErrorString(st));
  }

  return reinterpret_cast<uintptr_t>(ptr);
}

uintptr_t attach_shm_pinned_ptr(size_t size, const std::string& shm_name,
                                bool writable) {
  if (size == 0)
    throw std::runtime_error("attach_shm_pinned_ptr requires size > 0 for " +
                             shm_name);
  if (shm_name.empty())
    throw std::runtime_error("attach_shm_pinned_ptr requires a shm_name");

  int fd = shm_open(shm_name.c_str(), writable ? O_RDWR : O_RDONLY, 0600);
  if (fd < 0)
    throw std::runtime_error(std::string("shm_open attach failed for ") +
                             shm_name + ": " + strerror(errno));

  int prot = writable ? (PROT_READ | PROT_WRITE) : PROT_READ;
  void* ptr = mmap(nullptr, size, prot, MAP_SHARED, fd, 0);
  close(fd);
  if (ptr == MAP_FAILED) {
    throw std::runtime_error(std::string("mmap attach failed for ") + shm_name +
                             ": " + strerror(errno));
  }

  cudaError_t st = cudaHostRegister(ptr, size, 0);
  if (st != cudaSuccess) {
    munmap(ptr, size);
    throw std::runtime_error(std::string("cudaHostRegister attach failed for ") +
                             shm_name + ": " + cudaGetErrorString(st));
  }

  return reinterpret_cast<uintptr_t>(ptr);
}

void free_shm_pinned_ptr(uintptr_t ptr, size_t size,
                         const std::string& shm_name) {
  if (ptr == 0)
    throw std::runtime_error("free_shm_pinned_ptr requires non-null ptr");
  if (size == 0)
    throw std::runtime_error("free_shm_pinned_ptr requires size > 0 for " +
                             shm_name);

  void* p = reinterpret_cast<void*>(ptr);
  cudaError_t st = cudaHostUnregister(p);
  if (st != cudaSuccess) {
    munmap(p, size);
    shm_unlink(shm_name.c_str());
    throw std::runtime_error(std::string("cudaHostUnregister failed: ") +
                             cudaGetErrorString(st));
  }
  if (munmap(p, size) != 0) {
    shm_unlink(shm_name.c_str());
    throw std::runtime_error(std::string("munmap failed: ") + strerror(errno));
  }
  shm_unlink(shm_name.c_str());
}

void detach_shm_pinned_ptr(uintptr_t ptr, size_t size) {
  if (ptr == 0)
    throw std::runtime_error("detach_shm_pinned_ptr requires non-null ptr");
  if (size == 0)
    throw std::runtime_error("detach_shm_pinned_ptr requires size > 0");

  void* p = reinterpret_cast<void*>(ptr);
  cudaError_t st = cudaHostUnregister(p);
  if (st != cudaSuccess) {
    munmap(p, size);
    throw std::runtime_error(std::string("cudaHostUnregister detach failed: ") +
                             cudaGetErrorString(st));
  }
  if (munmap(p, size) != 0) {
    throw std::runtime_error(std::string("munmap detach failed: ") +
                             strerror(errno));
  }
}

void unlink_shm(const std::string& shm_name) {
  if (shm_unlink(shm_name.c_str()) != 0 && errno != ENOENT) {
    throw std::runtime_error(std::string("shm_unlink failed for ") + shm_name +
                             ": " + strerror(errno));
  }
}
