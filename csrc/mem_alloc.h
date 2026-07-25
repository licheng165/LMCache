#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

uintptr_t alloc_pinned_ptr(size_t size, unsigned int flags);
uintptr_t alloc_numa_ptr(size_t size, int node);
uintptr_t alloc_pinned_numa_ptr(size_t size, int node);
uintptr_t alloc_shm_pinned_ptr(
    size_t size, const std::string& shm_name,
    const std::vector<int>& interleave_nodes = {});
uintptr_t attach_shm_pinned_ptr(size_t size, const std::string& shm_name,
                                bool writable);

void free_pinned_ptr(uintptr_t ptr);
void free_numa_ptr(uintptr_t ptr, size_t size);
void free_pinned_numa_ptr(uintptr_t ptr, size_t size);
void free_shm_pinned_ptr(uintptr_t ptr, size_t size,
                         const std::string& shm_name);
void detach_shm_pinned_ptr(uintptr_t ptr, size_t size);
void unlink_shm(const std::string& shm_name);
