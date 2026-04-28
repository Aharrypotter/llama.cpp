#pragma once

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

int prepare_tensor_rpcmem_mapping(const struct ggml_tensor * dst);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#    include <list>
#    include <unordered_map>

struct RpcMemMapper {
    // Callback type for flushing pending unmaps via DSP message channel.
    // Called between LRU eviction and new fastrpc_mmap when defer_unmap is true.
    // The callback must: (1) send unmap fds to DSP, (2) wait for DSP ack,
    // (3) call unmap_all_pending_buffers() to do host-side fastrpc_munmap.
    using DspFlushFn = void (*)(RpcMemMapper & mapper);

    RpcMemMapper() : max_active_map_size{ (size_t) -1 }, active_map_size{ 0 }, defer_unmap{ false } {}

    RpcMemMapper(size_t max_size, bool defer_unmap_ops) :
        max_active_map_size{ max_size },
        active_map_size{ 0 },
        defer_unmap{ defer_unmap_ops } {}

    void set_dsp_flush_callback(DspFlushFn fn) { dsp_flush_fn = fn; }

    void                    validate(const struct ggml_tensor * dst);
    std::pair<int, ssize_t> get_tensor_mapping(const struct ggml_tensor *) const;  // returns <mapping fd, offset>

    using UnmapRequest = std::tuple<int, void *, size_t>;

    const std::list<UnmapRequest> & get_pending_unmap_reqs() const { return pending_unmap_reqs; }

    void unmap_all_pending_buffers();

    void dump_state() const; // for debugging

  private:
    int n_map_ops = 0; // for debugging only, remove later

    size_t max_active_map_size;
    size_t active_map_size;

    bool                    defer_unmap;
    DspFlushFn              dsp_flush_fn = nullptr;
    std::list<UnmapRequest> pending_unmap_reqs;

    std::unordered_map<void *, std::pair<int, size_t>>      buf_mapping;
    std::unordered_map<void *, std::list<void *>::iterator> buf_iters;      // for LRU replacement
    std::list<void *>                                       accessed_bufs;  // for LRU replacement
};
#endif
