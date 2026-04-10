// mock libcsprpc.so content
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

// rpcmem
void   rpcmem_init(void);
void   rpcmem_deinit(void);
void * rpcmem_alloc(int heap_id, uint32_t flags, int size);  // NOTE: `size` is indeed an int here (< 2GB)
// NOTE(hzx): we don't mock rpcmem_alloc2 since it is not implemented in actual libcdsprpc.so (at least this is true on my phone)
void   rpcmem_free(void * p);
int    rpcmem_to_fd(void * p);

// fastrpc mmap & munmap
enum fastrpc_map_flags {
    FASTRPC_MAP_STATIC,
    FASTRPC_MAP_RESERVED,
    FASTRPC_MAP_FD,
    FASTRPC_MAP_FD_DELAYED,
};
int fastrpc_mmap(int domain, int fd, void *addr, int offset, size_t length, enum fastrpc_map_flags flags);
int fastrpc_munmap(int domain, int fd, void *addr, size_t length);

// rpcmem impl
void rpcmem_init(void) {}

void rpcmem_deinit(void) {}

// returns pointer to the buffer on success; NULL on failure
void * rpcmem_alloc(int heap_id, uint32_t flags, int size) {
    void * p = NULL;
    posix_memalign(&p, 128, size);
    return p;

    (void) heap_id;
    (void) flags;
}

// free a buffer and ignore invalid buffers
void rpcmem_free(void * p) {
    free(p);
}

// returns an associated file descriptor
int rpcmem_to_fd(void * p) {
    return 123;

    (void) p;
}

// Creates a mapping on remote process for an ION buffer with file descriptor. New fastrpc session will be opened if not already opened for the domain.
int fastrpc_mmap(int domain, int fd, void *addr, int offset, size_t length, enum fastrpc_map_flags flags) {
    return 0;

    (void) domain;
    (void) fd;
    (void) addr;
    (void) offset;
    (void) length;
    (void) flags;
}

// Removes a mapping associated with file descriptor.
int fastrpc_munmap(int domain, int fd, void *addr, size_t length) {
    return 0;

    (void) domain;
    (void) fd;
    (void) addr;
    (void) length;
}
