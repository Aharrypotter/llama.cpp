// SPDX-License-Identifier: BSD-3-Clause

#include <assert.h>
#include <stdint.h>
#include <string.h>

#if defined(HTP_EDGEKV_RECONSTRUCT_AOT) || defined(HTP_EDGEKV_DIRECT_AOT)
struct edgekv_unranked_memref {
    int64_t rank;
    void *  descriptor;
};

struct edgekv_dynamic_memref {
    char *  allocated;
    char *  aligned;
    int64_t offset;
    int64_t sizes_and_strides[];
};

// Hexagon-MLIR AOT objects reference the generic helper when a lowered copy is
// not proven contiguous. Simulator wrappers normally provide it; the HTP skel
// needs an equivalent implementation without linking the test runtime.
void memrefCopy(int64_t elem_size, struct edgekv_unranked_memref * src_arg, struct edgekv_unranked_memref * dst_arg) {
    const int64_t rank = src_arg->rank;
    assert(rank >= 0 && rank <= 8);
    assert(dst_arg->rank == rank);

    const struct edgekv_dynamic_memref * src         = (const struct edgekv_dynamic_memref *) src_arg->descriptor;
    const struct edgekv_dynamic_memref * dst         = (const struct edgekv_dynamic_memref *) dst_arg->descriptor;
    const int64_t *                      src_sizes   = src->sizes_and_strides;
    const int64_t *                      src_strides = src_sizes + rank;
    const int64_t *                      dst_strides = dst->sizes_and_strides + rank;

    for (int64_t axis = 0; axis < rank; ++axis) {
        if (src_sizes[axis] == 0) {
            return;
        }
    }

    char * src_ptr = src->aligned + src->offset * elem_size;
    char * dst_ptr = dst->aligned + dst->offset * elem_size;
    if (rank == 0) {
        memcpy(dst_ptr, src_ptr, (size_t) elem_size);
        return;
    }

    int64_t indices[8]    = { 0 };
    int64_t src_stride[8] = { 0 };
    int64_t dst_stride[8] = { 0 };
    for (int64_t axis = 0; axis < rank; ++axis) {
        src_stride[axis] = src_strides[axis] * elem_size;
        dst_stride[axis] = dst_strides[axis] * elem_size;
    }

    int64_t read_index  = 0;
    int64_t write_index = 0;
    for (;;) {
        memcpy(dst_ptr + write_index, src_ptr + read_index, (size_t) elem_size);
        for (int64_t axis = rank - 1; axis >= 0; --axis) {
            const int64_t next = ++indices[axis];
            read_index += src_stride[axis];
            write_index += dst_stride[axis];
            if (src_sizes[axis] != next) {
                break;
            }
            if (axis == 0) {
                return;
            }
            indices[axis] = 0;
            read_index -= src_sizes[axis] * src_stride[axis];
            write_index -= src_sizes[axis] * dst_stride[axis];
        }
    }
}
#endif
