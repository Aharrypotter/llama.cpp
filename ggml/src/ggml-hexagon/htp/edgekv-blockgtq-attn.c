// SPDX-License-Identifier: BSD-3-Clause

#include "edgekv-blockgtq-attn.h"

#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>

#ifdef EDGEKV_BLOCKGTQ_ATTRIBUTION
static const char * const edgekv_blockgtq_stage_names[] = {
    "core_input_validation",
    "static_package_validation",
    "catalog_construction",
    "query_rotation",
    "first_k_decode_dot",
    "diagnostics_off_second_k_softmax_v_fused",
    "diagnostics_on_logit_reuse_softmax_v_fused",
    "normalization",
    "inverse_v_rotation",
};

_Static_assert(
    sizeof(edgekv_blockgtq_stage_names) /
            sizeof(edgekv_blockgtq_stage_names[0]) ==
        EDGEKV_BLOCKGTQ_STAGE_COUNT,
    "Block-GTQ attribution stage inventory drift");

const char * edgekv_blockgtq_stage_name(enum edgekv_blockgtq_stage stage) {
    return stage >= 0 && stage < EDGEKV_BLOCKGTQ_STAGE_COUNT
               ? edgekv_blockgtq_stage_names[stage]
               : NULL;
}

static uint64_t attribution_start(
    const struct edgekv_blockgtq_attribution * attribution) {
    return attribution != NULL && attribution->clock != NULL
               ? attribution->clock(attribution->clock_context)
               : 0;
}

static void attribution_stop(
    struct edgekv_blockgtq_attribution * attribution,
    enum edgekv_blockgtq_stage stage, uint64_t start) {
    if (attribution == NULL || attribution->clock == NULL) {
        return;
    }
    const uint64_t stop = attribution->clock(attribution->clock_context);
    attribution->pcycles[stage] += stop - start;
    ++attribution->intervals[stage];
}
#endif

enum {
    TRANSFORM_K_ROTATIONS = 0,
    TRANSFORM_V_ROTATIONS = 817728,

    CONSUMER_K_LUT = 0,
    CONSUMER_K_LUT_OFFSET = 57728,
    CONSUMER_K_NORM_GROUP = 94592,
    CONSUMER_V_LUT = 131456,
    CONSUMER_K_DESCRIPTORS = 133760,
    CONSUMER_V_ROTATION_INDEX = 142976,

    SHARED_K_PERMUTATION = 0,
    SHARED_K_INVERSE_PERMUTATION = 9216,
    SHARED_K_SOURCE_ALLOCATION = 18432,
    SHARED_SEQUENCE_DESCRIPTOR = 23040,

    K_ROTATION_FLOATS = 204432,
    K_LUT_HALF_VALUES = 28864,
};

static uint16_t read_u16(const uint8_t * p) {
    return (uint16_t) p[0] | ((uint16_t) p[1] << 8);
}

static int16_t read_i16(const uint8_t * p) {
    return (int16_t) read_u16(p);
}

static uint32_t read_u32(const uint8_t * p) {
    return (uint32_t) p[0] | ((uint32_t) p[1] << 8) |
           ((uint32_t) p[2] << 16) | ((uint32_t) p[3] << 24);
}

static int32_t read_i32(const uint8_t * p) {
    return (int32_t) read_u32(p);
}

static float read_f32(const uint8_t * p) {
    const uint32_t bits = read_u32(p);
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

#ifdef EDGEKV_BLOCKGTQ_TARGET_OBSERVABILITY
static uint32_t float_bits(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static uint64_t fnv1a64_u32(uint64_t hash, uint32_t bits) {
    for (int byte = 0; byte < 4; ++byte) {
        hash ^= (bits >> (byte * 8)) & 0xffu;
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

#ifdef EDGEKV_BLOCKGTQ_TARGET_LOGIT_FORENSICS
static uint64_t fnv1a64_u8(uint64_t hash, uint8_t value) {
    hash ^= value;
    hash *= UINT64_C(1099511628211);
    return hash;
}
#endif
#endif

static float half_to_float(uint16_t bits) {
    const uint32_t sign = (uint32_t) (bits & 0x8000u) << 16;
    uint32_t exponent = (bits >> 10) & 0x1fu;
    uint32_t mantissa = bits & 0x3ffu;
    uint32_t output;
    if (exponent == 0) {
        if (mantissa == 0) {
            output = sign;
        } else {
            int shift = 0;
            while ((mantissa & 0x400u) == 0) {
                mantissa <<= 1;
                ++shift;
            }
            mantissa &= 0x3ffu;
            output = sign |
                     ((uint32_t) (127 - 15 - shift) << 23) |
                     (mantissa << 13);
        }
    } else if (exponent == 31) {
        output = sign | 0x7f800000u | (mantissa << 13);
    } else {
        output = sign | ((exponent + 112u) << 23) | (mantissa << 13);
    }
    float value;
    memcpy(&value, &output, sizeof(value));
    return value;
}

static int descriptor(const uint8_t * consumer, int head, int segment,
                      int field) {
    const size_t offset = CONSUMER_K_DESCRIPTORS +
                          ((size_t) head * 8u * 8u +
                           (size_t) segment * 8u + (size_t) field) *
                              2u;
    return (int) read_i16(consumer + offset);
}

static int permutation(const uint8_t * shared, int head, int dim) {
    return (int) shared[SHARED_K_PERMUTATION +
                        (size_t) head * EDGEKV_BLOCKGTQ_HEAD_DIM +
                        (size_t) dim];
}

static uint8_t packed_code(const uint8_t * codes, int pack_offset,
                           int logical_index, int bits) {
    if (bits == 8) {
        return codes[pack_offset + logical_index];
    }
    const uint8_t value = codes[pack_offset + logical_index / 2];
    return (logical_index & 1) == 0 ? value & 15u : value >> 4;
}

static size_t k_codes_offset(int token, int head) {
    return (size_t) token * EDGEKV_BLOCKGTQ_HEADS *
               EDGEKV_BLOCKGTQ_K_CODE_STRIDE +
           (size_t) head * EDGEKV_BLOCKGTQ_K_CODE_STRIDE;
}

static size_t k_norms_pool_offset(int capacity) {
    return (size_t) capacity * EDGEKV_BLOCKGTQ_HEADS *
           EDGEKV_BLOCKGTQ_K_CODE_STRIDE;
}

static size_t k_norms_offset(int capacity, int token, int head) {
    return k_norms_pool_offset(capacity) +
           (size_t) token * EDGEKV_BLOCKGTQ_HEADS *
               EDGEKV_BLOCKGTQ_K_NORM_STRIDE +
           (size_t) head * EDGEKV_BLOCKGTQ_K_NORM_STRIDE;
}

static size_t v_codes_pool_offset(int capacity) {
    return k_norms_pool_offset(capacity) +
           (size_t) capacity * EDGEKV_BLOCKGTQ_HEADS *
               EDGEKV_BLOCKGTQ_K_NORM_STRIDE;
}

static size_t v_codes_offset(int capacity, int token, int head) {
    return v_codes_pool_offset(capacity) +
           (size_t) token * EDGEKV_BLOCKGTQ_HEADS *
               EDGEKV_BLOCKGTQ_V_CODE_STRIDE +
           (size_t) head * EDGEKV_BLOCKGTQ_V_CODE_STRIDE;
}

static size_t v_norms_pool_offset(int capacity) {
    return v_codes_pool_offset(capacity) +
           (size_t) capacity * EDGEKV_BLOCKGTQ_HEADS *
               EDGEKV_BLOCKGTQ_V_CODE_STRIDE;
}

static size_t v_norms_offset(int capacity, int token, int head) {
    return v_norms_pool_offset(capacity) +
           (size_t) token * EDGEKV_BLOCKGTQ_HEADS *
               EDGEKV_BLOCKGTQ_V_NORM_STRIDE +
           (size_t) head * EDGEKV_BLOCKGTQ_V_NORM_STRIDE;
}

static size_t dynamic_bytes(int capacity) {
    return v_norms_pool_offset(capacity) +
           (size_t) capacity * EDGEKV_BLOCKGTQ_HEADS *
               EDGEKV_BLOCKGTQ_V_NORM_STRIDE;
}

int edgekv_blockgtq_history_addresses(
    int capacity, int token, int head,
    struct edgekv_blockgtq_history_addresses * addresses) {
    if (addresses == NULL || capacity != EDGEKV_BLOCKGTQ_CAPACITY ||
        token < 0 || token >= capacity || head < 0 ||
        head >= EDGEKV_BLOCKGTQ_HEADS) {
        return EDGEKV_BLOCKGTQ_INVALID_ARGUMENT;
    }
    addresses->k_codes = k_codes_offset(token, head);
    addresses->k_norms = k_norms_offset(capacity, token, head);
    addresses->v_codes = v_codes_offset(capacity, token, head);
    addresses->v_norms = v_norms_offset(capacity, token, head);
    const size_t bytes = dynamic_bytes(capacity);
    if (addresses->k_codes + EDGEKV_BLOCKGTQ_K_CODE_STRIDE > bytes ||
        addresses->k_norms + EDGEKV_BLOCKGTQ_K_NORM_STRIDE > bytes ||
        addresses->v_codes + EDGEKV_BLOCKGTQ_V_CODE_STRIDE > bytes ||
        addresses->v_norms + EDGEKV_BLOCKGTQ_V_NORM_STRIDE > bytes) {
        return EDGEKV_BLOCKGTQ_DYNAMIC_LAYOUT_MISMATCH;
    }
    return EDGEKV_BLOCKGTQ_OK;
}

static int build_catalogs(const uint8_t * consumer,
                          size_t rotation_offsets[85],
                          size_t head_lut_bases[EDGEKV_BLOCKGTQ_HEADS]) {
    int rotation_dims[85] = {0};
    int codebook_bits[55] = {0};
    int semantic_groups = 0;
    size_t global_lut = 0;
    for (int head = 0; head < EDGEKV_BLOCKGTQ_HEADS; ++head) {
        size_t dim_cursor = 0;
        size_t pack_cursor = 0;
        size_t local_lut = 0;
        bool inactive = false;
        head_lut_bases[head] = global_lut;
        for (int segment = 0; segment < EDGEKV_BLOCKGTQ_MAX_SEGMENTS;
             ++segment) {
            const int source_bits = descriptor(consumer, head, segment, 0);
            const int bits = descriptor(consumer, head, segment, 1);
            const int start = descriptor(consumer, head, segment, 2);
            const int length = descriptor(consumer, head, segment, 3);
            const int packed_at = descriptor(consumer, head, segment, 4);
            const int packed_bytes = descriptor(consumer, head, segment, 5);
            const int rotation = descriptor(consumer, head, segment, 6);
            const int codebook = descriptor(consumer, head, segment, 7);
            if (start < 0) {
                if (source_bits != 0 || bits != 0 || start != -1 ||
                    length != 0 || packed_at != 0 || packed_bytes != 0 ||
                    rotation != -1 ||
                    descriptor(consumer, head, segment, 7) != -1) {
                    return EDGEKV_BLOCKGTQ_STATIC_LAYOUT_MISMATCH;
                }
                inactive = true;
                continue;
            }
            if (inactive || source_bits < 1 || source_bits > 8 ||
                (bits != 4 && bits != 8) || start != (int) dim_cursor ||
                length <= 0 || (length & 1) != 0 ||
                packed_at != (int) pack_cursor ||
                packed_bytes != (bits == 4 ? length / 2 : length) ||
                rotation < 0 || rotation >= 85 ||
                codebook < 0 || codebook >= 55) {
                return EDGEKV_BLOCKGTQ_STATIC_LAYOUT_MISMATCH;
            }
            if (rotation_dims[rotation] == 0) {
                rotation_dims[rotation] = length;
            }
            if (rotation_dims[rotation] != length) {
                return EDGEKV_BLOCKGTQ_STATIC_LAYOUT_MISMATCH;
            }
            if (codebook_bits[codebook] == 0) {
                codebook_bits[codebook] = bits;
            }
            if (codebook_bits[codebook] != bits) {
                return EDGEKV_BLOCKGTQ_STATIC_LAYOUT_MISMATCH;
            }
            dim_cursor += (size_t) length;
            pack_cursor += (size_t) packed_bytes;
            local_lut += (size_t) 1u << bits;
            ++semantic_groups;
        }
        if (dim_cursor != EDGEKV_BLOCKGTQ_HEAD_DIM ||
            pack_cursor > EDGEKV_BLOCKGTQ_K_CODE_STRIDE) {
            return EDGEKV_BLOCKGTQ_STATIC_LAYOUT_MISMATCH;
        }
        global_lut += local_lut;
    }
    if (semantic_groups != 319 || global_lut != K_LUT_HALF_VALUES) {
        return EDGEKV_BLOCKGTQ_STATIC_LAYOUT_MISMATCH;
    }
    size_t codebook_cursor = 0;
    for (int index = 0; index < 55; ++index) {
        if (codebook_bits[index] != 4 && codebook_bits[index] != 8) {
            return EDGEKV_BLOCKGTQ_STATIC_LAYOUT_MISMATCH;
        }
        codebook_cursor += (size_t) 1u << codebook_bits[index];
    }
    if (codebook_cursor != 2320) {
        return EDGEKV_BLOCKGTQ_STATIC_LAYOUT_MISMATCH;
    }
    size_t rotation_cursor = 0;
    for (int index = 0; index < 85; ++index) {
        if (rotation_dims[index] <= 0) {
            return EDGEKV_BLOCKGTQ_STATIC_LAYOUT_MISMATCH;
        }
        rotation_offsets[index] = rotation_cursor;
        rotation_cursor +=
            (size_t) rotation_dims[index] * (size_t) rotation_dims[index];
    }
    return rotation_cursor == K_ROTATION_FLOATS
               ? EDGEKV_BLOCKGTQ_OK
               : EDGEKV_BLOCKGTQ_STATIC_LAYOUT_MISMATCH;
}

int edgekv_blockgtq_validate_static(const uint8_t * transform,
                                    size_t transform_bytes,
                                    const uint8_t * consumer,
                                    size_t consumer_bytes,
                                    const uint8_t * shared,
                                    size_t shared_bytes) {
    if (transform == NULL || consumer == NULL || shared == NULL ||
        transform_bytes != EDGEKV_BLOCKGTQ_TRANSFORM_BYTES ||
        consumer_bytes != EDGEKV_BLOCKGTQ_CONSUMER_BYTES ||
        shared_bytes != EDGEKV_BLOCKGTQ_SHARED_BYTES) {
        return EDGEKV_BLOCKGTQ_INVALID_ARGUMENT;
    }
    size_t rotation_offsets[85];
    size_t head_lut_bases[EDGEKV_BLOCKGTQ_HEADS];
    int status = build_catalogs(consumer, rotation_offsets, head_lut_bases);
    if (status != EDGEKV_BLOCKGTQ_OK) {
        return status;
    }
    bool original_seen[EDGEKV_BLOCKGTQ_HEAD_DIM];
    int semantic_groups = 0;
    for (int head = 0; head < EDGEKV_BLOCKGTQ_HEADS; ++head) {
        memset(original_seen, 0, sizeof(original_seen));
        size_t local_lut = 0;
        int active_segments = 0;
        for (int segment = 0; segment < EDGEKV_BLOCKGTQ_MAX_SEGMENTS;
             ++segment) {
            const int bits = descriptor(consumer, head, segment, 1);
            const int start = descriptor(consumer, head, segment, 2);
            const int length = descriptor(consumer, head, segment, 3);
            if (start < 0) {
                continue;
            }
            const int source_bits = descriptor(consumer, head, segment, 0);
            for (int local = 0; local < length; ++local) {
                const int dim = start + local;
                const int original = permutation(shared, head, dim);
                if (original < 0 || original >= EDGEKV_BLOCKGTQ_HEAD_DIM ||
                    original_seen[original] ||
                    shared[SHARED_K_INVERSE_PERMUTATION +
                           (size_t) head * EDGEKV_BLOCKGTQ_HEAD_DIM +
                           (size_t) original] != dim ||
                    read_i32(consumer + CONSUMER_K_LUT_OFFSET +
                             ((size_t) head * EDGEKV_BLOCKGTQ_HEAD_DIM +
                              (size_t) dim) *
                                 4u) != (int32_t) local_lut ||
                    read_i32(consumer + CONSUMER_K_NORM_GROUP +
                             ((size_t) head * EDGEKV_BLOCKGTQ_HEAD_DIM +
                              (size_t) dim) *
                                 4u) != segment) {
                    return EDGEKV_BLOCKGTQ_STATIC_LAYOUT_MISMATCH;
                }
                original_seen[original] = true;
                if ((local & 1) == 0) {
                    if (original >= 64 ||
                        permutation(shared, head, dim + 1) != original + 64 ||
                        shared[SHARED_K_SOURCE_ALLOCATION +
                               (size_t) head * 64u + (size_t) original] !=
                            source_bits) {
                        return EDGEKV_BLOCKGTQ_STATIC_LAYOUT_MISMATCH;
                    }
                }
            }
            local_lut += (size_t) 1u << bits;
            ++active_segments;
            ++semantic_groups;
        }
        if (active_segments < 4 ||
            active_segments > EDGEKV_BLOCKGTQ_MAX_SEGMENTS) {
            return EDGEKV_BLOCKGTQ_STATIC_LAYOUT_MISMATCH;
        }
        if (read_u16(consumer + CONSUMER_V_ROTATION_INDEX +
                     (size_t) head * 2u) >= 2) {
            return EDGEKV_BLOCKGTQ_STATIC_LAYOUT_MISMATCH;
        }
    }
    for (int layer = 0; layer < EDGEKV_BLOCKGTQ_LAYERS; ++layer) {
        const uint8_t * row =
            shared + SHARED_SEQUENCE_DESCRIPTOR + (size_t) layer * 16u;
        if (read_i16(row) != layer || read_i16(row + 2) != layer * 2 ||
            read_i16(row + 4) != layer * 2) {
            return EDGEKV_BLOCKGTQ_STATIC_LAYOUT_MISMATCH;
        }
        for (int i = 6; i < 16; ++i) {
            if (row[i] != 0) {
                return EDGEKV_BLOCKGTQ_STATIC_LAYOUT_MISMATCH;
            }
        }
    }
    return semantic_groups == 319 ? EDGEKV_BLOCKGTQ_OK
                                  : EDGEKV_BLOCKGTQ_STATIC_LAYOUT_MISMATCH;
}

#ifdef EDGEKV_BLOCKGTQ_TARGET_QUERY_ROTATION_FORENSICS
__attribute__((noinline))
static void capture_query_rotation_forensics(
    const struct edgekv_blockgtq_inputs * inputs,
    const size_t rotation_offsets[85], int head, const float * rotated_q,
    struct edgekv_blockgtq_target_observability * observability) {
    const int segment = 0;
    const int out = 2;
    const int start = descriptor(inputs->consumer, head, segment, 2);
    const int length = descriptor(inputs->consumer, head, segment, 3);
    const int rotation = descriptor(inputs->consumer, head, segment, 6);
    if (start != 0 || length != 22 || rotation != 34) {
        return;
    }
    const size_t rotation_base = rotation_offsets[rotation];
    volatile float separate_accumulator = 0.0f;
    float fma_accumulator = 0.0f;

    for (int in = 0; in < length; ++in) {
        const int source_dim =
            permutation(inputs->shared, head, start + in);
        const float query_value =
            inputs->query[(size_t) 11 * EDGEKV_BLOCKGTQ_HEAD_DIM +
                          (size_t) source_dim];
        const float rotation_value =
            read_f32(inputs->transform + TRANSFORM_K_ROTATIONS +
                     (rotation_base + (size_t) out * (size_t) length +
                      (size_t) in) *
                         4u);
        volatile float product = query_value * rotation_value;
        separate_accumulator = separate_accumulator + product;
        fma_accumulator =
            fmaf(query_value, rotation_value, fma_accumulator);

        observability->h5_source_dims[in] = (uint32_t) source_dim;
        observability->h5_query_bits[in] = float_bits(query_value);
        observability->h5_rotation_bits[in] =
            float_bits(rotation_value);
        observability->h5_product_bits[in] = float_bits(product);
        observability->h5_separate_accumulator_bits[in] =
            float_bits(separate_accumulator);
        observability->h5_fma_accumulator_bits[in] =
            float_bits(fma_accumulator);
    }

    uint64_t query_hash = UINT64_C(14695981039346656037);
    for (int dim = 0; dim < EDGEKV_BLOCKGTQ_HEAD_DIM; ++dim) {
        query_hash = fnv1a64_u32(query_hash, float_bits(rotated_q[dim]));
    }
    observability->h5_original_query_fnv1a64 = query_hash;
    observability->h5_original_out_bits = float_bits(rotated_q[start + out]);
    observability->h5_separate_final_bits =
        float_bits(separate_accumulator);
    observability->h5_fma_final_bits = float_bits(fma_accumulator);
}
#endif

static int edgekv_blockgtq_attn_decode_impl(
    const struct edgekv_blockgtq_inputs * inputs,
    float * output,
    struct edgekv_blockgtq_diagnostics * diagnostics
#ifdef EDGEKV_BLOCKGTQ_ATTRIBUTION
    , struct edgekv_blockgtq_attribution * attribution
#endif
#ifdef EDGEKV_BLOCKGTQ_TARGET_OBSERVABILITY
    , struct edgekv_blockgtq_target_observability * observability
#endif
    ) {
#ifdef EDGEKV_BLOCKGTQ_ATTRIBUTION
    uint64_t stage_start = attribution_start(attribution);
#endif
    if (inputs == NULL || output == NULL || inputs->query == NULL ||
        inputs->history == NULL || inputs->layer_index < 0 ||
        inputs->layer_index >= EDGEKV_BLOCKGTQ_LAYERS ||
        inputs->capacity != EDGEKV_BLOCKGTQ_CAPACITY ||
        inputs->sequence_length <= 0 ||
        inputs->sequence_length > inputs->capacity) {
        return EDGEKV_BLOCKGTQ_INVALID_ARGUMENT;
    }
    if (inputs->history_bytes != dynamic_bytes(inputs->capacity) ||
        inputs->history_bytes != EDGEKV_BLOCKGTQ_DYNAMIC_BYTES) {
        return EDGEKV_BLOCKGTQ_DYNAMIC_LAYOUT_MISMATCH;
    }
#ifdef EDGEKV_BLOCKGTQ_ATTRIBUTION
    attribution_stop(
        attribution, EDGEKV_BLOCKGTQ_STAGE_CORE_INPUT_VALIDATION,
        stage_start);
    stage_start = attribution_start(attribution);
#endif
    int status = edgekv_blockgtq_validate_static(
        inputs->transform, inputs->transform_bytes, inputs->consumer,
        inputs->consumer_bytes, inputs->shared, inputs->shared_bytes);
    if (status != EDGEKV_BLOCKGTQ_OK) {
        return status;
    }
#ifdef EDGEKV_BLOCKGTQ_ATTRIBUTION
    attribution_stop(
        attribution, EDGEKV_BLOCKGTQ_STAGE_STATIC_PACKAGE_VALIDATION,
        stage_start);
    stage_start = attribution_start(attribution);
#endif
    size_t rotation_offsets[85];
    size_t head_lut_bases[EDGEKV_BLOCKGTQ_HEADS];
    status = build_catalogs(inputs->consumer, rotation_offsets, head_lut_bases);
    if (status != EDGEKV_BLOCKGTQ_OK) {
        return status;
    }
#ifdef EDGEKV_BLOCKGTQ_ATTRIBUTION
    attribution_stop(
        attribution, EDGEKV_BLOCKGTQ_STAGE_CATALOG_CONSTRUCTION,
        stage_start);
#endif
    const float scale = 0.08838834764831845f;
    for (int query_head = 0; query_head < EDGEKV_BLOCKGTQ_QUERY_HEADS;
         ++query_head) {
        const int kv_head = query_head / EDGEKV_BLOCKGTQ_GQA;
        const int head = inputs->layer_index * EDGEKV_BLOCKGTQ_KV_HEADS +
                         kv_head;
        float rotated_q[EDGEKV_BLOCKGTQ_HEAD_DIM] = {0};
#ifdef EDGEKV_BLOCKGTQ_ATTRIBUTION
        stage_start = attribution_start(attribution);
#endif
        for (int segment = 0; segment < EDGEKV_BLOCKGTQ_MAX_SEGMENTS;
             ++segment) {
            const int start = descriptor(inputs->consumer, head, segment, 2);
            if (start < 0) {
                break;
            }
            const int length =
                descriptor(inputs->consumer, head, segment, 3);
            const int rotation =
                descriptor(inputs->consumer, head, segment, 6);
            const size_t rotation_base = rotation_offsets[rotation];
            for (int out = 0; out < length; ++out) {
                float value = 0.0f;
                for (int in = 0; in < length; ++in) {
                    const int source_dim =
                        permutation(inputs->shared, head, start + in);
                    value +=
                        inputs->query[(size_t) query_head *
                                          EDGEKV_BLOCKGTQ_HEAD_DIM +
                                      (size_t) source_dim] *
                        read_f32(inputs->transform +
                                 TRANSFORM_K_ROTATIONS +
                                 (rotation_base +
                                  (size_t) out * (size_t) length +
                                  (size_t) in) *
                                     4u);
                }
                rotated_q[start + out] = value;
            }
        }
#ifdef EDGEKV_BLOCKGTQ_TARGET_LOGIT_FORENSICS
        const bool capture_logit_forensics =
            observability != NULL && inputs->layer_index == 17 &&
            inputs->sequence_length == 128 && query_head == 11;
        if (capture_logit_forensics) {
            for (int dim = 0; dim < EDGEKV_BLOCKGTQ_HEAD_DIM; ++dim) {
                observability->rotated_query_bits[dim] =
                    float_bits(rotated_q[dim]);
            }
        }
#endif
#ifdef EDGEKV_BLOCKGTQ_TARGET_QUERY_ROTATION_FORENSICS
        if (observability != NULL && inputs->layer_index == 17 &&
            inputs->sequence_length == 128 && query_head == 11) {
            capture_query_rotation_forensics(
                inputs, rotation_offsets, head, rotated_q, observability);
        }
#endif
#ifdef EDGEKV_BLOCKGTQ_ATTRIBUTION
        attribution_stop(
            attribution, EDGEKV_BLOCKGTQ_STAGE_QUERY_ROTATION,
            stage_start);
        stage_start = attribution_start(attribution);
#endif

        float maximum = -FLT_MAX;
#ifdef EDGEKV_BLOCKGTQ_TARGET_OBSERVABILITY
        uint64_t logit_bits_hash = UINT64_C(14695981039346656037);
#endif
        for (int token = 0; token < inputs->sequence_length; ++token) {
            const uint8_t * codes =
                inputs->history + k_codes_offset(token, head);
            const uint8_t * norms =
                inputs->history +
                k_norms_offset(inputs->capacity, token, head);
            float dot = 0.0f;
            for (int segment = 0;
                 segment < EDGEKV_BLOCKGTQ_MAX_SEGMENTS; ++segment) {
                const int start =
                    descriptor(inputs->consumer, head, segment, 2);
                if (start < 0) {
                    break;
                }
                const int bits =
                    descriptor(inputs->consumer, head, segment, 1);
                const int length =
                    descriptor(inputs->consumer, head, segment, 3);
                const int pack_offset =
                    descriptor(inputs->consumer, head, segment, 4);
                const float norm =
                    half_to_float(read_u16(norms + (size_t) segment * 2u));
                for (int local = 0; local < length; ++local) {
                    const uint8_t code =
                        packed_code(codes, pack_offset, local, bits);
                    const int local_lut =
                        read_i32(inputs->consumer + CONSUMER_K_LUT_OFFSET +
                                 ((size_t) head *
                                      EDGEKV_BLOCKGTQ_HEAD_DIM +
                                  (size_t) start + (size_t) local) *
                                     4u);
                    const size_t lut_index =
                        head_lut_bases[head] + (size_t) local_lut + code;
                    const float lut =
                        half_to_float(read_u16(inputs->consumer +
                                               CONSUMER_K_LUT +
                                               lut_index * 2u));
                    dot += rotated_q[start + local] * lut * norm;
                }
#ifdef EDGEKV_BLOCKGTQ_TARGET_LOGIT_FORENSICS
                if (capture_logit_forensics) {
                    observability->cumulative_dot_bits[token][segment] =
                        float_bits(dot);
                    observability->segment_count = (uint32_t) segment + 1u;
                }
#endif
            }
            const float logit = dot * scale;
#ifdef EDGEKV_BLOCKGTQ_TARGET_LOGIT_FORENSICS
            if (capture_logit_forensics) {
                observability->pre_scale_dot_bits[token] = float_bits(dot);
                observability->logit_bits[token] = float_bits(logit);
                for (int segment = 0;
                     segment < (int) observability->segment_count;
                     ++segment) {
                    const int start =
                        descriptor(inputs->consumer, head, segment, 2);
                    const int bits =
                        descriptor(inputs->consumer, head, segment, 1);
                    const int length =
                        descriptor(inputs->consumer, head, segment, 3);
                    const int pack_offset =
                        descriptor(inputs->consumer, head, segment, 4);
                    const float norm = half_to_float(
                        read_u16(norms + (size_t) segment * 2u));
                    uint64_t code_hash =
                        UINT64_C(14695981039346656037);
                    uint64_t lut_hash =
                        UINT64_C(14695981039346656037);
                    for (int local = 0; local < length; ++local) {
                        const uint8_t code =
                            packed_code(codes, pack_offset, local, bits);
                        const int local_lut = read_i32(
                            inputs->consumer + CONSUMER_K_LUT_OFFSET +
                            ((size_t) head *
                                 EDGEKV_BLOCKGTQ_HEAD_DIM +
                             (size_t) start + (size_t) local) *
                                4u);
                        const size_t lut_index =
                            head_lut_bases[head] +
                            (size_t) local_lut + code;
                        const float lut = half_to_float(read_u16(
                            inputs->consumer + CONSUMER_K_LUT +
                            lut_index * 2u));
                        code_hash = fnv1a64_u8(code_hash, code);
                        lut_hash =
                            fnv1a64_u32(lut_hash, float_bits(lut));
                    }
                    observability->code_fnv1a64[token][segment] =
                        code_hash;
                    observability->lut_bits_fnv1a64[token][segment] =
                        lut_hash;
                    observability->norm_bits[token][segment] =
                        float_bits(norm);
                }
            }
#endif
#ifdef EDGEKV_BLOCKGTQ_TARGET_OBSERVABILITY
            logit_bits_hash =
                fnv1a64_u32(logit_bits_hash, float_bits(logit));
#endif
            if (diagnostics != NULL && diagnostics->logits != NULL) {
                diagnostics
                    ->logits[(size_t) query_head *
                                 (size_t) inputs->sequence_length +
                             (size_t) token] = logit;
            }
            if (logit > maximum) {
                maximum = logit;
            }
        }
#ifdef EDGEKV_BLOCKGTQ_TARGET_OBSERVABILITY
        if (observability != NULL) {
            observability->maximum_logit_bits[query_head] =
                float_bits(maximum);
            observability->logit_bits_fnv1a64[query_head] =
                logit_bits_hash;
        }
        uint64_t weight_bits_hash = UINT64_C(14695981039346656037);
#endif
#ifdef EDGEKV_BLOCKGTQ_ATTRIBUTION
        attribution_stop(
            attribution, EDGEKV_BLOCKGTQ_STAGE_FIRST_K_DECODE_DOT,
            stage_start);
        stage_start = attribution_start(attribution);
#endif

        float denominator = 0.0f;
        float rotated_output[EDGEKV_BLOCKGTQ_HEAD_DIM] = {0};
        for (int token = 0; token < inputs->sequence_length; ++token) {
            float logit;
            if (diagnostics != NULL && diagnostics->logits != NULL) {
                logit =
                    diagnostics
                        ->logits[(size_t) query_head *
                                     (size_t) inputs->sequence_length +
                                 (size_t) token];
            } else {
                const uint8_t * codes =
                    inputs->history + k_codes_offset(token, head);
                const uint8_t * norms =
                    inputs->history +
                    k_norms_offset(inputs->capacity, token, head);
                float dot = 0.0f;
                for (int segment = 0;
                     segment < EDGEKV_BLOCKGTQ_MAX_SEGMENTS; ++segment) {
                    const int start =
                        descriptor(inputs->consumer, head, segment, 2);
                    if (start < 0) {
                        break;
                    }
                    const int bits =
                        descriptor(inputs->consumer, head, segment, 1);
                    const int length =
                        descriptor(inputs->consumer, head, segment, 3);
                    const int pack_offset =
                        descriptor(inputs->consumer, head, segment, 4);
                    const float norm = half_to_float(
                        read_u16(norms + (size_t) segment * 2u));
                    for (int local = 0; local < length; ++local) {
                        const uint8_t code = packed_code(
                            codes, pack_offset, local, bits);
                        const int local_lut = read_i32(
                            inputs->consumer + CONSUMER_K_LUT_OFFSET +
                            ((size_t) head *
                                 EDGEKV_BLOCKGTQ_HEAD_DIM +
                             (size_t) start + (size_t) local) *
                                4u);
                        const size_t lut_index =
                            head_lut_bases[head] +
                            (size_t) local_lut + code;
                        const float lut = half_to_float(read_u16(
                            inputs->consumer + CONSUMER_K_LUT +
                            lut_index * 2u));
                        dot += rotated_q[start + local] * lut * norm;
                    }
                }
                logit = dot * scale;
            }
            const float weight = expf(logit - maximum);
#ifdef EDGEKV_BLOCKGTQ_TARGET_OBSERVABILITY
            weight_bits_hash =
                fnv1a64_u32(weight_bits_hash, float_bits(weight));
#endif
            if (diagnostics != NULL && diagnostics->weights != NULL) {
                diagnostics
                    ->weights[(size_t) query_head *
                                  (size_t) inputs->sequence_length +
                              (size_t) token] = weight;
            }
            denominator += weight;
            const uint8_t * codes =
                inputs->history +
                v_codes_offset(inputs->capacity, token, head);
            const float norm = half_to_float(read_u16(
                inputs->history +
                v_norms_offset(inputs->capacity, token, head)));
            for (int dim = 0; dim < EDGEKV_BLOCKGTQ_HEAD_DIM; ++dim) {
                const uint8_t packed = codes[dim / 2];
                const uint8_t code =
                    (dim & 1) == 0 ? packed & 15u : packed >> 4;
                const float lut = half_to_float(read_u16(
                    inputs->consumer + CONSUMER_V_LUT +
                    ((size_t) head * 16u + code) * 2u));
                rotated_output[dim] += weight * lut * norm;
            }
        }
#ifdef EDGEKV_BLOCKGTQ_ATTRIBUTION
        attribution_stop(
            attribution,
            diagnostics != NULL && diagnostics->logits != NULL
                ? EDGEKV_BLOCKGTQ_STAGE_DIAGNOSTICS_ON_LOGIT_REUSE_SOFTMAX_V_FUSED
                : EDGEKV_BLOCKGTQ_STAGE_DIAGNOSTICS_OFF_SECOND_K_SOFTMAX_V_FUSED,
            stage_start);
        stage_start = attribution_start(attribution);
#endif
        if (!(denominator > 0.0f) || !isfinite(denominator)) {
            return EDGEKV_BLOCKGTQ_DYNAMIC_LAYOUT_MISMATCH;
        }
        if (diagnostics != NULL && diagnostics->denominators != NULL) {
            diagnostics->denominators[query_head] = denominator;
        }
#ifdef EDGEKV_BLOCKGTQ_TARGET_OBSERVABILITY
        if (observability != NULL) {
            observability->denominator_bits[query_head] =
                float_bits(denominator);
            observability->weight_bits_fnv1a64[query_head] =
                weight_bits_hash;
        }
#endif
        for (int dim = 0; dim < EDGEKV_BLOCKGTQ_HEAD_DIM; ++dim) {
            rotated_output[dim] /= denominator;
            if (diagnostics != NULL && diagnostics->rotated_v != NULL) {
                diagnostics
                    ->rotated_v[(size_t) query_head *
                                    EDGEKV_BLOCKGTQ_HEAD_DIM +
                                (size_t) dim] = rotated_output[dim];
            }
        }
#ifdef EDGEKV_BLOCKGTQ_ATTRIBUTION
        attribution_stop(
            attribution, EDGEKV_BLOCKGTQ_STAGE_NORMALIZATION,
            stage_start);
        stage_start = attribution_start(attribution);
#endif
        const int rotation = (int) read_u16(
            inputs->consumer + CONSUMER_V_ROTATION_INDEX +
            (size_t) head * 2u);
        const size_t rotation_base =
            TRANSFORM_V_ROTATIONS +
            (size_t) rotation * EDGEKV_BLOCKGTQ_HEAD_DIM *
                EDGEKV_BLOCKGTQ_HEAD_DIM * 4u;
        for (int original = 0; original < EDGEKV_BLOCKGTQ_HEAD_DIM;
             ++original) {
            float value = 0.0f;
            for (int rotated = 0; rotated < EDGEKV_BLOCKGTQ_HEAD_DIM;
                 ++rotated) {
                value +=
                    rotated_output[rotated] *
                    read_f32(inputs->transform + rotation_base +
                             ((size_t) rotated *
                                  EDGEKV_BLOCKGTQ_HEAD_DIM +
                              (size_t) original) *
                                 4u);
            }
            output[(size_t) query_head * EDGEKV_BLOCKGTQ_HEAD_DIM +
                   (size_t) original] = value;
        }
#ifdef EDGEKV_BLOCKGTQ_ATTRIBUTION
        attribution_stop(
            attribution, EDGEKV_BLOCKGTQ_STAGE_INVERSE_V_ROTATION,
            stage_start);
#endif
    }
    return EDGEKV_BLOCKGTQ_OK;
}

int edgekv_blockgtq_attn_decode(
    const struct edgekv_blockgtq_inputs * inputs,
    float * output,
    struct edgekv_blockgtq_diagnostics * diagnostics
#ifdef EDGEKV_BLOCKGTQ_TARGET_OBSERVABILITY
    , struct edgekv_blockgtq_target_observability * observability
#endif
    ) {
    return edgekv_blockgtq_attn_decode_impl(
        inputs, output, diagnostics
#ifdef EDGEKV_BLOCKGTQ_ATTRIBUTION
        , NULL
#endif
#ifdef EDGEKV_BLOCKGTQ_TARGET_OBSERVABILITY
        , observability
#endif
    );
}

#ifdef EDGEKV_BLOCKGTQ_ATTRIBUTION
int edgekv_blockgtq_attn_decode_profiled(
    const struct edgekv_blockgtq_inputs * inputs,
    float * output,
    struct edgekv_blockgtq_diagnostics * diagnostics,
    struct edgekv_blockgtq_attribution * attribution
#ifdef EDGEKV_BLOCKGTQ_TARGET_OBSERVABILITY
    , struct edgekv_blockgtq_target_observability * observability
#endif
    ) {
    if (attribution == NULL || attribution->clock == NULL) {
        return EDGEKV_BLOCKGTQ_INVALID_ARGUMENT;
    }
    memset(attribution->pcycles, 0, sizeof(attribution->pcycles));
    memset(attribution->intervals, 0, sizeof(attribution->intervals));
    return edgekv_blockgtq_attn_decode_impl(
        inputs, output, diagnostics, attribution
#ifdef EDGEKV_BLOCKGTQ_TARGET_OBSERVABILITY
        , observability
#endif
    );
}
#endif
