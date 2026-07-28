// SPDX-License-Identifier: MIT

#include "edgekv-blockgtq-runtime.h"

#include <math.h>
#include <stdbool.h>
#include <string.h>

enum {
    PRODUCER_K_ROTATIONS = 0,
    PRODUCER_K_CODEBOOKS = 817728,
    PRODUCER_V_ROTATIONS = 827008,
    PRODUCER_V_CODEBOOK = 958080,

    CONSUMER_K_DESCRIPTORS = 133760,
    CONSUMER_V_ROTATION_INDEX = 142976,
    SHARED_K_PERMUTATION = 0,

    K_ROTATION_COUNT = 85,
    K_CODEBOOK_COUNT = 55,
    K_ROTATION_FLOATS = 204432,
    K_CODEBOOK_FLOATS = 2320,

    APPEND_K_CODES = 0,
    APPEND_K_NORMS = 152,
    APPEND_V_CODES = 184,
    APPEND_V_NORMS = 312,
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

static float read_f32(const uint8_t * p) {
    const uint32_t bits = read_u32(p);
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static void write_u16(uint8_t * p, uint16_t value) {
    p[0] = (uint8_t) value;
    p[1] = (uint8_t) (value >> 8);
}

static uint16_t float_to_half(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const uint32_t exponent = (bits >> 23) & 0xffu;
    const uint32_t mantissa = bits & 0x7fffffu;
    if (exponent == 0xffu) {
        return (uint16_t) (sign | (mantissa == 0 ? 0x7c00u : 0x7e00u));
    }
    int half_exp = (int) exponent - 127 + 15;
    if (half_exp >= 31) {
        return (uint16_t) (sign | 0x7c00u);
    }
    if (half_exp <= 0) {
        if (half_exp < -10) {
            return (uint16_t) sign;
        }
        uint32_t rounded = mantissa | 0x800000u;
        const unsigned shift = (unsigned) (14 - half_exp);
        const uint32_t halfway = 1u << (shift - 1);
        const uint32_t low = rounded & ((1u << shift) - 1u);
        rounded >>= shift;
        if (low > halfway || (low == halfway && (rounded & 1u))) {
            ++rounded;
        }
        return (uint16_t) (sign | rounded);
    }
    uint32_t rounded = mantissa;
    const uint32_t low = rounded & 0x1fffu;
    rounded >>= 13;
    if (low > 0x1000u || (low == 0x1000u && (rounded & 1u))) {
        ++rounded;
        if (rounded == 0x400u) {
            rounded = 0;
            ++half_exp;
            if (half_exp >= 31) {
                return (uint16_t) (sign | 0x7c00u);
            }
        }
    }
    return (uint16_t) (
        sign | ((uint32_t) half_exp << 10) | rounded);
}

static int descriptor(const uint8_t * consumer, int head, int segment,
                      int field) {
    return (int) read_i16(
        consumer + CONSUMER_K_DESCRIPTORS +
        ((size_t) head * 8u * 8u + (size_t) segment * 8u +
         (size_t) field) *
            2u);
}

static int permutation(const uint8_t * shared, int head, int dim) {
    return (int) shared[
        SHARED_K_PERMUTATION +
        (size_t) head * EDGEKV_BLOCKGTQ_HEAD_DIM + (size_t) dim];
}

static int build_catalogs(
    const uint8_t * consumer,
    size_t rotation_offsets[K_ROTATION_COUNT],
    size_t codebook_offsets[K_CODEBOOK_COUNT]) {
    int rotation_dims[K_ROTATION_COUNT] = {0};
    int codebook_bits[K_CODEBOOK_COUNT] = {0};
    for (int head = 0; head < EDGEKV_BLOCKGTQ_HEADS; ++head) {
        for (int segment = 0; segment < EDGEKV_BLOCKGTQ_MAX_SEGMENTS;
             ++segment) {
            const int start = descriptor(consumer, head, segment, 2);
            if (start < 0) {
                break;
            }
            const int bits = descriptor(consumer, head, segment, 1);
            const int length = descriptor(consumer, head, segment, 3);
            const int rotation = descriptor(consumer, head, segment, 6);
            const int codebook = descriptor(consumer, head, segment, 7);
            if ((bits != 4 && bits != 8) || length <= 0 ||
                rotation < 0 || rotation >= K_ROTATION_COUNT ||
                codebook < 0 || codebook >= K_CODEBOOK_COUNT) {
                return EDGEKV_BLOCKGTQ_STATIC_LAYOUT_MISMATCH;
            }
            if (rotation_dims[rotation] != 0 &&
                rotation_dims[rotation] != length) {
                return EDGEKV_BLOCKGTQ_STATIC_LAYOUT_MISMATCH;
            }
            if (codebook_bits[codebook] != 0 &&
                codebook_bits[codebook] != bits) {
                return EDGEKV_BLOCKGTQ_STATIC_LAYOUT_MISMATCH;
            }
            rotation_dims[rotation] = length;
            codebook_bits[codebook] = bits;
        }
    }
    size_t rotation_cursor = 0;
    for (int i = 0; i < K_ROTATION_COUNT; ++i) {
        if (rotation_dims[i] <= 0) {
            return EDGEKV_BLOCKGTQ_STATIC_LAYOUT_MISMATCH;
        }
        rotation_offsets[i] = rotation_cursor;
        rotation_cursor +=
            (size_t) rotation_dims[i] * (size_t) rotation_dims[i];
    }
    size_t codebook_cursor = 0;
    for (int i = 0; i < K_CODEBOOK_COUNT; ++i) {
        if (codebook_bits[i] != 4 && codebook_bits[i] != 8) {
            return EDGEKV_BLOCKGTQ_STATIC_LAYOUT_MISMATCH;
        }
        codebook_offsets[i] = codebook_cursor;
        codebook_cursor += (size_t) 1u << codebook_bits[i];
    }
    return rotation_cursor == K_ROTATION_FLOATS &&
                   codebook_cursor == K_CODEBOOK_FLOATS
               ? EDGEKV_BLOCKGTQ_OK
               : EDGEKV_BLOCKGTQ_STATIC_LAYOUT_MISMATCH;
}

static int quantize_code(
    const uint8_t * producer, size_t codebook_base, int levels,
    float value) {
    int code = 0;
    while (code + 1 < levels) {
        const float lhs = read_f32(
            producer + PRODUCER_K_CODEBOOKS +
            (codebook_base + (size_t) code) * 4u);
        const float rhs = read_f32(
            producer + PRODUCER_K_CODEBOOKS +
            (codebook_base + (size_t) code + 1u) * 4u);
        if (value <= (lhs + rhs) * 0.5f) {
            break;
        }
        ++code;
    }
    return code;
}

int edgekv_blockgtq_pack_token_v2(
    const float * k,
    const float * v,
    const uint8_t * producer,
    size_t producer_bytes,
    const uint8_t * consumer,
    size_t consumer_bytes,
    const uint8_t * shared,
    size_t shared_bytes,
    int layer_index,
    uint8_t output[EDGEKV_BLOCKGTQ_APPEND_BYTES]) {
    if (k == NULL || v == NULL || output == NULL || layer_index < 0 ||
        layer_index >= EDGEKV_BLOCKGTQ_LAYERS ||
        producer_bytes != EDGEKV_BLOCKGTQ_PRODUCER_BYTES ||
        consumer_bytes != EDGEKV_BLOCKGTQ_CONSUMER_BYTES ||
        shared_bytes != EDGEKV_BLOCKGTQ_SHARED_BYTES) {
        return EDGEKV_BLOCKGTQ_INVALID_ARGUMENT;
    }
    int status = edgekv_blockgtq_validate_static_v2(
        producer, producer_bytes, consumer, consumer_bytes, shared,
        shared_bytes);
    if (status != EDGEKV_BLOCKGTQ_OK) {
        return status;
    }
    size_t rotation_offsets[K_ROTATION_COUNT];
    size_t codebook_offsets[K_CODEBOOK_COUNT];
    status = build_catalogs(
        consumer, rotation_offsets, codebook_offsets);
    if (status != EDGEKV_BLOCKGTQ_OK) {
        return status;
    }

    memset(output, 0, EDGEKV_BLOCKGTQ_APPEND_BYTES);
    for (int kv_head = 0; kv_head < EDGEKV_BLOCKGTQ_KV_HEADS; ++kv_head) {
        const int head =
            layer_index * EDGEKV_BLOCKGTQ_KV_HEADS + kv_head;
        uint8_t packed_k[EDGEKV_BLOCKGTQ_K_CODE_STRIDE] = {0};
        uint16_t packed_k_norms[EDGEKV_BLOCKGTQ_MAX_SEGMENTS] = {0};
        const float * input_k =
            k + (size_t) kv_head * EDGEKV_BLOCKGTQ_HEAD_DIM;

        for (int segment = 0; segment < EDGEKV_BLOCKGTQ_MAX_SEGMENTS;
             ++segment) {
            const int start = descriptor(consumer, head, segment, 2);
            if (start < 0) {
                break;
            }
            const int bits = descriptor(consumer, head, segment, 1);
            const int length = descriptor(consumer, head, segment, 3);
            const int pack_offset =
                descriptor(consumer, head, segment, 4);
            const int rotation =
                descriptor(consumer, head, segment, 6);
            const int codebook =
                descriptor(consumer, head, segment, 7);
            float normalized[EDGEKV_BLOCKGTQ_HEAD_DIM] = {0};
            float norm_squared = 0.0f;
            for (int i = 0; i < length; ++i) {
                const float value =
                    input_k[permutation(shared, head, start + i)];
                normalized[i] = value;
                norm_squared += value * value;
            }
            const float norm = sqrtf(norm_squared);
            const float safe_norm = fmaxf(norm, 1.0e-10f);
            for (int i = 0; i < length; ++i) {
                normalized[i] /= safe_norm;
            }
            const size_t rotation_base = rotation_offsets[rotation];
            const size_t codebook_base = codebook_offsets[codebook];
            const int levels = 1 << bits;
            float correction_squared = 0.0f;
            for (int out = 0; out < length; ++out) {
                float rotated = 0.0f;
                for (int in = 0; in < length; ++in) {
                    const float coefficient = read_f32(
                        producer + PRODUCER_K_ROTATIONS +
                        (rotation_base + (size_t) out * (size_t) length +
                         (size_t) in) *
                            4u);
                    rotated += normalized[in] * coefficient;
                }
                const int code = quantize_code(
                    producer, codebook_base, levels, rotated);
                const float quantized = read_f32(
                    producer + PRODUCER_K_CODEBOOKS +
                    (codebook_base + (size_t) code) * 4u);
                correction_squared += quantized * quantized;
                if (bits == 8) {
                    packed_k[pack_offset + out] = (uint8_t) code;
                } else if ((out & 1) == 0) {
                    packed_k[pack_offset + out / 2] = (uint8_t) code;
                } else {
                    packed_k[pack_offset + out / 2] |=
                        (uint8_t) (code << 4);
                }
            }
            const float correction =
                fmaxf(sqrtf(correction_squared), 1.0e-10f);
            packed_k_norms[segment] =
                float_to_half(norm / correction);
        }

        uint8_t packed_v[EDGEKV_BLOCKGTQ_V_CODE_STRIDE] = {0};
        const float * input_v =
            v + (size_t) kv_head * EDGEKV_BLOCKGTQ_HEAD_DIM;
        float v_norm_squared = 0.0f;
        for (int i = 0; i < EDGEKV_BLOCKGTQ_HEAD_DIM; ++i) {
            v_norm_squared += input_v[i] * input_v[i];
        }
        const float v_norm = sqrtf(v_norm_squared);
        const float v_safe_norm = fmaxf(v_norm, 1.0e-10f);
        const int v_rotation = (int) read_u16(
            consumer + CONSUMER_V_ROTATION_INDEX + (size_t) head * 2u);
        const size_t v_rotation_base =
            (size_t) v_rotation * EDGEKV_BLOCKGTQ_HEAD_DIM *
            EDGEKV_BLOCKGTQ_HEAD_DIM;
        float v_correction_squared = 0.0f;
        for (int out = 0; out < EDGEKV_BLOCKGTQ_HEAD_DIM; ++out) {
            float rotated = 0.0f;
            for (int in = 0; in < EDGEKV_BLOCKGTQ_HEAD_DIM; ++in) {
                const float coefficient = read_f32(
                    producer + PRODUCER_V_ROTATIONS +
                    (v_rotation_base +
                     (size_t) out * EDGEKV_BLOCKGTQ_HEAD_DIM +
                     (size_t) in) *
                        4u);
                rotated += (input_v[in] / v_safe_norm) * coefficient;
            }
            int code = 0;
            while (code < 15) {
                const float lhs = read_f32(
                    producer + PRODUCER_V_CODEBOOK +
                    (size_t) code * 4u);
                const float rhs = read_f32(
                    producer + PRODUCER_V_CODEBOOK +
                    ((size_t) code + 1u) * 4u);
                if (rotated <= (lhs + rhs) * 0.5f) {
                    break;
                }
                ++code;
            }
            const float quantized = read_f32(
                producer + PRODUCER_V_CODEBOOK + (size_t) code * 4u);
            v_correction_squared += quantized * quantized;
            if ((out & 1) == 0) {
                packed_v[out / 2] = (uint8_t) code;
            } else {
                packed_v[out / 2] |= (uint8_t) (code << 4);
            }
        }
        const float v_correction =
            fmaxf(sqrtf(v_correction_squared), 1.0e-10f);
        const uint16_t packed_v_norm =
            float_to_half(v_norm / v_correction);

        memcpy(
            output + APPEND_K_CODES +
                (size_t) kv_head * EDGEKV_BLOCKGTQ_K_CODE_STRIDE,
            packed_k, sizeof(packed_k));
        for (int segment = 0; segment < EDGEKV_BLOCKGTQ_MAX_SEGMENTS;
             ++segment) {
            write_u16(
                output + APPEND_K_NORMS +
                    (size_t) kv_head * EDGEKV_BLOCKGTQ_K_NORM_STRIDE +
                    (size_t) segment * 2u,
                packed_k_norms[segment]);
        }
        memcpy(
            output + APPEND_V_CODES +
                (size_t) kv_head * EDGEKV_BLOCKGTQ_V_CODE_STRIDE,
            packed_v, sizeof(packed_v));
        write_u16(
            output + APPEND_V_NORMS +
                (size_t) kv_head * EDGEKV_BLOCKGTQ_V_NORM_STRIDE,
            packed_v_norm);
    }
    return EDGEKV_BLOCKGTQ_OK;
}

int edgekv_blockgtq_commit_token_v2(
    uint8_t * history,
    size_t history_bytes,
    int capacity,
    int layer_index,
    int token_index,
    const uint8_t current[EDGEKV_BLOCKGTQ_APPEND_BYTES],
    size_t current_bytes) {
    if (history == NULL || current == NULL ||
        history_bytes != EDGEKV_BLOCKGTQ_DYNAMIC_BYTES ||
        current_bytes != EDGEKV_BLOCKGTQ_APPEND_BYTES ||
        layer_index < 0 || layer_index >= EDGEKV_BLOCKGTQ_LAYERS) {
        return EDGEKV_BLOCKGTQ_INVALID_ARGUMENT;
    }
    for (int kv_head = 0; kv_head < EDGEKV_BLOCKGTQ_KV_HEADS; ++kv_head) {
        const int head =
            layer_index * EDGEKV_BLOCKGTQ_KV_HEADS + kv_head;
        struct edgekv_blockgtq_history_addresses addresses;
        const int status = edgekv_blockgtq_history_addresses(
            capacity, token_index, head, &addresses);
        if (status != EDGEKV_BLOCKGTQ_OK) {
            return status;
        }
        memcpy(
            history + addresses.k_codes,
            current + APPEND_K_CODES +
                (size_t) kv_head * EDGEKV_BLOCKGTQ_K_CODE_STRIDE,
            EDGEKV_BLOCKGTQ_K_CODE_STRIDE);
        memcpy(
            history + addresses.k_norms,
            current + APPEND_K_NORMS +
                (size_t) kv_head * EDGEKV_BLOCKGTQ_K_NORM_STRIDE,
            EDGEKV_BLOCKGTQ_K_NORM_STRIDE);
        memcpy(
            history + addresses.v_codes,
            current + APPEND_V_CODES +
                (size_t) kv_head * EDGEKV_BLOCKGTQ_V_CODE_STRIDE,
            EDGEKV_BLOCKGTQ_V_CODE_STRIDE);
        memcpy(
            history + addresses.v_norms,
            current + APPEND_V_NORMS +
                (size_t) kv_head * EDGEKV_BLOCKGTQ_V_NORM_STRIDE,
            EDGEKV_BLOCKGTQ_V_NORM_STRIDE);
    }
    return EDGEKV_BLOCKGTQ_OK;
}
