#include "llama-kv-blockgtq.h"

#include "ggml-alloc.h"
#include "ggml-cpp.h"
#include "llama-batch.h"
#include "llama-impl.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace {

constexpr size_t PACKAGE_BYTES  = 1128768;
constexpr size_t HEADER_BYTES   = 256;
constexpr size_t RECORD_BYTES   = 256;
constexpr size_t SECTION_COUNT  = 14;
constexpr size_t TABLE_OFFSET   = 256;
constexpr size_t PAYLOAD_OFFSET = 3840;
constexpr size_t PAYLOAD_BYTES  = 1124928;

constexpr char PACKAGE_SHA[] =
    "7159380b3b2bbc580c3e55173e1e7dacaa56c8b2c6a7b8b72f6c28f6ccb5bc5b";
constexpr char PARENT_SHA[] =
    "bb40c6179bc645096b1cf57057e91da5c012a14c6aaeb262afbe0adcfcbb2dc1";
constexpr char CONTRACT_SHA[] =
    "92f8aa7c6b1e64089f914cf353124dcb6f5275357b031b71c58607014f00c3c5";

struct expected_section {
    const char * name;
    uint32_t dtype;
    std::array<uint64_t, 4> shape;
    uint32_t rank;
    size_t value_bytes;
    size_t aligned_bytes;
    uint32_t ownership;
    uint32_t visibility;
};

constexpr std::array<expected_section, SECTION_COUNT> SECTIONS = {{
    {"k_rotation_values", 1, {204432, 0, 0, 0}, 1, 817728, 817728, 1, 1},
    {"k_producer_codebook_values", 1, {2320, 0, 0, 0}, 1, 9280, 9280, 1, 1},
    {"v_rotation_values", 1, {32768, 0, 0, 0}, 1, 131072, 131072, 1, 1},
    {"v_producer_codebook_values", 1, {16, 0, 0, 0}, 1, 64, 64, 1, 1},
    {"k_consumer_lut_values", 2, {28864, 0, 0, 0}, 1, 57728, 57728, 2, 2},
    {"k_lut_offset", 3, {72, 128, 0, 0}, 2, 36864, 36864, 2, 2},
    {"k_norm_group", 3, {72, 128, 0, 0}, 2, 36864, 36864, 2, 2},
    {"v_consumer_lut", 2, {72, 16, 0, 0}, 2, 2304, 2304, 2, 2},
    {"k_fixed_segment_descriptors", 4, {72, 8, 8, 0}, 3, 9216, 9216, 2, 2},
    {"v_rotation_index", 5, {72, 0, 0, 0}, 1, 144, 192, 2, 2},
    {"k_permutation", 6, {72, 128, 0, 0}, 2, 9216, 9216, 3, 3},
    {"k_inverse_permutation", 6, {72, 128, 0, 0}, 2, 9216, 9216, 3, 3},
    {"k_source_allocation", 6, {72, 64, 0, 0}, 2, 4608, 4608, 3, 3},
    {"sequence_layout_descriptor", 4, {36, 8, 0, 0}, 2, 576, 576, 3, 3},
}};

void require(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error("Block-GTQ package: " + message);
    }
}

uint32_t read_u32(const uint8_t * p) {
    return static_cast<uint32_t>(p[0]) |
           (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
}

uint64_t read_u64(const uint8_t * p) {
    return static_cast<uint64_t>(read_u32(p)) |
           (static_cast<uint64_t>(read_u32(p + 4)) << 32);
}

bool all_zero(const uint8_t * begin, const uint8_t * end) {
    return std::all_of(begin, end, [](uint8_t value) { return value == 0; });
}

std::string hex(const uint8_t * bytes, size_t size) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string result(size * 2, '\0');
    for (size_t i = 0; i < size; ++i) {
        result[2 * i] = digits[bytes[i] >> 4];
        result[2 * i + 1] = digits[bytes[i] & 15];
    }
    return result;
}

class sha256 {
public:
    void update(const uint8_t * data, size_t size) {
        total_ += size;
        while (size > 0) {
            const size_t take = std::min(size, buffer_.size() - buffered_);
            std::memcpy(buffer_.data() + buffered_, data, take);
            buffered_ += take;
            data += take;
            size -= take;
            if (buffered_ == buffer_.size()) {
                block(buffer_.data());
                buffered_ = 0;
            }
        }
    }

    std::array<uint8_t, 32> finish() {
        const uint64_t bit_length = total_ * 8;
        buffer_[buffered_++] = 0x80;
        if (buffered_ > 56) {
            std::fill(buffer_.begin() + static_cast<ptrdiff_t>(buffered_), buffer_.end(), 0);
            block(buffer_.data());
            buffered_ = 0;
        }
        std::fill(
            buffer_.begin() + static_cast<ptrdiff_t>(buffered_),
            buffer_.begin() + 56,
            0);
        for (int i = 0; i < 8; ++i) {
            buffer_[63 - i] = static_cast<uint8_t>(bit_length >> (8 * i));
        }
        block(buffer_.data());
        std::array<uint8_t, 32> result{};
        for (size_t i = 0; i < state_.size(); ++i) {
            result[4 * i] = static_cast<uint8_t>(state_[i] >> 24);
            result[4 * i + 1] = static_cast<uint8_t>(state_[i] >> 16);
            result[4 * i + 2] = static_cast<uint8_t>(state_[i] >> 8);
            result[4 * i + 3] = static_cast<uint8_t>(state_[i]);
        }
        return result;
    }

private:
    static uint32_t ror(uint32_t value, unsigned amount) {
        return (value >> amount) | (value << (32 - amount));
    }

    void block(const uint8_t * input) {
        static constexpr std::array<uint32_t, 64> k = {
            0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
            0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
            0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
            0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
            0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
            0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
            0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
            0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
            0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
            0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
            0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
            0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
            0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
            0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
            0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
            0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
        };
        std::array<uint32_t, 64> w{};
        for (size_t i = 0; i < 16; ++i) {
            w[i] = (static_cast<uint32_t>(input[4 * i]) << 24) |
                   (static_cast<uint32_t>(input[4 * i + 1]) << 16) |
                   (static_cast<uint32_t>(input[4 * i + 2]) << 8) |
                   static_cast<uint32_t>(input[4 * i + 3]);
        }
        for (size_t i = 16; i < 64; ++i) {
            const uint32_t s0 =
                ror(w[i - 15], 7) ^ ror(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 =
                ror(w[i - 2], 17) ^ ror(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3];
        uint32_t e = state_[4], f = state_[5], g = state_[6], h = state_[7];
        for (size_t i = 0; i < 64; ++i) {
            const uint32_t s1 = ror(e, 6) ^ ror(e, 11) ^ ror(e, 25);
            const uint32_t ch = (e & f) ^ ((~e) & g);
            const uint32_t t1 = h + s1 + ch + k[i] + w[i];
            const uint32_t s0 = ror(a, 2) ^ ror(a, 13) ^ ror(a, 22);
            const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t t2 = s0 + maj;
            h = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }
        state_[0] += a;
        state_[1] += b;
        state_[2] += c;
        state_[3] += d;
        state_[4] += e;
        state_[5] += f;
        state_[6] += g;
        state_[7] += h;
    }

    std::array<uint32_t, 8> state_ = {
        0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
        0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U,
    };
    std::array<uint8_t, 64> buffer_{};
    size_t buffered_ = 0;
    uint64_t total_ = 0;
};

std::string hash_bytes(const uint8_t * data, size_t size) {
    sha256 hash;
    hash.update(data, size);
    const auto digest = hash.finish();
    return hex(digest.data(), digest.size());
}

std::vector<uint8_t> read_file(const std::string & path) {
    std::ifstream stream(path, std::ios::binary);
    require(stream.good(), "cannot open " + path);
    stream.seekg(0, std::ios::end);
    const std::streamoff length = stream.tellg();
    require(length >= 0, "cannot determine file size");
    std::vector<uint8_t> bytes(static_cast<size_t>(length));
    stream.seekg(0);
    stream.read(reinterpret_cast<char *>(bytes.data()), length);
    require(stream.good() || stream.eof(), "cannot read complete file");
    return bytes;
}

void append(std::vector<uint8_t> & dst, const uint8_t * src, size_t size) {
    dst.insert(dst.end(), src, src + size);
}

}  // namespace

llama_kv_blockgtq_package llama_kv_blockgtq_package::load(
        const std::string & path) {
    const std::vector<uint8_t> bytes = read_file(path);
    require(bytes.size() == PACKAGE_BYTES, "file length drift");
    require(hash_bytes(bytes.data(), bytes.size()) == PACKAGE_SHA, "file SHA drift");
    const uint8_t * header = bytes.data();
    require(std::memcmp(header, "BGTQMRP1", 8) == 0, "magic drift");
    require(read_u32(header + 8) == 1, "schema drift");
    require(read_u32(header + 12) == 0x01020304U, "endian marker drift");
    require(read_u32(header + 16) == HEADER_BYTES, "header size drift");
    require(read_u32(header + 20) == SECTION_COUNT, "section count drift");
    require(read_u32(header + 24) == RECORD_BYTES, "record size drift");
    require(read_u64(header + 28) == TABLE_OFFSET, "table offset drift");
    require(read_u64(header + 36) == PAYLOAD_OFFSET, "payload offset drift");
    require(read_u64(header + 44) == PAYLOAD_BYTES, "payload size drift");
    require(read_u64(header + 52) == PACKAGE_BYTES, "package size field drift");
    require(hex(header + 60, 32) == PARENT_SHA, "parent artifact SHA drift");
    require(hex(header + 92, 32) == CONTRACT_SHA, "contract SHA drift");
    require(all_zero(header + 188, header + HEADER_BYTES), "header reserved bytes nonzero");
    require(
        hash_bytes(header + TABLE_OFFSET, SECTION_COUNT * RECORD_BYTES) ==
            hex(header + 156, 32),
        "section table SHA drift");
    require(
        hash_bytes(header + PAYLOAD_OFFSET, PAYLOAD_BYTES) ==
            hex(header + 124, 32),
        "payload SHA drift");

    llama_kv_blockgtq_package package;
    size_t expected_offset = PAYLOAD_OFFSET;
    for (size_t index = 0; index < SECTION_COUNT; ++index) {
        const uint8_t * record = header + TABLE_OFFSET + index * RECORD_BYTES;
        const expected_section & expected = SECTIONS[index];
        size_t name_length = 0;
        while (name_length < 64 && record[name_length] != 0) {
            ++name_length;
        }
        require(name_length < 64, "section name lacks terminator");
        require(
            std::string(reinterpret_cast<const char *>(record), name_length) ==
                expected.name,
            "section order or name drift");
        require(
            all_zero(record + name_length + 1, record + 64),
            std::string(expected.name) + " name padding nonzero");
        require(read_u32(record + 64) == expected.dtype, std::string(expected.name) + " dtype drift");
        require(read_u32(record + 68) == expected.rank, std::string(expected.name) + " rank drift");
        for (size_t dim = 0; dim < 4; ++dim) {
            require(
                read_u64(record + 72 + dim * 8) == expected.shape[dim],
                std::string(expected.name) + " shape drift");
        }
        const size_t offset = static_cast<size_t>(read_u64(record + 104));
        const size_t value_bytes = static_cast<size_t>(read_u64(record + 112));
        const size_t aligned_bytes = static_cast<size_t>(read_u64(record + 120));
        require(offset == expected_offset, std::string(expected.name) + " offset drift");
        require(value_bytes == expected.value_bytes, std::string(expected.name) + " value size drift");
        require(aligned_bytes == expected.aligned_bytes, std::string(expected.name) + " aligned size drift");
        require(read_u32(record + 128) == 64, std::string(expected.name) + " alignment drift");
        require(read_u32(record + 132) == expected.ownership, std::string(expected.name) + " ownership drift");
        require(read_u32(record + 136) == 1, std::string(expected.name) + " lifetime drift");
        require(read_u32(record + 140) == expected.visibility, std::string(expected.name) + " visibility drift");
        require(offset <= bytes.size() && aligned_bytes <= bytes.size() - offset, "section out of bounds");
        require(
            hash_bytes(header + offset, value_bytes) == hex(record + 144, 32),
            std::string(expected.name) + " value SHA drift");
        require(
            hash_bytes(header + offset, aligned_bytes) == hex(record + 176, 32),
            std::string(expected.name) + " padded SHA drift");
        require(
            all_zero(header + offset + value_bytes, header + offset + aligned_bytes),
            std::string(expected.name) + " padding nonzero");
        require(
            all_zero(record + 208, record + RECORD_BYTES),
            std::string(expected.name) + " record reserved bytes nonzero");
        if (expected.ownership == 1) {
            append(package.producer_, header + offset, aligned_bytes);
        } else if (expected.ownership == 2) {
            append(package.consumer_, header + offset, aligned_bytes);
        } else {
            append(package.shared_, header + offset, aligned_bytes);
        }
        expected_offset += aligned_bytes;
    }
    require(expected_offset == bytes.size(), "section coverage drift");
    require(package.producer_.size() == LLAMA_KV_BLOCKGTQ_PRODUCER_BYTES, "producer pool size drift");
    require(package.consumer_.size() == LLAMA_KV_BLOCKGTQ_CONSUMER_BYTES, "consumer pool size drift");
    require(package.shared_.size() == LLAMA_KV_BLOCKGTQ_SHARED_BYTES, "shared pool size drift");
    package.sha256_ = PACKAGE_SHA;
    return package;
}

const std::vector<uint8_t> & llama_kv_blockgtq_package::producer() const {
    return producer_;
}

const std::vector<uint8_t> & llama_kv_blockgtq_package::consumer() const {
    return consumer_;
}

const std::vector<uint8_t> & llama_kv_blockgtq_package::shared() const {
    return shared_;
}

const std::string & llama_kv_blockgtq_package::sha256() const {
    return sha256_;
}

struct llama_kv_blockgtq_runtime::impl {
    ggml_backend_t backend = nullptr;
    ggml_context_ptr tensor_ctx;
    ggml_backend_buffer_ptr buffer;
    ggml_tensor * producer = nullptr;
    ggml_tensor * consumer = nullptr;
    ggml_tensor * shared = nullptr;
    ggml_tensor * history = nullptr;
    int32_t sequence_length = 0;
    int32_t pending_start = 0;
    int32_t pending_count = 0;
    bool pending = false;
    bool valid = true;
};

llama_kv_blockgtq_runtime::llama_kv_blockgtq_runtime(
        std::unique_ptr<impl> impl) :
    impl_(std::move(impl)) {
}

llama_kv_blockgtq_runtime::~llama_kv_blockgtq_runtime() = default;

std::unique_ptr<llama_kv_blockgtq_runtime> llama_kv_blockgtq_runtime::create(
        const std::string & package_path,
        ggml_backend_t backend,
        std::string * error) {
    auto fail = [error](const std::string & message) {
        if (error) {
            *error = message;
        }
        return std::unique_ptr<llama_kv_blockgtq_runtime>{};
    };
    if (!backend) {
        return fail("CPU backend is null");
    }

    llama_kv_blockgtq_package package;
    try {
        package = llama_kv_blockgtq_package::load(package_path);
    } catch (const std::exception & exception) {
        return fail(exception.what());
    }

    ggml_init_params params = {
        /* .mem_size   = */ 5 * ggml_tensor_overhead(),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    auto value = std::make_unique<impl>();
    value->backend = backend;
    value->tensor_ctx.reset(ggml_init(params));
    if (!value->tensor_ctx) {
        return fail("could not allocate persistent tensor context");
    }
    value->producer = ggml_new_tensor_1d(
        value->tensor_ctx.get(), GGML_TYPE_I8, LLAMA_KV_BLOCKGTQ_PRODUCER_BYTES);
    value->consumer = ggml_new_tensor_1d(
        value->tensor_ctx.get(), GGML_TYPE_I8, LLAMA_KV_BLOCKGTQ_CONSUMER_BYTES);
    value->shared = ggml_new_tensor_1d(
        value->tensor_ctx.get(), GGML_TYPE_I8, LLAMA_KV_BLOCKGTQ_SHARED_BYTES);
    value->history = ggml_new_tensor_1d(
        value->tensor_ctx.get(), GGML_TYPE_I8, LLAMA_KV_BLOCKGTQ_HISTORY_BYTES);
    ggml_set_name(value->producer, "edgekv_blockgtq_producer");
    ggml_set_name(value->consumer, "edgekv_blockgtq_consumer");
    ggml_set_name(value->shared, "edgekv_blockgtq_shared");
    ggml_set_name(value->history, "edgekv_blockgtq_history");

    value->buffer.reset(
        ggml_backend_alloc_ctx_tensors(value->tensor_ctx.get(), backend));
    if (!value->buffer) {
        return fail("could not allocate persistent backend buffer");
    }
    ggml_backend_buffer_clear(value->buffer.get(), 0);
    ggml_backend_tensor_set(
        value->producer, package.producer().data(), 0, package.producer().size());
    ggml_backend_tensor_set(
        value->consumer, package.consumer().data(), 0, package.consumer().size());
    ggml_backend_tensor_set(
        value->shared, package.shared().data(), 0, package.shared().size());

    return std::unique_ptr<llama_kv_blockgtq_runtime>(
        new llama_kv_blockgtq_runtime(std::move(value)));
}

ggml_tensor * llama_kv_blockgtq_runtime::producer() const {
    return impl_->producer;
}

ggml_tensor * llama_kv_blockgtq_runtime::consumer() const {
    return impl_->consumer;
}

ggml_tensor * llama_kv_blockgtq_runtime::shared() const {
    return impl_->shared;
}

ggml_tensor * llama_kv_blockgtq_runtime::history() const {
    return impl_->history;
}

ggml_backend_t llama_kv_blockgtq_runtime::backend() const {
    return impl_->backend;
}

bool llama_kv_blockgtq_runtime::prepare(
        int32_t token_start,
        int32_t token_count,
        std::string * error) {
    auto reject = [error](const std::string & message) {
        if (error) {
            *error = message;
        }
        return false;
    };
    if (!impl_->valid) {
        return reject("runtime is invalid after an earlier failure");
    }
    if (impl_->pending) {
        return reject("a previous graph update is still pending");
    }
    if (token_start != impl_->sequence_length) {
        return reject(
            "token start does not match append-only history length");
    }
    if (token_count <= 0 ||
        token_start > LLAMA_KV_BLOCKGTQ_CAPACITY - token_count) {
        return reject("token range exceeds frozen history capacity");
    }
    impl_->pending_start = token_start;
    impl_->pending_count = token_count;
    impl_->pending = true;
    return true;
}

void llama_kv_blockgtq_runtime::finish() {
    GGML_ASSERT(impl_->valid && impl_->pending);
    GGML_ASSERT(impl_->pending_start == impl_->sequence_length);
    impl_->sequence_length += impl_->pending_count;
    impl_->pending = false;
    impl_->pending_count = 0;
}

void llama_kv_blockgtq_runtime::fail() {
    impl_->pending = false;
    impl_->pending_count = 0;
    impl_->valid = false;
}

bool llama_kv_blockgtq_runtime::valid() const {
    return impl_->valid;
}

int32_t llama_kv_blockgtq_runtime::sequence_length() const {
    return impl_->sequence_length;
}

bool llama_kv_blockgtq_runtime::reset(
        bool clear_data,
        std::string * error) {
    if (impl_->pending) {
        if (error) {
            *error = "cannot reset while a graph update is pending";
        }
        return false;
    }
    if (clear_data) {
        constexpr size_t ZERO_CHUNK_BYTES = 1024 * 1024;
        const std::vector<uint8_t> zeroes(ZERO_CHUNK_BYTES);
        const size_t history_bytes = ggml_nbytes(impl_->history);
        for (size_t offset = 0; offset < history_bytes;
             offset += ZERO_CHUNK_BYTES) {
            const size_t size =
                std::min(ZERO_CHUNK_BYTES, history_bytes - offset);
            ggml_backend_tensor_set(
                impl_->history, zeroes.data(), offset, size);
        }
    }
    impl_->sequence_length = 0;
    impl_->pending_start = 0;
    impl_->pending_count = 0;
    impl_->pending = false;
    impl_->valid = true;
    return true;
}

bool llama_kv_blockgtq_runtime::truncate(
        int32_t sequence_length,
        std::string * error) {
    if (!impl_->valid) {
        if (error) {
            *error = "runtime is invalid after an earlier failure";
        }
        return false;
    }
    if (impl_->pending) {
        if (error) {
            *error = "cannot truncate while a graph update is pending";
        }
        return false;
    }
    if (sequence_length < 0 ||
        sequence_length > impl_->sequence_length) {
        if (error) {
            *error = "truncate length is outside committed history";
        }
        return false;
    }
    impl_->sequence_length = sequence_length;
    return true;
}

namespace {

class llama_memory_blockgtq_context final : public llama_memory_context_i {
public:
    explicit llama_memory_blockgtq_context(llama_memory_status status) :
        status_(status) {
    }

    explicit llama_memory_blockgtq_context(
            std::vector<llama_ubatch> ubatches) :
        status_(LLAMA_MEMORY_STATUS_SUCCESS),
        ubatches_(std::move(ubatches)) {
    }

    bool next() override {
        assert(status_ == LLAMA_MEMORY_STATUS_SUCCESS);
        if (++current_ >= ubatches_.size()) {
            return false;
        }
        return true;
    }

    bool apply() override {
        return !llama_memory_status_is_fail(status_);
    }

    const llama_ubatch & get_ubatch() const override {
        assert(status_ == LLAMA_MEMORY_STATUS_SUCCESS);
        assert(current_ < ubatches_.size());
        return ubatches_[current_];
    }

    llama_memory_status get_status() const override {
        return status_;
    }

private:
    llama_memory_status status_;
    std::vector<llama_ubatch> ubatches_;
    size_t current_ = 0;
};

}  // namespace

llama_memory_blockgtq::llama_memory_blockgtq(
        llama_kv_blockgtq_runtime * runtime,
        uint32_t n_seq_max) :
    runtime_(runtime) {
    if (!runtime_) {
        throw std::runtime_error("Block-GTQ memory requires a runtime");
    }
    if (n_seq_max != 1) {
        throw std::runtime_error(
            "Block-GTQ memory supports exactly one sequence");
    }
}

llama_memory_context_ptr llama_memory_blockgtq::init_batch(
        llama_batch_allocr & balloc,
        uint32_t n_ubatch,
        bool embd_all) {
    GGML_UNUSED(embd_all);

    if (n_ubatch != 1) {
        return std::make_unique<llama_memory_blockgtq_context>(
            LLAMA_MEMORY_STATUS_FAILED_PREPARE);
    }

    balloc.split_reset();
    std::vector<llama_ubatch> ubatches;
    while (true) {
        auto ubatch = balloc.split_simple(1);
        if (ubatch.n_tokens == 0) {
            break;
        }
        if (ubatch.n_tokens != 1 || ubatch.n_seqs_unq != 1 ||
            ubatch.n_seq_id[0] != 1 || !ubatch.seq_id[0] ||
            ubatch.seq_id[0][0] != 0) {
            return std::make_unique<llama_memory_blockgtq_context>(
                LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }
        ubatches.push_back(std::move(ubatch));
    }
    if (ubatches.empty() ||
        balloc.get_n_used() != balloc.get_n_tokens()) {
        return std::make_unique<llama_memory_blockgtq_context>(
            LLAMA_MEMORY_STATUS_FAILED_PREPARE);
    }
    return std::make_unique<llama_memory_blockgtq_context>(
        std::move(ubatches));
}

llama_memory_context_ptr llama_memory_blockgtq::init_full() {
    return std::make_unique<llama_memory_blockgtq_context>(
        LLAMA_MEMORY_STATUS_SUCCESS);
}

llama_memory_context_ptr llama_memory_blockgtq::init_update(
        llama_context * lctx,
        bool optimize) {
    GGML_UNUSED(lctx);
    GGML_UNUSED(optimize);
    return std::make_unique<llama_memory_blockgtq_context>(
        LLAMA_MEMORY_STATUS_NO_UPDATE);
}

bool llama_memory_blockgtq::get_can_shift() const {
    return false;
}

void llama_memory_blockgtq::clear(bool data) {
    std::string error;
    if (!runtime_->reset(data, &error)) {
        LLAMA_LOG_ERROR(
            "%s: failed to clear Block-GTQ memory: %s\n",
            __func__,
            error.c_str());
        runtime_->fail();
    }
}

bool llama_memory_blockgtq::seq_rm(
        llama_seq_id seq_id,
        llama_pos p0,
        llama_pos p1) {
    if (seq_id != -1 && seq_id != 0) {
        return false;
    }
    const llama_pos length = runtime_->sequence_length();
    p0 = p0 < 0 ? 0 : p0;
    p1 = p1 < 0 ? length : std::min(p1, length);
    if (p0 >= p1) {
        return true;
    }
    if (p1 != length) {
        return false;
    }
    std::string error;
    if (!runtime_->truncate(p0, &error)) {
        LLAMA_LOG_ERROR(
            "%s: failed to truncate Block-GTQ memory: %s\n",
            __func__,
            error.c_str());
        return false;
    }
    return true;
}

void llama_memory_blockgtq::invalidate_unsupported(
        const char * operation) {
    LLAMA_LOG_ERROR(
        "%s: Block-GTQ memory does not support %s; runtime invalidated\n",
        __func__,
        operation);
    runtime_->fail();
}

void llama_memory_blockgtq::seq_cp(
        llama_seq_id seq_id_src,
        llama_seq_id seq_id_dst,
        llama_pos p0,
        llama_pos p1) {
    GGML_UNUSED(p0);
    GGML_UNUSED(p1);
    if (seq_id_src == 0 && seq_id_dst == 0) {
        return;
    }
    invalidate_unsupported("sequence copy");
}

void llama_memory_blockgtq::seq_keep(llama_seq_id seq_id) {
    if (seq_id == 0) {
        return;
    }
    invalidate_unsupported("sequence keep");
}

void llama_memory_blockgtq::seq_add(
        llama_seq_id seq_id,
        llama_pos p0,
        llama_pos p1,
        llama_pos shift) {
    GGML_UNUSED(p0);
    GGML_UNUSED(p1);
    if (seq_id == 0 && shift == 0) {
        return;
    }
    invalidate_unsupported("sequence position shift");
}

void llama_memory_blockgtq::seq_div(
        llama_seq_id seq_id,
        llama_pos p0,
        llama_pos p1,
        int d) {
    GGML_UNUSED(p0);
    GGML_UNUSED(p1);
    if (seq_id == 0 && d == 1) {
        return;
    }
    invalidate_unsupported("sequence position division");
}

llama_pos llama_memory_blockgtq::seq_pos_min(
        llama_seq_id seq_id) const {
    if (seq_id != 0 || runtime_->sequence_length() == 0) {
        return -1;
    }
    return 0;
}

llama_pos llama_memory_blockgtq::seq_pos_max(
        llama_seq_id seq_id) const {
    if (seq_id != 0 || runtime_->sequence_length() == 0) {
        return -1;
    }
    return runtime_->sequence_length() - 1;
}

std::map<ggml_backend_buffer_type_t, size_t>
llama_memory_blockgtq::memory_breakdown() const {
    return {};
}

void llama_memory_blockgtq::state_write(
        llama_io_write_i & io,
        llama_seq_id seq_id,
        llama_state_seq_flags flags) const {
    GGML_UNUSED(io);
    GGML_UNUSED(seq_id);
    GGML_UNUSED(flags);
    throw std::runtime_error(
        "Block-GTQ memory state serialization is not implemented");
}

void llama_memory_blockgtq::state_read(
        llama_io_read_i & io,
        llama_seq_id seq_id,
        llama_state_seq_flags flags) {
    GGML_UNUSED(io);
    GGML_UNUSED(seq_id);
    GGML_UNUSED(flags);
    throw std::runtime_error(
        "Block-GTQ memory state deserialization is not implemented");
}
