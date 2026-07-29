#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-hexagon.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

static void require(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        std::exit(1);
    }
}

int main() {
    ggml_backend_reg_t reg = ggml_backend_hexagon_reg();
    require(reg != nullptr, "Hexagon registry initialization");
    require(ggml_backend_reg_dev_count(reg) > 0, "Hexagon device enumeration");

    ggml_backend_t backend = ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, 0), nullptr);
    require(backend != nullptr, "Hexagon backend initialization");

    constexpr int64_t ne0 = 128;
    constexpr int64_t ne1 = 2;
    ggml_init_params  params = {
         /* .mem_size = */ ggml_tensor_overhead() * 4 + ggml_graph_overhead_custom(4, false),
         /* .mem_base = */ nullptr,
         /* .no_alloc = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr, "ggml context initialization");

    ggml_tensor * src0 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1);
    ggml_tensor * src1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1);
    ggml_tensor * dst  = ggml_add(ctx, src0, src1);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, "Hexagon tensor allocation");

    std::vector<float> input(static_cast<size_t>(ne0 * ne1), 1.0f);
    ggml_backend_tensor_set(src0, input.data(), 0, input.size() * sizeof(float));
    ggml_backend_tensor_set(src1, input.data(), 0, input.size() * sizeof(float));

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 4, false);
    ggml_build_forward_expand(graph, dst);

    const size_t valid_row_stride = src1->nb[1];
    src1->nb[1] = 0;
    require(
            ggml_backend_graph_compute(backend, graph) == GGML_STATUS_FAILED,
            "target NO_SUPPORT status did not fail the graph");

    src1->nb[1] = valid_row_stride;
    require(
            ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS,
            "session did not recover after a reported target failure");

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);

    std::puts("HTP target-to-host status propagation passed");
    return 0;
}
