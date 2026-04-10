#include <cassert>
#include <cstdio>
#include <memory>
#include <string>

#include "clip-impl.h"
#include "clip.h"
#include "mtmd-image.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

int main(int argc, char ** argv) {
    // const char * mmproj_path = "/home/hzx/learn/llama.cpp-20250523/extras/mmproj-qwen2.5-vl-3b-instruct.q8_0.gguf";
    // const char * image_path  = "/home/hzx/learn/llama.cpp-20250523/extras/test-1.jpeg";

    if (argc < 3) {
        fprintf(stderr, "Usage: %s <mmproj_path> <image_path>\n", argv[0]);
        return 1;
    }
    const char *mmproj_path = argv[1];
    const char *image_path  = argv[2];

    const int n_threads = 4;

    clip_context_params clip_params = {
        /* use_gpu           */ true,
        /* flash_attn_type   */ CLIP_FLASH_ATTN_TYPE_AUTO,
        /* image_min_tokens  */ 4,
        /* image_max_tokens  */ 16384,
        /* warmup            */ false,
        /* cb_eval           */ nullptr,
        /* cb_eval_user_data */ nullptr,
    };

    auto clip_init_res = clip_init(mmproj_path, clip_params);
    GGML_ASSERT(clip_init_res.ctx_v != nullptr);

    auto vision_ctx = clip_init_res.ctx_v;
    // load image from file and init clip_image_u8
    clip_image_u8 * img_u8(clip_image_u8_init());
    {
        int nx, ny, nc;

        unsigned char * data = stbi_load(image_path, &nx, &ny, &nc, 3);
        GGML_ASSERT(data != nullptr && "failed to load image");

        clip_build_img_from_pixels(data, nx, ny, img_u8);
        stbi_image_free(data);
    }

    // covert image to float32
    clip_image_f32_batch batch_f32;
    std::unique_ptr<mtmd_image_preprocessor> image_preproc;
    switch (clip_get_projector_type(vision_ctx)) {
        case PROJECTOR_TYPE_MLP:
        case PROJECTOR_TYPE_MLP_NORM:
        case PROJECTOR_TYPE_LDP:
        case PROJECTOR_TYPE_LDPV2:
        case PROJECTOR_TYPE_COGVLM:
        case PROJECTOR_TYPE_JANUS_PRO:
        case PROJECTOR_TYPE_GLM_EDGE:
        case PROJECTOR_TYPE_GEMMA3:
        case PROJECTOR_TYPE_GEMMA3NV:
            image_preproc = std::make_unique<mtmd_image_preprocessor_fixed_size>(vision_ctx);
            break;
        default:
            image_preproc = std::make_unique<mtmd_image_preprocessor_dyn_size>(vision_ctx);
            break;
    }
    GGML_ASSERT(image_preproc->preprocess(*img_u8, batch_f32));

    size_t n_tokens = 0;
    for (const auto & entry : batch_f32.entries) {
        n_tokens += clip_n_output_tokens(vision_ctx, entry.get());
    }
    printf("OK1 n_tokens=%zu\n", n_tokens);

    const size_t img_embd_n_elems = n_tokens * clip_n_mmproj_embd(vision_ctx);
    float *      img_embd         = new float[img_embd_n_elems];
    GGML_ASSERT(clip_image_batch_encode(vision_ctx, n_threads, &batch_f32, img_embd));
    printf("Done!\n");

    // dump img_embd to file
    {
        FILE * fp = fopen("img_embd.bin", "wb");
        GGML_ASSERT(fp != nullptr && "failed to open img_embd.bin for writing");
        size_t n_write = fwrite(img_embd, sizeof(float), img_embd_n_elems, fp);
        fclose(fp);
        printf("Wrote %zu floats to img_embd.bin\n", n_write);
    }

    delete[] img_embd;
    clip_image_u8_free(img_u8);
    clip_free(vision_ctx);
    return 0;
}
