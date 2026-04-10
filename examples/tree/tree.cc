#include <cstdio>
#include <cstdlib>
#include <unordered_map>

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "log.h"
// #include "sampling.h"

static std::vector<std::string> k_prompts = {
    "What is the meaning of life?",
    "Tell me an interesting fact about llamas.",
    "What is the best way to cook a steak?",
    "Are you familiar with the Special Theory of Relativity and can you explain it to me?",
};

int main(int argc, char ** argv) {
    srand(42);

    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    common_init();

    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init = common_init_from_params(params);

    llama_model *   model = llama_init.model;
    llama_context * ctx   = llama_init.context;

    const int n_ctx = llama_n_ctx(ctx);

    const int n_max_seqs = 16;
    auto      batch      = llama_batch_init(n_ctx, 0, n_max_seqs);

    /*
    auto shared_tokens =
        common_tokenize(ctx, "You are a helpful assistant that help people to find information.\n\n", false);
    const int n_shared_tokens = shared_tokens.size();
    
    // level 0 shared prompt
    {
        LLAMA_LOG("Level 0 shared prompt: %d tokens\n", n_shared_tokens);
        for (int i = 0; i < n_shared_tokens; ++i) {
            common_batch_add(batch, shared_tokens[i], i, { 0 }, false);
        }

        if (llama_decode(ctx, batch)) {
            LOG_ERR("llama_decode() failed\n");
            return 1;
        }
    }

    common_batch_clear(batch);
    for (int i = 0; i < (int) k_prompts.size(); ++i) {
        int n_keep = rand() % n_shared_tokens;
        LOG_INF("sequence %d keep %d shared tokens\n", i, n_keep);

        int prev_n_tokens = batch.n_tokens;
        int seq_id = i + 1;
        llama_kv_cache_seq_cp(ctx, 0, seq_id, 0, n_keep);
        for (int j = n_keep; j < n_shared_tokens; ++j) {
            common_batch_add(batch, shared_tokens[j], j, { seq_id }, false);
        }

        auto tokens = common_tokenize(ctx, k_prompts[i], false);
        for (int j = 0; j < (int) tokens.size(); ++j) {
            common_batch_add(batch, tokens[j], j + n_shared_tokens, { seq_id }, false);
        }

        if (batch.n_tokens > 0) {
            batch.logits[batch.n_tokens - 1] = true;
        }

        LOG_INF("sequence %d has %d new tokens in the batch\n", i, batch.n_tokens - prev_n_tokens);
    }

    if (llama_decode(ctx, batch)) {
        LOG_ERR("second llama_batch() failed\n");
    }
    */

    std::vector<std::vector<std::string>> multi_level_prompts = {
        { "You are a helpful assistant that help people to find information.\n\n" },
        { "The following contents are user inputs: ", "Reply politely and don't contain anything toxic or harmful. " },
        {
         "What is the meaning of life?", "Tell me an interesting fact about llamas.",
         "What is the best way to cook a steak?", "Are you familiar with the Special Theory of Relativity and can you explain it to me?",
         },
    };
    std::vector<std::vector<std::vector<llama_token>>> multi_level_tokens;

    int g_seq_id = 0;
    std::unordered_map<int, int> parent_seq_id;
    std::unordered_map<int, int> seq_lengths;
    std::vector<int> last_level_seq_ids = {-1};

    parent_seq_id[-1] = -1;
    seq_lengths[-1] = 0;

    const int n_levels = multi_level_prompts.size(); // outer size
    multi_level_tokens.resize(n_levels);

    for (int level = 0; level < n_levels; ++level) {
        common_batch_clear(batch);
        auto &cur_level_tokens = multi_level_tokens[level];
        std::vector<int> new_seq_ids;

        for (const auto &prompt : multi_level_prompts[level]) {
            std::vector<llama_token> tokens = common_tokenize(ctx, prompt, false);
            const int n_tokens = tokens.size();

            for (int parent_id : last_level_seq_ids) {
                int seq_id = g_seq_id++; // allocate new sequence id
                int parent_seq_len = seq_lengths[parent_id];
                new_seq_ids.push_back(seq_id);

                parent_seq_id[seq_id] = parent_id;
                seq_lengths[seq_id] = parent_seq_len + n_tokens;

                if (parent_id != -1) {
                    llama_kv_cache_seq_cp(ctx, parent_id, seq_id, -1, parent_seq_len);
                }
                for (int i = 0; i < n_tokens; ++i) {
                    common_batch_add(batch, tokens[i], parent_seq_len + i, { seq_id }, false);
                }
                batch.logits[batch.n_tokens - 1] = true;
            }

            cur_level_tokens.push_back(std::move(tokens));
        }

        for (int seq_id : new_seq_ids) {
            LOG_INF("level %d new sequence %d, parent %d, total length %d, new part length %d\n",
                level, seq_id, parent_seq_id[seq_id], seq_lengths[seq_id], seq_lengths[seq_id] - seq_lengths[parent_seq_id[seq_id]]);
        }
        LOG_INF("level %d batch contains %d tokens\n", level, batch.n_tokens);

        if (llama_decode(ctx, batch)) {
            LOG_ERR("level %d decode failed!\n", level);
            break;
        }

        last_level_seq_ids = std::move(new_seq_ids);
    }

    llama_batch_free(batch);
    llama_free(ctx);
    llama_free_model(model);

    llama_backend_free();

    return 0;
}
