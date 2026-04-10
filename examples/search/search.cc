#include <algorithm>
#include <chrono>
#include <cstdio>
#include <functional>
#include <future>
#include <string>
#include <vector>

// #include "arg.h"
#include "common.h"
#include "json.hpp"
// #include "llama-cpp.h" // for llama_sampler_ptr
#include "llama.h"
#include "reward_model.h"

namespace chrono = std::chrono;

struct GuardTimer {
    decltype(chrono::system_clock::now()) t;

    std::function<void(uint64_t)> cb;

    GuardTimer(const std::function<void(uint64_t)> & f) : cb(f) { t = chrono::system_clock::now(); }

    ~GuardTimer() {
        auto     tp_diff = chrono::system_clock::now() - t;
        uint64_t time_us = chrono::duration_cast<chrono::microseconds>(tp_diff).count();
        cb(time_us);
    }
};

struct SearchConfig {
    int n          = 4;               // seasch budget, max decode batch size
    int beam_width = 2;               // beam width
    int n_paths    = n / beam_width;  // number of search paths
    // TODO: assert(n == bean_width * n_paths);

    // search stop conditions
    int max_step_len = 256;
    int max_seq_len  = 4096;
    int max_n_iters  = 40;

    bool filter_duplicates = false;

    std::string step_scores_agg_method = "last";
};

struct Beam {
    int   id;
    int   seq_len;  // total length, including prompt
    float score;    // aggregated score, used for beam sorting (reranking)

    // beam state flags
    bool step_finished;
    bool eos_found;

    llama_token next_fwd_token;  // the token that need to be forwarded

    // tokens & text of current step
    std::vector<llama_token> cur_tokens;
    std::string              cur_str;

    // history completion steps
    std::vector<std::vector<llama_token>> history_tokens;
    std::vector<std::string>              history_strs;
    std::vector<float>                    step_scores;

    // sampler
    // llama_sampler_ptr sampler;

    void add_token(llama_token token, const std::string & token_str) {
        cur_tokens.push_back(token);
        cur_str.append(token_str);
        next_fwd_token = token;
        seq_len++;
    }

    bool match_step_delimiters(const std::vector<std::string> & step_delimiters) const {
        return std::any_of(step_delimiters.begin(), step_delimiters.end(), [this](auto & step_delimiter) {
            return cur_str.rfind(step_delimiter) != std::string::npos;
        });
    }

    // TODO: adjust to different tasks
    bool match_task_specific_eog() const {
        constexpr const char * s = "\\boxed";
        return cur_str.find(s) != std::string::npos;
    }

    float compute_aggregated_score(const std::string & agg_method) {
        GGML_ASSERT(step_scores.size() > 0);
        if (agg_method == "max") {
            return score = *std::max_element(step_scores.begin(), step_scores.end());
        }
        if (agg_method == "mean") {
            float sum = 0;
            for (auto s : step_scores) {
                sum += s;
            }
            return (score = sum / step_scores.size());
        }
        // default: use the score of the most recent step
        return score = step_scores.back();
    }

    Beam fork(int new_id) const {
        Beam child;
        child.id             = new_id;
        child.seq_len        = seq_len;
        child.next_fwd_token = next_fwd_token;

        child.history_tokens = history_tokens;  // copy
        child.history_strs   = history_strs;    // copy
        child.step_scores    = step_scores;     // copy

        // child.sampler = llama_sampler_ptr{ llama_sampler_clone(sampler.get()) };
        return child;
    }

    std::string completion() const {
        std::string result;
        for (const auto & step : history_strs) {
            result.append(step);
        }
        return result;
    }
};

// if some beams' current step is too long, return true
bool check_beams_step_length(const std::vector<Beam> & beams, const SearchConfig & cfg) {
    std::vector<size_t> lengths(beams.size());
    for (const auto & beam : beams) {
        lengths.push_back(beam.cur_tokens.size());
    }
    std::sort(lengths.begin(), lengths.end());

    int max_len = lengths.back();
    if (max_len > cfg.max_step_len) {
        return true;
    }

    // TODO: use proper parameters
    int median_len = lengths[lengths.size() / 2];
    int min_len    = lengths[0];
    if (min_len > 16 && max_len > median_len * 4) {
        return true;
    }
    return false;
}

int main(int argc, char ** argv) {
    bool debug_log  = false;
    bool eager_exit = false;

    // std::string system_prompt =
    //     "Solve the following math problem efficiently and clearly:\n\n- For simple problems (2 steps or "
    //     "fewer):\nProvide a concise solution with minimal explanation.\n\n- For complex problems (3 steps or "
    //     "more):\nUse this step-by-step format:\n\n## Step 1: [Concise description]\n[Brief explanation and "
    //     "calculations]\n\n## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n...\n\nRegardless "
    //     "of the approach, always conclude with:\n\nTherefore, the final answer is: $\\boxed{answer}$. I hope it is "
    //     "correct.\n\nWhere [answer] is just the final number or expression that solves the problem.";
    // std::string question = "How many positive whole-number divisors does 196 have?";
    // std::string prompt = system_prompt + "\n\nQuestion: " + question + "\n\nSolution: ";

    std::string question = "How many positive whole-number divisors does 196 have?";

    int n    = 4;
    int p    = 2;  // number of paths
    int port = 8080;

    if (argc > 1) {
        std::ifstream file(argv[1]);
        if (file) {
            std::string content{ std::istreambuf_iterator<char>(file), {} };
            question = content;
        }
    }
    if (argc > 2) {
        n = std::atoi(argv[2]);
    }
    if (argc > 3) {
        p = std::atoi(argv[3]);
    }
    if (argc > 4) {
        port = std::atoi(argv[4]);
    }

    // std::string prompt =
    //     "Please reason step by step, and put your final answer within \\boxed{}.\n\n" + question + "\n";

    // for chatml template used by qwen
    std::string prompt =
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n";
    prompt += "Please reason step by step, and put your final answer within \\boxed{}.\n\n" + question + "\n";
    prompt += "<|im_end|>\n";
    prompt += "<|im_start|>assistant\n";

    std::string model_path = "/home/hzx/learn/llama.cpp-20241028/extras/qwen2.5-1.5b-instruct.iq4_nl+q8_0.gguf";

    // if (model_path.find("llama3.2") != std::string::npos) {
    //     prompt += "## Step 1";
    // }
    prompt += "## Step 1:";

    int ngl = 999;  // server test

    // std::vector<std::string> step_delimiters = { ". ", ".\n", "\n\n" };
    std::vector<std::string> step_delimiters = { "\n\n" };

    // TODO: parse command line args

    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers       = ngl;

    llama_model * model = llama_load_model_from_file(model_path.c_str(), model_params);
    if (model == nullptr) {
        GGML_ABORT("%s: error: unable to load model\n", __func__);
    }

    // initialize the context
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx                = 65536;
    // default n_batch and n_ubatch should be enough for most cases
    ctx_params.flash_attn           = true;
    ctx_params.no_perf              = false;

    llama_context * ctx = llama_new_context_with_model(model, ctx_params);
    if (ctx == nullptr) {
        GGML_ABORT("%s: error: failed to create the llama_context\n", __func__);
    }

    // initialize the sampler
    llama_sampler_chain_params sampler_params = {
        .no_perf = false,
    };
    llama_sampler * sampler = llama_sampler_chain_init(sampler_params);
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.95f, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.8f));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(42));

    // prompt prefill
    auto prompt_tokens = common_tokenize(model, prompt, true);
    auto prompt_batch  = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
    if (llama_decode(ctx, prompt_batch)) {
        GGML_ABORT("prompt prefill failed\n");
    }

    SearchConfig      cfg;
    std::vector<Beam> beams;

    cfg.n          = n;
    cfg.n_paths    = p;
    cfg.beam_width = n / p;

    int seq_id_base = cfg.n;

    llama_batch batch = llama_batch_init(cfg.n, 0, cfg.n * 2);

    // metadata, records
    std::vector<std::vector<int>> step_lengths;
    std::vector<uint64_t>         step_tot_time_ms;
    std::vector<uint64_t>         step_gen_time_ms;
    std::vector<uint64_t>         step_score_time_ms;

    // init remote scorer
    std::string endpoint = "http://localhost:" + std::to_string(port);
    RemotePRM   prm(endpoint, cfg.n);

    // make initial beams
    for (int i = 0; i < cfg.n; ++i) {
        Beam beam;
        beam.id = seq_id_base + i;

        beam.step_finished = false;
        beam.eos_found     = false;

        // "copy" shared prompt prefix
        llama_kv_cache_seq_cp(ctx, 0, beam.id, 0, prompt_tokens.size());
        beam.seq_len = prompt_tokens.size();

        // llama_sampler * new_sampler = llama_sampler_clone(sampler);
        llama_token token = llama_sampler_sample(sampler, ctx, -1);
        beam.add_token(token, common_token_to_piece(ctx, token));
        // beam.sampler = llama_sampler_ptr{ new_sampler };
        // printf("beam %d initial sampled token: %s\n", beam.id, beam.cur_str.c_str());

        beams.push_back(std::move(beam));
    }

    // remove seq_id 0 (prompt part) from kv cache
    llama_kv_cache_seq_rm(ctx, 0, 0, -1);

    auto gen_start = chrono::system_clock::now();

    for (int n_iters = 0; n_iters < cfg.max_n_iters; ++n_iters) {
        GuardTimer step_tot_timer{ [&step_tot_time_ms](uint64_t time_us) {
            step_tot_time_ms.push_back(time_us / 1000);
        } };

        // generation loop
        {
            GuardTimer step_gen_timer{ [&step_gen_time_ms](uint64_t time_us) {
                step_gen_time_ms.push_back(time_us / 1000);
            } };

            while (true) {
                // auto t7 = chrono::system_clock::now();

                // length balance check
                bool some_step_too_long = check_beams_step_length(beams, cfg);
                if (some_step_too_long) {
                    break;
                }

                // batch construction
                common_batch_clear(batch);
                for (const auto & beam : beams) {
                    // TODO: allow speculation, early termination
                    if (beam.step_finished) {
                        continue;
                    }

                    int  pos   = beam.seq_len - 1;  // last token
                    auto token = beam.next_fwd_token;
                    common_batch_add(batch, token, pos, { beam.id }, true);
                }
                // auto t8 = chrono::system_clock::now();
                // auto e4 = chrono::duration_cast<chrono::microseconds>(t8 - t7).count();
                // printf("$%d: prepare time: %.4f s\n", cnt, e4 * 1e-6f);

                // auto t1 = chrono::system_clock::now();
                if (llama_decode(ctx, batch)) {
                    GGML_ABORT("batch decode failed");
                }
                // auto t2 = chrono::system_clock::now();
                // auto e1 = chrono::duration_cast<chrono::microseconds>(t2 - t1).count();
                // printf("    decode time: %.4f s\n", e1 * 1e-6f);

                // auto t3              = chrono::system_clock::now();

                int intra_batch_idx = 0;

                std::vector<std::future<void>> futures;
                for (size_t i = 0; i < beams.size(); ++i) {
                    if (beams[i].step_finished) {
                        continue;
                    }

                    const int output_idx = intra_batch_idx++;

                    auto future = std::async(std::launch::async, [&, i, output_idx]() {
                        auto token = llama_sampler_sample(/* beams[i].sampler.get() */ sampler, ctx, output_idx);
                        beams[i].add_token(token, common_token_to_piece(ctx, token));
                    });
                    futures.push_back(std::move(future));
                }

                for (auto & future : futures) {
                    future.wait();
                }
                // auto t4 = chrono::system_clock::now();
                // auto e2 = chrono::duration_cast<chrono::microseconds>(t4 - t3).count();
                // printf("    sample time: %.4f s\n", e2 * 1e-6f);

                // auto   t5         = chrono::system_clock::now();
                size_t n_finished = 0;
                for (auto & beam : beams) {
                    if (!beam.step_finished) {
                        auto last_token = beam.cur_tokens.back();
                        bool is_eog     = llama_token_is_eog(model, last_token);
                        if (beam.match_step_delimiters(step_delimiters) || is_eog) {
                            beam.step_finished = true;
                        }
                        if (is_eog || beam.match_task_specific_eog()) {
                            beam.eos_found = true;
                        }
                    }
                    n_finished += beam.step_finished ? 1 : 0;
                }
                // auto t6 = chrono::system_clock::now();
                // auto e3 = chrono::duration_cast<chrono::microseconds>(t6 - t5).count();
                // printf("    check EOS time: %.4f s\n", e3 * 1e-6f);
                if (n_finished == beams.size()) {
                    break;
                }
            }
        }

        std::vector<int> local_step_lengths;
        local_step_lengths.reserve(beams.size());
        for (const auto & beam : beams) {
            local_step_lengths.push_back((int) beam.cur_tokens.size());
        }
        step_lengths.emplace_back(std::move(local_step_lengths));

        // move recently generated step into history
        for (auto & beam : beams) {
            beam.history_tokens.emplace_back(std::move(beam.cur_tokens));
            beam.history_strs.emplace_back(std::move(beam.cur_str));
        }

        // TODO: duplicate filter logic
        if (cfg.filter_duplicates) {
            // ...
        }

        // score logic
        {
            GuardTimer step_score_timer{ [&step_score_time_ms](uint64_t time_us) {
                step_score_time_ms.push_back(time_us / 1000);
            } };

            // collect completion steps
            std::vector<std::vector<std::string>> completion_steps;
            completion_steps.reserve(beams.size());
            for (const auto & beam : beams) {
                completion_steps.emplace_back(beam.history_strs);  // copy vector of string
            }

            auto scores = prm.score(question, completion_steps);

            if (debug_log) {
                printf("score process dump:\n");
                for (size_t i = 0; i < completion_steps.size(); ++i) {
                    printf("seq #%ld:\n", i);
                    for (size_t j = 0; j < completion_steps[i].size(); ++j) {
                        printf("    step #%ld: %s\n", j, completion_steps[i][j].c_str());
                    }
                }

                printf("\n==============\nscores:");
                for (auto f : scores) {
                    printf(" %g", f);
                }
                printf("\n");
            }

            GGML_ASSERT(scores.size() == beams.size());
            for (size_t i = 0; i < beams.size(); ++i) {
                beams[i].step_scores.push_back(scores[i]);
                GGML_ASSERT(beams[i].step_scores.size() == beams[i].history_strs.size());

                beams[i].compute_aggregated_score(cfg.step_scores_agg_method);
            }

            std::sort(beams.begin(), beams.end(), [](const Beam & a, const Beam & b) { return a.score > b.score; });
        }

        // check stop conditions
        // stop condition #1: sequences already too long
        if (std::any_of(beams.begin(), beams.end(),
                        [&cfg](const Beam & beam) { return beam.seq_len > cfg.max_seq_len; })) {
            break;
        }

        // stop condition #2: EOS handling
        auto has_eos = [](const Beam & beam) {
            return beam.eos_found;
        };
        if (eager_exit) {
            // simple exit logic: stop generation once an EOS is found in any beam
            if (std::any_of(beams.begin(), beams.end(), has_eos)) {
                break;
            }
        } else {
            // if any beam with EOS is ranked within top n_paths, stop
            int n_candidates = std::min(cfg.n_paths, (int) beams.size());
            if (std::any_of(beams.begin(), beams.begin() + n_candidates, has_eos)) {
                break;
            }
        }

        // remove beams with lowest scores
        if (beams.size() > (size_t) cfg.n_paths) {
            beams.resize(cfg.n_paths);
        }

        // discard finished sequences
        auto new_end = std::remove_if(beams.begin(), beams.end(), has_eos);
        beams.erase(new_end, beams.end());

        // expand remaining beams
        int              f = cfg.n / beams.size();
        std::vector<int> expand_factors(beams.size(), f);

        int r = cfg.n - f * beams.size();
        for (int i = 0; i < r; ++i) {
            expand_factors[i]++;
        }

        // fork beams & remap sequence ids
        seq_id_base    = cfg.n - seq_id_base;  // 0 <=> n
        int id_counter = 0;

        std::vector<Beam> new_beams;
        for (size_t i = 0; i < beams.size(); ++i) {
            const Beam & beam = beams[i];
            for (int j = 0; j < expand_factors[i]; ++j) {
                int seq_id_offset = id_counter++;
                GGML_ASSERT(seq_id_offset < cfg.n);

                int  seq_id   = seq_id_base + seq_id_offset;
                Beam new_beam = beam.fork(seq_id);

                new_beam.step_finished = false;
                new_beam.eos_found     = false;

                llama_kv_cache_seq_cp(ctx, beam.id, new_beam.id, 0, -1);

                new_beams.emplace_back(std::move(new_beam));
            }

            llama_kv_cache_seq_rm(ctx, beam.id, 0, -1);
        }
        beams = std::move(new_beams);
    }

    auto gen_end    = chrono::system_clock::now();
    auto elapsed_ms = chrono::duration_cast<chrono::milliseconds>(gen_end - gen_start).count();

    // for (auto & beam : beams) {
    //     printf("beam %d, score %g, completion: %s\n", beam.id, beam.score, beam.completion().c_str());
    // }

    // select final sequence
    const Beam * selected_beam = nullptr;
    for (auto & beam : beams) {
        if (beam.eos_found) {
            selected_beam = &beam;
            break;
        }
    }
    if (!selected_beam) {
        selected_beam = beams.data();
    }
    // printf("final selected sequence: id %d\n", selected_beam->id);

    // collect metrics
    nlohmann::json log_dict;
    log_dict["gen_time"]        = elapsed_ms * 1e-3;
    log_dict["n_tokens"]        = selected_beam->seq_len;
    log_dict["completion"]      = selected_beam->completion();
    log_dict["step_lengths"]    = step_lengths;
    log_dict["step_tot_time"]   = step_tot_time_ms;
    log_dict["step_gen_time"]   = step_gen_time_ms;
    log_dict["step_score_time"] = step_score_time_ms;

    std::cout << log_dict << "\n";

    // llama_perf_context_print(ctx);
    // llama_perf_sampler_print(sampler);

    // release resources
    llama_batch_free(batch);
    llama_sampler_free(sampler);
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
    return 0;
}
