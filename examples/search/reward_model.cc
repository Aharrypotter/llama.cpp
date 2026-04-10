#include "reward_model.h"

#include <stdexcept>

#include "common.h"
#include "json.hpp"
#include "llama.h"

std::string LocalPRM::build_prompt(const std::string & question, const std::vector<std::string> & steps) const {
    // example, not for any specific model
    std::string prompt = question + "\n\n";
    for (const auto & step : steps) {
        prompt.append(step);
        prompt.append("\n");
    }
    return prompt;
}

// TODO: finish implementation
std::vector<float> LocalPRM::score(const std::string &                           question,
                                   const std::vector<std::vector<std::string>> & completion_steps) {
    throw std::runtime_error("not implemented");
    
    std::vector<std::vector<llama_token>> token_seqs(completion_steps.size());

    llama_batch batch = llama_batch_init(llama_n_batch(ctx), 0, max_n_seqs * 2);

    // NOTE: intra-batch prefix sharing is disabled
    for (size_t i = 0; i < completion_steps.size(); ++i) {
        const auto & steps = completion_steps[i];

        auto prompt = build_prompt(question, steps);
        auto tokens = common_tokenize(ctx, prompt, true);

        int seq_id = seq_id_base + i;

        auto [prefix_len, src_seq_id] = trie.match_one(tokens);
        if (prefix_len == (int) tokens.size()) {  // will this happen? always keep one token to compute
            prefix_len--;
        }
        if (prefix_len > 0) {
            llama_kv_cache_seq_cp(ctx, src_seq_id, seq_id, 0, prefix_len);
        }
        // TODO: make sure at least one step tag token exist in this partial sequence
        for (size_t p = prefix_len; p < tokens.size(); ++p) {
            // TODO: only set logits=true when this token is a step marker
            common_batch_add(batch, tokens[p], p, { seq_id }, true);
        }

        token_seqs[i] = std::move(tokens);
    }

    if (llama_decode(ctx, batch)) {
        throw std::runtime_error("llama_decode failed");
    }

    // TODO: extract step scores

    // add sequences of this round to the trie
    for (size_t i = 0; i < token_seqs.size(); ++i) {
        int seq_id = seq_id_base + i;
        trie.insert(token_seqs[i], seq_id);
    }

    // remove sequences of previous round
    for (auto & [seq, seq_id] : active_seqs) {
        trie.remove(seq, seq_id);
    }
    active_seqs.clear();

    // exchange
    for (size_t i = 0; i < token_seqs.size(); ++i) {
        int seq_id = seq_id_base + i;
        active_seqs.emplace_back(std::move(token_seqs[i]), seq_id);
    }

    seq_id_base = max_n_seqs - seq_id_base;  // 0 <=> n

    llama_batch_free(batch);

    ///////
    std::vector<float> scores;
    return scores;
}

using json = nlohmann::json;

RemotePRM::RemotePRM(const std::string & endpoint, int max_n_seqs) : client(endpoint), session_id(-1) {
    client.set_read_timeout(20, 0);  // 20 seconds

    // post session create request to the backend server
    json req;
    req["max_n_seqs"] = max_n_seqs;

    auto res = client.Post("/session/create", req.dump(), "application/json");
    if (res.error() != httplib::Error::Success) {
        throw std::runtime_error("remote session create request failed");
    }

    auto res_body = json::parse(res->body);
    if (res_body["status"] != "OK") {
        throw std::runtime_error("status not OK");
    }
    session_id = res_body["session"];
}

std::vector<float> RemotePRM::score(const std::string &                           question,
                                    const std::vector<std::vector<std::string>> & completion_steps) {
    // issue incremental score request
    json req;
    req["session"]          = session_id;
    req["question"]         = question;
    req["completion_steps"] = completion_steps;

    auto res = client.Post("/session/score", req.dump(), "application/json");
    if (res.error() != httplib::Error::Success) {
        throw std::runtime_error("remote session score request failed");
    }

    auto res_body = json::parse(res->body);
    if (res_body["status"] != "OK") {
        throw std::runtime_error("status not OK");
    }
    return res_body["scores"];
}

RemotePRM::~RemotePRM() {
    // destroy remote session
    json req;
    req["session"] = session_id;

    auto res = client.Post("/session/destroy", req.dump(), "application/json");
    // TODO: warn if result not ok
}
