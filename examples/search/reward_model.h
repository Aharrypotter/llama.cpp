#pragma once

#include <string>
#include <vector>

#include "httplib.h"
#include "llama.h"
#include "trie.h"

struct PRM {
    virtual std::vector<float> score(const std::string &                           question,
                                     const std::vector<std::vector<std::string>> & completion_steps) = 0;

    virtual ~PRM() {}
};

struct LocalPRM : PRM {
    llama_context * ctx;  // own this llama_context?

    Trie<llama_token> trie;

    std::vector<std::pair<std::vector<llama_token>, int>> active_seqs;

    int max_n_seqs;
    int seq_id_base;  // ping-pong buffer

    LocalPRM(llama_context * lctx, int n_seqs) : ctx(lctx), max_n_seqs(n_seqs), seq_id_base(0) {}

    virtual std::string        build_prompt(const std::string & question, const std::vector<std::string> & steps) const;
    virtual std::vector<float> score(const std::string &                           question,
                                     const std::vector<std::vector<std::string>> & completion_steps);
};

struct RemotePRM : PRM {
    httplib::Client client;
    int             session_id;

    RemotePRM(const std::string & endpoint, int n_seqs);
    ~RemotePRM();

    std::vector<float> score(const std::string &                           question,
                             const std::vector<std::vector<std::string>> & completion_steps);
};
