#pragma once

#include <map>
#include <memory>
#include <set>
#include <vector>

template <typename T> class Trie {
    struct Node {
        std::map<T, std::unique_ptr<Node>> children;
        std::set<int>                      seq_ids;
    };

  private:
    std::unique_ptr<Node> root;
    std::set<int>         all_seq_ids;

    // TODO: non-recursive version
    // returns whether we should remove the child node
    bool remove_aux(Node * node, const std::vector<T> & seq, int seq_id, int pos) {
        if (!node) {
            return false;
        }

        if (pos == (int) seq.size()) {
            if (!node->seq_ids.count(seq_id)) {
                return false;
            }
            node->seq_ids.erase(seq_id);

            return node->seq_ids.empty() && node->children.empty();
        }

        const T & v  = seq[pos];
        auto      it = node->children.find(v);
        if (it == node->children.end()) {
            return false;
        }

        bool should_remove_child = remove_aux(it->second.get(), seq, seq_id, pos + 1);
        if (should_remove_child) {
            node->children.erase(it);
            return node->children.empty() && node->seq_ids.empty();
        }
        return false;
    }

  public:
    Trie() : root{ std::make_unique<Node>() } {}

    void insert(const std::vector<T> & seq, int seq_id) {
        Node * p = root.get();
        for (auto & v : seq) {
            if (!p->children.count(v)) {
                p->children[v] = std::make_unique<Node>();
            }
            p->seq_ids.insert(seq_id);
            p = p->children.at(v).get();
        }
        all_seq_ids.insert(seq_id);
    }

    void remove(const std::vector<T> & seq, int seq_id) {
        remove_aux(root.get(), seq, seq_id, 0);
        all_seq_ids.erase(seq_id);
    }

    // returns {length of the longest common prefix, one matched sequence id}
    std::pair<int, int> match_one(const std::vector<T> & seq) const {
        int    i;
        Node * p = root.get();
        for (i = 0; i < (int) seq.size(); ++i) {
            auto it = p->children.find(seq[i]);
            if (it == p->children.end()) {
                break;
            }
            p = it->second.get();
        }
        int seq_id = p->seq_ids.empty() ? -1 : *(p->seq_ids.begin());
        return { i, seq_id };
    }
};
