#!/bin/bash
set -e
MODEL=/Users/hoyi/paper/real_model/Qwen3-1.7B-Q4_K_M.gguf
PROMPT_FILE=/Users/hoyi/paper/llama.cpp/tests/phase5_prompt.txt
BIN=/Users/hoyi/paper/llama.cpp/build/bin

python3 -c "print('The quick brown fox jumps over the lazy dog. ' * 50)" > "$PROMPT_FILE"

echo "=== dense baseline ==="
"$BIN/llama-perplexity" -m "$MODEL" -f "$PROMPT_FILE" -c 128 -ngl 0 -t 4 2>&1 | tee /tmp/ppl_dense.log

echo "=== xKV backend ==="
"$BIN/llama-perplexity" -m "$MODEL" -f "$PROMPT_FILE" -c 128 -ngl 0 -t 4 \
  --kv-blocksvd --kv-blocksvd-backend --kv-blocksvd-cross-layer \
  --kv-blocksvd-block-size 16 --kv-blocksvd-rank 8 2>&1 | tee /tmp/ppl_xkv.log

echo "dense ppl:" $(grep -oE 'Final estimate: PPL = [0-9.]+' /tmp/ppl_dense.log | tail -1)
echo "xkv   ppl:" $(grep -oE 'Final estimate: PPL = [0-9.]+' /tmp/ppl_xkv.log | tail -1)
