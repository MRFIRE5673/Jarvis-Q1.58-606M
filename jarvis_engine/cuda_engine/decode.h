#pragma once
// =============================================================================
// decode.h — Native CUDA Single-Token Decode & Generation Engine
// =============================================================================
#include <vector>
#include <functional>

void init_decode_engine();
void reset_decode_state();
int  decode_step_native(int input_tok, int pos, float temperature, float top_p, int top_k, float rep_penalty = 1.15f, bool update_recent = true);
std::vector<int> generate_native(
    const std::vector<int>& prompt,
    int max_new_tokens,
    float temperature,
    float top_p,
    int top_k,
    std::function<void(int)> token_callback = nullptr,
    float rep_penalty = 1.15f
);
