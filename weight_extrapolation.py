# https://github.com/deepseek-ai/DeepSeek-V3/tree/f09f5fa/inference/configs
configs = [
    {
        "vocab_size": 102400,
        "dim": 2048,
        "inter_dim": 10944,
        "moe_inter_dim": 1408,
        "n_layers": 27,
        "n_dense_layers": 1,
        "n_heads": 16,
        "n_routed_experts": 64,
        "n_shared_experts": 2,
        "n_activated_experts": 6,
        "route_scale": 1.0,
        "q_lora_rank": 0,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "mscale": 0.707
    },
    {
        "vocab_size": 102400,
        "dim": 5120,
        "inter_dim": 12288,
        "moe_inter_dim": 1536,
        "n_layers": 60,
        "n_dense_layers": 1,
        "n_heads": 128,
        "n_routed_experts": 160,
        "n_shared_experts": 2,
        "n_activated_experts": 6,
        "n_expert_groups": 8,
        "n_limited_groups": 3,
        "route_scale": 16.0,
        "q_lora_rank": 1536,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128
    },
    {
        "vocab_size": 129280,
        "dim": 7168,
        "inter_dim": 18432,
        "moe_inter_dim": 2048,
        "n_layers": 61,
        "n_dense_layers": 3,
        "n_heads": 128,
        "n_routed_experts": 256,
        "n_shared_experts": 1,
        "n_activated_experts": 8,
        "n_expert_groups": 8,
        "n_limited_groups": 4,
        "route_scale": 2.5,
        "score_func": "sigmoid",
        "q_lora_rank": 1536,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "dtype": "fp8"
    },
]

transposed = {
        k: [i.get(k, None) for i in configs]
        for k in set(sum((list(i.keys()) for i in configs), []))}

from pprint import pprint
pprint(transposed)
