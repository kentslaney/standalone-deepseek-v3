from collections import OrderedDict

# https://github.com/deepseek-ai/DeepSeek-V3/tree/f09f5fa/inference/configs
configs = OrderedDict([
    ("6M", {
        "dim": 64,
        "inter_dim": 128,
        "moe_inter_dim": 32,
        "n_layers": 4,
        "n_heads": 3,
        "n_routed_experts": 8,
        "max_batch_size": 128,
        "learning_rate": 1e-5,
    }), ("6M_moe_impl", {
        "dim": 64,
        "inter_dim": 128,
        "moe_inter_dim": 32,
        "n_layers": 4,
        "n_heads": 3,
        "n_routed_experts": 8,
        "max_batch_size": 128,
        "learning_rate": 1e-5,
        "moe_impl": "distribution",
    }), ("19M", {
        "dim": 128,
        "inter_dim": 256,
        "moe_inter_dim": 64,
        "n_layers": 8,
        "n_heads": 6,
        "n_routed_experts": 16,
        "max_batch_size": 64,
    }), ("19M_moe_impl", {
        "dim": 128,
        "inter_dim": 256,
        "moe_inter_dim": 64,
        "n_layers": 8,
        "n_heads": 6,
        "n_routed_experts": 16,
        "max_batch_size": 64,
        "moe_impl": "distribution",
    }), ("500M", {
        "dim": 512,
        "inter_dim": 2048,
        "moe_inter_dim": 256,
        "n_layers": 16,
        "n_heads": 8,
    }), ("16B", {
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
        "mscale": 0.707,
    }), ("236B", {
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
        "v_head_dim": 128,
    }), ("671B", {
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
        "dtype": "fp8",
    })
])

if __name__ == "__main__":
    show = ("6M", "19M", "500M", "16B", "236B", "671B")
    transposed = {
            k: [configs[i].get(k, None) for i in show]
            for k in set(sum((list(configs[i].keys()) for i in show), []))}

    from pprint import pprint
    pprint(transposed)
