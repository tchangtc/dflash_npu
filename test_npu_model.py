"""NPU 模型加载与推理测试"""
import os, sys
sys.path.insert(0, '/mnt/data/t00911745/code/dflash')

print("=" * 60)
print("DFlash NPU 模型测试")
print("=" * 60)

import torch
import torch_npu
from dflash.device import get_device_type, get_device, set_device

set_device(0)
device = get_device(0)
print(f"Device: {device} ({torch.npu.get_device_name(0)})")

# ---- 测试 1: Qwen3DFlashAttention 单元测试 ----
print("\n[1/3] Qwen3DFlashAttention 单元测试...")
try:
    from dflash.model import Qwen3DFlashAttention, apply_rotary_pos_emb
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding, Qwen3Config

    # 构造一个小型 Qwen3Config 用于测试
    config = Qwen3Config(
        hidden_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        intermediate_size=512,
        num_hidden_layers=2,
        vocab_size=1000,
        rms_norm_eps=1e-6,
        attention_bias=False,
        attention_dropout=0.0,
        sliding_window=None,
        layer_types=["full_attention", "full_attention"],
    )
    config._attn_implementation = "sdpa"

    attn = Qwen3DFlashAttention(config, layer_idx=0).to(device).eval()
    rotary = Qwen3RotaryEmbedding(config).to(device)

    bsz, q_len, ctx_len = 1, 4, 8
    hidden = torch.randn(bsz, q_len, 256, dtype=torch.bfloat16, device=device)
    target_hidden = torch.randn(bsz, ctx_len, 256, dtype=torch.bfloat16, device=device)
    # position_ids 必须覆盖 context + noise 的完整范围
    pos_ids = torch.arange(ctx_len + q_len, device=device).unsqueeze(0)

    pos_emb = rotary(hidden, pos_ids)
    out, _ = attn(
        hidden_states=hidden,
        target_hidden=target_hidden,
        position_embeddings=pos_emb,
        attention_mask=None,
    )
    print(f"  ✅ Attention forward: output shape={out.shape}")

except Exception as e:
    print(f"  ❌ Attention 测试失败: {e}")
    import traceback; traceback.print_exc()

# ---- 测试 2: DFlashDraftModel 构造与 forward ----
print("\n[2/3] DFlashDraftModel 构造与 forward...")
try:
    from dflash.model import DFlashDraftModel

    config.block_size = 4
    config.num_target_layers = 8
    config.dflash_config = {
        "target_layer_ids": [1, 3, 5, 7],
        "mask_token_id": 0,
    }

    draft = DFlashDraftModel(config).to(device).eval()
    print(f"  ✅ Draft model 构造成功")
    print(f"     layers={len(draft.layers)}, block_size={draft.block_size}")
    print(f"     target_layer_ids={draft.target_layer_ids}")
    print(f"     params={sum(p.numel() for p in draft.parameters()):,}")

    # forward pass
    noise = torch.randn(bsz, q_len, 256, dtype=torch.bfloat16, device=device)
    target_h = torch.randn(bsz, ctx_len, 4 * 256, dtype=torch.bfloat16, device=device)  # 4 layers * hidden_size
    # position_ids 覆盖 context + noise
    full_pos_ids = torch.arange(ctx_len + q_len, device=device).unsqueeze(0)

    out = draft(
        position_ids=full_pos_ids,
        noise_embedding=noise,
        target_hidden=target_h,
        use_cache=False,
    )
    print(f"  ✅ Draft forward: output shape={out.shape}")

except Exception as e:
    print(f"  ❌ Draft model 测试失败: {e}")
    import traceback; traceback.print_exc()

# ---- 测试 3: dflash_generate 完整流程（模拟 target 模型）----
print("\n[3/3] dflash_generate 模拟测试...")
try:
    from dflash.model import dflash_generate, extract_context_feature, sample
    from transformers import DynamicCache

    # 构造一个极简的 mock target 模型
    class MockTargetLayer(torch.nn.Module):
        def __init__(self, hidden_size, num_heads):
            super().__init__()
            self.attn = torch.nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        def forward(self, x):
            out, _ = self.attn(x, x, x)
            return x + out

    class MockTarget(torch.nn.Module):
        def __init__(self, vocab_size, hidden_size, num_layers, num_heads):
            super().__init__()
            self.hidden_size = hidden_size
            self.model = torch.nn.Module()
            self.model.embed_tokens = torch.nn.Embedding(vocab_size, hidden_size)
            self.model.layers = torch.nn.ModuleList([
                MockTargetLayer(hidden_size, num_heads) for _ in range(num_layers)
            ])
            self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)
            self._hidden_states_buf = None

        def forward(self, input_ids, position_ids=None, past_key_values=None,
                    use_cache=False, output_hidden_states=False, logits_to_keep=None, **kwargs):
            h = self.model.embed_tokens(input_ids)
            hidden_states_list = [h] if output_hidden_states else None
            for layer in self.model.layers:
                h = layer(h)
                if hidden_states_list is not None:
                    hidden_states_list.append(h)

            logits = self.lm_head(h)
            if logits_to_keep is not None:
                logits = logits[:, -logits_to_keep:, :]

            result = torch.nn.Module()
            result.logits = logits
            if output_hidden_states:
                result.hidden_states = hidden_states_list
            return result

        @property
        def device(self):
            return next(self.parameters()).device

    vocab_size = 1000
    hidden_size = 256
    num_target_layers = 8
    num_heads = 4

    target = MockTarget(vocab_size, hidden_size, num_target_layers, num_heads).to(device).eval()
    print(f"  ✅ Mock target 构造成功")

    # 重新配置 draft
    config2 = Qwen3Config(
        hidden_size=hidden_size,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        intermediate_size=512,
        num_hidden_layers=2,
        vocab_size=vocab_size,
        rms_norm_eps=1e-6,
        attention_bias=False,
        attention_dropout=0.0,
        sliding_window=None,
        layer_types=["full_attention", "full_attention"],
    )
    config2._attn_implementation = "sdpa"
    config2.block_size = 4
    config2.num_target_layers = num_target_layers
    config2.dflash_config = {
        "target_layer_ids": [1, 3, 5, 7],
        "mask_token_id": 0,
    }
    draft2 = DFlashDraftModel(config2).to(device).eval()

    # 运行生成
    input_ids = torch.randint(0, vocab_size, (1, 8), device=device)
    output = dflash_generate(
        draft2,
        target=target,
        input_ids=input_ids,
        max_new_tokens=16,
        stop_token_ids=[2],
        temperature=0.0,
        block_size=4,
        mask_token_id=0,
        return_stats=True,
    )
    print(f"  ✅ dflash_generate 成功!")
    print(f"     input tokens: {output.num_input_tokens}")
    print(f"     output tokens: {output.num_output_tokens}")
    print(f"     TTFT: {output.time_to_first_token*1000:.1f}ms")
    print(f"     TPOT: {output.time_per_output_token*1000:.1f}ms")
    print(f"     acceptance lengths: {output.acceptance_lengths}")

except Exception as e:
    print(f"  ❌ dflash_generate 测试失败: {e}")
    import traceback; traceback.print_exc()

print("\n" + "=" * 60)
print("模型测试完成")
print("=" * 60)
