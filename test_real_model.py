"""真实模型端到端测试：Qwen3-8B + Qwen3-8B-DFlash-b16 on NPU"""
import os, sys, time
sys.path.insert(0, '/mnt/data/t00911745/code/dflash')

print("=" * 70)
print("DFlash NPU 真实模型端到端测试")
print("=" * 70)

import torch
import torch_npu
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from dflash.model import DFlashDraftModel, dflash_generate
from dflash.device import get_device, set_device, synchronize

# ---- 配置 ----
TARGET_MODEL_PATH = "/mnt/data/models/Qwen/Qwen3-8B"
DRAFT_MODEL_PATH = "/mnt/data/models/z-lab/Qwen3-8B-DFlash-b16"
DEVICE_ID = 0
DTYPE = torch.bfloat16

set_device(DEVICE_ID)
device = get_device(DEVICE_ID)
print(f"Device: {device} ({torch.npu.get_device_name(DEVICE_ID)})")

# ---- 1. 加载 tokenizer ----
print("\n[1/5] 加载 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL_PATH, trust_remote_code=True)
print(f"  ✅ Tokenizer loaded: vocab_size={tokenizer.vocab_size}")

# ---- 2. 加载 target 模型 ----
print("\n[2/5] 加载 target 模型 (Qwen3-8B)...")
t0 = time.time()
target = AutoModelForCausalLM.from_pretrained(
    TARGET_MODEL_PATH,
    torch_dtype=DTYPE,
    trust_remote_code=True,
    attn_implementation="sdpa",
)
target = target.to(device).eval()
load_time_target = time.time() - t0
num_params_target = sum(p.numel() for p in target.parameters())
print(f"  ✅ Target model loaded in {load_time_target:.1f}s")
print(f"     Parameters: {num_params_target:,} ({num_params_target/1e9:.2f}B)")
print(f"     Device: {next(target.parameters()).device}")

# ---- 3. 加载 draft 模型 ----
print("\n[3/5] 加载 draft 模型 (Qwen3-8B-DFlash-b16)...")
t0 = time.time()
draft = DFlashDraftModel.from_pretrained(
    DRAFT_MODEL_PATH,
    torch_dtype=DTYPE,
    trust_remote_code=True,
    attn_implementation="sdpa",
)
draft = draft.to(device).eval()
load_time_draft = time.time() - t0
num_params_draft = sum(p.numel() for p in draft.parameters())
print(f"  ✅ Draft model loaded in {load_time_draft:.1f}s")
print(f"     Parameters: {num_params_draft:,} ({num_params_draft/1e6:.1f}M)")
print(f"     Block size: {draft.block_size}")
print(f"     Target layer IDs: {draft.target_layer_ids}")

# ---- 4. Baseline 生成（无推测解码）----
print("\n[4/5] Baseline 生成 (block_size=1, 无推测解码)...")
prompts = [
    "请解释什么是快速排序算法，并给出一个 Python 实现。",
    "计算 1 到 100 之间所有偶数的和。",
]

for i, prompt in enumerate(prompts):
    print(f"\n  --- Prompt {i+1}: {prompt[:50]}... ---")
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    # Baseline
    result_baseline = dflash_generate(
        draft,
        target=target,
        input_ids=input_ids,
        max_new_tokens=64,
        stop_token_ids=[tokenizer.eos_token_id],
        temperature=0.0,
        block_size=1,
        return_stats=True,
    )

    output_text_bl = tokenizer.decode(
        result_baseline.output_ids[0, result_baseline.num_input_tokens:],
        skip_special_tokens=True,
    )
    print(f"  [Baseline]")
    print(f"    Tokens: {result_baseline.num_output_tokens}")
    print(f"    TTFT: {result_baseline.time_to_first_token*1000:.1f}ms")
    print(f"    TPOT: {result_baseline.time_per_output_token*1000:.1f}ms")
    print(f"    Throughput: {1/result_baseline.time_per_output_token:.2f} tok/s")
    print(f"    Output: {output_text_bl[:100]}...")

# ---- 5. DFlash 推测解码生成 ----
print("\n[5/5] DFlash 推测解码生成...")
for i, prompt in enumerate(prompts):
    print(f"\n  --- Prompt {i+1}: {prompt[:50]}... ---")
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    # DFlash
    result_dflash = dflash_generate(
        draft,
        target=target,
        input_ids=input_ids,
        max_new_tokens=64,
        stop_token_ids=[tokenizer.eos_token_id],
        temperature=0.0,
        block_size=draft.block_size,
        return_stats=True,
    )

    output_text_df = tokenizer.decode(
        result_dflash.output_ids[0, result_dflash.num_input_tokens:],
        skip_special_tokens=True,
    )

    # 计算接受率统计
    accept_lengths = result_dflash.acceptance_lengths
    mean_accept = sum(accept_lengths) / len(accept_lengths) if accept_lengths else 0
    max_accept = max(accept_lengths) if accept_lengths else 0

    print(f"  [DFlash block_size={draft.block_size}]")
    print(f"    Tokens: {result_dflash.num_output_tokens}")
    print(f"    TTFT: {result_dflash.time_to_first_token*1000:.1f}ms")
    print(f"    TPOT: {result_dflash.time_per_output_token*1000:.1f}ms")
    print(f"    Throughput: {1/result_dflash.time_per_output_token:.2f} tok/s")
    print(f"    Mean acceptance length: {mean_accept:.2f}")
    print(f"    Max acceptance length: {max_accept}")
    print(f"    Acceptance lengths: {accept_lengths[:20]}...")
    print(f"    Output: {output_text_df[:100]}...")

    # 对比 baseline 和 DFlash 的输出一致性
    if i == 0:
        print(f"\n  [一致性检查]")
        print(f"    Baseline output == DFlash output: {output_text_bl == output_text_df}")

print("\n" + "=" * 70)
print("测试完成!")
print("=" * 70)
