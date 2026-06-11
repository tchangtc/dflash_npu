"""KV Cache 性能分析脚本"""
import sys
sys.path.insert(0, '/mnt/data/t00911745/code/dflash')

print("=" * 70)
print("NPU KV Cache 性能分析")
print("=" * 70)

import torch
import torch_npu
import time
from transformers import DynamicCache
from dflash.device import get_device, set_device

set_device(0)
device = get_device(0)
print(f"Device: {device}\n")

# 测试参数
batch_size = 1
num_heads = 32
head_dim = 128
seq_lengths = [64, 128, 256, 512, 1024, 2048]
num_layers = 36  # Qwen3-8B 层数

print(f"测试配置: batch={batch_size}, heads={num_heads}, head_dim={head_dim}, layers={num_layers}")
print("-" * 70)

# 1. DynamicCache 基础操作性能
print("\n[1] DynamicCache 操作延迟分析")
print("-" * 70)

for seq_len in seq_lengths:
    cache = DynamicCache()

    # 预热
    for layer_idx in range(num_layers):
        k = torch.randn(batch_size, num_heads, seq_len, head_dim,
                       dtype=torch.bfloat16, device=device)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim,
                       dtype=torch.bfloat16, device=device)
        cache.update(k, v, layer_idx)
    torch.npu.synchronize()

    # 测试 update 性能
    n_iter = 100
    torch.npu.synchronize()
    t0 = time.time()
    for _ in range(n_iter):
        cache_test = DynamicCache()
        for layer_idx in range(num_layers):
            k = torch.randn(batch_size, num_heads, 1, head_dim,
                           dtype=torch.bfloat16, device=device)
            v = torch.randn(batch_size, num_heads, 1, head_dim,
                           dtype=torch.bfloat16, device=device)
            cache_test.update(k, v, layer_idx)
    torch.npu.synchronize()
    update_time = (time.time() - t0) / n_iter * 1000

    # 测试 crop 性能
    cache_full = DynamicCache()
    for layer_idx in range(num_layers):
        k = torch.randn(batch_size, num_heads, seq_len, head_dim,
                       dtype=torch.bfloat16, device=device)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim,
                       dtype=torch.bfloat16, device=device)
        cache_full.update(k, v, layer_idx)
    torch.npu.synchronize()

    t0 = time.time()
    for _ in range(n_iter):
        cache_full.crop(seq_len // 2)
        cache_full.crop(seq_len)  # 恢复
    torch.npu.synchronize()
    crop_time = (time.time() - t0) / n_iter * 1000

    # 测试 get_seq_length 性能
    t0 = time.time()
    for _ in range(n_iter * 10):
        _ = cache.get_seq_length()
    torch.npu.synchronize()
    get_len_time = (time.time() - t0) / (n_iter * 10) * 1000

    print(f"  seq_len={seq_len:4d}: update={update_time:6.2f}ms, "
          f"crop={crop_time:6.2f}ms, get_seq_length={get_len_time:.3f}ms")

# 2. 内存占用分析
print("\n[2] KV Cache 内存占用分析")
print("-" * 70)

for seq_len in seq_lengths:
    cache = DynamicCache()
    torch.npu.synchronize()

    # 记录初始内存
    torch.npu.reset_peak_memory_stats(device)
    mem_before = torch.npu.memory_allocated(device) / 1024**3

    # 填充 cache
    for layer_idx in range(num_layers):
        k = torch.randn(batch_size, num_heads, seq_len, head_dim,
                       dtype=torch.bfloat16, device=device)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim,
                       dtype=torch.bfloat16, device=device)
        cache.update(k, v, layer_idx)

    torch.npu.synchronize()
    mem_after = torch.npu.memory_allocated(device) / 1024**3
    peak_mem = torch.npu.max_memory_allocated(device) / 1024**3

    cache_mem = mem_after - mem_before
    theoretical = (batch_size * num_heads * seq_len * head_dim * 2 *
                   num_layers * 2) / 1024**3  # 2 for K+V, 2 for bfloat16 bytes

    print(f"  seq_len={seq_len:4d}: cache={cache_mem:.3f}GB, "
          f"theoretical={theoretical:.3f}GB, peak={peak_mem:.3f}GB")

    del cache
    torch.npu.empty_cache()

# 3. DFlash 推测解码中的 cache 使用模式分析
print("\n[3] DFlash 推测解码中的 KV Cache 使用模式")
print("-" * 70)

from dflash.model import DFlashDraftModel, dflash_generate
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载小型测试模型
print("  加载 Qwen3-8B 和 DFlash draft 模型...")
target = AutoModelForCausalLM.from_pretrained(
    "/mnt/data/models/Qwen/Qwen3-8B",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="sdpa",
).to(device).eval()

draft = DFlashDraftModel.from_pretrained(
    "/mnt/data/models/z-lab/Qwen3-8B-DFlash-b16",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="sdpa",
).to(device).eval()

tokenizer = AutoTokenizer.from_pretrained("/mnt/data/models/Qwen/Qwen3-8B")

# 准备输入
prompt = "请解释什么是快速排序算法。"
messages = [{"role": "user", "content": prompt}]
input_text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
)
input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

print(f"  输入长度: {input_ids.shape[1]} tokens")

# 监控内存使用
torch.npu.reset_peak_memory_stats(device)
mem_before = torch.npu.memory_allocated(device) / 1024**3

# 运行生成
result = dflash_generate(
    draft,
    target=target,
    input_ids=input_ids,
    max_new_tokens=64,
    stop_token_ids=[tokenizer.eos_token_id],
    temperature=0.0,
    block_size=draft.block_size,
    return_stats=True,
)

torch.npu.synchronize()
mem_after = torch.npu.memory_allocated(device) / 1024**3
peak_mem = torch.npu.max_memory_allocated(device) / 1024**3

print(f"\n  生成统计:")
print(f"    输出 tokens: {result.num_output_tokens}")
print(f"    平均接受长度: {sum(result.acceptance_lengths)/len(result.acceptance_lengths):.2f}")
print(f"    最大接受长度: {max(result.acceptance_lengths)}")
print(f"    TPOT: {result.time_per_output_token*1000:.2f}ms")
print(f"    吞吐量: {1/result.time_per_output_token:.2f} tok/s")
print(f"\n  内存统计:")
print(f"    生成前: {mem_before:.3f}GB")
print(f"    生成后: {mem_after:.3f}GB")
print(f"    峰值: {peak_mem:.3f}GB")
print(f"    增量: {mem_after - mem_before:.3f}GB")

# 4. 优化建议
print("\n[4] NPU KV Cache 优化建议")
print("-" * 70)
print("""
  基于以上分析，NPU 上的 KV Cache 优化方向：

  1. 预分配策略 (Pre-allocation)
     - 当前：DynamicCache 动态增长，频繁分配/释放
     - 优化：预估最大序列长度，一次性分配
     - 预期收益：减少 30-50% 的 update 开销

  2. 减少 crop 操作
     - 当前：每次推测解码后都 crop cache
     - 优化：使用滑动窗口或标记机制，避免物理裁剪
     - 预期收益：减少 20-40% 的 cache 管理开销

  3. NPU 特定优化
     - 使用 torch_npu 的 pinned memory 减少 H2D 传输
     - 探索 NPU 的 KV Cache 专用算子（如果 torch_npu 2.5+ 支持）
     - 使用 NPU 的 graph mode 减少 kernel launch 开销

  4. 分层缓存
     - 将频繁访问的 cache 保留在 NPU HBM
     - 较少访问的 cache offload 到 CPU 内存
     - 需要 NPU 支持 async copy 操作
""")

print("=" * 70)
print("KV Cache 分析完成")
print("=" * 70)
