"""NPU 适配测试脚本"""
import sys
sys.path.insert(0, '/mnt/data/t00911745/code/dflash')

print("=" * 60)
print("DFlash NPU 适配测试")
print("=" * 60)

# 1. 环境检查
print("\n[1/5] 环境检查...")
import torch
import torch_npu
print(f"  torch: {torch.__version__}")
print(f"  torch_npu: {torch_npu.__version__}")
print(f"  NPU available: {torch.npu.is_available()}")
print(f"  NPU count: {torch.npu.device_count()}")
if torch.npu.is_available():
    print(f"  Current device: {torch.npu.current_device()}")
    print(f"  Device name: {torch.npu.get_device_name(0)}")

# 2. 导入 DFlash 模型
print("\n[2/5] 导入 DFlash 模型...")
try:
    from dflash.model import DFlashDraftModel, build_target_layer_ids
    from dflash.device import get_device_type, get_device
    print("  ✅ 导入成功")
    print(f"  Device type: {get_device_type()}")
except Exception as e:
    print(f"  ❌ 导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 3. 测试基础张量操作
print("\n[3/5] 测试基础张量操作...")
try:
    device = get_device(0)
    print(f"  Using device: {device}")

    # 张量创建和移动
    x = torch.randn(2, 3, dtype=torch.bfloat16).to(device)
    print(f"  ✅ 张量创建: {x.shape}, dtype={x.dtype}, device={x.device}")

    # Softmax
    y = torch.softmax(x, dim=-1)
    print(f"  ✅ Softmax: {y.shape}")

    # Argmax
    z = torch.argmax(x, dim=-1)
    print(f"  ✅ Argmax: {z.shape}")

    # Concat
    a = torch.randn(1, 8, 4, 64, dtype=torch.bfloat16).to(device)
    b = torch.randn(1, 8, 2, 64, dtype=torch.bfloat16).to(device)
    c = torch.cat([a, b], dim=2)
    print(f"  ✅ Cat: {c.shape}")

    # DynamicCache
    from transformers import DynamicCache
    cache = DynamicCache()
    k_new = torch.randn(1, 8, 4, 64, dtype=torch.bfloat16).to(device)
    v_new = torch.randn(1, 8, 4, 64, dtype=torch.bfloat16).to(device)
    cache.update(k_new, v_new, 0)
    print(f"  ✅ DynamicCache update: seq_len={cache.get_seq_length()}")

    cache.crop(2)
    print(f"  ✅ DynamicCache crop: seq_len={cache.get_seq_length()}")

except Exception as e:
    print(f"  ❌ 基础操作失败: {e}")
    import traceback
    traceback.print_exc()

# 4. 测试 SDPA 注意力（非因果）
print("\n[4/5] 测试 SDPA 注意力（非因果）...")
try:
    q = torch.randn(1, 8, 16, 64, dtype=torch.bfloat16).to(device)
    k = torch.randn(1, 8, 32, 64, dtype=torch.bfloat16).to(device)
    v = torch.randn(1, 8, 32, 64, dtype=torch.bfloat16).to(device)

    # 非因果注意力（无 mask）
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
    print(f"  ✅ SDPA (is_causal=False): {out.shape}")

    # 非因果注意力（显式全 0 mask）
    mask = torch.zeros(1, 1, 16, 32, dtype=torch.bfloat16).to(device)
    out2 = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    print(f"  ✅ SDPA (explicit mask): {out2.shape}")

    # 验证两者结果一致
    diff = (out - out2).abs().max().item()
    print(f"  差异: {diff:.6f} ({'一致' if diff < 1e-4 else '不一致'})")

except Exception as e:
    print(f"  ❌ SDPA 失败: {e}")
    import traceback
    traceback.print_exc()

# 5. 测试 multinomial 采样
print("\n[5/5] 测试 multinomial 采样...")
try:
    probs = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float32).to(device)
    result = torch.multinomial(probs, num_samples=1)
    print(f"  ✅ Multinomial on NPU: {result.item()}")
except Exception as e:
    print(f"  ⚠️ Multinomial on NPU 失败: {e}")
    print("  尝试 CPU 回退...")
    try:
        result = torch.multinomial(probs.cpu(), num_samples=1).to(device)
        print(f"  ✅ CPU 回退成功: {result.item()}")
    except Exception as e2:
        print(f"  ❌ CPU 回退也失败: {e2}")

print("\n" + "=" * 60)
print("基础测试完成")
print("=" * 60)
