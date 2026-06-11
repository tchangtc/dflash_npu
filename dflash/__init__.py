__all__ = [
    "DFlashDraftModel",
    "extract_context_feature",
    "get_device",
    "get_device_type",
    "get_dist_backend",
    "load_and_process_dataset",
    "manual_seed_all",
    "sample",
    "set_device",
    "synchronize",
]


def __getattr__(name):
    if name == "load_and_process_dataset":
        from .benchmark import load_and_process_dataset

        return load_and_process_dataset

    if name in {"DFlashDraftModel", "extract_context_feature", "sample"}:
        # 根据设备类型选择导入 CUDA 或 NPU 版本
        from .device import get_device_type
        if get_device_type() == "npu":
            from .model_npu import DFlashDraftModel, sample
            from .model import extract_context_feature
        else:
            from .model import DFlashDraftModel, extract_context_feature, sample

        return {
            "DFlashDraftModel": DFlashDraftModel,
            "extract_context_feature": extract_context_feature,
            "sample": sample,
        }[name]

    if name in {
        "get_device",
        "get_device_type",
        "get_dist_backend",
        "manual_seed_all",
        "set_device",
        "synchronize",
    }:
        from . import device

        return getattr(device, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
