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
        # All implementations are now in model.py with device-specific branches
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
