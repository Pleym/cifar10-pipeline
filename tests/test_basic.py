import os
import sys
import torch
import importlib.util
import pytest

# Compute project root (directory containing 'scripts' and 'src')
THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))

# Ensure project root is on sys.path so that 'src.*' imports work
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Load training module from file path to avoid package import issues
train_py = os.path.join(PROJECT_ROOT, "scripts", "train.py")
spec = importlib.util.spec_from_file_location("train_module", train_py)
assert spec and spec.loader, f"Cannot load module spec from {train_py}"
train_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_mod)

_resolve_model_class = getattr(train_mod, "_resolve_model_class")
_topk_acc = getattr(train_mod, "_topk_acc")

# Import DeepCNN directly from src.model_zoo
from src.model_zoo.deep_cnn import DeepCNN


def make_deepcnn(num_classes: int = 10) -> torch.nn.Module:
    return DeepCNN(
        num_classes=num_classes,
        dropout=0.1,
        conv_layers=[
            {"channels": 32, "kernel_size": 3, "stride": 1, "padding": 1},
            {"channels": 64, "kernel_size": 3, "stride": 1, "padding": 1},
            {"channels": 64, "kernel_size": 3, "stride": 1, "padding": 1},
        ],
        fc_layers=[{"size": 128}],
        in_channels=3,
        pool_every=2,
        use_batchnorm=True,
    )


def test_deepcnn_forward_shape():
    model = make_deepcnn(num_classes=10)
    x = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (4, 10)


def test_model_resolver_returns_classes():
    cls_simple = _resolve_model_class("SimpleCNN")
    cls_deep = _resolve_model_class("DeepCNN")
    # They should be classes (subclasses of nn.Module)
    assert isinstance(cls_simple, type)
    assert isinstance(cls_deep, type)


def test_topk_acc_basic_cases():
    # Batch of 3, 5 classes
    logits = torch.tensor([
        [10.0, 0.1, 0.2, 0.3, 0.4],  # pred=0, target=0 -> correct
        [0.1, 0.2, 5.0, 0.3, 0.4],    # pred=2, target=2 -> correct
        [0.1, 0.2, 0.3, 4.0, 0.4],    # pred=3, target=1 -> wrong for top1, correct for top2 if 1 is second
    ])
    targets = torch.tensor([0, 2, 1])

    acc1 = _topk_acc(logits, targets, k=1)
    assert pytest.approx(acc1, rel=0, abs=1e-6) == 2/3

    acc2 = _topk_acc(logits, targets, k=2)
    assert pytest.approx(acc2, rel=0, abs=1e-6) == 1.0

    # k > num_classes -> NaN
    acc_bigk = _topk_acc(logits, targets, k=10)
    assert acc_bigk != acc_bigk  # NaN check
