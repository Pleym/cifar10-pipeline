import torch
import torch.nn as nn
import argparse
import sys
import os
import mlflow
import mlflow.pytorch

# Make project src importable when running scripts/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.model import SimpleCNN
from src.data.dataset import create_dataloaders
from src.utils.config import load_config

def _resolve_model_class(model_name: str):
    """Return a model class for a given config name."""
    if model_name == "SimpleCNN":
        return SimpleCNN
    if model_name == "DeepCNN":
        # Lazy import: try src.models first, then src.model_zoo
        try:
            from src.models.deep_cnn import DeepCNN  # type: ignore
            return DeepCNN
        except Exception:
            from src.model_zoo.deep_cnn import DeepCNN  # type: ignore
            return DeepCNN
    raise ValueError(f"Unknown model.name '{model_name}'. Use 'SimpleCNN' or 'DeepCNN'.")


def _topk_acc(logits: torch.Tensor, targets: torch.Tensor, k: int = 5) -> float:
    """Compute top-k accuracy; returns NaN if k > num_classes."""
    if logits.size(1) < k:
        return float('nan')
    _, pred = logits.topk(k, dim=1)
    return pred.eq(targets.view(-1, 1)).any(dim=1).float().mean().item()


def _train_one_epoch(model, loader, optimizer, criterion, device, epoch, log_interval):
    """Train for a single epoch and return (loss, accuracy). Also logs step loss to MLflow."""
    model.train()
    running_loss, running_correct, total = 0.0, 0, 0
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        bs = data.size(0)
        running_loss += loss.item() * bs
        running_correct += (output.argmax(dim=1) == target).sum().item()
        total += bs

        if batch_idx % log_interval == 0:
            step = epoch * len(loader) + batch_idx
            mlflow.log_metric("train_loss", loss.item(), step=step)

    return running_loss / max(total, 1), running_correct / max(total, 1)


def _evaluate(model, loader, criterion, device):
    """Evaluate on a loader and return (val_loss, val_acc, val_top5_acc_or_nan)."""
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    top5_vals = []
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            val_loss += loss.item() * data.size(0)
            val_correct += (output.argmax(dim=1) == target).sum().item()
            val_total += data.size(0)
            acc5 = _topk_acc(output, target, k=5)
            if acc5 == acc5:  # not NaN
                top5_vals.append(acc5)

    val_loss /= max(val_total, 1)
    val_acc = val_correct / max(val_total, 1)
    val_top5 = (sum(top5_vals) / len(top5_vals)) if top5_vals else float('nan')
    return val_loss, val_acc, val_top5


def train_model(config_path: str = "configs/model.yaml"):
    # Load config and device
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model from config
    model_cfg = dict(config.model.__dict__)
    model_name = model_cfg.pop("name", "DeepCNN")
    model_cls = _resolve_model_class(model_name)
    model = model_cls(**model_cfg).to(device)

    # Data, optimizer, loss
    train_loader, test_loader = create_dataloaders(config.data.data_path, config.training.batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # MLflow setup
    suffix = "_simple" if model_name == "SimpleCNN" else "_deep"
    mlflow.set_experiment(config.logging.experiment_name + suffix)
    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_params(config.training.__dict__)
        mlflow.log_params(config.model.__dict__)

        for epoch in range(config.training.num_epochs):
            # Train
            tr_loss, tr_acc = _train_one_epoch(
                model, train_loader, optimizer, criterion, device, epoch, config.logging.log_interval
            )
            mlflow.log_metric("epoch_train_loss", tr_loss, step=epoch)
            mlflow.log_metric("epoch_train_acc", tr_acc, step=epoch)
            mlflow.log_metric("lr", optimizer.param_groups[0]["lr"], step=epoch)

            # Validate
            val_loss, val_acc, val_top5 = _evaluate(model, test_loader, criterion, device)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_acc", val_acc, step=epoch)
            if val_top5 == val_top5:  # not NaN
                mlflow.log_metric("val_top5_acc", val_top5, step=epoch)

            print(
                f"Epoch {epoch+1}/{config.training.num_epochs} | "
                f"train_loss: {tr_loss:.4f}, train_acc: {tr_acc:.4f} | "
                f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}"
            )

        # Save model artifact with distinct name
        mlflow_path = "model_simple" if model_name == "SimpleCNN" else "model_deep"
        mlflow.pytorch.log_model(model, mlflow_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CIFAR-10 model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/model.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()
    train_model(args.config)