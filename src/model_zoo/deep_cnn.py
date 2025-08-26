import torch
import torch.nn as nn
from typing import List, Dict, Any


class DeepCNN(nn.Module):
    """
    Deeper configurable CNN built from YAML config fields:
      - num_classes: int
      - dropout: float
      - conv_layers: List[{channels, kernel_size, stride, padding}]
      - fc_layers: List[{size}]

    Design:
      - Repeats Conv2d -> BatchNorm2d -> ReLU blocks per conv_layers item
      - Applies MaxPool2d(kernel_size=2) every 2 conv blocks to downsample
      - Uses AdaptiveAvgPool2d((1,1)) to avoid manual flatten size calculation
      - Classifier: Linear stacks from pooled channels -> fc sizes -> num_classes
    """

    def __init__(
        self,
        num_classes: int,
        dropout: float,
        conv_layers: List[Dict[str, Any]],
        fc_layers: List[Dict[str, Any]],
        # Optional config extensions with safe defaults
        in_channels: int = 3,
        pool_every: int = 2,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        assert len(conv_layers) > 0, "conv_layers must contain at least one layer definition"

        layers: List[nn.Module] = []
        current_c = in_channels
        conv_count = 0
        for i, spec in enumerate(conv_layers):
            out_c = int(spec.get("channels", 64))
            k = int(spec.get("kernel_size", 3))
            s = int(spec.get("stride", 1))
            p = int(spec.get("padding", 1))

            layers.append(nn.Conv2d(current_c, out_c, kernel_size=k, stride=s, padding=p, bias=not use_batchnorm))
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout2d(p=dropout))

            conv_count += 1
            current_c = out_c

            # Downsample every 'pool_every' conv blocks
            if pool_every > 0 and conv_count % pool_every == 0:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.features = nn.Sequential(*layers)
        # Normalize spatial dimension to 1x1 so classifier input is just channel count
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Build classifier from last channels -> fc sizes -> num_classes
        classifier_layers: List[nn.Module] = []
        in_dim = current_c
        for j, spec in enumerate(fc_layers):
            size = int(spec.get("size", 256))
            classifier_layers.append(nn.Linear(in_dim, size))
            classifier_layers.append(nn.ReLU(inplace=True))
            classifier_layers.append(nn.Dropout(p=dropout))
            in_dim = size
        classifier_layers.append(nn.Linear(in_dim, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
