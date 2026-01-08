from __future__ import annotations
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = F.relu(out, inplace=True)
        return out


class ResNet18_CIFAR(nn.Module):
    """
    CIFAR ResNet-18 variant:
    - 3x3 conv stem, stride 1, padding 1
    - no initial maxpool
    - blocks [2,2,2,2]
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(512, num_blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        # FedEvo sentinel is attached later if needed.

        self._init_weights()

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides: List[int] = [stride] + [1] * (num_blocks - 1)
        blocks: List[nn.Module] = []
        for s in strides:
            blocks.append(BasicBlock(self.in_planes, planes, stride=s))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*blocks)

    def _init_weights(self) -> None:
        # Kaiming init for conv; default BN; linear init.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


class SplitResNet18CIFAR(nn.Module):
    """
    FedImpro split wrapper (after layer2).
    - low: conv1 -> bn1 -> relu -> layer1 -> layer2
    - high: layer3 -> layer4 -> avgpool -> flatten -> fc
    """
    def __init__(self, low: nn.Module, high: nn.Module) -> None:
        super().__init__()  # 부모 클래스(nn.Module) 먼저 초기화 (필수!)
        self.low = low      # 그 다음 모듈 할당
        self.high = high

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_map = self.low(x)
        logits = self.high(feat_map)
        return logits


def build_split_resnet18_cifar(model: ResNet18_CIFAR) -> SplitResNet18CIFAR:
    class Low(nn.Module):
        def __init__(self, conv1: nn.Module, bn1: nn.Module, layer1: nn.Module, layer2: nn.Module) -> None:
            super().__init__()
            self.conv1 = conv1
            self.bn1 = bn1
            self.layer1 = layer1
            self.layer2 = layer2

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = F.relu(self.bn1(self.conv1(x)), inplace=True)
            out = self.layer1(out)
            out = self.layer2(out)
            return out

    class High(nn.Module):
        def __init__(self, layer3: nn.Module, layer4: nn.Module, avgpool: nn.Module, fc: nn.Module) -> None:
            super().__init__()
            self.layer3 = layer3
            self.layer4 = layer4
            self.avgpool = avgpool
            self.fc = fc

        def forward(self, feat_map: torch.Tensor) -> torch.Tensor:
            out = self.layer3(feat_map)
            out = self.layer4(out)
            out = self.avgpool(out)
            out = torch.flatten(out, 1)
            out = self.fc(out)
            return out

    low = Low(model.conv1, model.bn1, model.layer1, model.layer2)
    high = High(model.layer3, model.layer4, model.avgpool, model.fc)
    return SplitResNet18CIFAR(low=low, high=high)


def attach_fedevo_sentinel(model: nn.Module, nu: float, sentinel_bits: torch.Tensor) -> None:
    """
    Adds model.sentinel as a requires_grad=False Parameter, not used in forward.
    sentinel_bits: tensor of shape [64] with entries in {-1, +1}.
    """
    assert sentinel_bits.shape == (64,)
    sentinel = (nu * sentinel_bits).detach().clone()
    p = nn.Parameter(sentinel, requires_grad=False)
    setattr(model, "sentinel", p)


def fc_weight_std(model: ResNet18_CIFAR) -> float:
    w = model.fc.weight.detach()
    return float(w.std().item())