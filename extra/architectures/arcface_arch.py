import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

# Совместимость старого pytorch-insightface
# с современной torchvision
from torch.hub import load_state_dict_from_url

utils_module = types.ModuleType("torchvision.models.utils")
utils_module.load_state_dict_from_url = load_state_dict_from_url
sys.modules["torchvision.models.utils"] = utils_module

from insightface import iresnet50


class Embedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = iresnet50(pretrained=False)

    def forward(self, x):
        # ArcFace IResNet50 ожидает 112x112
        if x.shape[-2:] != (112, 112):
            x = F.interpolate(
                x,
                size=(112, 112),
                mode="bilinear",
                align_corners=False,
            )

        # Из [0, 1] -> [-1, 1]
        x = x * 2.0 - 1.0

        return self.model(x)

    def get_embeddings(self, x):
        with torch.no_grad():
            return self.forward(x)