import torch
import torch.nn as nn

from .TTABase import TTABase
from models.ModelForTTA import CLIPModelForTTA


class NoAdapt(TTABase):
    def __init__(self, model, datasets, args):
        super().__init__()

        self.cfg = args.config

        self.model = CLIPModelForTTA(model, datasets.classes, self.cfg, args)

    @torch.no_grad()
    def forward(self, x):
        logits = self.model(x)

        return logits

    def reset(self):
        pass
