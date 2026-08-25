import torch
import torch.nn as nn

from .TTABase import TTABase
from models.ModelForTTA import CLIPModelForTTA


class NoAdapt(TTABase):
    def __init__(self, model, datasets, args):
        super().__init__()

        self.cfg = args.config

        self.model = CLIPModelForTTA(model, datasets.classes, self.cfg, args)
        self.last_diagnostics = {}

    @torch.no_grad()
    def forward(self, x):
        logits = self.model(x)
        self.last_diagnostics = {
            'pre_adaptation_ood_score': -torch.logsumexp(logits.detach(), dim=1),
        }

        return logits

    def get_diagnostics(self):
        return dict(self.last_diagnostics)

    def reset(self):
        pass
