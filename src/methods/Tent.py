import torch
import torch.nn as nn

from .losses import softmax_entropy
from .TTABase import TTABase
from models.ModelForTTA import CLIPModelForTTA


class Tent(TTABase):
    def __init__(self, model, datasets, args):
        super().__init__()

        self.cfg = args.config

        self.model = CLIPModelForTTA(model, datasets.classes, self.cfg, args)
        self.last_diagnostics = {}

    def forward(self, x):
        logits = self.model(x)
        self.last_diagnostics = {
            'pre_adaptation_ood_score': -torch.logsumexp(logits.detach(), dim=1),
        }
        loss = softmax_entropy(logits, reduction='mean')
        loss.backward()
        self.model.step_and_zero_grad()

        return logits.detach()

    def get_diagnostics(self):
        return dict(self.last_diagnostics)

    def reset(self):
        self.model.reset_parameters()
