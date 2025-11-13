import torch
import torch.nn as nn
from copy import deepcopy


class TTABase(nn.Module):

    def copy_model_and_optimizer(self):
        """Copy the model and optimizer states for resetting after adaptation."""
        self.model_state = deepcopy(self.model.state_dict())
        self.optimizer_state = deepcopy(self.optimizer.state_dict())

    def load_model_and_optimizer(self):
        """Restore the model and optimizer states from copies."""
        self.model.load_state_dict(self.model_state, strict=True)
        self.optimizer.load_state_dict(self.optimizer_state)
