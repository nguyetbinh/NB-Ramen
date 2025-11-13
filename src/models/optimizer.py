import torch.optim as optim

import math
from typing import Iterable, Optional
import torch
from torch.optim.optimizer import Optimizer


class SignSGD(Optimizer):
    r"""
    SignSGD with an SGD-compatible interface.

    Update rule mirrors torch.optim.SGD preprocessing (weight_decay -> momentum -> nesterov),
    but applies sign() to the preconditioned gradient before the parameter update.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        weight_decay (float, optional): L2 penalty (added to grad, NOT decoupled) (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
        maximize (bool, optional): maximize the objective (default: False)
    """

    def __init__(
            self,
            params: Iterable[torch.nn.Parameter],
            lr: float = 1e-3,
            momentum: float = 0.0,
            dampening: float = 0.0,
            weight_decay: float = 0.0,
            nesterov: bool = False,
            maximize: bool = False,
            # keep kwargs for API compatibility (foreach, capturable, differentiable, etc.)
            **kwargs,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if nesterov and (momentum <= 0 or dampening != 0):
            # same constraint as torch SGD
            raise ValueError("Nesterov momentum requires a positive momentum and zero dampening")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            maximize=maximize,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            weight_decay = group["weight_decay"]
            nesterov = group["nesterov"]
            maximize = group.get("maximize", False)

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                d_p = grad if not maximize else -grad

                # L2-style weight decay: add to grad (same semantics as torch.optim.SGD)
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)

                # Momentum
                if momentum != 0:
                    param_state = self.state[p]
                    buf = param_state.get("momentum_buffer", None)
                    if buf is None:
                        buf = torch.clone(d_p).detach()
                        param_state["momentum_buffer"] = buf
                    else:
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)  # Nesterov lookahead
                    else:
                        d_p = buf

                # Sign step (elementwise)
                step_dir = d_p.sign()

                # Zero sign means no move for that element (matches signSGD convention)
                p.add_(step_dir, alpha=-lr)

        return loss


def get_optimizer_class(optimizer_name):
    if optimizer_name == 'sgd':
        return optim.SGD
    elif optimizer_name == 'signsgd':
        return SignSGD
    else:
        raise ValueError('Unknown optimizer: {}'.format(optimizer_name))
