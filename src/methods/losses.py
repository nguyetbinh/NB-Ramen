import torch


def softmax_entropy(logits, reduction='mean'):
    loss = -(logits.softmax(dim=1) * logits.log_softmax(dim=1)).sum(dim=1)

    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    elif reduction == 'none':
        pass
    else:
        raise ValueError

    return loss

