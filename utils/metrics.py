import torch
import numpy as np
from scipy.stats import wasserstein_distance as wasserstein

_SQRT2 = np.sqrt(2)
Softmax = torch.nn.Softmax(dim=1)


def js_divergence(p, q):
    p, q = p.detach(), q.detach()
    p, q = Softmax(p), Softmax(q)
    KLDivLoss = torch.nn.KLDivLoss(reduction="batchmean")

    mean = ((p + q) / 2).log()
    result = (KLDivLoss(mean, p) + KLDivLoss(mean, q)) / 2

    return result.item()


def wasserstein_distance(p, q):
    p, q = Softmax(p.detach()).cpu().numpy().reshape(-1), Softmax(q.detach()).cpu().numpy().reshape(-1)

    result = wasserstein(p, q)

    return result


def hellinger_distance(p, q):
    p, q = Softmax(p.detach()).cpu().numpy().reshape(-1), Softmax(q.detach()).cpu().numpy().reshape(-1)

    sqrt_p, sqrt_q = np.sqrt(p), np.sqrt(q)

    result = np.sqrt(np.sum(sqrt_p - sqrt_q) ** 2) / _SQRT2

    return result
