# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import torch

from ..utils import flatten_or_unsqueeze, updated_ewm_average
from .handler import KFACModuleHandler


def _bincount_3d1(inp, weights, minlength=0):
    m = max(inp.max().item() + 1, minlength)
    N, _, D = weights.shape
    idx = torch.arange(m * N * D, device=inp.device).view(m * N, D)
    offset = torch.arange(0, m * N, m, device=inp.device)
    return torch.bincount(
        idx[inp + offset[:, None]].flatten(),
        weights.flatten(),
        minlength=N * m * D,
    ).view(N, m, D)


class KFACEmbedding(KFACModuleHandler):
    mod_class = 'Embedding'

    def update_fisher(self, group, state):
        idx = [idx for (idx,) in self._buffer.pop('input')]
        g = [g for (g,) in reversed(self._buffer.pop('grad_output'))]
        assert len(idx) == len(g)
        assert all(idx.shape == g.shape[:-1] for idx, g, in zip(idx, g))
        idx = torch.cat([idx.flatten(start_dim=1) for idx in idx], dim=1)
        g = torch.cat([flatten_or_unsqueeze(g, 1, -2) for g in g], dim=1)
        g = g * g.shape[0]  # multiplication by batch size assumes mean-reduced loss
        dW = _bincount_3d1(idx, g, minlength=len(group['params'][0]))
        dW = dW.mean(dim=1)
        if group['centered_cov']:
            dW = dW - dW.mean(dim=0)
        state['dW'] = dW
        dW = dW.flatten(start_dim=1)
        state['F'] = updated_ewm_average(
            state.get('F'), dW.t() @ dW / len(dW), group['cov_ema_decay'], state['k']
        )

    def update_inverse(self, group, state):
        super().update_inverse(group, state)

    def precondition(self, group, state):
        weight = group['params'][0]
        state['grad'] = grad = weight.grad.detach().clone()
        grad = (state['iF'] @ grad.flatten()).view_as(grad)
        weight.grad.detach().copy_(grad)

    def norms(self, group, state):
        grad = group['params'][0].grad.detach()
        fnorm = ((state['dW'] * grad).sum(dim=(-1, -2)) ** 2).mean(dim=0)
        gnorm = (grad * state['grad']).sum()
        return fnorm, gnorm
