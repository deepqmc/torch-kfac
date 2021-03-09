# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from abc import abstractmethod
from math import sqrt

import torch
import torch.nn.functional as F

from ..utils import flatten_or_unsqueeze, updated_ewm_average
from .handler import KFACModuleHandler


class KFACLinearHandler(KFACModuleHandler):
    mod_class = 'Linear'

    @abstractmethod
    def update_fisher_linear(self, group, state, a, g):
        pass

    @abstractmethod
    def precondition_linear(self, group, state, grad):
        pass

    def update_fisher(self, group, state):
        a = [a for (a,) in self._buffer.pop('input')]
        g = [g for (g,) in reversed(self._buffer.pop('grad_output'))]
        assert len(a) == len(g)
        assert all(a.shape[:-1] == g.shape[:-1] for a, g, in zip(a, g))
        assert all(x.shape[0] == a[0].shape[0] for x in a + g)
        a = torch.cat([flatten_or_unsqueeze(a, 1, -2) for a in a], dim=1)
        g = torch.cat([flatten_or_unsqueeze(g, 1, -2) for g in g], dim=1)
        g = g * g.shape[0]  # multiplication by batch size assumes mean-reduced loss
        self.update_fisher_linear(group, state, a, g)

    def _get_params(self, group):
        weight = group['params'][0]
        bias = group['params'][1] if len(group['params']) == 2 else None
        return weight, bias

    def _get_grad(self, group):
        weight, bias = self._get_params(group)
        grad = weight.grad.detach()
        if bias is not None:
            grad = torch.cat([grad, bias.grad.detach()[..., None]], dim=-1)
        return grad

    def _set_grad(self, group, grad):
        weight, bias = self._get_params(group)
        if bias is not None:
            grad_weight, grad_bias = grad[..., :-1], grad[..., -1]
        else:
            grad_weight = grad
        weight.grad.detach().copy_(grad_weight)
        if bias is not None:
            bias.grad.detach().copy_(grad_bias)

    def precondition(self, group, state):
        state['grad'] = grad = self._get_grad(group).clone()
        grad = self.precondition_linear(group, state, grad)
        self._set_grad(group, grad)

    def norms(self, group, state):
        grad = self._get_grad(group)
        fnorm = self.fisher_norm_linear(group, state, grad)
        gnorm = (grad * state['grad']).sum()
        return fnorm, gnorm


class KFACLinearFull(KFACLinearHandler):
    def update_fisher_linear(self, group, state, a, g):
        if len(group['params']) == 2:
            a = F.pad(a, (0, 1), value=1)
        dW = a[:, :, None, :] * g[:, :, :, None]
        dW = dW.mean(dim=1) if group['average_loc'] else dW.flatten(end_dim=1)
        if group['centered_cov']:
            dW = dW - dW.mean(dim=0)
        state['dW'] = dW
        dW = dW.flatten(start_dim=1)
        state['F'] = updated_ewm_average(
            state.get('F'), dW.t() @ dW / len(dW), group['cov_ema_decay'], state['k']
        )

    def update_inverse(self, group, state):
        super().update_inverse(group, state)

    def precondition_linear(self, group, state, grad):
        return (state['iF'] @ grad.flatten()).view_as(grad)

    def fisher_norm_linear(self, group, state, v):
        return ((state['dW'] * v).sum(dim=(-1, -2)) ** 2).mean(dim=0)


class KFACLinearFactored(KFACLinearHandler):
    def update_fisher_linear(self, group, state, a, g):
        if group['centered_cov']:
            a = a - a.mean(dim=(0, 1))
            g = g - g.mean(dim=(0, 1))
        if len(group['params']) == 2:
            a = F.pad(a, (0, 1), value=1)
        if group['average_loc']:
            a, g = a.mean(dim=1), g.mean(dim=1)
        else:
            a, g = a.flatten(end_dim=1), g.flatten(end_dim=1)
        state['a'], state['g'] = a, g
        state['A'] = updated_ewm_average(
            state.get('A'), a.t() @ a / len(a), group['cov_ema_decay'], state['k']
        )
        state['G'] = updated_ewm_average(
            state.get('G'), g.t() @ g / len(g), group['cov_ema_decay'], state['k']
        )

    def update_inverse(self, group, state):
        A, G = state['A'], state['G']
        pi = torch.sqrt((A.trace() / A.shape[0]) / (G.trace() / G.shape[0]))
        if group['max_pi']:
            pi = torch.minimum(pi, pi.new_tensor(group['max_pi']))
        sqrt_lam = sqrt(group['damping'])
        A = A + torch.diag(A.new_full((A.shape[0],), sqrt_lam * pi))
        G = G + torch.diag(G.new_full((G.shape[0],), sqrt_lam / pi))
        state['iA'], state['iG'] = A.inverse(), G.inverse()

    def precondition_linear(self, group, state, grad):
        return state['iG'] @ grad @ state['iA']

    def fisher_norm_linear(self, group, state, v):
        return (((state['g'] @ v) * state['a']).sum(dim=-1) ** 2).mean(dim=0)
