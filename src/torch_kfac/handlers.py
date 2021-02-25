import logging
from abc import ABC, abstractmethod
from collections import defaultdict

import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)


def debug(*args):
    log.debug(', '.join(str(x) for x in args))


def flatten_or_unsqueeze(x, start_dim=0, end_dim=-1):
    try:
        return x.flatten(start_dim, end_dim)
    except RuntimeError as exp:
        if 'start_dim cannot come after end_dim' not in exp.args[0]:
            raise
        return x.unsqueeze(start_dim)


def bincount_3d1(inp, weights, minlength=0):
    m = max(inp.max().item() + 1, minlength)
    N, _, D = weights.shape
    idx = torch.arange(m * N * D, device=inp.device).view(m * N, D)
    offset = torch.arange(0, m * N, m, device=inp.device)
    return torch.bincount(
        idx[inp + offset[:, None]].flatten(),
        weights.flatten(),
        minlength=N * m * D,
    ).view(N, m, D)


class KFACModuleHandler(ABC):
    mod_class = None

    @abstractmethod
    def update_fisher(self, group, state, grad_weight=None):
        pass

    @abstractmethod
    def update_inverse(self, group, state):
        F = state['F']
        F = F + torch.diag(F.new_full((F.shape[0],), group['damping'] ** 2))
        state['iF'] = F.inverse()
        debug('iF mean/std', state['iF'].mean(), state['iF'].std())

    @abstractmethod
    def precondition(self, group, state):
        pass

    @abstractmethod
    def norms(self, group, state, v):
        pass

    def __init__(self, mod):
        assert mod.__class__.__name__ == self.mod_class
        self._buffer = defaultdict(list)
        self._handles = [
            mod.register_forward_pre_hook(self._forward_pre_hook),
            mod.register_backward_hook(self._backward_hook),
        ]
        self.tracking_forward = False
        self.tracking_backward = False

    def _forward_pre_hook(self, mod, inp):
        if self.tracking_forward:
            self._buffer['input'].append(tuple(x.detach() for x in inp))

    def _backward_hook(self, mod, grad_inp, grad_out):
        if self.tracking_backward:
            self._buffer['grad_output'].append(tuple(x.detach() for x in grad_out))

    def close(self):
        while self._handles:
            self._handles.pop().remove()


class KFACEmbedding(KFACModuleHandler):
    mod_class = 'Embedding'

    def update_fisher(self, group, state, grad_weight=None):
        idx = [idx for (idx,) in self._buffer.pop('input')]
        g = [g for (g,) in reversed(self._buffer.pop('grad_output'))]
        # bs = idx[0].shape[0]  # batch size
        debug('len', len(idx), len(g))
        assert len(idx) == len(g)
        assert all(idx.shape == g.shape[:-1] for idx, g, in zip(idx, g))
        debug('idx shapes', [x.shape for x in idx])
        debug('g shapes', [x.shape for x in g])
        idx = torch.cat([idx.flatten(start_dim=1) for idx in idx], dim=1)
        g = torch.cat([flatten_or_unsqueeze(g, 1, -2) for g in g], dim=1)
        if grad_weight is not None:
            g = g / grad_weight[:, None, None]
        dW = bincount_3d1(idx, g, minlength=len(group['params'][0]))
        state['dW'] = dW
        if group['centered_cov']:
            dW = dW - dW.mean(dim=0)
        dW = dW.flatten(start_dim=1)
        F = dW.t() @ dW / len(dW)
        if 'F' not in state:
            state['F'] = F
        else:
            eps = min(1 - 1 / state['k'], group['cov_ema_decay'])
            state['F'] = eps * state['F'] + (1 - eps) * F

    def update_inverse(self, group, state):
        super().update_inverse(group, state)

    def precondition(self, group, state):
        weight = group['params'][0]
        grad = weight.grad.detach()
        debug('grad shape', grad.shape)
        state['grad'] = grad
        grad = (state['iF'] @ grad.flatten()).view_as(grad)
        weight.grad.detach().copy_(grad)

    def norms(self, group, state):
        grad = group['params'][0].grad.detach()
        dW = state['dW']
        fnorms = (dW * grad).sum(dim=(1, 2))
        gnorm = (grad * state.get('grad', grad)).sum()
        return fnorms, gnorm


class KFACLinearHandler(KFACModuleHandler):
    mod_class = 'Linear'

    @abstractmethod
    def update_fisher_linear(self, group, state, a, g):
        pass

    @abstractmethod
    def precondition_linear(self, group, state, grad):
        pass

    def update_fisher(self, group, state, grad_weight=None):
        a = [a for (a,) in self._buffer.pop('input')]
        g = [g for (g,) in reversed(self._buffer.pop('grad_output'))]
        assert len(a) == len(g)
        assert all(a.shape[:-1] == g.shape[:-1] for a, g, in zip(a, g))
        assert all(x.shape[0] == a[0].shape[0] for x in a + g)
        debug('a shapes', [x.shape for x in a])
        debug('g shapes', [x.shape for x in g])
        a = torch.cat([flatten_or_unsqueeze(a, 1, -2) for a in a], dim=1)
        g = torch.cat([flatten_or_unsqueeze(g, 1, -2) for g in g], dim=1)
        if grad_weight is not None:
            g = g / grad_weight[:, None, None]
        debug('a shape', a.shape, 'g shape', g.shape)
        debug('a mean/std', a.mean(dim=(0, 1)), a.std(dim=(0, 1)))
        debug('g mean/std', g.mean(dim=(0, 1)), g.std(dim=(0, 1)))
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
        grad = self._get_grad(group)
        debug('grad shape', grad.shape)
        debug('grad W mean/std', grad[..., :-1].mean(), grad[..., :-1].std())
        debug('grad b mean/std', grad[..., -1].mean(), grad[..., -1].std())
        state['grad'] = grad
        grad = self.precondition_linear(group, state, grad)
        self._set_grad(group, grad)
        # grad_norm = (grad * state['grad']).sum()
        # debug('grad shape', grad.shape)
        # debug('F-1 grad W mean/std', grad[..., :-1].mean(), grad[..., :-1].std())
        # debug('F-1 grad b mean/std', grad[..., -1].mean(), grad[..., -1].std())
        # return fisher_norm, grad_norm

    def norms(self, group, state):
        grad = self._get_grad(group)
        fnorms = self.fisher_norm_linear(group, state, grad)
        gnorm = (grad * state.get('grad', grad)).sum()
        return fnorms, gnorm


class KFACLinearDirect(KFACLinearHandler):
    def update_fisher_linear(self, group, state, a, g):
        if len(group['params']) == 2:
            a = F.pad(a, (0, 1), value=1)
            debug('a shape', a.shape)
        dW = (a[:, :, None, :] * g[:, :, :, None]).sum(dim=1)
        debug('dW shape', dW.shape)
        debug('dW mean/std', dW.mean(), dW.std())
        debug('dW shape', dW.shape)
        state['dW'] = dW
        if group['centered_cov']:
            dW = dW - dW.mean(dim=0)
            debug('dW mean/std', dW.mean(), dW.std())
        dW = dW.flatten(start_dim=1)
        state['F'] = dW.t() @ dW / len(dW)
        debug('F shape', state['F'].shape)
        debug('F mean/std', state['F'].mean(), state['F'].std())

    def update_inverse(self, group, state):
        super().update_inverse(group, state)

    def precondition_linear(self, group, state, grad):
        return (state['iF'] @ grad.flatten()).view_as(grad)

    def fisher_norm_linear(self, group, state, v):
        return ((state['dW'] * v).sum(dim=(1, 2)) ** 2).mean(dim=0)


class KFACLinearFactored(KFACLinearHandler):
    def update_fisher_linear(self, group, state, a, g):
        state['a'], state['g'] = a, g
        if group['centered_cov']:
            if a.shape[-1] > 1:
                a = a - a.mean(dim=(0, 1))
            if g.shape[-1] > 1:
                g = g - g.mean(dim=(0, 1))
            debug('a mean/std', a.mean(dim=(0, 1)), a.std(dim=(0, 1)))
            debug('g mean/std', g.mean(dim=(0, 1)), g.std(dim=(0, 1)))
        if len(group['params']) == 2:
            a = F.pad(a, (0, 1), value=1)
            debug('a shape', a.shape)
        M, a, g = a.shape[1], a.flatten(end_dim=1), g.flatten(end_dim=1)
        if 'A' not in state:
            state['A'] = a.t() @ a / len(a)
            state['G'] = g.t() @ g / len(g) * M ** 2
        else:
            eps = min(1 - 1 / state['k'], group['cov_ema_decay'])
            state['A'].addmm_(beta=eps, alpha=(1 - eps) / len(a), mat1=a.t(), mat2=a)
            state['G'].addmm_(beta=eps, alpha=(1 - eps) / len(a), mat1=g.t(), mat2=g)
        debug('A shape', state['A'].shape, 'G shape', state['G'].shape)
        debug('A mean/std', state['A'].mean(), state['A'].std())
        debug('G mean/std', state['G'].mean(), state['G'].std())

    def update_inverse(self, group, state):
        A, G = state['A'], state['G']
        debug('A trace mean', A.trace() / A.shape[0])
        debug('G trace mean', G.trace() / G.shape[0])
        pi = torch.sqrt((A.trace() / A.shape[0]) / (G.trace() / G.shape[0]))
        debug('pi', pi)
        debug('diags', pi * group['damping'], 1 / pi * group['damping'])
        A = A + torch.diag(A.new_full((A.shape[0],), pi * group['damping']))
        G = G + torch.diag(G.new_full((G.shape[0],), 1 / pi * group['damping']))
        try:
            state['iA'], state['iG'] = A.inverse(), G.inverse()
        except Exception:
            print(pi)
            print(A.shape)
            print(A)
            print(G.shape)
            print(G)
            raise
        debug('iA mean/std', state['iA'].mean(), state['iA'].std())
        debug('iG mean/std', state['iG'].mean(), state['iG'].std())

    def precondition_linear(self, group, state, grad):
        return state['iG'] @ grad @ state['iA']

    def fisher_norm_linear(self, group, state, v):
        a = state['a']
        if len(group['params']) == 2:
            a = F.pad(a, (0, 1), value=1)
        return ((state['g'] @ v) * a).sum(dim=(1, 2))
