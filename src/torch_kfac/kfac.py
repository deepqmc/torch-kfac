from contextlib import AbstractContextManager, contextmanager

import torch
from torch.optim.optimizer import Optimizer

from .handlers import KFACEmbedding, KFACLinearFactored, KFACLinearFull


class ModuleTracker(AbstractContextManager):
    def __init__(self):
        self._handlers = {}

    def _tracking(self, what, tracking):
        for handler in self._handlers.values():
            setattr(handler, f'tracking_{what}', tracking)

    @contextmanager
    def track_forward(self):
        self._tracking('forward', True)
        try:
            yield
        finally:
            self._tracking('forward', False)

    @contextmanager
    def track_backward(self):
        self._tracking('backward', True)
        try:
            yield
        finally:
            self._tracking('backward', False)

    def close(self):
        for handler in self._handlers.values():
            handler.close()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class KFAC(Optimizer, ModuleTracker):
    def __init__(
        self,
        modules,
        lr,
        damping,
        cov_ema_decay=0.95,
        norm_constraint=None,
        update_cov_manually=False,
        centered_cov=False,
        handler_factories=None,
        exact_norm=False,
        exact_fisher_linear=False,
    ):
        if handler_factories is None:
            handler_factories = [
                KFACLinearFull if exact_fisher_linear else KFACLinearFactored,
                KFACEmbedding,
            ]
        defaults = {
            'lr': lr,
            'damping': damping,
            'cov_ema_decay': cov_ema_decay,
            'centered_cov': centered_cov,
        }
        factories = {f.mod_class: f for f in handler_factories}
        self._handlers = {}
        param_groups = []
        for mod in list(modules):
            if isinstance(mod, tuple):
                mod_name, mod = mod
            else:
                mod_name = None
            mod_class = mod.__class__.__name__
            if mod_class not in factories:
                for name, param in mod.named_parameters(recurse=False):
                    if param.requires_grad:
                        raise ValueError(
                            f'{mod_name or mod_class}.{name} requires '
                            'gradient but no handler factory'
                        )
                continue
            params = list(mod.parameters(recurse=False))
            grad_requirements = [p.requires_grad for p in params]
            if not params or not any(grad_requirements):
                continue
            assert all(grad_requirements)
            group = {'params': params}
            if mod_name:
                group['name'] = mod_name
            param_groups.append(group)
            self._handlers[params[0]] = factories[mod_class](mod)
        super().__init__(param_groups, defaults)
        self.state['norm_constraint'] = norm_constraint
        self.state['exact_norm'] = exact_norm
        self.state['update_cov_manually'] = update_cov_manually

    def _iter_groups(self):
        for group in self.param_groups:
            group_id = group['params'][0]
            state = self.state[group_id]
            handler = self._handlers[group_id]
            yield group, handler, state

    def step_update_cov(self):
        for group, handler, state in self._iter_groups():
            state['k'] = state.get('k', 0) + 1
            handler.update_fisher(group, state)
            handler.update_inverse(group, state)

    update_cov = step_update_cov

    def step_precondition(self):
        for group, handler, state in self._iter_groups():
            handler.precondition(group, state)

    def step(self):
        if not self.state['update_cov_manually']:
            self.step_update_cov()
        self.step_precondition()
        if self.state['norm_constraint']:
            fnorms, gnorms = [], []
            for group, handler, state in self._iter_groups():
                fnorm, gnorm = handler.norms(group, state)
                fnorms.append(fnorm * group['lr'] ** 2)
                gnorms.append(gnorm * group['lr'] ** 2)
            gnorms = torch.stack(gnorms)
            gnorm = gnorms.sum()
            fnorms = torch.stack(fnorms)
            fnorm = fnorms.sum()
            norm = fnorm if self.state['exact_norm'] else gnorm
            epsilon = min(1, (self.state['norm_constraint'] / norm).sqrt())
        else:
            epsilon = 1.0
        for group in self.param_groups:
            for p in group['params']:
                p.detach().add_(-epsilon * group['lr'] * p.grad.detach())
