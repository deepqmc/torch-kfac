import logging
from contextlib import AbstractContextManager, contextmanager

import torch
from torch.optim.optimizer import Optimizer

from .handlers import KFACEmbedding, KFACLinearFactored

log = logging.getLogger(__name__)


class ModuleTracker:
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


class KFAC(Optimizer, ModuleTracker, AbstractContextManager):
    def __init__(
        self,
        modules,
        damping,
        norm_constraint,
        cov_ema_decay=0.95,
        centered_cov=False,
        precond=True,
        global_precond=False,
        handler_factories=None,
    ):
        if handler_factories is None:
            handler_factories = [KFACLinearFactored, KFACEmbedding]
        defaults = {
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
        self.state['precond'] = precond
        self.state['global_precond'] = global_precond

    def _iter_groups(self):
        for group in self.param_groups:
            group_id = group['params'][0]
            state = self.state[group_id]
            handler = self._handlers[group_id]
            yield group, handler, state

    def step_update(self):
        for group, handler, state in self._iter_groups():
            state['k'] = state.get('k', 0) + 1
            handler.update_fisher(group, state)
            handler.update_inverse(group, state)

    def step_precondition(self):
        if not self.state['precond']:
            return
        for group, handler, state in self._iter_groups():
            handler.precondition(group, state)

    def step_rescale(self):
        fnorms, gnorms = [], []
        for i, (group, handler, state) in enumerate(self._iter_groups()):
            fn, gnorm = handler.norms(group, state)
            fnorm = ((fn - fn.mean()) ** 2).mean()
            log.debug(f'{group.get("name", i)}: KL: {fnorm}, Δℒ: {-gnorm}')
            fnorms.append(fn)
            gnorms.append(gnorm)
        gnorms = torch.stack(gnorms)
        gnorm = -gnorms.sum()
        fnorms = torch.stack(fnorms, dim=-1)
        fnorms = fnorms - fnorms.mean(dim=0)
        fnorm = (fnorms.sum(dim=-1) ** 2).mean()
        log.debug(f'total: KL: {fnorm}, Δℒ: {gnorm}')
        scale = (self.state['norm_constraint'] / fnorm).sqrt()
        if scale < 1:
            for param in self.parameters():
                param.grad.detach().copy_(scale * param.grad)
            fnorms, gnorms = scale * fnorms, scale * gnorms
            fnorm, gnorm = scale ** 2 * fnorm, scale * gnorm
            log.debug(f'total: KL: {fnorm}, Δℒ: {gnorm}')
        if self.state['global_precond']:
            F = fnorms.t() @ fnorms / len(fnorms)
            scales = F.inverse() @ gnorms
            for scale, group in zip(scales, self.param_groups):
                for param in group['params']:
                    param.grad.detach().copy_(scale * param.grad)
            fnorms, gnorms = scales * fnorms, scales * gnorms
            fnorm = (fnorms.sum(dim=-1) ** 2).mean()
            gnorm = -gnorms.sum()
            log.debug(f'total: KL: {fnorm}, Δℒ: {gnorm}')
            scale = (self.state['norm_constraint'] / fnorm).sqrt()
            if scale < 1:
                for param in self.parameters():
                    param.grad.detach().copy_(scale * param.grad)
                fnorm, gnorm = scale ** 2 * fnorm, scale * gnorm
                log.debug(f'total: KL: {fnorm}, Δℒ: {gnorm}')
        return fnorm, gnorm

    def step(self):
        self.step_update()
        self.step_precondition()
        return self.step_rescale()

    def parameters(self):
        for group in self.param_groups:
            yield from group['params']
