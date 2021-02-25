# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from abc import ABC, abstractmethod
from collections import defaultdict

import torch


class KFACModuleHandler(ABC):
    mod_class = None

    @abstractmethod
    def update_fisher(self, group, state):
        pass

    @abstractmethod
    def update_inverse(self, group, state):
        F = state['F']
        F = F + torch.diag(F.new_full((F.shape[0],), group['damping']))
        state['iF'] = F.inverse()

    @abstractmethod
    def precondition(self, group, state):
        pass

    @abstractmethod
    def norms(self, group, state):
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
