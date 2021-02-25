# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.


def flatten_or_unsqueeze(x, start_dim=0, end_dim=-1):
    if end_dim < 0:
        end_dim = len(x.shape) + end_dim
    if end_dim < start_dim:
        return x.unsqueeze(start_dim)
    return x.flatten(start_dim, end_dim)


def updated_ewm_average(current, update, decay, step):
    if current is None:
        return update
    kappa = min(1 - 1 / step, decay)
    return kappa * current + (1 - kappa) * update
