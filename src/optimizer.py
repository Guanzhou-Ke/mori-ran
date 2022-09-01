"""
Optimizer.
------

Author: Guanzhou Ke.
Email: guanzhouk@gmail.com
Date: 2022/08/14
"""
import torch


class Optimizer:
    def __init__(self, params, lr=1e-3, type='adam', schedule=True):

        self.params = params
        self.schedule = schedule
        if type == 'sgd':
            self._opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-6)
        elif type == 'adam':
            self._opt = torch.optim.Adam(params, lr=lr, weight_decay=1e-6)
        else:
            raise ValueError('optimizer must be sgd or adam.')
        if self.schedule:
            self._sch = torch.optim.lr_scheduler.CosineAnnealingLR(self._opt, T_max=5, eta_min=1e-4)

    def zero_grad(self):
        return self._opt.zero_grad()

    def step(self):
        out = self._opt.step()
        if self.schedule:
            # Only step the scheduler at integer epochs, and don't step on the first epoch.
            self._sch.step()
        return out
    
    
def get_optimizer(params, lr=1e-3, type='adam'):

    if type == 'sgd':
        opt = torch.optim.SGD(params, lr=lr, momentum=0.9)
    elif type == 'adam':
        opt = torch.optim.Adam(params, lr=lr)
    else:
        raise ValueError('optimizer must be sgd or adam.')
    return opt
    