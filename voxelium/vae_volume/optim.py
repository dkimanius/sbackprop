import math
from typing import List, TypeVar

import torch
from torch.optim.optimizer import Optimizer


Tensor = TypeVar('torch.tensor')


class ExtendedAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        if not 0.0 < lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 < eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        params = list(params)

        sparse_params = []
        for index, param in enumerate(params):
            if isinstance(param, dict):
                for d_index, d_param in enumerate(param.get("params", [])):
                    if d_param.is_sparse:
                        sparse_params.append([index, d_index])
            elif param.is_sparse:
                sparse_params.append(index)
        if sparse_params:
            raise ValueError(
                f"Sparse params at indices {sparse_params}: SparseAdam requires dense parameter tensors"
            )

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(ExtendedAdam, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None, regularize=0):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                state['step'] += 1
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                step = state['step']

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                if grad.is_sparse:
                    if regularize > 0:
                        raise NotImplementedError("no_moment not supported for sparse gradient.")

                    grad = grad.coalesce()  # the update is non-linear so indices must be unique
                    grad_indices = grad._indices()
                    grad_values = grad._values()
                    size = grad.size()

                    def make_sparse(values):
                        constructor = grad.new
                        if grad_indices.dim() == 0 or values.dim() == 0:
                            return constructor().resize_as_(grad)
                        return constructor(grad_indices, values, size)

                    # Decay the first and second moment running average coefficient
                    #      old <- b * old + (1 - b) * new
                    # <==> old += (1 - b) * (new - old)
                    old_exp_avg_values = exp_avg.sparse_mask(grad)._values()
                    exp_avg_update_values = grad_values.sub(old_exp_avg_values).mul_(1 - beta1)
                    exp_avg.add_(make_sparse(exp_avg_update_values))
                    old_exp_avg_sq_values = exp_avg_sq.sparse_mask(grad)._values()
                    exp_avg_sq_update_values = grad_values.pow(2).sub_(old_exp_avg_sq_values).mul_(1 - beta2)
                    exp_avg_sq.add_(make_sparse(exp_avg_sq_update_values))

                    # Dense addition again is intended, avoiding another sparse_mask
                    numer = exp_avg_update_values.add_(old_exp_avg_values)
                    exp_avg_sq_update_values.add_(old_exp_avg_sq_values)
                    denom = exp_avg_sq_update_values.sqrt_().add_(group['eps'])
                    del exp_avg_update_values, exp_avg_sq_update_values
                    step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                    if group['weight_decay'] != 0:
                        p.add_(
                            -step_size * (
                                    make_sparse(numer.div_(denom)) +
                                    group['weight_decay'] * p.sparse_mask(grad)
                            )
                        )
                    else:
                        p.add_(-step_size * make_sparse(numer.div_(denom)))
                else:
                    if regularize > 0:
                        p.add_(grad, alpha=-regularize)
                    else:
                        if group['weight_decay'] != 0:
                            grad = grad.add(p, alpha=group['weight_decay'])

                        # Decay the first and second moment running average coefficient
                        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                        exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)
                        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                        step_size = group['lr'] / bias_correction1

                        p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
