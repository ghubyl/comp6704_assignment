
import torch
import copy
from classifier import objective
from torch.nn import functional as F


class Adam:
    def __init__(self, model, learning_rate, regularization):
        self.model = model
        self.regularization = regularization

        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=0
        )

    def step(self, features, labels):

        self.optimizer.zero_grad()
        loss = objective(self.model, features, labels, self.regularization)
        loss.backward()
        self.optimizer.step()


class Nesterov:
    def __init__(self, model, learning_rate, momentum, regularization):
        self.model = model
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.regularization = regularization
        self.previous_parameters = [p.detach().clone() for p in model.parameters()]

    def step(self, features, labels):

        current_parameters = [p.detach().clone() for p in self.model.parameters()]
        with torch.no_grad():
            for p, current, previous in zip(
                self.model.parameters(),
                current_parameters,
                self.previous_parameters,
            ):
                lookahead = current + self.momentum * (current - previous)
                p.copy_(lookahead)

        self.model.zero_grad()

        loss = objective(
            self.model, features, labels, self.regularization
        )
        loss.backward()

        with torch.no_grad():
            for p in self.model.parameters():
                p.add_(-self.learning_rate * p.grad)

        self.previous_parameters =current_parameters


class SVRG:
    def __init__(self, model, learning_rate, regularization):
        self.model = model
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.snapshot_model = None
        self.snapshot_gradient = None

    def refresh_snapshot(self, features, labels):
        self.snapshot_model = copy.deepcopy(self.model)
        self.snapshot_model.zero_grad()

        scores = self.snapshot_model(features)
        loss = F.cross_entropy(scores, labels)
        loss.backward()

        self.snapshot_gradient = [
            p.grad.detach().clone()
            for p in self.snapshot_model.parameters()
        ]

    def step(self, features, labels):

        self.model.zero_grad()
        loss = F.cross_entropy(self.model(features), labels)
        loss.backward()


        self.snapshot_model.zero_grad()
        snapshot_loss = F.cross_entropy(
            self.snapshot_model(features), labels
        )
        snapshot_loss.backward()

        with torch.no_grad():
            for p, snapshot_p, full_gradient in zip(
                self.model.parameters(),
                self.snapshot_model.parameters(),
                self.snapshot_gradient,
            ):
                corrected_gradient = (
                    p.grad
                    - snapshot_p.grad
                    + full_gradient
                    + self.regularization * p
                )
                p.add_(-self.learning_rate * corrected_gradient)



class BlockCoordinate:
    def __init__(self, model, learning_rate, regularization, block_type='class'):
        if block_type != 'class':
            raise ValueError('Only class blocks are currently supported.')
        self.model = model
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.block_index = 0
        self.scores = None

    def step(self, features, labels):

        if self.scores is None:
            with torch.no_grad():
                self.scores = self.model(features)

        c = self.block_index
        w = self.model.weight[c].detach().clone().requires_grad_()
        b = self.model.bias[c].detach().clone().requires_grad_()
        scores = self.scores.detach().clone()
        scores[:, c] = features @ w + b
        loss = F.cross_entropy(scores, labels)
        loss = loss + 0.5 * self.regularization * (w.square().sum() + b.square())
        weight_gradient, bias_gradient = torch.autograd.grad(loss, (w, b))

        with torch.no_grad():
            weight_change = -self.learning_rate * weight_gradient
            bias_change = -self.learning_rate * bias_gradient
            self.model.weight[c].add_(weight_change)
            self.model.bias[c].add_(bias_change)
            self.scores[:, c].add_(features @ weight_change + bias_change)

        self.block_index = (c + 1) % self.model.out_features
