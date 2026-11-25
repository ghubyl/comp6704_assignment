from torch import nn
from torch.nn import functional as F


def make_classifier(input_dim=384, num_classes=23):
    return nn.Linear(input_dim, num_classes)


def objective(model, features, labels, regularization):
    scores = model(features)
    data_loss = F.cross_entropy(scores, labels)

    penalty = sum(
        parameter.square().sum()
        for parameter in model.parameters()
    )

    return data_loss + 0.5 * regularization * penalty
