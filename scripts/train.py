
import time
import math
import itertools
import torch
from torch.nn import functional as F
from classifier import objective
from optimizers import Adam, Nesterov, SVRG, BlockCoordinate


def record_metrics(model, features, labels, validation_features,
                   validation_labels, regularization):
    training_objective = objective(model, features, labels, regularization)

    gradients = torch.autograd.grad(training_objective, model.parameters())
    full_gradient_norm = sum(g.square().sum() for g in gradients).sqrt()

    with torch.no_grad():
        scores = model(validation_features)
        validation_loss = F.cross_entropy(scores, validation_labels)
        predictions = scores.argmax(dim=1)
        validation_accuracy = (predictions == validation_labels).float().mean()

    return {
        'training_objective': training_objective.item(),
        'full_gradient_norm': full_gradient_norm.item(),
        'validation_loss': validation_loss.item(),
        'validation_accuracy': validation_accuracy.item(),
    }


def validation_confusion(model, features, labels):

    with torch.no_grad():
        scores = model(features)
        classes = scores.shape[1]
        predictions = scores.argmax(dim=1)
        counts = torch.bincount(labels * classes + predictions,
                                minlength=classes * classes)
    return counts.reshape(classes, classes).cpu().tolist()


def train(model, features, labels, validation_features, validation_labels,
          optimizer_name, learning_rate, regularization,
          momentum=0.9, batch_size=128, snapshot_interval=100,
          block_type='class', seed=0, record_every_passes=1,
          gradient_tolerance=1e-4, max_updates=10000,
          max_equivalent_passes=100, best_checkpoint_path=None,
          max_optimizer_seconds=None, record_every_seconds=1.0,
          stop_on_gradient=None):

    time_mode = max_optimizer_seconds is not None
    if time_mode and (not math.isfinite(max_optimizer_seconds) or max_optimizer_seconds <= 0):
        raise ValueError('max_optimizer_seconds must be positive and finite.')
    if not math.isfinite(record_every_seconds) or record_every_seconds <= 0:
        raise ValueError('record_every_seconds must be positive and finite.')
    if stop_on_gradient is None:
        stop_on_gradient = not time_mode
    if features.is_cuda:
        torch.cuda.synchronize(features.device)
    setup_start = time.perf_counter()




    if optimizer_name == 'adam':
        optimizer = Adam(model, learning_rate, regularization)
    elif optimizer_name == 'nesterov':
        optimizer = Nesterov(model, learning_rate, momentum, regularization)
    elif optimizer_name == 'svrg':
        optimizer = SVRG(model, learning_rate, regularization)
    elif optimizer_name == 'block':
        optimizer = BlockCoordinate(model, learning_rate, regularization, block_type)
    else:
        raise ValueError('Choose adam, nesterov, svrg or block.')

    rng = torch.Generator(device=features.device).manual_seed(seed)
    if features.is_cuda:
        torch.cuda.synchronize(features.device)
    setup_seconds = time.perf_counter() - setup_start
    n = len(features)
    batch_start = n
    history = []
    gradient_examples = 0
    optimizer_seconds = 0.0
    gradient_threshold_first_reached = None
    snapshot_refreshes = 0
    monitoring_evaluations = 0
    best_validation = None
    updates_done = 0
    blocks_per_sweep = model.out_features if optimizer_name == 'block' else 1
    next_checkpoint = record_every_seconds if time_mode else record_every_passes
    termination = 'max_updates'

    def synchronize():
        if features.is_cuda:
            torch.cuda.synchronize(features.device)

    def checkpoint():
        nonlocal monitoring_evaluations, best_validation, gradient_threshold_first_reached
        metrics = record_metrics(model, features, labels, validation_features,
                                 validation_labels, regularization)
        monitoring_evaluations += 1
        if (gradient_threshold_first_reached is None
                and metrics['full_gradient_norm'] <= gradient_tolerance):
            gradient_threshold_first_reached = dict(
                optimizer_seconds=optimizer_seconds, update=updates_done,
                full_gradient_norm=metrics['full_gradient_norm'])
        history.append(dict(update=updates_done, equivalent_passes=gradient_examples / n,
                            gradient_examples=gradient_examples,
                            full_gradient_examples=gradient_examples if optimizer_name != 'block' else 0,
                            block_gradient_examples=gradient_examples if optimizer_name == 'block' else 0,
                            optimizer_seconds=optimizer_seconds,
                            snapshot_refreshes=snapshot_refreshes,
                            block_updates=updates_done if optimizer_name == 'block' else 0,
                            block_sweeps=updates_done / blocks_per_sweep if optimizer_name == 'block' else 0,
                            monitoring_gradient_examples=monitoring_evaluations * n,
                            **metrics))
        if (all(math.isfinite(value) for value in metrics.values())
                and (best_validation is None
                     or metrics['validation_loss'] < best_validation['validation_loss'])):
            best_validation = dict(update=updates_done,
                                   equivalent_passes=gradient_examples / n,
                                   optimizer_seconds=optimizer_seconds, **metrics)
            if best_checkpoint_path is not None:
                torch.save(model.state_dict(), best_checkpoint_path)
        return metrics

    synchronize()
    wall_start = time.perf_counter()

    metrics = checkpoint()
    initial_confusion = validation_confusion(model, validation_features, validation_labels)

    optimizer_seconds = setup_seconds
    updates = itertools.count(1) if time_mode else range(1, max_updates + 1)
    for update in updates:
        if not all(math.isfinite(value) for value in metrics.values()):
            termination = 'nonfinite'
            break
        if stop_on_gradient and metrics['full_gradient_norm'] <= gradient_tolerance:
            termination = 'converged'
            break

        if time_mode and optimizer_seconds >= max_optimizer_seconds:
            metrics = checkpoint()
            termination = 'time_limit'
            break

        synchronize()
        start = time.perf_counter()
        if optimizer_name == 'svrg' and (update - 1) % snapshot_interval == 0:
            optimizer.refresh_snapshot(features, labels)
            gradient_examples += n
            snapshot_refreshes += 1

        if optimizer_name in ('adam', 'svrg'):
            if optimizer_name == 'svrg':

                indices = torch.randint(n, (batch_size,), device=features.device, generator=rng)
            else:
                if batch_start >= n:
                    order = torch.randperm(n, device=features.device, generator=rng)
                    batch_start = 0
                indices = order[batch_start:batch_start + batch_size]
                batch_start += len(indices)
            optimizer.step(features[indices], labels[indices])
            gradient_examples += len(indices) * (2 if optimizer_name == 'svrg' else 1)
        else:
            optimizer.step(features, labels)

            gradient_examples += n
        synchronize()
        optimizer_seconds += time.perf_counter() - start
        updates_done = update

        equivalent_passes = gradient_examples / n
        time_limit = time_mode and optimizer_seconds >= max_optimizer_seconds
        work_limit = not time_mode and equivalent_passes >= max_equivalent_passes
        progress = optimizer_seconds if time_mode else equivalent_passes
        interval = record_every_seconds if time_mode else record_every_passes
        if progress >= next_checkpoint or time_limit or work_limit or (not time_mode and update == max_updates):

            metrics = checkpoint()
            next_checkpoint = (int(progress / interval) + 1) * interval

        if time_limit or work_limit:
            termination = 'time_limit' if time_limit else 'max_equivalent_passes'
            break

    if not all(math.isfinite(value) for value in metrics.values()):
        termination = 'nonfinite'
    elif stop_on_gradient and metrics['full_gradient_norm'] <= gradient_tolerance:
        termination = 'converged'
    final_confusion = (validation_confusion(model, validation_features, validation_labels)
                       if termination != 'nonfinite' else None)
    synchronize()
    summary = {'termination': termination,
               'max_optimizer_seconds': max_optimizer_seconds,
               'time_budget_overshoot_seconds': max(0.0, optimizer_seconds - max_optimizer_seconds) if time_mode else 0.0,
               'optimizer_setup_seconds': setup_seconds,
               'gradient_threshold_first_reached': gradient_threshold_first_reached,
               'best_validation': best_validation,
               'updates': updates_done,
               'gradient_examples': gradient_examples,
               'full_gradient_examples': gradient_examples if optimizer_name != 'block' else 0,
               'block_gradient_examples': gradient_examples if optimizer_name == 'block' else 0,
               'work_unit': 'block_gradient_example' if optimizer_name == 'block' else 'full_gradient_example',
               'equivalent_passes': gradient_examples / n,
               'optimizer_seconds': optimizer_seconds,
               'monitoring_evaluations': monitoring_evaluations,
               'monitoring_gradient_examples': monitoring_evaluations * n,
               'snapshot_refreshes': snapshot_refreshes,
               'initial_validation_confusion': initial_confusion,
               'final_validation_confusion': final_confusion,
               'final_metrics': metrics,
               'wall_seconds': time.perf_counter() - wall_start}

    return model, history, summary
