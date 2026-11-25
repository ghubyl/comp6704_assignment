
import copy
import json
from pathlib import Path

import torch

from classifier import make_classifier
from make_incremental_splits import read_data, make_increments
from train import train


def experiment(optimizer_name, data_settings, training_settings,
               start_mode='warm', initial_only=False, initialization_seed=0,
               device='cuda', output_folder='results/run_001',
               initial_checkpoint=None):
    if start_mode not in ('warm', 'cold'):
        raise ValueError('Choose warm or cold start.')
    if initial_checkpoint is not None and initial_only:
        raise ValueError('Use initial_only=False when supplying an initial checkpoint.')

    cache_path = Path(__file__).resolve().parents[1] / 'features' / 'minc_dinov2_cls_patch.pt'

    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    features = cache["features"]
    labels = cache["labels"]
    image_paths = cache["image_paths"]
    class_to_index = cache["class_to_index"]

    path_to_row = {
        path: row
        for row, path in enumerate(image_paths)
    }
    increment_settings = data_settings.copy()
    split = increment_settings.pop("split")

    training_paths, validation_paths, test_paths = read_data(split=split)

    initial_paths, incremental_paths = make_increments(
        training_paths,
        **increment_settings,
    )
    torch.manual_seed(initialization_seed)

    model = make_classifier(
        input_dim=features.shape[1],
        num_classes=len(class_to_index),
    ).to(device)

    initial_weights = copy.deepcopy(model.state_dict())

    if initial_checkpoint is not None:
        initial_checkpoint = Path(initial_checkpoint).resolve()
        model.load_state_dict(torch.load(initial_checkpoint, map_location=device,
                                         weights_only=True))
    run_folder = Path(output_folder)
    run_folder.mkdir(parents=True, exist_ok=False)

    settings = {
        'optimizer_name': optimizer_name,
        'data_settings': data_settings,
        'training_settings': training_settings,
        'start_mode': start_mode,
        'initial_only': initial_only,
        'initial_checkpoint': str(initial_checkpoint) if initial_checkpoint is not None else None,
        'initialization_seed': initialization_seed,
        'device': str(device),
        'cache_path': str(cache_path),
    }
    with (run_folder / 'settings.json').open('w', encoding='utf-8') as file:
        json.dump(settings, file, indent=2)

    split_manifest = {
        'class_to_index': class_to_index,
        'initial_paths': initial_paths,
        'incremental_paths': incremental_paths,
        'validation_paths': validation_paths,
        'test_paths': test_paths,
    }
    with (run_folder / 'splits.json').open('w', encoding='utf-8') as file:
        json.dump(split_manifest, file, indent=2)

    torch.save(initial_weights, run_folder / 'initial_weights.pt')
    if initial_checkpoint is not None:
        torch.save(model.state_dict(), run_folder / 'pretrained_weights.pt')
    stages = [initial_paths] + incremental_paths

    if initial_only:
        stages = stages[:1]

    current_paths = []
    first_stage = 0
    if initial_checkpoint is not None:
        current_paths = initial_paths.copy()
        stages = incremental_paths
        first_stage = 1

    for stage_index, new_paths in enumerate(stages, start=first_stage):

        current_paths.extend(new_paths)


        training_rows = [path_to_row[path] for path in current_paths]
        stage_features = features[training_rows].to(device)
        stage_labels = labels[training_rows].to(device)


        if data_settings["category_mode"] == "new_categories":
            seen_categories = {
                path.split("/")[1] for path in current_paths
            }

            stage_validation_paths = [
                path for path in validation_paths
                if path.split("/")[1] in seen_categories
            ]
        else:
            stage_validation_paths = validation_paths


        validation_rows = [
            path_to_row[path] for path in stage_validation_paths
        ]
        stage_validation_features = features[validation_rows].to(device)
        stage_validation_labels = labels[validation_rows].to(device)

        if start_mode == "cold":
            model.load_state_dict(initial_weights)

        stage_folder = run_folder / f"stage_{stage_index}"
        stage_folder.mkdir()
        model, history, summary = train(
            model=model,
            features=stage_features,
            labels=stage_labels,
            validation_features=stage_validation_features,
            validation_labels=stage_validation_labels,
            optimizer_name=optimizer_name,
            best_checkpoint_path=stage_folder / 'best_validation_weights.pt',
            **training_settings,
        )

        with (stage_folder / "history.json").open("w") as file:
            json.dump(history, file, indent=2)

        with (stage_folder / "summary.json").open("w") as file:
            json.dump(summary, file, indent=2)

        torch.save(model.state_dict(), stage_folder / "weights.pt")

        stage_info = {
            "stage_index": stage_index,
            "training_samples": len(current_paths),
            "validation_samples": len(stage_validation_paths),
            "seen_categories": sorted({
                path.split("/")[1] for path in current_paths
            }),
        }

        with (stage_folder / "stage_info.json").open("w") as file:
            json.dump(stage_info, file, indent=2)

        if summary["termination"] == "nonfinite":
            break

    return run_folder


def evaluate_test(run_folder, device='cpu'):
    run_folder = Path(run_folder)
    cache_path = Path(__file__).resolve().parents[1] / 'features' / 'minc_dinov2_cls_patch.pt'
    cache = torch.load(cache_path, map_location='cpu', weights_only=True)
    manifest = json.loads((run_folder / 'splits.json').read_text())
    lookup = {path: i for i, path in enumerate(cache['image_paths'])}
    indices = torch.tensor([lookup[path] for path in manifest['test_paths']])
    stages = sorted(run_folder.glob('stage_*'), key=lambda p: int(p.name.split('_')[1]))
    checkpoint = stages[-1] / 'weights.pt'
    model = make_classifier(cache['features'].shape[1], len(cache['class_to_index'])).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    classes = len(cache['class_to_index'])
    confusion = torch.zeros((classes, classes), dtype=torch.long)
    loss_sum = 0.0
    
    with torch.inference_mode():
        for batch in indices.split(1024):
            labels = cache['labels'][batch].to(device)
            scores = model(cache['features'][batch].to(device))
            loss_sum += torch.nn.functional.cross_entropy(scores, labels, reduction='sum').item()
            predictions = scores.argmax(dim=1)
            confusion += torch.bincount(labels * classes + predictions,
                                        minlength=classes * classes).reshape(classes, classes).cpu()
            
    result = dict(checkpoint=str(checkpoint), test_samples=len(indices),
                  test_accuracy=confusion.diag().sum().item() / len(indices),
                  test_cross_entropy=loss_sum / len(indices),
                  confusion_matrix=confusion.tolist(), class_to_index=cache['class_to_index'])
    
    (run_folder / 'test_evaluation.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in ('confusion_matrix', 'class_to_index')}, indent=2))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluate-run')
    parser.add_argument('--optimizer', choices=['adam', 'nesterov', 'svrg', 'block'])
    parser.add_argument('--protocol', choices=['balanced', 'category', 'new_categories'])
    parser.add_argument('--start', choices=['warm', 'cold'], default='warm')
    parser.add_argument('--seed', type=int, choices=[0, 1, 2], default=0)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output')
    args = parser.parse_args()
    if args.evaluate_run:
        evaluate_test(args.evaluate_run, args.device)
        raise SystemExit(0)
    if not all([args.optimizer, args.protocol, args.output]):
        parser.error('Training requires --optimizer, --protocol and --output.')
    stage_counts = {'balanced': 5, 'category': 9, 'new_categories': 7}
    data_settings = dict(split=1, split_ratio=0.2,
                         category_mode={'balanced': False, 'category': True,
                                        'new_categories': 'new_categories'}[args.protocol],
                         increments=4, seed=args.seed, initial_categories=5,
                         categories_per_increment=3, category_group_size=3)
    parameters = {
        'adam': dict(learning_rate=0.00225, batch_size=32),
        'nesterov': dict(learning_rate=1.757, momentum=0.991),
        'svrg': dict(learning_rate=2.0, batch_size=64, snapshot_interval=400),
        'block': dict(learning_rate=40.0),
    }
    training_settings = dict(parameters[args.optimizer], regularization=1e-5,
                             seed=args.seed, gradient_tolerance=1e-4,
                             max_optimizer_seconds=300 / stage_counts[args.protocol],
                             record_every_seconds=2.0, stop_on_gradient=False)
    experiment(args.optimizer, data_settings, training_settings,
               start_mode=args.start, initialization_seed=args.seed,
               device=args.device, output_folder=args.output)
