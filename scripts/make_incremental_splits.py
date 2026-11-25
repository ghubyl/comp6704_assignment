from pathlib import Path
import random


def read_data(split=1):
    project_folder = Path(__file__).resolve().parents[1]
    data_root = project_folder / "data" / "minc-2500"

    train_paths = (
        data_root / "labels" / f"train{split}.txt"
    ).read_text().splitlines()

    validation_paths = (
        data_root / "labels" / f"validate{split}.txt"
    ).read_text().splitlines()

    test_paths = (
        data_root / "labels" / f"test{split}.txt"
    ).read_text().splitlines()

    paths_by_category = {}

    for path in train_paths:
        category = path.split("/")[1]

        if category not in paths_by_category:
            paths_by_category[category] = []

        paths_by_category[category].append(path)

    return paths_by_category, validation_paths, test_paths

def make_increments(paths_by_category, split_ratio, category_mode, increments, seed,
                    initial_categories=5, categories_per_increment=2, category_group_size=1):
    rng = random.Random(seed)

    if category_mode == "new_categories":
        categories = list(paths_by_category)
        rng.shuffle(categories)

        initial_paths = []
        for category in categories[:initial_categories]:
            initial_paths.extend(paths_by_category[category])

        incremental_paths = []
        for start in range(initial_categories, len(categories), categories_per_increment):
            new_paths = []
            for category in categories[start:start + categories_per_increment]:
                new_paths.extend(paths_by_category[category])
            incremental_paths.append(new_paths)
        return initial_paths, incremental_paths

    initial_by_category = {}
    remaining_by_category = {}
    remaining_by_increment= {}

    for category , paths in paths_by_category.items():
        shuffled_paths = paths.copy()
        rng.shuffle(shuffled_paths)
        split_index = int(len(shuffled_paths) * split_ratio)
        initial_by_category[category] = shuffled_paths[:split_index]
        remaining_by_category[category] = shuffled_paths[split_index:]

    if category_mode == True:
        categories = list(remaining_by_category)
        rng.shuffle(categories)
        if category_group_size < 1:
            raise ValueError('category_group_size must be positive.')
        for i, start in enumerate(range(0, len(categories), category_group_size)):
            remaining_by_increment[i] = [
                path
                for category in categories[start:start + category_group_size]
                for path in remaining_by_category[category]
            ]

    else:
        for i in range(increments):
            remaining_by_increment[i] = []

        for category, paths in remaining_by_category.items():
            x = len(paths)

            for i in range(increments):
                start = i * x // increments
                end = (i + 1) * x // increments
                remaining_by_increment[i].extend(paths[start:end])

    initial_paths = []

    for paths in initial_by_category.values():
        initial_paths.extend(paths)

    incremental_paths = list(remaining_by_increment.values())
    return initial_paths, incremental_paths



def main():

    split = 1
    split_ratio = 0.2
    category_mode = False
    initial_categories = 5
    categories_per_increment = 2
    increments = 4
    seed = 0

    grouped_paths, validation_paths, test_paths = read_data(split)

    initial_paths, incremental_paths = make_increments(
        grouped_paths,
        split_ratio,
        category_mode,
        increments,
        seed,
        initial_categories,
        categories_per_increment,
    )


    print("Initial training:", len(initial_paths))

    for i, paths in enumerate(incremental_paths):
        print(f"Increment {i}: {len(paths)}")


if __name__ == "__main__":
    main()






