SETUP AND INSTALL

Requires Python 3.11, Git, and an NVIDIA GPU.
Run commands from the project folder.

1. Create and activate an environment

Linux:
  python3.11 -m venv .venv
  source .venv/bin/activate

Windows PowerShell:
  py -3.11 -m venv .venv
  .\.venv\Scripts\Activate.ps1

2. Install dependencies

  python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
  python -m pip install -r requirements.txt

3. Prepare MINC-2500

Download from:
  http://opensurfaces.cs.cornell.edu/publications/minc/

Extract to:
  data/minc-2500/categories.txt
  data/minc-2500/images/
  data/minc-2500/labels/train1.txt
  data/minc-2500/labels/validate1.txt
  data/minc-2500/labels/test1.txt

4. Prepare DINOv2

  git clone https://github.com/facebookresearch/dinov2.git vendor/dinov2
  git -C vendor/dinov2 checkout 7764ea0f912e53c92e82eb78a2a1631e92725fc8

Download:
  https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth

Save as:
  checkpoints/dinov2_vits14_pretrain.pth

Extract features:
  python scripts/extract_features.py

Skip this step if the matching cache is already available at:
  features/minc_dinov2_cls_patch.pt

5. Run experiments

Training:
  python scripts/experiment.py --optimizer svrg --protocol balanced --start warm --seed 0 --device cuda:0 --output results/example

Test evaluation:
  python scripts/experiment.py --evaluate-run results/example --device cpu

Options:
  --optimizer: adam, nesterov, svrg, block
  --protocol: balanced, category, new_categories
  --start: warm, cold
  --seed: 0, 1, 2

The scripts cover feature extraction, data splitting, training, and evaluation. `extract_features.py` caches DINOv2 features, and `make_incremental_splits.py` creates the three data-arrival protocols. `classifier.py` defines the linear classifier, while `optimizers.py` implements Adam, Nesterov, SVRG, and class-block updates. `train.py` handles training and metric recording, and `experiment.py` runs warm- or cold-start experiments and evaluates saved models on the test set.
