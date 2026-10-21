# DirectCLR: installation and CPU reproduction check

This is a small execution check using generated images. It checks installation,
training, checkpoint restoration and frozen evaluation. It does not measure
representation quality or reproduce the ImageNet experiments.

## Install from the reviewed source

From a clean checkout containing this PR, create a separate environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -c benchmarks/imagenet100/results/directclr/data/reproduction-constraints.txt . pytest
python -m pip check
```

The constraints record the macOS arm64/Python 3.12 CPU validation environment.
They are not a promise that all platforms or optional backends are equivalent.
This uses a non-editable installation; the original experiment environment is
not required. The synthetic commands below need no data download or GPU.

## Train, stop at four updates, then resume to eight

Use a new output directory. The two invocations retain the same two-epoch
schedule and global batch of eight; the first invocation is an intentional
interruption rather than a completed experiment.

```bash
RUN_DIR=$(mktemp -d)
python benchmarks/imagenet100/directclr.py train \
  --dataset fake --backbone resnet18 --dim 64 \
  --accelerator cpu --precision 32-true --workers 0 --cpu-threads 2 \
  --epochs 2 --batch-size 8 --val-every 1 --max-steps 4 \
  --output "$RUN_DIR/train"

CKPT=$(python -c 'import json, sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$RUN_DIR/train/training_summary.json")
python benchmarks/imagenet100/directclr.py train \
  --dataset fake --backbone resnet18 --dim 64 \
  --accelerator cpu --precision 32-true --workers 0 --cpu-threads 2 \
  --epochs 2 --batch-size 8 --val-every 1 --max-steps 8 \
  --resume "$CKPT" \
  --output "$RUN_DIR/train"
```

The first `training_summary.json` should have `global_step=4`,
`planned_total_steps=8` and `complete_schedule=false`. After recovery it should
have `global_step=8`, `updates_this_invocation=4` and `complete_schedule=true`.
The checkpoint includes encoder state, BatchNorm buffers, both optimizers and
both schedulers. Separate Manager regressions compare interrupted and
uninterrupted training for the class and composable APIs; a successful command
alone does not establish cross-device deterministic equivalence.
Read the actual checkpoint path from the summary: Manager/registry setup can
place checkpoints in a run-specific directory rather than `output/checkpoints`.

## Fit a new classifier on frozen full-width features

```bash
CKPT=$(python -c 'import json, sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$RUN_DIR/train/training_summary.json")
python benchmarks/imagenet100/directclr.py evaluate \
  --dataset fake --backbone resnet18 \
  --accelerator cpu --precision 32-true --workers 0 --cpu-threads 2 \
  --checkpoint "$CKPT" \
  --probe-epochs 2 --probe-batch-size 8 --probe-lr 1 \
  --output "$RUN_DIR/evaluate"
```

The generated dataset has 32 training and 16 evaluation rows. The ResNet-18
feature cache has width 512, even though the contrastive loss used only the
first 64 coordinates. The new classifier trains on frozen cached features;
the encoder checkpoint SHA remains unchanged. Any accuracy on these generated
images is an execution artifact and must not enter the scientific result table.

## Check the native interfaces

```bash
python -m pytest -q -p no:cacheprovider \
  stable_pretraining/tests/unit/test_directclr.py \
  stable_pretraining/tests/unit/test_directclr_distributed.py \
  stable_pretraining/tests/unit/test_directclr_benchmark.py \
  stable_pretraining/tests/unit/test_directclr_probe_sweep.py \
  stable_pretraining/tests/regression/test_directclr_native_lifecycle.py
```

For distributed tests on macOS, the prior validation used the loopback interface
`GLOO_SOCKET_IFNAME=lo0`; use the appropriate interface for your own system.
Hosted CI and the complete optional GPU/download matrix remain separate checks.
