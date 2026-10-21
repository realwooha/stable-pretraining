DirectCLR
=========

DirectCLR implements the fixed-subspace objective from Jing et al.,
`Understanding Dimensional Collapse in Contrastive Self-supervised Learning
<https://arxiv.org/abs/2110.09348>`_ (ICLR2022).
It uses the library's native Module/Manager, dict batches and evaluation
callbacks. The trainable encoder has no trainable projector.

Method and interfaces
---------------------

Two views pass through the backbone separately. Training applies L2-normalized,
temperature-scaled symmetric InfoNCE to the first ``subspace_dim`` coordinates.
The opposite view supplies candidates; the matching sample is the positive.
The full backbone vector is returned under ``embedding`` for downstream use.
Labels are not used in the SSL objective.

.. code-block:: python

   import stable_pretraining as spt
   from torch import nn

   backbone = spt.backbone.from_torchvision("resnet50", weights=None)
   backbone.fc = nn.Identity()
   model = spt.DirectCLR(backbone=backbone, subspace_dim=360, temperature=0.1)

The composable interface is ``spt.Module(forward=spt.forward.directclr,
backbone=..., subspace_dim=360,
directclr_loss=spt.losses.DirectCLRLoss(0.1), optim=...)``.
Use two dictionaries under ``batch["views"]`` for training, a single
``batch["image"]`` for evaluation, and ``spt.Manager`` for the lifecycle.
The default SGD LR0.03 is an API starting point, not the paper's full recipe.

For an isolated installation and a complete small CPU train/restart/frozen-readout
check, follow :doc:`guides/directclr_reproduction`. That generated-data check is
separate from the measured ImageNet and CIFAR experiments.

The loss uses autograd-aware gather with local anchors and rank-offset
positive indices. Ranks require equal nonempty batches and feature widths.
Half-precision normalization/similarities are computed in float32.
The benchmark's ``--batch-size`` is global; accumulation does not enlarge
the actual contrastive negative pool.

Runnable training and evaluation
--------------------------------

``benchmarks/imagenet100/directclr.py`` provides ``prepare``, ``train`` and
``evaluate`` stages. The following small CPU smoke test checks execution;
synthetic accuracy is not a model-quality result:

.. code-block:: bash

   python benchmarks/imagenet100/directclr.py train \
       --dataset fake --backbone resnet18 --dim 64 \
       --accelerator cpu --precision 32-true --workers 0 \
       --epochs 2 --batch-size 8 --max-steps 2 \
       --output /tmp/directclr-smoke

For a new full encoder experiment on suitably provisioned hardware:

.. code-block:: bash

   python benchmarks/imagenet100/directclr.py prepare \
       --data-dir /data/imagenet100 \
       --data-revision 0519dc2f402a3a18c6e57f7913db059215eee25b
   python benchmarks/imagenet100/directclr.py train \
       --data-dir /data/imagenet100 \
       --data-revision 0519dc2f402a3a18c6e57f7913db059215eee25b \
       --method directclr --backbone resnet50 --dim 256 \
       --temperature 0.2 --epochs 400 --batch-size 512 --lr 0.6 \
       --augmentation reference --precision 16-mixed --devices 4 \
       --seed 0 --output /data/runs/directclr-fresh400

The actual LARS peak is1.2 because the CLI base rate is scaled by
batch-size/256. This recipe differs from the paper's100-epoch ImageNet1K,
batch4096,LR4.8 experiment. Training stops according to successful optimizer
updates, not restored Lightning epoch labels. Do not extend a completed
200-epoch cosine schedule into400; use a new full schedule.

The ``evaluate`` stage supports frozen full-representation evaluation:

.. code-block:: bash

   python benchmarks/imagenet100/directclr.py evaluate \
       --data-dir /data/imagenet100 \
       --data-revision 0519dc2f402a3a18c6e57f7913db059215eee25b \
       --checkpoint /data/runs/directclr-fresh400/path/to/last.ckpt \
       --accelerator gpu --devices 1 --probe-epochs 100 \
       --probe-lr 60 --probe-batch-size 1024 \
       --output /data/runs/directclr-frozen-eval

Supply a trusted actual checkpoint path. The staged evaluator is an entry
point, not a reconstruction of every historical dev-selection/refit operation.
Use the measured split and selection protocol in the linked report when
interpreting the audited results. The commands above are reproduction
instructions, not new jobs launched while preparing this submission.

Results and interpretation
--------------------------

The dev-selected temperature0.2/fresh400-epoch encoder reaches79.80% final-val
Top-1 after a new full-train frozen classifier. CIFAR-10 supervised transfer
of the same pretrained encoder reaches93.57% on one final test; this is not
CIFAR SSL or frozen evaluation. The separate cached-center readout diagnostic
is70.94%/75.07%/77.13%/81.19% for first256/last256/fixedQ256/full2048.
Full2048 has more head parameters; Q is posthoc and is never part of SSL.

All four paired spectra have full numerical rank under the fixed thresholds.
Spectral uniformity does not monotonically rank classification accuracy.
Single seeds, repeated dev and prior val/test exposure preclude significance
or unique causal claims. The source160px/crop224 preprocessing, SSL exposure
to unlabeled dev images, mixed precision, DDP and recovery differences are
recorded. The early matched100-epoch control is a separate negative result,
not a baseline for the later400-epoch score.

.. toctree::
   :maxdepth: 1

   guides/directclr_experiments
   guides/directclr_reproduction
