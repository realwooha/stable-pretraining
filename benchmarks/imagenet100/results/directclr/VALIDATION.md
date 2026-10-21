# DirectCLR validation scopes

## Existing source-wide local checks

The earlier v3 source packet has four PyTorch unit shards totaling **3790 passed,
0 failed**, separate JAX CPU **183 passed, 2 skipped**, and complete local
regression **54 passed, 1 skipped**. Its five actual coverage files combine to
**97.52% (19909/20416 production statements)** and pass the unchanged 95% gate.
No correctness assertion, tolerance, coverage exclusion or threshold was weakened.

All existing production and test Python files in this v4 candidate are byte-for-byte
identical to v3. This update adds documentation, tested dependency constraints and
review material. The source-wide counts above are retained evidence; they are not
misreported as a new whole-project run for this documentation update.

## New installation and command-level check

A new Python 3.12 virtual environment has system site-packages disabled. A
non-editable wheel was built from the exact public PR commit plus the reviewed
patch, installed with declared dependencies, and passes `pip check`. Imports
resolve inside the new environment, not the original checkout or environment.
The tested constraints are recorded in `data/reproduction-constraints.txt`.

The documented CPU commands were executed on generated images, including:

- Four initial encoder updates on a planned eight-update schedule.
- Checkpoint restoration followed by exactly four more updates; both optimizer
  and scheduler states are saved and checkpoint tensors are finite.
- A new two-epoch classifier using frozen full-width 512-dimensional ResNet-18
  features, while the contrastive prefix has width 64.
- Unchanged encoder checkpoint SHA across evaluation; 32 generated fit rows and
  16 generated evaluation rows. These synthetic scores are not scientific results.

The actual checkpoint path is read from the training summary, because Manager
may use a run-specific registry directory. The original hard-coded path assumption
was caught and corrected in the new instructions, then the commands were rerun.

The fresh-environment DirectCLR unit/distributed/benchmark/probe tests plus native
Manager resume regressions pass with **64 passed, 1 CUDA-only skip**. Initially,
the restricted sandbox denied a local Gloo socket (`uv_bind: operation not
permitted`), giving 63 passes, that one failure and one skip. Allowing local
process communication gives the full passing selection; no code/assertion change
or new skip was used to hide the restriction.

## Separate general-purpose fixture repairs

`REVIEW-SCOPE.md` explains the two pre-existing fixture portability changes.
They are independently reviewable and do not change the DirectCLR objective,
production callbacks or original test assertions. The prior baseline reproduced
the same three whole-suite failures before the fixture repairs.

## Remaining validation limits

Hosted CI has not been run by the agent. The complete optional GPU/download
matrix is not claimed to pass. The earlier offline integration selection had
24 passes, two unmarked network-dependent failures and one skip. With public data
access the MNIST case passed; the existing video test skipped, and a direct
check found a missing optional torchcodec decoder. That skip does not prove
video decoding works.

Historical experiment audits, their CPU regression counts and actual job-release
checks are separate from these source tests and are not added to the pass totals.
The extra 400-epoch linear control was cancelled before allocation, with no score;
CIFAR controls remain optional pending independently audited results. Single seeds,
repeated dev and prior held-out exposure remain scientific limitations.

The human must review, commit, push and update the original PR according to the
course rule. No agent commit, push or PR update was performed.
