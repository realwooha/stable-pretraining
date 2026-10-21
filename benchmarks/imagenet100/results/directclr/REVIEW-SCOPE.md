# Review scope and test-fixture portability

The scientific method is implemented in the DirectCLR loss, method class and
dict-native forward, with exports, focused tests and the staged benchmark.
No test-fixture fix changes that loss, its temperature, gradient checks,
distributed tolerance or downstream scores.

Two existing general-purpose test files contain separate portability repairs:

| File | Observed issue | Repair and preserved contract |
|---|---|---|
| `stable_pretraining/tests/unit/test_dino_visualizers.py` | Synthetic noise/attention were created on CPU while features could be on MPS | Build noise like the token tensor and attention with the pooled feature's device/dtype; retain the original visualizer assertions |
| `stable_pretraining/tests/unit/test_latent_viz.py` | Fork-based CPU DDP failed in whole-suite execution; spawn needs a serializable model | Use spawn and a class-defined forward; isolate optional entry-point callbacks owning locks inside this test; retain distributed projection checks and tolerances |

The latter fixture does not test optional logger/environment callback startup;
that scope exclusion is explicit. No production callback factory is disabled.
The prior baseline comparison reproduced the same three failures before these
two fixture repairs; individual-test success was not treated as whole-suite
success. Tests are not skipped and assertions are not relaxed by these repairs.

The human submission packet supplies a separate test-fixture patch and file
list so these changes can be reviewed independently. The complete patch and ZIP
include them once; do not apply the fixture patch a second time after applying
the complete patch. Follow the course's human-only commit and PR-update rule.

Local PyTorch unit shards, JAX CPU, combined coverage and regression evidence
are described in `VALIDATION.md`. A fresh installed-wheel CPU check adds
installation and command-level evidence; it does not replace hosted CI or
optional GPU/download validation. The incomplete 400-epoch matched baseline
and additional encoder seeds remain scientific limitations, not test failures.
