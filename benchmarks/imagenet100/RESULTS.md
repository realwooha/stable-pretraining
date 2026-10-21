# ImageNet-100: DirectCLR

The current audited 400-epoch ResNet-50 experiment reaches **79.80% Top-1 /94.98% Top-5** on one final 5,000-image validation evaluation. Temperature/budget selection uses dev; final validation does not select the model. Source images are pre-resized to 160 px and actual SSL/cache transforms crop 224. This is an ImageNet-100 course-scale validation, not the paper's ImageNet-1K reproduction.

See the [training and analysis report](results/directclr/TRAINING-REPORT.md) for SSL loss traces, temperature/budget dev selection, frozen-head evaluation, separate CIFAR supervised transfer, matched-capacity readouts, paired spectra and limitations. [Validation status](results/directclr/VALIDATION.md) distinguishes fresh focused-source tests from historical experiment audits and the whole-project unit-suite check.

All figure inputs and their provenance are under[results/directclr/data](results/directclr/data). Figures do not contain invented per-epoch dev/test accuracy. The earlier matched 100-epoch DirectCLR 62.26% versus linear-projector 64.42% result is retained; it cannot be compared directly with the later 400-epoch result.
