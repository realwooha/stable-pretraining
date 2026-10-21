"""DirectCLR from Understanding Dimensional Collapse (Jing et al., ICLR 2022).

Reference: https://arxiv.org/abs/2110.09348
"""

from typing import Any

from torch import nn

from ..forward import directclr
from ..losses import DirectCLRLoss
from ..module import Module


class DirectCLR(Module):
    """Compose DirectCLR with a user-supplied backbone and the native trainer.

    Args:
        backbone: Encoder returning flat ``[N, D]`` features, without a classifier.
        subspace_dim: Number of leading feature coordinates used in the loss.
            Must be between one and the backbone width. Defaults to 360 for
            the reference ResNet-50's 2048-dimensional representation.
        temperature: Positive finite contrastive temperature.
        optim: Module optimizer/scheduler configuration. Defaults to SGD with
            learning rate 0.03, momentum 0.9 and weight decay 1e-6. This default
            is a starting point, not the paper's large-batch training recipe.
        hparams: Additional experiment metadata recorded in the run registry.

    Note:
        Training uses dict batches with two views; evaluation uses a single
        ``image`` batch and returns the complete backbone representation.
        Labels never enter the SSL loss. Online probes detach their inputs.

    Example:
        Use the same Module/Manager API as the composable forward function::

            import stable_pretraining as spt
            from torch import nn

            backbone = spt.backbone.from_torchvision("resnet50", weights=None)
            backbone.fc = nn.Identity()
            model = spt.DirectCLR(backbone=backbone, subspace_dim=360)
    """

    def __init__(
        self,
        backbone: nn.Module,
        subspace_dim: int = 360,
        temperature: float = 0.1,
        optim: dict[str, Any] | None = None,
        hparams: dict[str, Any] | None = None,
    ) -> None:
        if (
            isinstance(subspace_dim, bool)
            or not isinstance(subspace_dim, int)
            or subspace_dim < 1
        ):
            raise ValueError("subspace_dim must be a positive integer")
        if not isinstance(backbone, nn.Module):
            raise TypeError("backbone must be a torch.nn.Module")
        super().__init__(
            forward=directclr,
            backbone=backbone,
            subspace_dim=subspace_dim,
            directclr_loss=DirectCLRLoss(temperature),
            optim=optim
            if optim is not None
            else {
                "optimizer": {
                    "type": "SGD",
                    "lr": 0.03,
                    "momentum": 0.9,
                    "weight_decay": 1e-6,
                }
            },
            hparams={
                **(hparams or {}),
                "method": "directclr",
                "subspace_dim": subspace_dim,
                "temperature": temperature,
            },
        )
