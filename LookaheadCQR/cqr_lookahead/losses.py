from torch import nn, Tensor, cat

import torch


class PinballLoss(nn.Module):
    """
    A pinball loss module
    """

    def __init__(self, tau: float = 0.1):
        """
        :param tau: (float) The target quantile
        """

        super().__init__()

        self._tau = tau

    def forward(self, y: Tensor, y_pred: Tensor) -> Tensor:
        loss = cat(
            (
                ((y - y_pred) * self._tau).unsqueeze(-1),
                ((y_pred - y) * (1 - self._tau)).unsqueeze(-1)
            ),
            dim=-1,
        )
        loss = torch.max(loss, dim=-1)[0]
        loss = loss.mean()

        return loss
