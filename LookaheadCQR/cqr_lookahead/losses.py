from typing import Dict
from torch import nn, Tensor, cat
from abc import ABC, abstractmethod
from LookaheadCQR.cqr_lookahead.defaults import OUTPUT_KEY, PREDICTION_KEY

import torch


class LossComponent(nn.Module, ABC):
    """
    An abstract API class for loss models
    """

    def __init__(self):
        super(LossComponent, self).__init__()

    @abstractmethod
    def forward(self, inputs: Dict) -> Tensor:
        """
        The forward logic of the loss class.

        :param inputs: (Dict) Either a dictionary with the predictions from the forward pass and the ground truth
        outputs. The possible keys are specified by the following variables:
            INPUT_KEY
            OUTPUT_KEY
            PREDICTION_KEY

        Which can be found under .../LookaheadCQR/cqr_lookahead/defaults.py

        :return: (Tensor) A scalar loss.
        """

        raise NotImplemented

    def __call__(self, inputs: Dict) -> Tensor:
        return self.forward(inputs=inputs)


class ModuleLoss(LossComponent):
    """
    A LossComponent which takes in a PyTorch loss Module and decompose the inputs according to the module's
    expected API.
    """

    def __init__(self, model: nn.Module):
        """
        :param model: (PyTorch Module) The loss model, containing the computation logic.
        """

        super().__init__()

        self._model = model

    def forward(self, inputs: Dict) -> Tensor:
        """
        Basically a wrapper around the forward of the inner model, which decompose the inputs to the expected
        structure expected by the PyTorch module.

        :param inputs: (dict) The outputs of the forward pass of the model along with the ground-truth labels.
        :return: (Tensor) A scalar Tensor representing the aggregated loss
        """

        y_pred = inputs[PREDICTION_KEY]
        y = inputs[OUTPUT_KEY]

        loss = self._model(y, y_pred)

        return loss


class PinballLoss(LossComponent):
    """
    A pinball loss module
    """

    def __init__(self, tau: float = 0.1):
        """
        :param tau: (float) The target quantile
        """

        super().__init__()

        self._tau = tau

    def forward(self, inputs: Dict) -> Tensor:
        y_pred = inputs[PREDICTION_KEY]
        y = inputs[OUTPUT_KEY]

        loss = cat(
            (
                ((y - y_pred) * self._tau).unsqueeze(-1),
                ((y_pred - y) * (1 - self._tau)).unsqueeze(-1)
            ),
            dim=-1,
        )
        loss = torch.max(loss, dim=-1)
        loss = loss.mean()

        return loss


class LpLossModel(nn.Module):
    """
    A quick wrapper implementation of the Lp loss
    """

    def __init__(self, p: int = 3):
        """
        :param p: degree of the norm
        """

        super().__init__()

        self._p = p

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        loss = (x - y).abs().pow(self._p).pow(1 / self._p).mean()
        return loss


class LpLoss(ModuleLoss):
    """
    A loss module for Lp losses
    """

    def __init__(self, p: int = 2):
        """
        :param p: degree of the loss
        """

        self._p = p

        if p == 1:
            model = torch.nn.L1Loss()

        elif p == 2:
            model = torch.nn.MSELoss()

        else:
            model = LpLossModel(p=p)

        super().__init__(model=model)
