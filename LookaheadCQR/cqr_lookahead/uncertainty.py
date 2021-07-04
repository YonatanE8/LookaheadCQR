from LookaheadCQR.cqr_lookahead.losses import PinballLoss
from LookaheadCQR.cqr_lookahead.models import CQRQuantileRegressor, init_weights
from LookaheadCQR.lookahead.models.models import *
from LookaheadCQR.lookahead.models.uncertainty import UncertModel, get_l2_reg

import torch
import numpy as np


class CQR(UncertModel):
    def __init__(
            self,
            num_features: int,
            tau: (int, int) = (0.1, 0.9),
            alpha: float = 0.,
            lr: float = 0.1,
            num_iter: int = 100,
            train_calib_split: float = 0.8,
            batch_size: int = 16,
            n_forwards: int = 25,
    ):
        self.model = CQRQuantileRegressor(
            in_channels=num_features,
        )
        super().__init__(num_features, model=self.model)

        self.init_models()
        self.tune_alpha = alpha < 0
        self.alpha = float(alpha)
        self.tau = tau
        self.lr = lr
        self.num_iter = num_iter
        self.train_calib_split = train_calib_split
        self.pinball_loss_low = PinballLoss(tau[0])
        self.pinball_loss_high = PinballLoss(tau[1])
        self.conformity_quantile = 0
        self.batch_size = batch_size
        self.n_forwards = n_forwards

    def init_models(self):
        """ Initializes the models """
        init_weights(self.model)

    def fit(
            self,
            x: np.ndarray,
            y: np.ndarray,
    ):
        """ Trains Quantile Regression Model for Uncertainty """
        n = x.shape[0]
        train_set_size = int(n * self.train_calib_split)
        inputs = Variable(torch.from_numpy(x)).float().requires_grad_(True)
        labels = Variable(torch.from_numpy(y).float().unsqueeze(1))

        # Divide to train & calibration sets
        train_set_inputs = inputs[:train_set_size]
        train_set_labelsl = labels[:train_set_size]
        calib_set_inputs = inputs[train_set_size:]
        calib_set_labelsl = labels[train_set_size:]

        # Define the optimizer
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr
        )

        # Fit the quantile regressor
        n_epochs = max((self.num_iter // train_set_size), 1)
        n_batchs = (
            (train_set_size // self.batch_size) +
            (1 if (train_set_size % self.batch_size) else 0)
        )
        for epoch in range(n_epochs):
            # Prepare for fitting the quantile regressor
            random_inds = np.random.choice(
                np.arange(train_set_size),
                size=train_set_size,
                replace=False,
            )
            random_inds = torch.from_numpy(random_inds)

            # Divide training data to batches
            random_training_inputs = train_set_inputs[random_inds]
            random_training_labels = train_set_labelsl[random_inds]

            for iter in range(n_batchs):
                optimizer.zero_grad()

                batch_start_ind = iter * self.batch_size
                batch_end_ind = (iter + 1) * self.batch_size

                if batch_end_ind > train_set_size:
                    batch_end_ind = train_set_size

                inputs_batch = random_training_inputs[
                               batch_start_ind:batch_end_ind
                               ]
                y = random_training_labels[
                               batch_start_ind:batch_end_ind
                               ].squeeze()

                # Generate multiple predictions with the same X via the MC dropout
                y_pred = [
                    self.model(inputs_batch)
                    for _ in range(self.n_forwards)
                ]
                y_pred = torch.cat(y_pred, dim=0)
                pred_y_low = y_pred[..., 0]
                pred_y_high = y_pred[..., 1]

                extended_y = y.repeat(self.n_forwards)

                loss_low = self.pinball_loss_low(extended_y, pred_y_low)
                loss_high = self.pinball_loss_high(extended_y, pred_y_high)
                loss = loss_low + loss_high

                l2_reg = get_l2_reg(self.model, self.alpha, train_set_size)
                objective = loss + self.alpha * l2_reg

                # objective.backward()
                objective.backward(retain_graph=True)
                optimizer.step()

        training_loss = loss.item()
        objectives = objective.item()

        # --- Calibrate the quantile regressor
        # Estimate the quantiles over the calibration set
        q_hat = self.model(calib_set_inputs)
        q_hat_low = q_hat[..., 0]
        q_hat_high = q_hat[..., 1]

        # Compute the conformity scores over the calibration set
        conformity_scores = self._compute_conformity_scores(
            q_hat_low,
            q_hat_high,
            calib_set_labelsl.squeeze(),
        )

        # Compute the (1 - alpha) * (1 + 1 / |I2|)-th empirical quantile over the
        # calibration set
        q = (1 - self.tau[0]) * (1 + (1 / calib_set_labelsl.shape[0]))
        self.conformity_quantile = conformity_scores.quantile(q=q, dim=0)

        metrics = [
            training_loss,
            0 if self.alpha == 0 else l2_reg,
            objectives,
        ]

        return metrics

    def _compute_conformity_scores(
            self,
            q_hat_low: torch.Tensor,
            q_hat_high: torch.Tensor,
            y: torch.Tensor,
    ) -> torch.Tensor:
        low_error = q_hat_low - y
        high_error = y - q_hat_high
        scores = torch.max(low_error, high_error)
        return scores

    def lu(self, inputs):
        """ Computes lower bound and upper bound uncertainty values """
        # Run N forwards and average to produce better estimates using the MC dropout
        preds = [
            self.model(inputs).unsqueeze(0)
            for _ in range(self.n_forwards)
        ]
        pred = torch.cat(preds, 0).mean(dim=0)
        low_pred = pred[..., 0]
        high_pred = pred[..., 1]
        lower = low_pred - self.conformity_quantile
        upper = high_pred + self.conformity_quantile

        return lower, upper
