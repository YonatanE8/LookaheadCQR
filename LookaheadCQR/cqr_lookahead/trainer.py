from abc import ABC
from torch import nn
from typing import List, Callable, Tuple, Sequence
from torch.utils.data.dataloader import DataLoader
from LookaheadCQR.cqr_lookahead.loggers import Logger
from LookaheadCQR.cqr_lookahead.optim import Optimizer
from LookaheadCQR.cqr_lookahead.losses import LossComponent
from LookaheadCQR.cqr_lookahead.defaults import INPUT_KEY, OUTPUT_KEY


import os
import tqdm
import torch
import numpy as np


class BaseTrainer(ABC):
    """
    A class abstracting the various tasks of training models.
    Provides methods at multiple levels of granularity:
    - Multiple epochs (fit)
    - Single epoch (train_epoch/test_epoch)
    - Single batch (train_batch/test_batch)
    """

    def __init__(
            self,
            model: nn.Module,
            loss_fn: LossComponent,
            evaluation_metric: LossComponent,
            optimizer: Optimizer,
            logger: Logger,
            device: torch.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu'
            ),
            max_iterations_per_epcoch: int = float('inf'),
    ):
        """
        Initialize the trainer.

        :param model: Instance of the _model to train.
        :param loss_fn: A LossComponent object, which serves as the loss
        function to evaluate with.
        :param evaluation_metric: A LossComponent object,
        which takes in the predictions and ground truth, and returns float,
        representing an 'accuracy' score.
        :param optimizer: The optimizer to train with.
        :param device: torch.device to run training on (CPU or GPU).
        :param max_iterations_per_epcoch: Upper limit on the number of iterations (i.e.
        batches) to compute in each epoch.
        :param logger: (Logger) A logger for saving all of the required
        intermediate results.
        """

        self._model = model.to(device)
        self._loss_fn = loss_fn
        self._evaluation_metric = evaluation_metric
        self._optimizer = optimizer
        self._device = device
        self._max_iterations_per_epcoch = max_iterations_per_epcoch
        self._logger = logger
        self._save_path_dir = logger.save_dir

    def fit(self,
            dl_train: DataLoader,
            dl_val: DataLoader,
            num_epochs: int = 10,
            checkpoints: bool = False,
            checkpoints_mode: str = 'min',
            early_stopping: int = None,
            ) -> Tuple[Sequence[float], Sequence[float],
                       Sequence[float], Sequence[float]]:
        """
        Trains the _model for multiple epochs with a given training set,
        and calculates validation loss over a given validation set.

        :param dl_train: Dataloader for the training set.
        :param dl_val: Dataloader for the validation set.
        :param num_epochs: Number of epochs to train for.
        :param checkpoints: Whether to save _model to file every time the
            tests set accuracy improves. Should be a string containing a
            filename without extension.
        :param checkpoints_mode: (str) Whether to optimize for minimum, or maximum,
        score.
        :param early_stopping: Whether to stop training early if there is no
            tests loss improvement for this number of epochs.

        :return: A tuple with four lists, containing, in that order, the train loss and
         accuracy and the tests loss and accuracy.
        """

        best_acc = None
        epochs_without_improvement = 0
        train_loss, train_acc, eval_loss, eval_acc = [], [], [], [],
        for epoch in range(num_epochs):
            print(f'\n--- EPOCH {epoch + 1}/{num_epochs} ---')

            loss, acc = self.train_epoch(dl_train=dl_train)
            train_loss.append(loss)
            train_acc.append(acc)
            self._logger.log_variable(loss, f"train_loss")
            self._logger.log_variable(acc, f"train_acc")

            # Perform the schedulers step - if relevant
            self._optimizer.schedulers_step(np.mean(loss).item())

            # Run an evaluation of the _model & save the results
            loss, acc = self.test_epoch(dl_test=dl_val)
            eval_loss.append(loss)
            eval_acc.append(acc)
            self._logger.log_variable(loss, f"eval_loss")
            self._logger.log_variable(acc, f"eval_acc")

            if epoch == 0:
                best_acc = acc
                epochs_without_improvement = 0
                save_checkpoint = True

            else:
                if checkpoints_mode == 'max' and acc > best_acc:
                    best_acc = acc
                    save_checkpoint = True
                    epochs_without_improvement = 0

                elif checkpoints_mode == 'min' and acc < best_acc:
                    best_acc = acc
                    save_checkpoint = True
                    epochs_without_improvement = 0

                else:
                    save_checkpoint = False
                    epochs_without_improvement += 1

            # Create a checkpoint after each epoch if applicable
            if checkpoints and save_checkpoint:
                print(f"\nSaving Checkpont at epoch {epoch + 1}")

                saved_object = {
                    'model': self._model.state_dict(),
                }
                saving_path = os.path.join(
                    self._save_path_dir, f"BestModel.PyTorchModule"
                )
                torch.save(
                    obj=saved_object,
                    f=saving_path
                )
                self._logger.flush(
                    variables=[n for n in self._logger.logged_vars if 'train' in n],
                    save_dir=self._logger.save_dir_train,
                )
                self._logger.flush(
                    variables=[n for n in self._logger.logged_vars if 'eval' in n],
                    save_dir=self._logger.save_dir_eval,
                )

            # We haven't improved at all in the last 'early_stopping' epochs
            if (
                    early_stopping is not None and
                    epochs_without_improvement == early_stopping
            ):
                print(f"\nSaving Checkpont at epoch {epoch + 1}")
                saved_object = {
                    'model': self._model.state_dict(),
                }
                saving_path = os.path.join(
                    self._save_path_dir, f"Checkpoint_Epoch_{epoch}.PyTorchModule"
                )
                torch.save(
                    obj=saved_object,
                    f=saving_path
                )
                self._logger.flush(
                    variables=[n for n in self._logger.logged_vars if 'train' in n],
                    save_dir=self._logger.save_dir_train,
                )
                self._logger.flush(
                    variables=[n for n in self._logger.logged_vars if 'eval' in n],
                    save_dir=self._logger.save_dir_eval,
                )

                print("Reached the Early Stop condition.\nStopping the training.")
                break

        print(f"\nSaving the final Checkpont")
        saved_object = {
            'model': self._model.state_dict(),
        }
        saving_path = os.path.join(
            self._save_path_dir, "LastModel.PyTorchModule"
        )
        torch.save(
            obj=saved_object,
            f=saving_path
        )

        self._logger.log_variable(train_loss, "fit_train_loss", ignore_cap=True)
        self._logger.log_variable(train_acc, "fit_train_acc", ignore_cap=True)
        self._logger.log_variable(eval_loss, "fit_eval_loss", ignore_cap=True)
        self._logger.log_variable(eval_acc, "fit_eval_acc", ignore_cap=True)

        self._logger.flush(
            variables=[n for n in self._logger.logged_vars if 'train' in n],
            save_dir=self._logger.save_dir_train,
        )
        self._logger.flush(
            variables=[n for n in self._logger.logged_vars if 'eval' in n],
            save_dir=self._logger.save_dir_eval,
        )

        return train_loss, train_acc, eval_loss, eval_acc

    def evaluate(
            self,
            dl_test: DataLoader,
            ignore_cap: bool = False,
    ) -> Sequence[float]:
        """
        Run a single evaluation epoch on an held out test set.

        :param dl_test: Dataloader for the test set.
        :param ignore_cap: (bool) Whether to ignore the logger cap or not.
        """

        print(f'\n--- Evaluating Test Set ---')
        loss, acc = self.test_epoch(dl_test=dl_test)
        self._logger.log_variable(loss, f"eval_loss", ignore_cap=ignore_cap)
        self._logger.log_variable(acc, f"eval_acc", ignore_cap=ignore_cap)

        self._logger.flush(
            variables=[n for n in self._logger.logged_vars if 'eval' in n],
            save_dir=self._logger.save_dir_eval,
        )

        return loss, acc

    def train_epoch(self, dl_train: DataLoader) -> Tuple[float, float]:
        """
        Train once over a training set (single epoch).

        :param dl_train: DataLoader for the training set.

        :return: A tuple containing the aggregated training epoch loss and accuracy
        """

        self._model.train(True)
        loss, accuracy = self._foreach_batch(dl_train, self.train_batch)

        return np.mean(loss).item(), np.mean(accuracy).item()

    def test_epoch(self, dl_test: DataLoader) -> Tuple[float, float]:
        """
        Evaluate a model once over a tests set (single epoch).

        :param dl_test: DataLoader for the tests set.

        :return: A tuple containing the aggregated tests epoch loss and accuracy
        """

        self._model.train(False)
        loss, accuracy = self._foreach_batch(dl_test, self.test_batch)

        return np.mean(loss).item(), np.mean(accuracy).item()

    def train_batch(self, batch) -> Tuple[float, float]:
        """
        Runs a single batch forward through the model, calculates loss,
        preforms back-propagation and uses the optimizer to update weights.

        :param batch: A dict, representing a single batch of data with the keys,
        'x' and 'y' representing the inputs and ground truth outputs.

        :return: A tuple containing the train loss and accuracy on the current batch
        """

        # Unpack the batch
        x = batch[INPUT_KEY]
        x = x.to(self._device)

        # Run the forward pass
        outputs = self._model.forward(x)

        # Zero the gradients after each step
        self._optimizer.zero_grad()

        # Compute the loss with respect to the true labels
        batch[OUTPUT_KEY] = outputs
        loss = self._loss_fn(batch)

        # Run the backwards pass
        loss.backward()

        # Perform the optimization step
        self._optimizer.step()

        # Compute the 'accuracy'
        accuracy = self._evaluation_metric(batch)

        return loss.item(), accuracy.item()

    def test_batch(self, batch) -> Tuple[float, float]:
        """
        Runs a single batch forward through the model, and calculates loss and accuracy.

        :param batch: A dict, representing a single batch of data with the keys,
        'x' and 'y' representing the inputs and ground truth outputs.

        :return: A tuple containing the tests loss and accuracy on the current batch
        """

        # Unpack the batch
        x = batch[INPUT_KEY]
        x = x.to(self._device)

        with torch.no_grad():
            # Run the forward pass
            outputs = self._model.forward(x)

            # Compute the loss with respect to the true labels
            batch[OUTPUT_KEY] = outputs
            loss = self._loss_fn(batch)

            # Compute the 'accuracy'
            accuracy = self._evaluation_metric(batch)

        return loss.item(), accuracy.item()

    @staticmethod
    def _foreach_batch(dl: DataLoader, forward_fn: Callable) -> Tuple[List, List]:
        """
        Evaluates the given forward-function on batches from the given
        dataloader, and prints progress along the way.

        :param dl: The DataLoader object from which to query batches.
        :param forward_fn: The forward method to apply to each batch,
        i.e. `train_batch` or `test_batch`.

        :return: A tuple of two lists, the first contains the losses over all batches in
        the current epoch, and the second ones contains all of the accuracies.
        """

        losses = []
        accuracies = []
        num_batches = len(dl.batch_sampler)
        pbar_name = forward_fn.__name__
        with tqdm.tqdm(desc=pbar_name, total=num_batches) as pbar:
            for batch_idx, data in enumerate(dl):
                loss, acc = forward_fn(data)

                pbar.set_description(f'{pbar_name} ({loss:.3f})')
                pbar.update()

                losses.append(loss)
                accuracies.append(acc)

            avg_loss = np.mean(losses).item()
            avg_acc = np.mean(accuracies).item()
            pbar.set_description(f'{pbar_name} (Avg. Loss {avg_loss:.3f}, '
                                 f'Avg. Accuracy {avg_acc:.3f})')

        return losses, accuracies