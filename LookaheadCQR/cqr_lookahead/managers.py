from abc import ABC
from DynamicalSystems import LOGS_DIR
from typing import Dict, Any, Type, Sequence
from LookaheadCQR.cqr_lookahead.loggers import Logger
from LookaheadCQR.cqr_lookahead.trainer import BaseTrainer
from LookaheadCQR.cqr_lookahead.optim import OptimizerInitializer

import os
import torch


class CrossValidationExperimentManager(ABC):
    """
    A class for managing cross-validated experiments.
    """

    def __init__(
            self,
            trainer_type: Type[BaseTrainer],
            trainer_params: Dict[str, Any],
            model_type: Type[torch.nn.Module],
            model_params: Dict[str, Any],
            optimizer: OptimizerInitializer,
            n_fold: int = 1,
            log_dir: str = LOGS_DIR,
            device: torch.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu'
            ),
            max_elements_to_save: int = 500,
    ):
        """
        :param trainer_type: (Type[BaseTrainer]) The type of trainer to use.
        :param trainer_params: (Dict[str, Any]) All the parameters of the trainer,
        except 'model', 'optimizer', 'device' and 'logger'.
        :param model_type: (Type[torch.nn.Module]) The type of model to use.
        Should be sub-class of nn.Module.
        :param model_params: (Dict[str, Any]) The parameters for the model.
        :param optimizer: (OptimizerInitializer) An OptimizerInitializer class that can
        instantiate the required optimizers & schedulers.
        :param n_fold: (int) Number of folds to run
        :param log_dir: (str) The directory in which to save all results,
        from all folds.
        :param device: (torch.device) The device to work on.
        :param max_elements_to_save: (int) Maximal number of logs to save in the logger.
        """

        self._trainer_type = trainer_type
        self._trainer_params = trainer_params
        self._model_type = model_type
        self._model_params = model_params
        self._optimizer = optimizer
        self._n_fold = n_fold
        self._log_dir = log_dir
        self._device = device
        self._max_elements_to_save = max_elements_to_save

    def _train_fold(
            self,
            fold: int,
            dl_train: torch.utils.data.DataLoader,
            dl_valid: torch.utils.data.DataLoader,
            num_epochs: int,
            checkpoints: bool = True,
            checkpoints_mode: str = 'min',
            early_stopping: int = None,
    ) -> None:
        """
        Train a single fold and save the best model on the validation set

        :param fold: (int) The fold number.
        :param dl_train: Dataloader for the training set.
        :param dl_valid: Dataloader for the validation set.
        :param num_epochs: Number of epochs to train for.
        :param checkpoints: Whether to save _model to file every time the
            tests set accuracy improves. Should be a string containing a
            filename without extension.
        :param checkpoints_mode: (str) Whether to optimize for minimum, or maximum,
        score.
        :param early_stopping: (int) Number of epochs with no improvement after-which to stop training early
        if there is no tests loss improvement for this number of epochs.

        :return: None
        """

        # Define the fold's logger
        logger = Logger(
            log_dir=self._log_dir,
            experiment_name=f"Fold_{fold}",
            max_elements=self._max_elements_to_save,
        )

        # Instantiate the model for the current fold
        model = self._model_type(
            **self._model_params
        ).to(self._device)

        # Instantiate the optimizer for the current fold
        model_parameters = (model.parameters(),)
        optimizer = self._optimizer.initialize(model_parameters)

        trainer_params = self._trainer_params.copy()
        trainer_params['logger'] = logger
        trainer_params['model'] = model
        trainer_params['optimizer'] = optimizer
        trainer_params['device'] = self._device
        trainer = self._trainer_type(**trainer_params)

        # Fit the model
        trainer.fit(
            dl_train=dl_train,
            dl_val=dl_valid,
            num_epochs=num_epochs,
            checkpoints=checkpoints,
            checkpoints_mode=checkpoints_mode,
            early_stopping=early_stopping,
        )

    def _eval(self, trainer: BaseTrainer, dl_test: torch.utils.data.DataLoader) -> None:
        """
        A wrapper for calling the correct evaluation method.

        :param trainer: (BaseTrainer) An instantiated trainer, with a loaded model,
        which is ready to start the evaluation.
        :param dl_test: Dataloader for the test set.
        """

        trainer.evaluate(dl_test=dl_test)

    def _eval_fold(
            self,
            dl_test: torch.utils.data.DataLoader,
            fold: int,
    ) -> None:
        """
        Evaluate a single fold using the best model on the validation set

        :param dl_test: Dataloader for the test set.
        :param fold: (int) The fold number.

        :return: None
        """

        # Define the test's set logger
        logger = Logger(
            log_dir=os.path.join(self._log_dir, f'Fold_{fold}'),
            experiment_name=f"Test",
            max_elements=float("inf"),
        )

        # Load the best model
        best_model_path = os.path.join(
            self._log_dir,
            f"Fold_{fold}",
            "BestModel.PyTorchModule",
        )
        ckpt = torch.load(best_model_path)['model']
        model = self._model_type(**self._model_params)
        model.load_state_dict(ckpt)
        model = model.to(self._device)

        # Define the test's set trainer
        trainer_params = self._trainer_params.copy()
        trainer_params['logger'] = logger
        trainer_params['model'] = model
        trainer_params['optimizer'] = None
        trainer_params['device'] = self._device
        trainer = self._trainer_type(**trainer_params)

        # Evaluate
        self._eval(
            trainer=trainer,
            dl_test=dl_test,
        )

    def cross_validate(
            self,
            train_dls: Sequence[torch.utils.data.DataLoader],
            val_dls: Sequence[torch.utils.data.DataLoader],
            test_dls: Sequence[torch.utils.data.DataLoader],
            num_epochs: int,
            checkpoints: bool = True,
            checkpoints_mode: str = 'min',
            early_stopping: int = None,
    ):
        """
        :param train_dls: (Sequence[torch.utils.data.DataLoader]) Dataloaders for the training sets.
        :param val_dls: (Sequence[torch.utils.data.DataLoader]) Dataloaders for the validation sets.
        :param test_dls: (Sequence[torch.utils.data.DataLoader]) Dataloaders for the test sets.
        :param num_epochs: Number of epochs to train for.
        :param checkpoints: Whether to save _model to file every time the
            tests set accuracy improves. Should be a string containing a
            filename without extension.
        :param checkpoints_mode: (str) Whether to optimize for minimum, or maximum,
        score.
        :param early_stopping: (int) Number of epochs with no improvement after-which to stop training early
        if there is no tests loss improvement for this number of epochs.

        :return: A tuple with four lists, containing, in that order, the train loss and
         accuracy and the tests loss and accuracy.
        """

        for f in range(self._n_fold):
            # Train over the fold
            print(f"\nTraining over fold {f + 1} / {self._n_fold}\n")
            self._train_fold(
                fold=f,
                dl_train=train_dls[f],
                dl_valid=val_dls[f],
                num_epochs=num_epochs,
                checkpoints=checkpoints,
                checkpoints_mode=checkpoints_mode,
                early_stopping=early_stopping,
            )

            # Evaluate over the fold
            print(f"\nEvaluating over fold {f + 1} / {self._n_fold}\n")
            self._eval_fold(
                dl_test=test_dls[f],
                fold=f,
            )

