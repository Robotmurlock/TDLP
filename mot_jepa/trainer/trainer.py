"""
Implementation of a multi-node multi-gpu trainer
"""
import gc
import logging
import os
import time
from pathlib import Path
from typing import Optional, Dict, Union, Any

import torch
from torch import distributed as dist
from torch import nn
# noinspection PyPep8Naming
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast

from mot_jepa.common.conventions import LAST_CKPT
from mot_jepa.trainer import torch_distrib_utils
from mot_jepa.trainer import torch_helper
from mot_jepa.trainer.losses.base import VideoClipLoss
from mot_jepa.trainer.metrics import LossDictMeter, AccuracyMeter

logger = logging.getLogger('Trainer')

class ContrastiveTrainer:
    """
    Trainer for training models on multiple nodes and multiple GPUs.
    """
    def __init__(
        self,
        model: nn.Module,
        loss_func: VideoClipLoss,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        n_epochs: int,
        tensorboard_log_dirpath: str,
        checkpoints_dirpath: str,
        metric_monitor: str = 'val-epoch/loss',
        metric_monitor_minimize: bool = True,
        gradient_clip: Optional[float] = None,
        mixed_precision: bool = False
    ):
        """
        Args:
            model: PyTorch model
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            n_epochs: Number of epochs to train
            tensorboard_log_dirpath: Path where Tensorboard logs will be saved.
            checkpoints_dirpath: Directory to save checkpoints
            metric_monitor: Metric to monitor for checkpoint saving
            metric_monitor_minimize: Whether to minimize the monitored metric

            gradient_clip: Clip gradient norm during training
            mixed_precision: Use automatic mixed precision training and evaluation
        """
        # Node info
        self._use_ddp = False  # Modified in _setup()
        self._rank = int(os.environ.get('RANK', -1))
        self._local_rank = int(os.environ.get('LOCAL_RANK', -1))
        self._world_size = int(os.environ.get('WORLD_SIZE', 1))
        self._device = f'cuda:{max(0, self._local_rank)}' if torch.cuda.is_available() else 'cpu'

        # Trainer state
        self._model = model
        self._loss_func = loss_func
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._metric_monitor = metric_monitor
        self._metric_monitor_minimize = metric_monitor_minimize
        self._writer = SummaryWriter(log_dir=tensorboard_log_dirpath)

        # Trainer parameters
        self._n_epochs = n_epochs
        self._gradient_clip = gradient_clip
        self._mixed_precision = mixed_precision and torch.cuda.is_available()
        self._scaler = GradScaler(enabled=self._mixed_precision)
        if self._mixed_precision:
            logger.info('Using mixed precision training.')

        # Checkpoints
        self._best_loss: Optional[float] = None
        self._best_metrics: Optional[Dict[str, float]] = None
        self._checkpoints_path = checkpoints_dirpath

        # State
        self._epoch = 0

        # Finish
        self._log_trainer_configuration()

    @property
    def model(self) -> Union[nn.Module, DDP]:
        """
        Returns:
            Trainer model
        """
        return self._model

    @torch_distrib_utils.rank_zero_only
    def _log(self, msg: str, level: int = logging.INFO) -> None:
        """
        Trainer rank zero logger.

        Args:
            msg: Log message
            level: Logging level

        """
        logger.log(level, msg)

    @torch_distrib_utils.rank_zero_only
    def _log_metrics(self, metrics: Dict[str, float], step: int, verbose: bool = True) -> None:
        """
        Log metrics to console and TensorBoard.

        Args:
            metrics: Metrics to log (dictionary)
        """
        for metric_name, metric_value in metrics.items():
            self._writer.add_scalar(metric_name, metric_value, step)

            if verbose:
                self._log(f'Epoch {step}: {metric_name} = {metric_value:.4f}')

    def _on_start(self) -> None:
        """
        Pre-process code before a train/eval process.
        """
        self._model.to(self._device)
        torch_helper.optimizer_to(self._optimizer, device=self._device)

        if self._use_ddp:
            self._model = DDP(self._model, find_unused_parameters=True)

    def _on_end(self):
        """
        Cleanup code after a train/eval process.
        """
        self._model.cpu()

    def train(self, train_loader: 'DataLoader', val_loader: Optional['DataLoader'] = None) -> Dict[str, Any]:
        """
        Main training loop.

        Args:
            train_loader: DataLoader for a training set.
            val_loader: DataLoader for a validation set.

        Returns:
            Metrics
        """
        if val_loader is None:
            self._log('Validation loader not set. Evaluation will be skipped!', level=logging.WARNING)

        self._ddp_setup()
        self._on_start()

        self._log('Training...')
        while self._epoch < self._n_epochs:
            start_time = time.time()

            # Training step
            train_metrics = self._train_epoch(train_loader)
            self._log_metrics(train_metrics, step=self._epoch)
            torch_distrib_utils.dist_barrier()  # Wait for all workers to finish training

            # Validation step
            if val_loader is not None:
                with torch.no_grad():
                    val_metrics = self._eval_epoch(val_loader)
                self._log_metrics(val_metrics, step=self._epoch)

            # Save model checkpoint
            self._save_checkpoint(val_metrics)

            elapsed_time = time.time() - start_time
            self._log(f'Epoch {self._epoch} completed in {elapsed_time:.2f}s')
            torch_distrib_utils.dist_barrier()  # Wait for all workers to finish evaluation

            self._epoch += 1

        self._on_end()
        self._ddp_cleanup()

        self._log('Training completed.')

        return self._best_metrics

    def eval(self, val_loader: 'DataLoader') -> Dict[str, Any]:
        """
        Model evaluation.

        Args:
            val_loader: DataLoader for validation set.

        Returns:
            Metrics
        """
        self._ddp_setup()
        self._on_start()

        self._log('Evaluating...')
        start_time = time.time()

        # Validation step
        with torch.no_grad():
            val_metrics = self._eval_epoch(val_loader)

        elapsed_time = time.time() - start_time
        self._log(f'Evaluation completed in {elapsed_time:.2f}s')

        self._on_end()
        self._ddp_cleanup()

        self._log('Evaluation completed.')

        return val_metrics

    def _forward_and_loss(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        track_x = data['observed']['features']
        track_mask = data['observed']['mask']
        track_ids = data['observed']['ids']
        det_x = data['unobserved']['features']
        det_mask = data['unobserved']['mask']
        det_ids = data['unobserved']['ids']

        with autocast(enabled=self._mixed_precision):
            model_output = self._model(track_x, track_mask, det_x, det_mask)
            if len(model_output) == 4:
                track_features, det_features, track_feat_dict, det_feat_dict = model_output
            else:
                track_features, det_features = model_output
                track_feat_dict = None
                det_feat_dict = None

            loss_dict = self._loss_func(
                track_features,
                det_features,
                track_mask,
                det_mask,
                track_feat_dict,
                det_feat_dict,
                track_ids,
                det_ids
            )

            return loss_dict

    def _train_epoch(self, train_loader: 'DataLoader') -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: DataLoader for a training set.

        Returns:
            Metrics
        """
        self._model.train()
        loss_meter = LossDictMeter()
        track_accuracy_meter = AccuracyMeter(use_percentages=True)
        det_accuracy_meter = AccuracyMeter(use_percentages=True)

        for data in torch_distrib_utils.rank_zero_tqdm(train_loader, desc='Training', unit='batch'):
            data = torch_helper.to_device(data, device=self._device)
            self._optimizer.zero_grad()
            loss_dict = self._forward_and_loss(data)
            loss = loss_dict['loss']
            self._scaler.scale(loss).backward()
            if self._gradient_clip is not None:
                self._scaler.unscale_(self._optimizer)
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._gradient_clip)
            self._scaler.step(self._optimizer)
            self._scaler.update()

            self._scheduler.step()

            track_accuracy_meter.push(loss_dict['track_predictions'], loss_dict['track_labels'], mask=loss_dict['track_mask'])
            det_accuracy_meter.push(loss_dict['det_predictions'], loss_dict['det_labels'], mask=loss_dict['det_mask'])
            step_metrics = loss_meter.push({f'train/{k}': v for k, v in loss_dict.items() if 'loss' in k})
            global_step = self._epoch * len(train_loader) + loss_meter.step
            # noinspection PyTypeChecker
            step_metrics['train/lr'] = self._optimizer.param_groups[0]['lr']
            self._log_metrics(
                metrics=step_metrics,
                step=global_step,
                verbose=False
            )

        epoch_metrics = {k.replace('train/', 'train-epoch/'): v for k, v in loss_meter.aggregate_and_flush().items()}
        epoch_metrics.update({
            'train-epoch/track-accuracy': track_accuracy_meter.aggregate_and_flush(),
            'train-epoch/detection-accuracy': det_accuracy_meter.aggregate_and_flush(),
        })

        torch.cuda.empty_cache()
        gc.collect()

        return epoch_metrics

    def _eval_epoch(self, val_loader: 'DataLoader') -> Dict[str, float]:
        """
        Evaluate on validation set.

        Args:
            val_loader: DataLoader for validation set.

        Returns:
            Metrics
        """
        loss_meter = LossDictMeter()
        track_accuracy_meter = AccuracyMeter(use_percentages=True)
        det_accuracy_meter = AccuracyMeter(use_percentages=True)

        self._model.eval()
        for data in torch_distrib_utils.rank_zero_tqdm(val_loader, desc='Evaluation', unit='batch'):
            data = torch_helper.to_device(data, device=self._device)
            loss_dict = self._forward_and_loss(data)

            track_accuracy_meter.push(loss_dict['track_predictions'], loss_dict['track_labels'], mask=loss_dict['track_mask'])
            det_accuracy_meter.push(loss_dict['det_predictions'], loss_dict['det_labels'], mask=loss_dict['det_mask'])
            step_metrics = loss_meter.push({f'val/{k}': v for k, v in loss_dict.items() if 'loss' in k})
            global_step = self._epoch * len(val_loader) + loss_meter.step

            self._log_metrics(
                metrics=step_metrics,
                step=global_step,
                verbose=False
            )

        epoch_metrics = {k.replace('val/', 'val-epoch/'): v for k, v in loss_meter.aggregate_and_flush().items()}
        epoch_metrics.update({
            'val-epoch/track-accuracy': track_accuracy_meter.aggregate_and_flush(),
            'val-epoch/detection-accuracy': det_accuracy_meter.aggregate_and_flush(),
        })

        torch.cuda.empty_cache()
        gc.collect()

        return epoch_metrics

    @torch_distrib_utils.rank_zero_only
    def _save_checkpoint(self, val_metrics: Dict[str, float]) -> None:
        """
        Save model checkpoints.

        Args:
            val_metrics: Validation metrics
        """
        model = torch_distrib_utils.get_model(self._model)
        trainer_state = {
            'epoch': self._epoch,
            'model': model.state_dict(),
            'optimizer': self._optimizer.state_dict(),
            'scheduler': self._scheduler.state_dict()
        }

        Path(self._checkpoints_path).mkdir(parents=True, exist_ok=True)
        checkpoint_path = os.path.join(self._checkpoints_path, f'ckpt_epoch_{self._epoch:03d}.pt')
        torch.save(trainer_state, checkpoint_path)

        # Save latest checkpoint
        latest_checkpoint_path = os.path.join(self._checkpoints_path, LAST_CKPT)
        torch.save(trainer_state, latest_checkpoint_path)

        # Save the best checkpoint
        metric = val_metrics[self._metric_monitor]
        if self._best_loss is None or (self._metric_monitor_minimize and metric < self._best_loss) or \
           (not self._metric_monitor_minimize and metric > self._best_loss):
            self._best_loss = metric
            self._best_metrics = val_metrics
            best_checkpoint_path = os.path.join(self._checkpoints_path, 'best.pt')
            torch.save(trainer_state, best_checkpoint_path)
            self._log(f'New best model saved at epoch {self._epoch}.')

    def from_checkpoint(self, path: str) -> None:
        data = torch.load(path)
        torch_distrib_utils.get_model(self._model).load_state_dict(data['model'])
        self._epoch = data['epoch'] + 1
        self._optimizer.load_state_dict(data['optimizer'])
        self._scheduler.load_state_dict(data['scheduler'])

    def _ddp_setup(self) -> None:
        """
        DDP setup.

        Returns:
            Model wrapper in DDP (if multi-gpu is used) or not (if single-gpu is used)
        """
        local_rank: Optional[int] = int(os.environ.get('LOCAL_RANK', self._local_rank))

        if local_rank == -1:
            logger.info('Using single-GPU training mode.')

            if torch.cuda.is_available():
                logger.info('Using "cuda:0" (default) for single-GPU training.')
                self._local_rank = 0
            else:
                logger.info('Using CPU for single-GPU training.')
                self._local_rank = 'cpu'
        else:
            if not dist.is_initialized():
                # Allows `_setup(model)` to be called multiple times with a different (or same) model
                logger.info(f'[{local_rank=}] Using multi-GPU training mode. Initializing...')
                dist.init_process_group('nccl', rank=self._rank, world_size=self._world_size)
                logger.info(f'[{local_rank=}] Initialization finished.')
            self._local_rank = local_rank
            self._use_ddp = True

    def _ddp_cleanup(self):
        """
        DDP cleanup.
        """
        if self._use_ddp:
            dist.destroy_process_group()

        torch.cuda.empty_cache()
        gc.collect()

    def _log_trainer_configuration(self):
        logger.info(
            f'Initialized trainer. TrainerConfiguration('
            f'device={self._device}, '
            f'world_size={self._world_size}, '
            f'rank={self._rank}, '
            f'local_rank={self._local_rank}, '
            f'mixed_precision={self._mixed_precision})'
        )
