import os
from typing import Optional

import hydra
import torch
from torch.utils.data import DataLoader, DistributedSampler, Sampler

from mot_jepa.common import conventions
from mot_jepa.common.project import CONFIGS_PATH
from mot_jepa.config_parser import GlobalConfig
from mot_jepa.datasets.dataset import dataset_index_factory, MOTClipDataset
from mot_jepa.trainer.torch_distrib_utils import DistributedSamplerWrapper
from mot_jepa.trainer.trainer import ContrastiveTrainer
from mot_jepa.utils import pipeline
from tools.utils import check_train_experiment_history, logger


def create_dataloader(
    dataset: MOTClipDataset,
    batch_size: int,
    num_workers: int,
    train: bool,
    sampler: Optional[Sampler] = None,
    use_batch_sampler: bool = False
) -> DataLoader:
    if use_batch_sampler:
        logger.warning('Using batch sampler. Configured batch size is ignored!')

    rank = int(os.environ.get('RANK', -1))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    is_ddp = (rank != -1)

    # Setup sampler
    if sampler is None:
        if is_ddp:
            shuffle = None
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=train
            )
        else:
            shuffle = train if use_batch_sampler else None
    else:
        shuffle = None
        if is_ddp:
            sampler = DistributedSamplerWrapper(
                sampler=sampler,
                num_replicas=world_size,
                rank=rank,
                shuffle=True
            )

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size if not use_batch_sampler else 1,
        num_workers=num_workers,
        shuffle=shuffle,
        sampler=sampler if not use_batch_sampler else None,
        batch_sampler=sampler if use_batch_sampler else None
    )


@hydra.main(config_path=CONFIGS_PATH, config_name='default', version_base='1.1')
@pipeline.task('train')
def main(cfg: GlobalConfig) -> None:
    checkpoint_path = check_train_experiment_history(
        train_resume=cfg.train.resume,
        train_truncate=cfg.train.truncate,
        master_path=cfg.path.master,
        dataset_name=cfg.dataset_name,
        experiment_name=cfg.experiment_name
    )

    train_index = dataset_index_factory(
        name=cfg.dataset.index.type,
        params=cfg.dataset.index.params,
        split='train',
        sequence_list=cfg.dataset.index.sequence_list
    )

    train_dataset = cfg.dataset.build_dataset(train_index)

    train_sampler = cfg.dataset.build_sampler(train_dataset)

    train_dataloader = create_dataloader(
        train_dataset,
        batch_size=cfg.resources.batch_size,
        num_workers=cfg.resources.num_workers,
        train=True,
        sampler=train_sampler,
        use_batch_sampler=cfg.dataset.use_batch_sampler
    )

    val_index = dataset_index_factory(
        name=cfg.dataset.index.type,
        params=cfg.dataset.index.params,
        split='val',
        sequence_list=cfg.dataset.index.sequence_list,
    )

    val_dataset = cfg.dataset.build_dataset(val_index)

    val_dataloader = create_dataloader(
        val_dataset,
        batch_size=cfg.resources.val_batch_size,
        num_workers=cfg.resources.num_workers,
        train=False,
        sampler=None,
        use_batch_sampler=False
    )

    model = cfg.build_model()
    if cfg.train.checkpoint_cfg.resume_from is not None:
        logger.warning(f'Using "{cfg.train.checkpoint_cfg.resume_from}" as starting checkpoint.')
        state_dict = torch.load(cfg.train.checkpoint_cfg.resume_from)
        model.load_state_dict(state_dict['model'])

    loss_func = cfg.train.build_loss_func()
    optimizer = cfg.train.build_optimizer(model.parameters())
    scheduler = cfg.train.build_scheduler(
        optimizer=optimizer,
        epoch_steps=len(train_dataloader),
    )

    experiment_path = conventions.get_experiment_path(cfg.path.master, cfg.dataset_name, cfg.experiment_name)
    tensorboard_log_dirpath = \
        conventions.get_tensorboard_logs_dirpath(cfg.path.master, cfg.dataset_name, cfg.experiment_name)
    checkpoints_dirpath = conventions.get_checkpoints_dirpath(experiment_path)
    trainer = ContrastiveTrainer(
        model=model,
        loss_func=loss_func,
        optimizer=optimizer,
        scheduler=scheduler,

        n_epochs=cfg.train.max_epochs,
        gradient_clip=cfg.train.gradient_clip,
        mixed_precision=cfg.train.mixed_precision,

        tensorboard_log_dirpath=tensorboard_log_dirpath,
        checkpoints_dirpath=checkpoints_dirpath,
        metric_monitor=cfg.train.checkpoint_cfg.metric_monitor
    )

    if checkpoint_path is not None:
        logger.info(f'Loading trainer state from path "{checkpoint_path}".')
        trainer.from_checkpoint(checkpoint_path)

    trainer.train(train_dataloader, val_dataloader)


if __name__ == '__main__':
    main()
