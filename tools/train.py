import os

import hydra
import torch
from torch.utils.data import DataLoader, DistributedSampler

from mot_jepa.common import conventions
from mot_jepa.common.project import CONFIGS_PATH
from mot_jepa.config_parser import GlobalConfig
from mot_jepa.datasets.dataset import dataset_index_factory, MOTClipDataset
from mot_jepa.trainer.trainer import ContrastiveTrainer
from mot_jepa.utils import pipeline
from tools.utils import check_train_experiment_history, logger


def create_dataloader(
    dataset: MOTClipDataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool
) -> DataLoader:
    rank = int(os.environ.get('RANK', -1))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    is_ddp = (rank != -1)

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle if not is_ddp else None,
        sampler=DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle
        ) if is_ddp else None
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

    train_dataloader = create_dataloader(
        train_dataset,
        batch_size=cfg.resources.batch_size,
        num_workers=cfg.resources.num_workers,
        shuffle=True
    )

    val_index = dataset_index_factory(
        name=cfg.dataset.index.type,
        params=cfg.dataset.index.params,
        split='val',
        sequence_list=cfg.dataset.index.sequence_list
    )

    val_dataset = cfg.dataset.build_dataset(val_index)

    val_dataloader = create_dataloader(
        val_dataset,
        batch_size=cfg.resources.batch_size,
        num_workers=cfg.resources.num_workers,
        shuffle=False
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

    experiments_path = conventions.get_experiment_path(cfg.path.master, cfg.dataset_name, cfg.experiment_name)
    tensorboard_log_dirpath = \
        conventions.get_tensorboard_logs_dirpath(cfg.path.master, cfg.dataset_name, cfg.experiment_name)
    checkpoints_dirpath = conventions.get_checkpoints_dirpath(experiments_path)
    trainer = ContrastiveTrainer(
        model=model,
        loss_func=loss_func,
        optimizer=optimizer,
        scheduler=scheduler,

        n_epochs=cfg.train.max_epochs,
        gradient_clip=cfg.train.gradient_clip,

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
