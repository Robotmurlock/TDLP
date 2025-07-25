import logging
from typing import Optional

import hydra
import torch
from tqdm import tqdm
import yaml

from mot_jepa.common.project import CONFIGS_PATH
from mot_jepa.config_parser import GlobalConfig
from mot_jepa.datasets.dataset import dataset_index_factory
from mot_jepa.utils import pipeline

logger = logging.getLogger('DatasetStatCalculation')


@torch.no_grad()
@hydra.main(config_path=CONFIGS_PATH, config_name='default', version_base='1.1')
@pipeline.task('bbox-stats')
def main(cfg: GlobalConfig) -> None:
    torch.set_printoptions(precision=3, sci_mode=None)

    train_index = dataset_index_factory(
        name=cfg.dataset.index.type,
        params=cfg.dataset.index.params,
        split='train',
        sequence_list=cfg.dataset.index.sequence_list
    )

    train_dataset = cfg.dataset.build_dataset(
        index=train_index,
        disable_transform=True,
        disable_augmentations=True
    )
    n_samples = len(train_dataset)

    sum_abs: Optional[torch.Tensor] = None
    sum_abs2: Optional[torch.Tensor] = None
    sum_fod: Optional[torch.Tensor] = None
    sum_fod2: Optional[torch.Tensor] = None
    n_bboxes = 0
    n_fod = 0

    for i in tqdm(range(n_samples), unit='sample', desc='Calculating bbox statistics', total=n_samples):
        data = train_dataset.get_raw(i)
        observed_bboxes = data.observed_bboxes[~data.observed_temporal_mask]
        unobserved_bboxes = data.unobserved_bboxes[~data.unobserved_temporal_mask]
        bboxes = torch.cat([observed_bboxes, unobserved_bboxes], dim=0)

        fod = torch.zeros_like(data.observed_bboxes)
        fod[:, 1:, :] = data.observed_bboxes[:, 1:, :] - data.observed_bboxes[:, :-1, :]
        fod[:, 1:, :] = fod[:, 1:, :] * (1 - data.observed_temporal_mask[:, :-1].unsqueeze(-1).repeat(1, 1, data.observed_bboxes.shape[-1]).float())
        fod = fod[~data.observed_temporal_mask]

        bboxes_sum = bboxes.sum(dim=0)
        bboxes_sum2 = torch.square(bboxes).sum(dim=0)
        bboxes_cnt = bboxes.shape[0]
        fod_sum = fod.sum(dim=0)
        fod_sum2 = torch.square(fod).sum(dim=0)
        fod_cnt = fod.shape[0]

        sum_abs = bboxes_sum if sum_abs is None else sum_abs + bboxes_sum
        sum_abs2 = bboxes_sum2 if sum_abs2 is None else sum_abs2 + bboxes_sum2
        sum_fod = fod_sum if sum_fod is None else sum_fod + fod_sum
        sum_fod2 = fod_sum2 if sum_fod2 is None else sum_fod2 + fod_sum2
        n_bboxes += bboxes_cnt
        n_fod += fod_cnt

    abs_mean = sum_abs / n_bboxes
    abs_mean2 = sum_abs2 / n_bboxes
    abs_std = (abs_mean2 - abs_mean ** 2) ** 0.5

    fod_mean = sum_fod / n_fod
    fod_mean2 = sum_fod2 / n_fod
    fod_std = (fod_mean2 - fod_mean ** 2) ** 0.5

    stats = {
        'absolute_mean': abs_mean.numpy().tolist(),
        'absolute_std': abs_std.numpy().tolist(),
        'fod_mean': fod_mean.numpy().tolist(),
        'fod_std': fod_std.numpy().tolist()
    }

    logger.info(f'Stats\n{yaml.dump(stats)}')


if __name__ == '__main__':
    main()
