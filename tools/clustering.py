import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from tqdm import tqdm

from mot_jepa.common.project import CONFIGS_PATH
from mot_jepa.config_parser import GlobalConfig
from mot_jepa.datasets.dataset import dataset_index_factory
from mot_jepa.trainer import torch_helper
from mot_jepa.utils import pipeline
from tools.train import create_dataloader


import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def visualize_embeddings_tsne_pca(
    det_embeddings: torch.Tensor,
    track_embeddings: torch.Tensor,
    perplexity: int = 30,
    tsne_n_iter: int = 1000,
    pca_components: int = 2,
    random_state: int = 42
):
    """
    Projects detection and track embeddings to 2D using t-SNE and PCA, then visualizes them.

    Args:
        det_embeddings (torch.Tensor): Detection embeddings of shape (N_d, D)
        track_embeddings (torch.Tensor): Track embeddings of shape (N_t, D)
        perplexity (int): Perplexity parameter for t-SNE.
        tsne_n_iter (int): Number of iterations for t-SNE optimization.
        pca_components (int): Number of components for PCA (should be 2).
        random_state (int): Random seed for reproducibility.
    """
    # Convert to numpy
    det_np = det_embeddings.detach().cpu().numpy()
    track_np = track_embeddings.detach().cpu().numpy()

    # Stack for joint projection
    all_embeddings = np.vstack([det_np, track_np])
    labels = np.array(["Detection"] * len(det_np) + ["Track"] * len(track_np))

    # --- t-SNE ---
    tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=tsne_n_iter, random_state=random_state)
    tsne_proj = tsne.fit_transform(all_embeddings)

    fig = plt.figure(figsize=(8, 6))
    for label, color in zip(["Detection", "Track"], ["tab:blue", "tab:orange"]):
        idx = labels == label
        plt.scatter(tsne_proj[idx, 0], tsne_proj[idx, 1], label=label, alpha=0.7, s=30, color=color)
    plt.title("t-SNE Projection")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    fig.savefig('/work/tsne-clusters.png')

    # --- PCA ---
    pca = PCA(n_components=pca_components, random_state=random_state)
    pca_proj = pca.fit_transform(all_embeddings)

    fig = plt.figure(figsize=(8, 6))
    for label, color in zip(["Detection", "Track"], ["tab:blue", "tab:orange"]):
        idx = labels == label
        plt.scatter(pca_proj[idx, 0], pca_proj[idx, 1], label=label, alpha=0.7, s=30, color=color)
    plt.title("PCA Projection")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    fig.savefig('/work/pca-clusters.png')


@torch.no_grad()
@hydra.main(config_path=CONFIGS_PATH, config_name='default', version_base='1.1')
@pipeline.task('clustering')
def main(cfg: GlobalConfig) -> None:
    val_index = dataset_index_factory(
        name=cfg.dataset.index.type,
        params=cfg.dataset.index.params,
        split='val',
        sequence_list=cfg.dataset.index.sequence_list
    )

    val_dataset = cfg.dataset.build_dataset(val_index)
    val_dataloader = create_dataloader(
        dataset=val_dataset,
        batch_size=cfg.resources.batch_size,
        num_workers=cfg.resources.num_workers,
        shuffle=False
    )

    model = cfg.build_model()
    state_dict = torch.load(cfg.eval.checkpoint)
    model.load_state_dict(state_dict['model'])
    model.to(cfg.resources.accelerator)
    model.eval()

    track_embedding_list = []
    det_embedding_list = []
    for data in tqdm(val_dataloader, desc='Inference', unit='batch'):
        data = torch_helper.to_device(data, device=cfg.resources.accelerator)
        track_features, det_features = model(
            data['observed_bboxes'],
            data['observed_temporal_mask'],
            data['unobserved_bboxes'],
            data['unobserved_temporal_mask'],
        )
        track_features = track_features[0].cpu()
        det_features = det_features[0].cpu()
        track_features = F.normalize(track_features, dim=-1)
        det_features = F.normalize(det_features, dim=-1)

        track_embedding_list.append(track_features)
        det_embedding_list.append(det_features)

    track_embeddings = torch.cat(track_embedding_list, dim=0)
    det_embeddings = torch.cat(det_embedding_list, dim=0)

    visualize_embeddings_tsne_pca(
        det_embeddings=det_embeddings,
        track_embeddings=track_embeddings
    )




if __name__ == '__main__':
    main()
