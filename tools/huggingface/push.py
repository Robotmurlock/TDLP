import logging

import hydra
import torch

from tdlp.common.project import CONFIGS_PATH
from tdlp.config_parser import GlobalConfig
from tdlp.utils import pipeline


logger = logging.getLogger('OnlineInference')
THRESHOLD = 1e-6


def _assert_models_match(
    model_a: torch.nn.Module,
    model_b: torch.nn.Module,
    threshold: float,
) -> None:
    """Assert that two models share identical parameters within a tolerance.

    Args:
        model_a: First model to compare.
        model_b: Second model to compare.
        threshold: Allowed absolute difference between parameters.

    Raises:
        AssertionError: If parameter names, shapes, or values do not match.
    """
    params_a = dict(model_a.named_parameters())
    params_b = dict(model_b.named_parameters())

    if params_a.keys() != params_b.keys():
        missing_in_a = params_b.keys() - params_a.keys()
        missing_in_b = params_a.keys() - params_b.keys()
        raise AssertionError(
            f'Parameter names mismatch. Missing in model_a: {missing_in_a}. Missing in model_b: {missing_in_b}.'
        )

    for name, param_a in params_a.items():
        param_b = params_b[name]
        if param_a.shape != param_b.shape:
            raise AssertionError(
                f'Parameter shape mismatch for {name}: {param_a.shape} vs {param_b.shape}.'
            )

        tensor_a = param_a.detach().cpu()
        tensor_b = param_b.detach().cpu()
        if not torch.allclose(tensor_a, tensor_b, atol=threshold, rtol=0.0):
            diff = torch.max(torch.abs(tensor_a - tensor_b)).item()
            raise AssertionError(f'Parameter {name} mismatch: max diff {diff} exceeds {threshold}.')



@torch.no_grad()
@hydra.main(config_path=CONFIGS_PATH, config_name='default', version_base='1.2')
@pipeline.task('inference')
def main(cfg: GlobalConfig) -> None:
    assert cfg.hf is not None, 'HuggingFace config is required for pushing to HuggingFace.'
    assert cfg.hf.model_name != '<MODEL>', 'Model name is required for pushing to HuggingFace.'

    model = cfg.build_hf_model()
    logger.info(f'Loading model from {cfg.eval.checkpoint}.')
    state_dict = torch.load(cfg.eval.checkpoint)
    model.model.load_state_dict(state_dict['model'])
    model_config = cfg.get_hf_model_params()

    model.push_to_hub(
        cfg.hf.model_path, 
        config=model_config
    )
    logger.info(f'Model pushed to {cfg.hf.model_path}.')

    hub_model = model.from_pretrained(cfg.hf.model_path)
    _assert_models_match(model.model, hub_model.model, THRESHOLD)
    logger.info('Hub model parameters match loaded checkpoint within tolerance.')



if __name__ == '__main__':
    main()
