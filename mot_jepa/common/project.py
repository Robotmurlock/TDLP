"""Project-relative paths for source code, configs and outputs."""
import os


ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SRC_PATH = os.path.join(ROOT_PATH, 'rt_motip')
ASSETS_PATH = os.path.join(ROOT_PATH, 'assets')
CONFIGS_PATH = os.path.join(ROOT_PATH, 'configs')
OUTPUTS_PATH = os.path.join(ROOT_PATH, 'outputs')
PLAYGROUND_PATH = os.path.join(ROOT_PATH, 'playground')
