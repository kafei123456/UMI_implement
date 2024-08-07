"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from omegaconf import OmegaConf
import pathlib

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy','config'))#去到./diffusion_policy/config/
)
def main(cfg: OmegaConf):#接收
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.resolve(cfg)

if __name__ == "__main__":
    #python train.py --config-name=train_diffusion_unet_timm_umi_workspace task.dataset_path=example_demo_session/dataset.zarr.zip
    #去到./diffusion_policy/config/train_diffusion_unet_timm_umi_workspace.yaml文件
    main()
