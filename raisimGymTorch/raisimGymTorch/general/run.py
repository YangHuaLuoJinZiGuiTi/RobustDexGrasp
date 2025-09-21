import hydra
from omegaconf import DictConfig, OmegaConf
from train import train
from quantitative_eval import quantitative_eval

# Usage: In raisim/RobustDexGrasp/raisimGymTorch, Run python train.py task_name=allegro_teacher
@hydra.main(config_path="../", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    exp_name_default = 'teacher' 
    task_name_default = 'allegro_teacher'
    if cfg.exp_name is None:
        cfg.exp_name = exp_name_default
    if cfg.task_name is None:
        cfg.task_name = task_name_default
        
    cfg.task_path = f'/home/hang/raisim/RobustDexGrasp/raisimGymTorch/raisimGymTorch/env/envs/{cfg.task_name}/cfgs/cfg_reg.yaml'
    # cfg.weight = f'../{cfg.exp_name}_ckpt/full_12500_r.pt'

    print("===== Config =====")
    print(OmegaConf.to_yaml(cfg))

    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        quantitative_eval(cfg)
 
    

if __name__ == "__main__":
    main()