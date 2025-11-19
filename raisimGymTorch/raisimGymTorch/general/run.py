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
        
    cfg.task_path = f'/home/hang/raisim/SafeGrasp/raisimGymTorch/raisimGymTorch/env/envs/{cfg.task_name}/cfgs/cfg_reg.yaml'
    # cfg.weight = f'../{cfg.exp_name}_ckpt/full_12500_r.pt'

    print("===== Config =====")
    print(OmegaConf.to_yaml(cfg))

    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        __ = quantitative_eval(cfg)
    elif cfg.mode == "batched_eval":
        mass_list = [0.1, 0.25, 0.5, 0.8]
        result_list = []
        for mass in mass_list:
            cfg.env_settings.mass = mass
            result = quantitative_eval(cfg)
            result_list.append(result)
        print("Batched Evaluation Results:")
        import sys
        import numpy as np
        for result in result_list:
            obj_mass, total_succ_rate, F_max, F_max_mean, F_max_mean_lift = result["obj_mass"], result["total_succ_rate"], result["F_max"], result["F_max_mean"], result["F_max_mean_lift"]
            print("\n============================")
            print("\nEval Name: ", cfg.eval_name)
            print("\nObj Average Gravity: {:.2f}N".format(obj_mass))
            print("\nTotal success rate: {:.2f}%".format(total_succ_rate), file=sys.stdout)
            print("\nTotal F_max: {:.2f}N".format(F_max), file=sys.stdout)
            print("\nTotal F_max_mean: {:.2f}N".format(F_max_mean), file=sys.stdout)
            print("\nTotal F_max_mean in lift: {:.2f}N".format(F_max_mean_lift), file=sys.stdout)
            

if __name__ == "__main__":
    main()