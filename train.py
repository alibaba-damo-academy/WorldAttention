# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0
import argparse
import os
from omegaconf import OmegaConf
import wandb

from trainer import ScoreDistillationTrainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--no_visualize", action="store_true")
    parser.add_argument("--logdir", type=str, default="", help="Path to the directory to save logs")
    parser.add_argument("--wandb-save-dir", type=str, default="", help="Path to the directory to save wandb logs")
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--no-auto-resume", action="store_true", help="Disable auto resume from latest checkpoint in logdir")
    parser.add_argument("--no-one-logger", action="store_true", help="Disable One Logger (enabled by default)")

    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load("configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)
    config.no_save = args.no_save
    config.no_visualize = args.no_visualize

    config_name = os.path.dirname(args.config_path).split("/")[-1]
    config.config_name = config_name
    config.logdir = args.logdir
    config.wandb_save_dir = args.wandb_save_dir
    config.disable_wandb = bool(args.disable_wandb or getattr(config, "disable_wandb", False))
    if not config.disable_wandb:
        for key in ("wandb_key", "wandb_entity", "wandb_project"):
            value = str(getattr(config, key, "") or "")
            if not value or value.startswith("YOUR_"):
                raise ValueError(f"set {key} in the config, or pass --disable-wandb")
    config.auto_resume = not args.no_auto_resume
    config.use_one_logger = not args.no_one_logger

    if config.trainer == "score_distillation":
        trainer = ScoreDistillationTrainer(config)
    trainer.train()

    wandb.finish()

if __name__ == "__main__":
    main()
