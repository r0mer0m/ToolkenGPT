# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

import os
import hydra
import torch
import random
import numpy as np
from fuse.utils import load
from omegaconf import OmegaConf
import torch.distributed as dist
from pytorch_lightning import Trainer
from fuse import PLDataModule, PLModel
from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DeepSpeedStrategy
from torch.distributed.elastic.multiprocessing.errors import record
from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict

def dist_setup():
    # initialize the process group
    dist.init_process_group("nccl")

def cleanup():
    dist.destroy_process_group()

@record
@hydra.main(version_base=None, config_path="./configs/runs")
def main(config):

    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.set_float32_matmul_precision('medium')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(1)
    np.random.seed(1)
    
    dist_setup()
    
    local_rank = int(os.environ['LOCAL_RANK'])
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    
    wandb_logger = WandbLogger(project=f"tool-augmentation-tests-{config.data.dataset_name}", name=config.run_name)
        
    tokenizer, model, _model = load(config.model, rank=rank, 
                            training_config=config.training, 
                            augmentation_config=config.augmentation)
    data_module = PLDataModule(tokenizer=tokenizer, data_args=config.data,
                               rank=rank, world_size=world_size)

    print(OmegaConf.to_yaml(config))
    print(f"{tokenizer.aug_vocab_size=}")
    print(f"{tokenizer.vocab_size=}")
    print(f"{tokenizer.id_to_api=}")
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.out_checkpoint_dir,
        every_n_epochs=1,
        save_on_train_epoch_end=False,
        # save_on_train_epoch_end=True,
        save_top_k=2,
        # monitor="loss",
        monitor="val_loss",
        mode="min",
        # filename="checkpoint-{epoch:02d}-{val_loss:.2f}"
    )
    trainer = Trainer(
        accelerator="gpu",
        default_root_dir=config.out_checkpoint_dir,
        enable_checkpointing=True,
        max_epochs=config.training.num_epochs,
        callbacks=checkpoint_callback,
        logger=wandb_logger,
        check_val_every_n_epoch=1,
        devices=world_size,
        strategy=DeepSpeedStrategy(config=config.training.deepspeed_config),
        precision=16,
    )
    trainer.fit(model, datamodule=data_module)
    
    if trainer.is_global_zero:
        # magically converts the folder into a single lightning loadable pytorch file (for ZeRO 1,2 and 3)
        convert_zero_checkpoint_to_fp32_state_dict(
            checkpoint_callback.best_model_path, config.best_checkpoint_filepath)
        # load check
        # PLModel.load_from_checkpoint(config.best_checkpoint_filepath, tokenizer=tokenizer, model=_model)
        # print("Re-loading successful")
    
    cleanup()

if __name__ == "__main__":
    main()