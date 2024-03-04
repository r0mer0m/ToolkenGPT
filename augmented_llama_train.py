# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

import os
import hydra
import torch
import random
import numpy as np
from os import path as osp
import torch.distributed as dist
from pytorch_lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DeepSpeedStrategy
from torch.distributed.elastic.multiprocessing.errors import record
from fuse import (
    AugmentedLM,
    AugmentedConfig,
    AugmentedTokenizer,
    PLModel,
    PLDataModule)
from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict

def setup():
    # initialize the process group
    dist.init_process_group("nccl")

def cleanup():
    dist.destroy_process_group()


def load(model_config, augmentation_config, rank: int, checkpoint_filepath: str = '') -> AugmentedLM:
    
    if rank == 0: print("Loading tokenizer")
    tokenizer = AugmentedTokenizer.from_pretrained(
        model_config.base_model_id,
        augmentation_config=augmentation_config,
        )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    if rank == 0: print("Loading config")
    augmented_model_config = AugmentedConfig.from_pretrained(
        model_config.base_model_id,
        augment=True,
        aug_vocab_size = tokenizer.aug_vocab_size # tokenizer.n_aug_words,
    )
    
    if rank == 0: print("Loading model")
    model = AugmentedLM.from_pretrained(
        model_config.base_model_id,
        load_in_8bit=False,
        torch_dtype=torch.bfloat16
        )
    
    if rank == 0: print("Augmenting model")
    model.augment(augmented_model_config)
    
    if checkpoint_filepath:
        if rank == 0: print("Loading checkpoint")
        model.load_state_dict(torch.load(checkpoint_filepath, map_location='cpu'))
    
    return tokenizer, model


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
    
    setup()
    import os; 
    print(os.getcwd())
    local_rank = int(os.environ['LOCAL_RANK'])
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    
    if rank == 0:
        print(f"{local_rank=}\n{rank=}\n{world_size=}")

    # if local_rank == 0:
        # wandb.init(project="funcllama", name=f"{config.data.dataset_name}-{world_size}-load")
        # wandb.init(project="opt", name='tests')
    wandb_logger = WandbLogger(project="funcllama", name=f"{config.data.dataset_name}-{world_size}-load")
        
    tokenizer, model = load(config.model, config.augmentation, rank=rank)
    data_module = PLDataModule(tokenizer=tokenizer, data_args=config.data,
                               rank=rank, world_size=world_size)

    model = PLModel(model=model, tokenizer=tokenizer, rank=rank, config=config.training)
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.out_checkpoint_dir,
        every_n_epochs=1,
        save_on_train_epoch_end=False,
        # save_on_train_epoch_end=True,
        save_top_k=-1,
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
    
    cleanup()
    
    if trainer.is_global_zero:
        single_ckpt_path = "single_model.pt"

        # magically converts the folder into a single lightning loadable pytorch file (for ZeRO 1,2 and 3)
        convert_zero_checkpoint_to_fp32_state_dict(
            checkpoint_callback.best_model_path, osp.join(config.out_checkpoint_dir, 'best_model.pt'))
        # loaded_parameters = BoringModel.load_from_checkpoint(single_ckpt_path).parameters()

        # model = model.cpu()
        # # Assert model parameters are identical after loading
        # for orig_param, saved_model_param in zip(model.parameters(), loaded_parameters):
        #     if model.dtype == torch.half:
        #         # moved model to float32 for comparison with single fp32 saved weights
        #         saved_model_param = saved_model_param.half()
        #     assert torch.equal(orig_param, saved_model_param)

if __name__ == "__main__":
    main()