import os
import hydra
import torch
import wandb
import random
import numpy as np
from fuse.utils import load
from fuse import PLDataModule
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record

def setup():
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
    
    setup()
    
    local_rank = int(os.environ['LOCAL_RANK'])
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    
    if rank == 0:
        print(f"{local_rank=}\n{rank=}\n{world_size=}")

    if local_rank == 0:
        wandb.init(project="funcllama", name=f"{config.data.dataset_name}-{world_size}-load")
        
    tokenizer, model = load(config.model, config.augmentation, rank=rank, world_size=world_size)
    data_module = PLDataModule(tokenizer=tokenizer, data_args=config.data,
                               rank=rank, world_size=world_size)
    data_module.setup("test")
    test_dataloader = data_module.test_dataloader()
    
    model.eval()
    for i, batch in enumerate(test_dataloader):
        target = batch['target']
        with torch.no_grad():
            output = model.augmented_generation(i, batch['input_ids'], batch['template_len'])
            print(output)
            print(target)
            break
    

    # model = PLModel(model=model, tokenizer=tokenizer, rank=rank, config=config.training)
    # checkpoint_callback = ModelCheckpoint(
    #     dirpath=f'./checkpoints/{config.data.dataset_name}/', # <--- specify this on the trainer itself for version control
    #     filename="fa_classifier_{epoch:02d}",
    #     every_n_epochs=1,
    #     save_top_k=-1,
    #     monitor="val_loss",
    # )
    # trainer = Trainer(
    #     accelerator="gpu",
    #     max_epochs=config.training.num_epochs,
    #     callbacks=[checkpoint_callback],
    #     check_val_every_n_epoch=1,
    #     devices=world_size,
    #     strategy=DeepSpeedStrategy(config=config.training.deepspeed_config),
    #     # strategy="deepspeed_stage_3",
    #     precision=16,
    # )
    # trainer.fit(model, datamodule=data_module)
    
    cleanup()

if __name__ == "__main__":
    main()