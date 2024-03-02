# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

import os
import torch
import random
import wandb
import numpy as np
from augmentation_wrappers import AugmentedLM, AugmentedConfig, AugmentedTokenizer, PLModel, PLDataModule
from torch.distributed.elastic.multiprocessing.errors import record
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DeepSpeedStrategy
from types import SimpleNamespace
import torch.distributed as dist
import os
import torch
import torch.distributed as dist
import hydra

def setup():
    # initialize the process group
    dist.init_process_group("nccl")

def cleanup():
    dist.destroy_process_group()


def load(model_config, augmentation_config, rank: int, world_size: int) -> AugmentedLM:
    
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
    print(config.training.deepspeed_config)
    local_rank = int(os.environ['LOCAL_RANK'])
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    
    if rank == 0:
        print(f"{local_rank=}\n{rank=}\n{world_size=}")

    if local_rank == 0:
        wandb.init(project="funcllama", name=f"{config.data.dataset_name}-{world_size}-load")
        # wandb.init(project="opt", name='tests')
        
    tokenizer, model = load(config.model, config.augmentation, rank=rank, world_size=world_size)
    data_module = PLDataModule(tokenizer=tokenizer, data_args=config.data)#input_file=input_file, dataset_name=dataset, rank=rank, world_size=world_size)#train_dataloader, test_dataloader = process_data(input_file, dataset)

    model = PLModel(model=model, tokenizer=tokenizer, rank=rank, config=config.training)
    trainer = Trainer(
        accelerator="gpu",
        max_epochs=config.training.num_epochs,
        devices=world_size,
        strategy=DeepSpeedStrategy(config=config.training.deepspeed_config),
        # strategy="deepspeed_stage_3",
        precision=16,
    )
    trainer.fit(model, datamodule=data_module)
    
    # funcmodel.train()
    # for epoch in range(num_epochs):
    #     results = defaultdict(list)
        
    #     random.shuffle(trainset)
    #     for case_idx, prompt in tqdm(enumerate(trainset), desc=f"Epoch {epoch} - Training"):
            
    #         # if len(prompt['api_ids']) == 0:
    #         #     continue
    #         optimizer.zero_grad(set_to_none=True)
    #         # print([{'name':n, 'parameter': p, 'grad':p.grad} for n, p in funcmodel.named_parameters() if p.requires_grad])
    #         # loss, result = funcmodel.get_loss([prompt], only_functoken=only_functoken)
    #         with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    #             loss, _ = funcmodel.get_standard_loss([prompt])
    #             loss.backward()
    #         print("------------------ backwards done ------------------")
    #         optimizer.step()
    #         print([{'name':n, 'parameter': p, 'grad':p.grad} for n, p in funcmodel.named_parameters() if p.requires_grad])
            
    #         # tool_ouput_diff = torch.abs(funcmodel.prev_weigths, funcmodel.tool_output.weigth).sum()
    #         # print(f"Case idx {case_idx} tool_ouput_diff: {tool_ouput_diff}")
    #         # funcmodel.prev_weights = funcmodel.tool_output.weight.clone()

    #         # for i, r in result.items():
    #         #     results[i].append(r)
            
    #         # if (case_idx + 1) % 20 == 0:
                
    #         #     for i in range(n_fun + 1):
    #         #         if i != n_fun:
    #         #             tp, pred, true = sum([r[i] for r in results["tp"]]), sum([r[i] for r in results["pred"]]), sum([r[i] for r in results["true"]])
    #         #         else:
    #         #             tp, pred, true = sum([r.sum() for r in results["tp"]]), sum([r.sum() for r in results["pred"]]), sum([r.sum() for r in results["true"]])
    #         #         # print(f"tp: {tp}, pred: {pred}, true: {true}")
                    
    #         #         if local_rank == 0:
    #         #             if i != n_fun and log_each:
    #         #                 wandb.log({
    #         #                     f"precision-{i}": tp / (pred + 1e-8),
    #         #                     f"recall-{i}": tp / (true + 1e-8),
    #         #                     f"f1-{i}": 2 * tp / (pred + true + 1e-8)
    #         #                 })
    #         #             elif i == n_fun:
    #         #                 wandb.log({
    #         #                     f"precision": tp / (pred + 1e-8),
    #         #                     f"recall": tp / (true + 1e-8),
    #         #                     f"f1": 2 * tp / (pred + true + 1e-8)
    #         #                 })
    #         #             # save the parameters of func_embed
    #         #             # torch.save(funcmodel.func_embed.state_dict(), save_file)
    #         #     results = defaultdict(list)
            
    #         # if local_rank == 0:
    #         #     wandb.log({"loss": loss.item()})
        
    #     # test on validation set
    #     # results = defaultdict(list)
    #     # for case_idx, prompt in tqdm(enumerate(testset), desc=f"Epoch {epoch} - Validation"):
    #     #     funcmodel.eval()
            
    #     #     # if len(prompt['api_ids']) == 0:
    #     #     #     continue
            
    #     #     with torch.no_grad():
    #     #         loss, result = funcmodel.get_loss([prompt])
            
    #     #     for i, r in result.items():
    #     #         results[i].append(r)
            
    #     # for i in range(n_fun + 1):
    #     #     if i != n_fun:
    #     #         tp, pred, true = sum([r[i] for r in results["tp"]]), sum([r[i] for r in results["pred"]]), sum([r[i] for r in results["true"]])
    #     #     else:
    #     #         # 4 is for all functions
    #     #         tp, pred, true = sum([r.sum() for r in results["tp"]]), sum([r.sum() for r in results["pred"]]), sum([r.sum() for r in results["true"]])
    #     #     # print(f"tp: {tp}, pred: {pred}, true: {true}")
            
    #     #     if local_rank == 0:
    #     #         if i != n_fun and log_each:
    #     #             wandb.log({
    #     #                 f"test-precision-{i}": tp / (pred + 1e-8),
    #     #                 f"test-recall-{i}": tp / (true + 1e-8),
    #     #                 f"test-f1-{i}": 2 * tp / (pred + true + 1e-8)
    #     #             })
    #     #         elif i == n_fun:
    #     #             wandb.log({
    #     #                 f"test-precision": tp / (pred + 1e-8),
    #     #                 f"test-recall": tp / (true + 1e-8),
    #     #                 f"test-f1": 2 * tp / (pred + true + 1e-8)
    #     #             })

    #     # # save the parameters of func_embed every epoch
    #     # print("Saving augmented_model")
    #     # save_dir = f"checkpoints/{log_prefix}{dataset}/epoch_{epoch}"
    #     # # os.makedirs(save_dir, exist_ok=True)
    #     # funcmodel.save_augmentation(save_dir, local_rank=local_rank)
    #     # # torch.save(funcmodel.func_embed.state_dict(), f"{save_dir}/epoch_{epoch}.pth")
    #     results = defaultdict(list)
    
    cleanup()

if __name__ == "__main__":
    main()