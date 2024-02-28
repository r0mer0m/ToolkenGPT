# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

import os
import sys
import torch
import fire
import time
import json
import random
import wandb
import numpy as np
from tqdm import tqdm
# from typing import Tuple
# from pathlib import Path
# from fairscale.nn.model_parallel.initialize import initialize_model_parallel
# from llama_augmented import ModelArgs, Transformer, AugmentedTokenizer, AugmentedLM
from augmentation_wrappers import AugmentedLM, AugmentedConfig, AugmentedTokenizer, Trainer
from torch.distributed.elastic.multiprocessing.errors import record
from collections import defaultdict
from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaTokenizer,
    LlamaConfig,
    # BitsAndBytesConfig
)
from types import SimpleNamespace
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoTokenizer, GPT2TokenizerFast
from transformers import T5Tokenizer, T5ForConditionalGeneration
import functools
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from looseversion import LooseVersion
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
 checkpoint_wrapper,
 CheckpointImpl,
 apply_activation_checkpointing_wrapper)

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    enable_wrap,
    wrap,
)
from functools import partial
from torch.utils.data import DataLoader
from pathlib import Path
from summarization_dataset import *
from transformers.models.t5.modeling_t5 import T5Block
from typing import Type
import time
from datetime import datetime

    

# def setup_model_parallel() -> Tuple[int, int]:
#     local_rank = int(os.environ.get("LOCAL_RANK", -1))
#     world_size = int(os.environ.get("WORLD_SIZE", -1))

#     torch.distributed.init_process_group("nccl")
#     initialize_model_parallel(world_size)
#     torch.cuda.set_device(local_rank)
#     return local_rank, world_size


def setup():
    # initialize the process group
    dist.init_process_group("nccl")

def cleanup():
    dist.destroy_process_group()


def load(ckpt_dir: str, tokenizer_path: str, local_rank: int, world_size: int, func_dict: dict) -> AugmentedLM:
    tokenizer = AugmentedTokenizer.from_pretrained(
        'meta-llama/Llama-2-7b-chat-hf',
        augmentation_config_path='./augmented_tokenizer/augmentation_config.json'
        )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    
    augmented_config = AugmentedConfig.from_pretrained(
        'meta-llama/Llama-2-7b-chat-hf',
        augment=True,
        aug_vocab_size = 8 # tokenizer.n_aug_words,
    )
    compute_dtype = getattr(torch, "bfloat16")
    # quant_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=compute_dtype,
    #     bnb_4bit_use_double_quant=False,
    # )
    
    model = AugmentedLM.from_pretrained(
        'meta-llama/Llama-2-7b-chat-hf',
        # load_in_8bit=False,# if train_config.quantization else None,
        device_map='auto',#0,#"auto",# if train_config.quantization else None,
        low_cpu_mem_usage=True,
        # quantization_config=quant_config,
        # use_cache=False,
        attn_implementation=None, #"sdpa" if train_config.use_fast_kernels else None,
        torch_dtype=torch.bfloat16
        )
    
    model.augment(augmented_config)
    
    return tokenizer, model


def process_data(input_file:str, dataset:str, rank:int=0, world_size:int=1):
    if input_file.endswith(".json"):
        with open(input_file, "r") as f:
            prompts = json.load(f)
    
    else:
        with open(input_file, "r") as f:
            prompts = f.readlines()
        prompts = [prompt.strip().replace("\\n", "\n") for prompt in prompts if len(prompt) > 1]

    if dataset == "gsm8k-xl":
        # the last 1000 prompts are the testset
        test_len = 1000
    elif dataset == "funcqa":
        # the last 39 prompts are the testset
        test_len = 39
    elif dataset == "vh":
        test_len = 47
    elif dataset == "kamel":
        test_len = 1000
    
    print("Total data: ")
    testset = prompts[-test_len:]
    trainset = prompts[:-test_len]
    
    # train_dataset = wikihow(tokenizer, 'train', 1500, 512, 150, False)
    # val_dataset = wikihow(tokenizer, 'validation', 300, 512, 150, False)

    sampler1 = DistributedSampler(trainset, rank=rank, num_replicas=world_size, shuffle=True)
    sampler2 = DistributedSampler(testset, rank=rank, num_replicas=world_size)
    
    train_kwargs = {'batch_size': 4, 'sampler': sampler1}
    test_kwargs = {'batch_size': 4, 'sampler': sampler2}
    cuda_kwargs = {'num_workers': 2,
                    'pin_memory': True,
                    'shuffle': False}
    
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    train_loader = torch.utils.data.DataLoader(trainset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(testset, **test_kwargs)

    
    '''
    ############# Toy example for testing non-function training ##############
    # trainset = [
    #     {
    #         "text": "This is a sample text.",
    #         "tar_eq": ["target1", "target2"],
    #         "tar_number": [42, 77],
    #         "api_ids": [123, 456],
    #         "bor_idxs": [1, 3, 5],
    #         "eor_idxs": [2, 4, 6]
    #     },
    #     {
    #         "text": "Another example here.",
    #         "tar_eq": ["target3", "target4", "target5"],
    #         "tar_number": [101, 202, 303],
    #         "api_ids": [789, 987],
    #         "bor_idxs": [7, 9, 11],
        #     "eor_idxs": [8, 10, 12]
        # },
        # Add more records as needed
    # ]
    # trainset = trainset * 100
    '''
    if rank == 0:
        print(f"Trainset: {len(trainset)}, Testset: {len(testset)}")
    
    return train_loader, test_loader

@record
def main(ckpt_dir: str, tokenizer_path: str, input_file: str = None, lr: float = 1e-3, num_epochs: int = 20, dataset: str = "gsm8k-xl", log_prefix="", only_functoken=False, log_each=False):

    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(1)
    np.random.seed(1)
    
    local_rank = int(os.environ['LOCAL_RANK'])
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    
    func_dict_path = f"../data/{dataset}/func_dict.json"

    func_dict = json.load(open(func_dict_path, "r"))
    
    training_config = SimpleNamespace(lr=lr, num_epochs=num_epochs, log_each=log_each)
    
    # # local_rank, world_size = setup_model_parallel()
    # if local_rank > 0:
    #     sys.stdout = open(os.devnull, 'w')

    # if local_rank == 0:
    #     wandb.init(project="funcllama", name=f"{dataset}-{world_size}-load")
        # wandb.init(project="opt", name=save_name)
        
    train_dataloader, test_dataloader = process_data(input_file, dataset)
    tokenizer, model = load(ckpt_dir, tokenizer_path, local_rank=0, world_size=2, func_dict=func_dict)
    
    llama_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            LlamaDecoderLayer,
        },
    )
    sharding_strategy: ShardingStrategy = ShardingStrategy.SHARD_GRAD_OP #for Zero2 and FULL_SHARD for Zero3
    torch.cuda.set_device(local_rank)
    
    bf16_ready = (
        torch.version.cuda
        and torch.cuda.is_bf16_supported()
        and LooseVersion(torch.version.cuda) >= "11.0"
        and dist.is_nccl_available()
        and torch.cuda.nccl.version() >= (2, 10)
    )

    if bf16_ready:
        bfSixteen = MixedPrecision(
            param_dtype=torch.bfloat16,
            # Gradient communication precision.
            reduce_dtype=torch.bfloat16,
            # Buffer precision.
            buffer_dtype=torch.bfloat16,
        )
        mp_policy = bfSixteen
    else:
        mp_policy = None # defaults to fp32

    model = FSDP(model,
        auto_wrap_policy=llama_auto_wrap_policy,
        mixed_precision=mp_policy,
        sharding_strategy=sharding_strategy,
        device_id=torch.cuda.current_device())

    
    trainer = Trainer(
        tokenizer=tokenizer,
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        training_config = training_config
    )
    
    trainer.train()
    # optimizer = torch.optim.Adam([p for p in funcmodel.parameters() if p.requires_grad], lr=lr)
    
    
    
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

if __name__ == "__main__":
    fire.Fire(main)