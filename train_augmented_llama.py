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
from typing import Tuple
from pathlib import Path
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from llama_augmented import ModelArgs, Transformer, AugmentedTokenizer, AugmentedLM
from torch.distributed.elastic.multiprocessing.errors import record

def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)
    return local_rank, world_size


def load(ckpt_dir: str, tokenizer_path: str, local_rank: int, world_size: int, func_dict: dict) -> AugmentedLM:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert (
        world_size == len(checkpoints)
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    print(ckpt_path)
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(max_seq_len=2048, max_batch_size=1, **params)
    tokenizer = AugmentedTokenizer(tokenizer_dir=tokenizer_path)
    model_args.vocab_size = tokenizer.n_base_words
    model_args.aug_vocab_size = tokenizer.n_aug_words
    # torch.set_default_tensor_type(torch.cuda.HalfTensor)
    # model = Transformer(model_args).cuda().half()
    # torch.set_default_tensor_type(torch.FloatTensor)
    torch.set_default_tensor_type(torch.HalfTensor)
    model = Transformer(model_args).half()
    model.load_state_dict(checkpoint, strict=False)
    torch.set_default_tensor_type(torch.FloatTensor)
    funcmodel = AugmentedLM(model, tokenizer).cuda().float()
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    funcmodel.freeze_base_model()
    funcmodel.train()
    return funcmodel

@record
def main(ckpt_dir: str, tokenizer_path: str, input_file: str = None, lr: float = 1e-3, num_epochs: int = 20, dataset: str = "gsm8k-xl", log_prefix="", only_functoken=False, log_each=False):

    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(1)
    np.random.seed(1)

    func_dict_path = f"../data/{dataset}/func_dict.json"

    func_dict = json.load(open(func_dict_path, "r"))

    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, 'w')

    if local_rank == 0:
        wandb.init(project="funcllama", name=f"{dataset}-{world_size}-load")
        # wandb.init(project="opt", name=save_name)

    funcmodel = load(ckpt_dir, tokenizer_path, local_rank, world_size, func_dict=func_dict)
    n_fun = funcmodel.tokenizer.n_fun
    
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
    
    print(f"Trainset: {len(trainset)}, Testset: {len(testset)}")

    # only update tokens with gradients required
    # optimizer = torch.optim.Adam([p for p in funcmodel.parameters() if p.requires_grad], lr=lr)
    optimizer = torch.optim.Adam(funcmodel.parameters(), lr=lr)
    
    from collections import defaultdict
    funcmodel.train()
    for epoch in range(num_epochs):
        results = defaultdict(list)
        
        random.shuffle(trainset)
        for case_idx, prompt in tqdm(enumerate(trainset), desc=f"Epoch {epoch} - Training"):
            
            # if len(prompt['api_ids']) == 0:
            #     continue
            
            optimizer.zero_grad()
            loss, result = funcmodel.get_loss([prompt], only_functoken=only_functoken)
            loss.backward()
            print("------------------ backwards done ------------------")
            optimizer.step()
            # tool_ouput_diff = torch.abs(funcmodel.prev_weigths, funcmodel.tool_output.weigth).sum()
            # print(f"Case idx {case_idx} tool_ouput_diff: {tool_ouput_diff}")
            # funcmodel.prev_weights = funcmodel.tool_output.weight.clone()

            for i, r in result.items():
                results[i].append(r)
            
            if (case_idx + 1) % 20 == 0:
                
                for i in range(n_fun + 1):
                    if i != n_fun:
                        tp, pred, true = sum([r[i] for r in results["tp"]]), sum([r[i] for r in results["pred"]]), sum([r[i] for r in results["true"]])
                    else:
                        tp, pred, true = sum([r.sum() for r in results["tp"]]), sum([r.sum() for r in results["pred"]]), sum([r.sum() for r in results["true"]])
                    # print(f"tp: {tp}, pred: {pred}, true: {true}")
                    
                    if local_rank == 0:
                        if i != n_fun and log_each:
                            wandb.log({
                                f"precision-{i}": tp / (pred + 1e-8),
                                f"recall-{i}": tp / (true + 1e-8),
                                f"f1-{i}": 2 * tp / (pred + true + 1e-8)
                            })
                        elif i == n_fun:
                            wandb.log({
                                f"precision": tp / (pred + 1e-8),
                                f"recall": tp / (true + 1e-8),
                                f"f1": 2 * tp / (pred + true + 1e-8)
                            })
                        # save the parameters of func_embed
                        # torch.save(funcmodel.func_embed.state_dict(), save_file)
                results = defaultdict(list)
            
            if local_rank == 0:
                wandb.log({"loss": loss.item()})
        
        # test on validation set
        results = defaultdict(list)
        for case_idx, prompt in tqdm(enumerate(testset), desc=f"Epoch {epoch} - Validation"):
            funcmodel.eval()
            
            # if len(prompt['api_ids']) == 0:
            #     continue
            
            with torch.no_grad():
                loss, result = funcmodel.get_loss([prompt])
            
            for i, r in result.items():
                results[i].append(r)
            
        for i in range(n_fun + 1):
            if i != n_fun:
                tp, pred, true = sum([r[i] for r in results["tp"]]), sum([r[i] for r in results["pred"]]), sum([r[i] for r in results["true"]])
            else:
                # 4 is for all functions
                tp, pred, true = sum([r.sum() for r in results["tp"]]), sum([r.sum() for r in results["pred"]]), sum([r.sum() for r in results["true"]])
            # print(f"tp: {tp}, pred: {pred}, true: {true}")
            
            if local_rank == 0:
                if i != n_fun and log_each:
                    wandb.log({
                        f"test-precision-{i}": tp / (pred + 1e-8),
                        f"test-recall-{i}": tp / (true + 1e-8),
                        f"test-f1-{i}": 2 * tp / (pred + true + 1e-8)
                    })
                elif i == n_fun:
                    wandb.log({
                        f"test-precision": tp / (pred + 1e-8),
                        f"test-recall": tp / (true + 1e-8),
                        f"test-f1": 2 * tp / (pred + true + 1e-8)
                    })

        # save the parameters of func_embed every epoch
        print("Saving augmented_model")
        save_dir = f"checkpoints/{log_prefix}{dataset}/epoch_{epoch}"
        # os.makedirs(save_dir, exist_ok=True)
        funcmodel.save_augmentation(save_dir, local_rank=local_rank)
        # torch.save(funcmodel.func_embed.state_dict(), f"{save_dir}/epoch_{epoch}.pth")
        results = defaultdict(list)

if __name__ == "__main__":
    fire.Fire(main)