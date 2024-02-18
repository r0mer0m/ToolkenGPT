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


def load(ckpt_dir: str, tokenizer_path: str, local_rank: int, world_size: int) -> AugmentedLM:
    start_time = time.time()
    # checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    # assert (
    #     world_size == len(checkpoints)
    # ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    # ckpt_path = checkpoints[local_rank]
    # print("Loading")
    # checkpoint = torch.load(ckpt_path, map_location="cpu")
    
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(max_seq_len=2048, max_batch_size=1, **params)
    tokenizer = AugmentedTokenizer(tokenizer_dir=tokenizer_path)
    model_args.vocab_size = tokenizer.n_base_words
    model_args.aug_vocab_size = tokenizer.n_aug_words
    # torch.set_default_tensor_type(torch.cuda.HalfTensor)
    # model = Transformer(model_args).cuda().half()
    # torch.set_default_tensor_type(torch.FloatTensor)
    torch.set_default_tensor_type(torch.FloatTensor)
    model = Transformer(model_args).cuda()
    # model.load_state_dict(checkpoint, strict=False)
    funcmodel = AugmentedLM(model, tokenizer)
    print("Initialization emb size: ", funcmodel.model.tok_embeddings.weight.shape)
    print("Initialization output tool size: ", funcmodel.tool_output.weight.shape)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return funcmodel

@record
def main(ckpt_dir: str, tokenizer_path: str, input_file: str = None, lr: float = 1e-3, num_epochs: int = 20, dataset: str = "gsm8k-xl", log_prefix="", only_functoken=False, log_each=False):

    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(1)
    np.random.seed(1)

    # func_dict_path = f"../data/{dataset}/func_dict.json"

    # func_dict = json.load(open(func_dict_path, "r"))

    local_rank, world_size = setup_model_parallel()
    # if local_rank > 0:
    #     sys.stdout = open(os.devnull, 'w')

    # if local_rank == 0:
    #     wandb.init(project="funcllama", name=f"{dataset}-{world_size}-load")
        # wandb.init(project="opt", name=save_name)

    funcmodel = load(ckpt_dir, tokenizer_path, local_rank, world_size)
    print("Saving...")
    funcmodel.save_augmentation('./augmented_components', local_rank=local_rank)
    print("Loading...")
    funcmodel.load_augmentation('./augmented_components', local_rank=local_rank)
    print("Post-loading emb size: ", funcmodel.model.tok_embeddings.weight.shape)
    print("Post-loading output tool size: ", funcmodel.tool_output.weight.shape)

if __name__ == "__main__":
    fire.Fire(main)