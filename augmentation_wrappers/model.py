# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

# from typing import Optional, Tuple
# from dataclasses import dataclass
import math
from typing import List
import torch
from torch import nn
import torch.nn.functional as F
import os.path as osp
# import copy
# import random
# import json
# import fairscale.nn.model_parallel.initialize as fs_init
# from fairscale.nn.model_parallel.layers import (
#     ParallelEmbedding,
#     RowParallelLinear,
#     ColumnParallelLinear,
#     _initialize_affine_weight,
#     get_model_parallel_rank,
#     get_model_parallel_world_size
# )
# from transformers import AutoTokenizer, AutoModel
from typing import Dict, Union
from pathlib2 import Path
from transformers import LlamaForCausalLM
from pytorch_lightning import LightningModule
from deepspeed.ops.adam import DeepSpeedCPUAdam

def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)

    return next_token


class PLModel(LightningModule):
    
    def __init__(self, model: nn.Module, tokenizer, config: Dict[str, Union[str, int]]):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.ignore_index = -100
        self.loss_fun = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        self.last = None
        
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    def training_step(self, batch, batch_idx):
        if not isinstance(batch, dict):
            assert len(batch) == 1
            batch = batch[0]
        
        raw_input_ids = torch.tensor(self.tokenizer.encode(batch["text"], bos=True, eos=True))[:]
        labels = raw_input_ids.clone()
        
        inputs = raw_input_ids[:-1].expand(1, -1).to("cuda")
        labels = labels[1:].expand(1, -1).to("cuda")
        
        full_logits = self.model(inputs).logits
        
        loss = self.loss_fun(full_logits.view(-1, full_logits.shape[-1]), labels.view(-1))
        
        return loss
    
    def on_before_backward(self, _):
        test = [{'name':n, 'weight':p} for n, p in self.model.named_parameters() if p.requires_grad]
        print(test)
        if self.last:
            print((self.last - test['weight']).sum())
        self.last = test['weight']
    
    
    def on_epoch_begin(self,):
        self.model.augment()
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.config.lr)
        # return DeepSpeedCPUAdam(self.model.parameters(), lr=self.config.lr)


class AugmentedLM(LlamaForCausalLM):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._augmented = False
    
    def freeze_base(self,):
        for param in self.parameters(): param.requires_grad = False
        # self.model.embed_tokens.weight.requires_grad = True
        # self.lm_head.weight.requires_grad = True
    
    def augment_embeddings(self, config):
        """
        Augment base LLM, to be run after the model is initialized & loaded from base
        """
        # initialize augmented embeddings
        base_weights = self.model.embed_tokens.weight
        aug_weight = torch.Tensor(config.aug_vocab_size, config.hidden_size)
        aug_weight = aug_weight.to(base_weights.device).to(base_weights.dtype)
        
        # TODO: change initialzation for semantic initialization
        # _initialize_affine_weight(
        #     aug_weight,
        #     self.model.embed_tokens.num_embeddings,
        #     self.model.embed_tokens.embedding_dim,
        #     self.model.embed_tokens.embedding_dim_per_partition,
        #     1,
        #     lambda x: x,
        #     stride=1,
        #     return_master_weight=False,
        # )
        stdv = 1. / math.sqrt(aug_weight.size(1)) # linear layer intialization
        aug_weight.uniform_(-stdv, stdv)
        
        # Expand embedding layer
        self.model.embed_tokens.weight = nn.Parameter(
            torch.cat((base_weights.data, aug_weight), dim=0)
            )
        
        # unfreeze new embeddings & register hook
        self.model.embed_tokens.weight.requires_grad = True
        mask = torch.zeros_like(self.model.embed_tokens.weight)
        mask[-config.aug_vocab_size:] = 1.
        self.model.embed_tokens.weight.register_hook(lambda grad: grad*mask)
        
    def augment_projection(self, config):
        """
        Augment base LLM, to be run after the model is initialized & loaded from base
        """
        # initialize augmented embeddings
        base_weights = self.lm_head.weight
        print('projection weights: ', base_weights.shape)
        aug_weight = torch.Tensor(config.aug_vocab_size, config.hidden_size)
        aug_weight = aug_weight.to(base_weights.device).to(base_weights.dtype)
        
        # TODO: change initialzation for semantic initialization
        # _initialize_affine_weight(
        #     aug_weight,
        #     self.model.tok_embeddings.num_embeddings,
        #     self.model.tok_embeddings.embedding_dim,
        #     self.model.tok_embeddings.embedding_dim_per_partition,
        #     1,
        #     lambda x: x,
        #     stride=1,
        #     return_master_weight=False,
        # )
        stdv = 1. / math.sqrt(aug_weight.size(1)) # linear layer intialization
        aug_weight.uniform_(-stdv, stdv)
        
        # Expand embedding layer
        self.lm_head.weight = nn.Parameter(
            torch.cat((base_weights.data, aug_weight), dim=0)
            )
        
        # unfreeze new embeddings & register hook
        self.lm_head.weight.requires_grad = True
        mask = torch.zeros_like(self.lm_head.weight)
        mask[-config.aug_vocab_size:] = 1.
        self.lm_head.weight.register_hook(lambda grad: grad*mask)
        
    def augment(self, config):
        if not self._augmented:
            self.freeze_base()
            self.augment_embeddings(config)
            self.augment_projection(config)
            self._augmented = True
    
    def save_augmentation(self, save_dir:str, local_rank:int = 0):
        """
        Save the augmented mode components.
        TODO: update to save in different files per rank
        """
        raise NotImplementedError
        print("Saving augmented_model")
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        save_path = osp.join(save_dir, f'checkpoint_{local_rank}.pt')
        embedding_state_dict = self.model.tok_embeddings.state_dict()
        output_state_dict = self.tool_output.state_dict()
        
        # Only store augmentation embeddings
        embedding_state_dict['weight'] = embedding_state_dict['weight'][-self.model.params.aug_vocab_size:]
        augmented_state_dict = {
            "tool_embeddings": embedding_state_dict,
            "tool_output": output_state_dict
        }
        # only store tool output layer and func embedding
        torch.save(augmented_state_dict, save_path)
        
        
    def load_augmentation(self, load_dir:str, local_rank:int = 0):
        """
        Save the augmented embeddings
        TODO: update to load in different files per rank
        """
        raise NotImplementedError
        load_path = osp.join(load_dir, f'checkpoint_{local_rank}.pt')
        augmentation_state_dict = torch.load(load_path)
        assert isinstance(augmentation_state_dict, dict), f"expected a state dictionary, found {type(augmentation_state_dict)}"
        tool_embeddings_sd = augmentation_state_dict['tool_embeddings']
        tool_output_sd= augmentation_state_dict['tool_output']
        # load augmentation compenents
        
        base_vocab_weights = self.model.tok_embeddings.weight[:-self.model.params.aug_vocab_size]
        # print(f"Embedding weights rank {local_rank}: ", tool_embeddings_sd['weight'])
        self.model.tok_embeddings.weight = nn.Parameter(
            torch.cat((base_vocab_weights, tool_embeddings_sd['weight'].to(base_vocab_weights.device)), dim=0)
            )
        
        self.tool_output.load_state_dict(tool_output_sd)
    