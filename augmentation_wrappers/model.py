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
from collections import defaultdict
import wandb
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
        # self.last = None
        
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    def on_train_epoch_start(self) -> None:
        self.results = defaultdict(list)
        return super().on_train_epoch_start()
    
    def training_step(self, batch, batch_idx):
        if not isinstance(batch, dict):
            assert len(batch) == 1
            batch = batch[0]
        
        input_ids = batch['input_ids'].to("cuda")
        target_ids = batch['target_ids'].to("cuda")
        
        print("input_ids shape: ", input_ids.shape)
        print("target_ids shape: ", target_ids.shape)
        print("input_ids shape: ", input_ids)
        print("target_ids: ", target_ids)
        
        logits = self.model(input_ids).logits
        
        loss = self.loss_fun(logits.view(-1, logits.shape[-1]), target_ids.view(-1))
        
        return {'loss': loss, 'logits': logits}
    
    # def on_before_backward(self, _):
    #     test = [{'name':n, 'weight':p} for n, p in self.model.named_parameters() if p.requires_grad]
    #     print(test)
    #     if self.last:
    #         print((self.last - test['weight']).sum())
    #     self.last = test['weight']
    
    def on_epoch_begin(self,):
        self.model.register_backward_hooks()
        
    
    def _compute_trigger_metrics(self, label_funcs, pred_funcs):
        tp = torch.sum(label_funcs * pred_funcs, dim=-1).detach().cpu().numpy()
        pred_funcs = torch.sum(pred_funcs, dim=-1).detach().cpu().numpy()
        true = torch.sum(label_funcs, dim=-1).detach().cpu().numpy()
        results = {
            "tp": tp,
            "pred": pred_funcs,
            "true": true
        }
        return results
    
    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        '''
        Metrics recording and computation
        '''
        # return super().on_train_batch_end(outputs, batch, batch_idx)
        if self.config.rank == 0:
            wandb.log({"loss": outputs['loss'].item()})
        
        for i, r in metrics.items(): self.results[i].append(r)
            
        # (bsz, seqlen, aug_vocab_size) -> (bsz, seqlen)
        pred = torch.argmax(outputs['logits'], dim=-1)
        pred = pred.view(-1)
        labels = batch['target_ids'].view(-1)

        label_funcs = [labels == idx for idx in self.tokenizer.api_ids]
        pred_funcs = [pred == idx for idx in self.tokenizer.api_ids]
        label_funcs = torch.stack(label_funcs, dim=0)
        pred_funcs = torch.stack(pred_funcs, dim=0)
        
        metrics = self._compute_trigger_metrics(label_funcs, pred_funcs)
        
        
        if (batch_idx + 1) % 20 == 0 and self.config.rank == 0:
            
            for i, api_name in enumerate(self.tokenizer.api_names):
                tp = sum([r[i] for r in self.results["tp"]])
                pred = sum([r[i] for r in self.results["pred"]])
                true = sum([r[i] for r in self.results["true"]])
            
                wandb.log({
                    f"precision-{api_name}": tp / (pred + 1e-8),
                    f"recall-{api_name}": tp / (true + 1e-8),
                    f"f1-{api_name}": 2 * tp / (pred + true + 1e-8)
                })
            
            tp = sum([r.sum() for r in self.results["tp"]])
            pred = sum([r.sum() for r in self.results["pred"]])
            true = sum([r.sum() for r in self.results["true"]])
        
            wandb.log({
                f"precision": tp / (pred + 1e-8),
                f"recall": tp / (true + 1e-8),
                f"f1": 2 * tp / (pred + true + 1e-8)
            })
            
            self.results = defaultdict(list)
            
            
        
        
            
    
    def configure_optimizers(self):
        return torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=self.config.lr)
        # return DeepSpeedCPUAdam(self.model.parameters(), lr=self.config.lr)


class AugmentedLM(LlamaForCausalLM):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._hook_registred = False
    
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
        
    def augment(self, config):
        self.freeze_base()
        self.augment_embeddings(config)
        self.augment_projection(config)
        
        # Workaround for hooks
        self.aug_vocab_size = config.aug_vocab_size
        
            
    def _register_backward_hooks(self, ):
        # Embeddings
        mask = torch.zeros_like(self.model.embed_tokens.weight)
        mask[-self.aug_vocab_size:] = 1.
        self.model.embed_tokens.weight.register_hook(lambda grad: grad*mask)
        
        # LM Head
        mask = torch.zeros_like(self.lm_head.weight)
        mask[-self.aug_vocab_size:] = 1.
        self.lm_head.weight.register_hook(lambda grad: grad*mask)
    
    def register_backward_hooks(self,):        
        if not self._hook_registred:
            self._hook_registred = True
            self._register_backward_hooks(self)
    
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
    