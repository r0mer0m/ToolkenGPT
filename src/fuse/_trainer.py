# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

# from typing import Optional, Tuple
# from dataclasses import dataclass
import math
from typing import List
import torch
from torch import nn
import torch.nn.functional as F
import os
import os.path as osp
import torch.distributed as dist
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
import tqdm

def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)

    return next_token


class Trainer(object):
    
    def __init__(self, 
                tokenizer,
                model,
                train_dataloader,
                test_dataloader,
                training_config
                ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.config = training_config
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr)
    
    def train(self, ):
        # self.model.train()
        # for epoch in range(self.config.num_epochs):
        #     for batch in self.train_dataloader:
        #         print(type(batch))
        #         print(batch)
        #         self.optimizer.zero_grad()
        #         print('Text: ', batch['text'])
        #         with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        #             loss, results = self.get_standard_loss(batch)
        #             loss.backward()
        #         self.optimizer.step()
        #         print([{'name':n, 'grad':p.grad.sum()} for n, p in self.model.named_parameters() if p.requires_grad])
        #         print(f"Epoch: {epoch}, Loss: {loss.item()}")
        
        for epoch in range(self.config.num_epochs):
            self.model.train()
            local_rank = int(os.environ['LOCAL_RANK'])
            rank = int(os.environ['RANK'])
            fsdp_loss = torch.zeros(2).to(local_rank)

            if rank==0:
                inner_pbar = tqdm.tqdm(
                    range(len(self.train_dataloader)), colour="blue", desc="r0 Training Epoch"
                )
            
            for batch in self.train_dataloader:
                for key in batch.keys():
                    batch[key] = batch[key].to(local_rank)
                self.optimizer.zero_grad()
                output = self.model(input_ids=batch["source_ids"],attention_mask=batch["source_mask"],labels=batch["target_ids"] )
                loss = output["loss"]
                loss.backward()
                self.optimizer.step()
                fsdp_loss[0] += loss.item()
                fsdp_loss[1] += len(batch)
                if rank==0:
                    inner_pbar.update(1)

            dist.all_reduce(fsdp_loss, op=dist.ReduceOp.SUM)
            train_accuracy = fsdp_loss[0] / fsdp_loss[1]


            if rank == 0:
                inner_pbar.close()
                print(
                        f"Train Epoch: \t{epoch}, Loss: \t{train_accuracy:.4f}"
                    )
            return train_accuracy
            
    def get_standard_loss(
        self,
        raw_inputs:List[Dict[str, Union[str, int]]]
    ):
        if not isinstance(raw_inputs, dict):
            assert len(raw_inputs) == 1
            raw_inputs = raw_inputs[0]
        
        raw_input_ids = torch.tensor(self.tokenizer.encode(raw_inputs["text"], bos=True, eos=True))[:]
        labels = raw_input_ids.clone()
        
        inputs = raw_input_ids[:-1].expand(1, -1).to("cuda")
        labels = labels[1:].expand(1, -1).to("cuda")
        
        full_logits = self.model(inputs).logits
        
        loss = F.cross_entropy(full_logits.view(-1, full_logits.shape[-1]), labels.view(-1), ignore_index=-100)
        print(loss)
        results = {}

        return loss, results
    
    def get_loss(
        self,
        raw_inputs:List[Dict[str, Union[str, int]]],
        only_functoken:bool=False,
    ):
        assert len(raw_inputs) == 1
        raw_inputs = raw_inputs[0]
        
        raw_input_ids = torch.tensor(self.tokenizer.encode(raw_inputs["text"], bos=True, eos=True))[:]
        labels = raw_input_ids.clone()
        for bor_idx, eor_idx in zip(raw_inputs["bor_idxs"], raw_inputs["eor_idxs"]):
            labels[bor_idx+1: eor_idx] = -100 # mask out API results tokens
        
        inputs = raw_input_ids[:-1].expand(1, -1).to("cuda")
        labels = labels[1:].expand(1, -1).to("cuda")
        
        full_logits = self(inputs, 0)
        
        # print(full_logits.shape, labels.max(), labels.min())
        loss = F.cross_entropy(full_logits.view(-1, full_logits.shape[-1]), labels.view(-1), ignore_index=-100)
        
        # Compute in-batch metrics
        # pred = torch.argmax(full_logits.detach(), dim=-1) # (bsz, seqlen)
        # pred = pred.view(-1)
        # labels = labels.view(-1)
                
        # label_funcs = [labels == int(id) for id in self.tokenizer.fun_map.keys()]
        # pred_funcs = [pred == int(id) for id in self.tokenizer.fun_map.keys()]
        # label_funcs = torch.stack(label_funcs, dim=0)
        # pred_funcs = torch.stack(pred_funcs, dim=0)
        
        # tp = torch.sum(label_funcs * pred_funcs, dim=-1).detach().cpu().numpy()
        # pred_funcs = torch.sum(pred_funcs, dim=-1).detach().cpu().numpy()
        # true = torch.sum(label_funcs, dim=-1).detach().cpu().numpy()
        # results = {
        #     "tp": tp,
        #     "pred": pred_funcs,
        #     "true": true
        # }
        results = {}

        return loss, results
    
        
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
        stop_token: List[int] = [],
        return_top: int = 0,
        disable_func: List[int] = [], # func ids to diable
        disable_token: List[int] = [], # base tokens ids to disable
    ) -> List[str]:
        """
        Generation method with stopping condidtions in tools or `<EOC>`
        """
        generation_log = []
        stop_token_substr = [torch.tensor(x).cuda().long() for x in stop_token if isinstance(x, list)]
        stop_token_single = [x for x in stop_token if isinstance(x, int)]
        
        prompt_tokens = self.tokenizer.encode(prompt, bos=True, eos=False)
        prompt_len = len(prompt_tokens)
        max_len = min(self.model.max_seq_len, prompt_len + max_gen_len)
        
        tokens = torch.full((1, max_len), self.tokenizer.pad_id).cuda().long()
        tokens[0, :prompt_len] = torch.tensor(prompt_tokens).cuda().long()
        
        prev_pos = 0
        
        start_pos = prompt_len
        for cur_pos in range(start_pos, max_len):
            logits = self.model.forward(prompt_tokens[:, prev_pos:cur_pos], prev_pos)
            
            # ------ move to callbacks ------
            if self.inference_mode != "func_embedding":
                # reduce logits to "arbitrary small likelyhood" 
                logits[:, -self.model.aug_vocab_size:] = - 1e5
                
            if len(disable_token) > 0:
                logits[:, disable_token] = -1e5
            
            if len(disable_func) > 0:
                logits[:, disable_func] = -1e5
                
            logits[:-self.model.aug_vocab_size] += self.logits_bias
        
            # -------------------------------
            
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            
            # only replace token if the prompt is ended
            # next_token = torch.where(
            #     input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            # )
            if return_top > 0:
                generation_log.append(
                    (next_token[0].item(), [(i.item(), logits[0, i.item()].item()) for i in torch.argsort(logits[0, :], descending=True)[:return_top]])
                )
            
            tokens[:, cur_pos] = next_token
            prev_pos = cur_pos

            # loop breaking conditions: find a tool or a stop tooken/substring
            if next_token[0] == self.tokenizer.eoc_id or next_token[0] in stop_token_single:
                break
            
            if any([torch.equal(tokens[0, cur_pos - len(substr) + 1: cur_pos + 1], substr) for substr in stop_token_substr]):
                break
        
        decoded = []
        for i, t in enumerate(tokens.tolist()):
            # cut to max gen len
            t = t[: len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            
            decoded_sample = self.tokenizer.decode(t[:cur_pos + 1])
            if t[cur_pos] >= self.model.base_vocab_size:
                decoded_sample += "("
            
            decoded.append(decoded_sample)
        
        if return_top > 0:
            return decoded, generation_log
        else:
            return decoded

