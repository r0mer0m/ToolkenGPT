import re
import math
import torch
from torch import nn
from transformers import LlamaForCausalLM
from funchub.math import *
import deepspeed
# from deepspeed.ops import deepspeed_hook

def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)

    return next_token


class AugmentedLM(LlamaForCausalLM):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._hook_registred = False
        
        for param in self.parameters(): param.requires_grad = False
        self.model.embed_tokens.weight.requires_grad = True
        self.lm_head.weight.requires_grad = True
        
        # deepspeed.zero.register_external_parameter(self, self.model.embed_tokens.weight)
        # deepspeed.zero.register_external_parameter(self, self.lm_head.weight)
    
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
        # print('projection weights: ', base_weights.shape)
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
        # self.freeze_base()
        self.augment_embeddings = config.augment_embeddings
        if config.augment_embeddings:
            self.augment_embeddings(config)
        self.augment_projection(config)
        
        # Workaround for hooks
        self.aug_vocab_size = config.aug_vocab_size
        
            
    def _register_backward_hooks(self, ):
        # Embeddings
        class MaskTokens(object):
            def __init__(self, aug_vocab_size):
                self.aug_vocab_size = aug_vocab_size
            
            def mask(self, grad):
                mask = torch.zeros_like(grad)
                mask[-self.aug_vocab_size:] = 1.
                out = grad*mask
                # print("Gradient:", out)
                return out
                
            def full_mask(self, grad):
                mask = torch.zeros_like(grad)
                out = grad*mask
                # print("This grad should be all zeros:", out)
                return out
        
        mask_tokens = MaskTokens(self.aug_vocab_size)
        
        # def mask_non_tool_tokens(grad):
        #     print("Gradient:", grad)
        
        with deepspeed.zero.GatheredParameters(self.model.embed_tokens.weight,
                                               modifier_rank=0):
            if self.augment_embeddings:
                # print(f"Embedding weights shape: {self.model.embed_tokens.weight.shape}")
                # print(f"Masking all but the last {self.aug_vocab_size} rows")
                self.model.embed_tokens.weight.register_hook(mask_tokens.mask)
            else:
                # print(f"Embedding weights don't require grad. Skipping backwards hook for this parameter.")
                self.model.embed_tokens.weight.register_hook(mask_tokens.full_mask)
        
        # LM Head
        with deepspeed.zero.GatheredParameters(self.lm_head.weight,
                                               modifier_rank=0):
            # print(f"projection weights shape: {self.lm_head.weight.shape}")
            # print(f"Masking all but the last {self.aug_vocab_size} rows")
            # mask = torch.zeros_like(self.lm_head.weight)
            # mask[-self.aug_vocab_size:] = 1.
            self.lm_head.weight.register_hook(mask_tokens.mask)
    
    def register_backward_hooks(self):        
        if not self._hook_registred:
            # print("Registering hoooks...")
            self._hook_registred = True
            self._register_backward_hooks()
    
    def augmented_generation(self, case_idx, input_ids, template_len):
        generate = True
        func_calls = []
        question = self.tokenizer.decode(input_ids)
        try:
            while generate:
                output = self.generate(
                    input_ids,
                    max_length = self.args.max_gen_len,
                    temperature = self.args.temperature,
                    top_p = self.args.top_p,
                    stopping_criteria=self.stopping_criteria,
                )
                if output.endswith("<EOC>"):
                    text_call = re.findall(r'(?:<SOC>)(.*)(?:<EOC>)', output)[-1]
                    text_op = text_call.split('(')[0]
                    call = re.replace(text_op, 
                                      self.tokenizer.symbol_to_api[text_op], 
                                      text_call)
                    func_calls.append(call)
                    result = eval(call)
                    result_text = "<SOR>" + result + "<EOR>"
                    result_ids = self._tokenizer.encode(result_text)
                    input_ids = input_ids + result_ids
                
                else:
                    generate = False
                    cur_generation = self.tokenizer.decode(output[template_len:])
            
            log = {
                "case_idx": case_idx,
                "question": question,
                "func_calls": func_calls,
                "generation": cur_generation, #.replace("\n", "\\n").strip(),
                "status": "success"
            }
            
        except Exception as e:
            log = {
                "case_idx": case_idx,
                "question": question,
                "func_calls": func_calls,
                "generation": cur_generation, #.replace("\n", "\\n").strip(),
                "status": str(e)
            }
        return log