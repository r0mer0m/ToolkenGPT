import torch
from torch.utils.data import Dataset

class AugLMDataset(Dataset):
    
    def __init__(self, records, tokenizer):
        self.records = records
        self.tokenizer = tokenizer
        self.ignore_index = -100
        
    def __len__(self,):
        return len(self.records)
    
    
    def _get_token_indices(self, text_ids, token_id):
        position_idxs = (text_ids == token_id).nonzero()
        if len(position_idxs.shape) > 1:
            position_idxs = position_idxs.squeeze(1)
        return position_idxs.tolist()    
    
    def __getitem__(self, i):
        text = self.records[i]['text']
        
        input_ids = torch.tensor(self.tokenizer.encode(text))[:] # bos=True, eos=True
        target_ids = input_ids.clone()
        
        # ignore indices before first API call. 
        boc_idxs = self._get_token_indices(target_ids, self.tokenizer.boc_id)# (target_ids == self.tokenizer.boc_id).nonzero().squeeze().tolist()
        if boc_idxs:
            target_ids[:boc_idxs[0]] = self.ignore_index
            
            # ignore response indices
            bor_idxs = self._get_token_indices(target_ids, self.tokenizer.bor_id)
            eor_idxs = self._get_token_indices(target_ids, self.tokenizer.eor_id)
            for bor_idx, eor_idx in zip(bor_idxs, eor_idxs):
                target_ids[bor_idx + 1: eor_idx] = self.ignore_index
        
        input_ids = input_ids[:-1]
        target_ids = target_ids[1:]
        
        record = {
            "input_ids": input_ids,
            "target_ids": target_ids
        }
        
        return record
        
        
class TestAugLMDataset(Dataset):
    
    def __init__(self, records, tokenizer, template=None):
        self.records = records
        self.tokenizer = tokenizer
        self.ignore_index = -100
        self.template = template
        
    def __len__(self,):
        return len(self.records)
    
    
    def _get_token_indices(self, text_ids, token_id):
        position_idxs = (text_ids == token_id).nonzero()
        if len(position_idxs.shape) > 1:
            position_idxs = position_idxs.squeeze(1)
        return position_idxs.tolist()    
    
    def __getitem__(self, i):
        text = self.records[i]['text']
        text = self.template.replace("[QUESTION]", text)
        
        input_ids = torch.tensor(self.tokenizer.encode(text))[:] # bos=True, eos=True
        # target_ids = input_ids.clone()
        
        # # ignore indices before first API call. 
        # boc_idxs = self._get_token_indices(target_ids, self.tokenizer.boc_id)# (target_ids == self.tokenizer.boc_id).nonzero().squeeze().tolist()
        # if boc_idxs:
        #     target_ids[:boc_idxs[0]] = self.ignore_index
            
        #     # ignore response indices
        #     bor_idxs = self._get_token_indices(target_ids, self.tokenizer.bor_id)
        #     eor_idxs = self._get_token_indices(target_ids, self.tokenizer.eor_id)
        #     for bor_idx, eor_idx in zip(bor_idxs, eor_idxs):
        #         target_ids[bor_idx + 1: eor_idx] = self.ignore_index
        
        # input_ids = input_ids[:-1]
        # target_ids = target_ids[1:]
        
        record = {
            "input_ids": input_ids,
            "template_len": len(self.tokenizer.encode(self.template))
            # "target_ids": target_ids
        }
        
        return record
