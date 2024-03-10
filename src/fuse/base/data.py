import re
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
        
        for s, t, eq in zip(self.records[i]["start_token_idx"], self.records[i]["end_token_idx"], self.records[i]["tar_eq"]):
            # for different data formats
            if "[" in eq:
                op = re.search(r"(\[.*?\])", eq).group(1)
            elif "<" in eq:
                op = re.search(r"(<.*?>)", eq).group(1)
                # print(op)

            if op not in self.tokenizer.api_symbols:
                # op = op[1:-1]
                op = '<<' + op + '>>'
                            
            target_ids[s] = self.tokenizer.symbol_to_id[op]
            target_ids[s+1: t] = -100
        
        input_ids = input_ids[:-1]
        target_ids = target_ids[1:]
        
        # print(f'{text=}')
        # print(f'{input_ids=}')
        # print(f'{target_ids=}')
        
        record = {
            "input_ids": input_ids,
            "target_ids": target_ids
        }
        '''
        text = self.records[i]['text']
        print("Text: ", text)
        input_ids = torch.tensor(self.tokenizer.encode(text))[:] # bos=True, eos=True
        target_ids = torch.tensor(self.tokenizer.encode(text))[:]
        
        # ignore indices before first API call. 
        boc_idxs = self._get_token_indices(target_ids, self.tokenizer.boc_id)# (target_ids == self.tokenizer.boc_id).nonzero().squeeze().tolist()
        if boc_idxs:
            # target_ids[:boc_idxs[0]] = self.ignore_index
            
            # # ignore response indices
            # bor_idxs = self._get_token_indices(target_ids, self.tokenizer.bor_id)
            # eor_idxs = self._get_token_indices(target_ids, self.tokenizer.eor_id)
            # for bor_idx, eor_idx in zip(bor_idxs, eor_idxs):
            #     target_ids[bor_idx + 1: eor_idx] = self.ignore_index
            
            idxs = torch.ones_like(target_ids, dtype=torch.long)
            
            # ignore response indices
            boc_idxs = self._get_token_indices(target_ids, self.tokenizer.boc_id)
            eoc_idxs = self._get_token_indices(target_ids, self.tokenizer.eoc_id)
            for boc_idx, eoc_idx in zip(boc_idxs, eoc_idxs):
                idxs[boc_idx: eoc_idx + 1] = 0
            target_ids[idxs] = self.ignore_index
        
        
        input_ids = input_ids[:-1]
        target_ids = target_ids[1:]
        
        record = {
            "input_ids": input_ids,
            "target_ids": target_ids
        }
        '''
        return record
        
        
class TestAugLMDataset(Dataset):
    
    def __init__(self, records, tokenizer, template=''):
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
    
    def _mask_all_but_trigger(self, target_ids):
        
        idxs = torch.ones_like(target_ids, dtype=torch.long)
        boc_idxs = self._get_token_indices(target_ids, self.tokenizer.boc_id)
        idxs[boc_idxs] = 0
        target_ids[idxs] = self.ignore_index
        return target_ids
    
    def _mask_call_after_trigger(self, target_ids):
        
        idxs = torch.ones_like(target_ids, dtype=torch.long)
        boc_idxs = self._get_token_indices(target_ids, self.tokenizer.boc_id)
        idxs[boc_idxs] = 0
        target_ids[idxs] = self.ignore_index
        return target_ids
        
    def __getitem__(self, i):
        text = self.records[i]['text']
        
        input_ids = torch.tensor(self.tokenizer.encode(text))[:] # bos=True, eos=True
        target_ids = input_ids.clone()
        
        for s, t, eq in zip(self.records[i]["start_token_idx"], self.records[i]["end_token_idx"], self.records[i]["tar_eq"]):
            # for different data formats
            if "[" in eq:
                op = re.search(r"(\[.*?\])", eq).group(1)
            elif "<" in eq:
                op = re.search(r"(<.*?>)", eq).group(1)
                # print(op)

            if op not in self.tokenizer.api_symbols:
                op = op[1:-1]
                
            labels[s] = self.tokenizer.symbol_to_id[op]
            labels[s+1: t] = -100
        
        # text = self.template.replace("[QUESTION]", text)
        
        # target_ids = self._mask_all_but_trigger(target_ids)
        
        # ignore indices before first API call. 
        # boc_idxs = self._get_token_indices(target_ids, self.tokenizer.boc_id)
        # if boc_idxs:
        #     target_ids[:boc_idxs[0]] = self.ignore_index
            
        #     # ignore response indices
        #     bor_idxs = self._get_token_indices(target_ids, self.tokenizer.bor_id)
        #     eor_idxs = self._get_token_indices(target_ids, self.tokenizer.eor_id)
        #     for bor_idx, eor_idx in zip(bor_idxs, eor_idxs):
        #         target_ids[bor_idx + 1: eor_idx] = self.ignore_index
        
        input_ids = input_ids[:-1]
        target_ids = target_ids[1:]
        
        record = {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "template_len": len(self.tokenizer.encode(self.template)),
        }
        
        return record
