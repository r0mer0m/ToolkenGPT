import re
import torch
from torch.utils.data import Dataset

class AugLMDataset(Dataset):
    
    def __init__(self, records, tokenizer, data_args=None, template=''):
        self.records = records
        self.tokenizer = tokenizer
        self.ignore_index = -100
        self.augmented_data = data_args.augmented_data if data_args else False
        self.template = re.sub(r'Q: \[QUESTION\]\nA:', '', template)
        
        if self.template:
            self.template_offset = len(self.tokenizer.encode(self.template)) - 1
        else:
            self.template_offset = 0
        
        print("Template: ", self.template)
        print("Template offset: ", self.template_offset)
        print(records[0])
        
    def __len__(self,):
        return len(self.records)
    
    def _get_token_indices(self, text_ids, token_id):
        position_idxs = (text_ids == token_id).nonzero()
        if len(position_idxs.shape) > 1:
            position_idxs = position_idxs.squeeze(1)
        return position_idxs.tolist()  
    
    def get_mask_idxs(self, target_ids, func_name_ids, eoc_offsets):
        idxs = torch.ones_like(target_ids, dtype=torch.long)
        sample_ids = target_ids[self.template_offset:].tolist()
        for func_id, offset in zip(func_name_ids, eoc_offsets):
            idx = sample_ids.index(func_id) + self.template_offset
            idxs[idx:idx + offset] = 0
            print(self.tokenizer.convert_ids_to_tokens(target_ids[idx:idx + offset]))
        
        return idxs
    
    def __getitem__(self, i):
        record = self.records[i]
        text = record['text']
        if self.template:
            text = self.template + text
        
        if self.augmented_data:
            input_ids = torch.tensor(self.tokenizer.encode(text)) # bos=True, eos=True
            target_ids = input_ids.clone()
            
            mask_idxs = self.get_mask_idxs(target_ids, 
                          record['func_name_ids'], 
                          record['eoc_offsets'])
            
            target_ids[mask_idxs] = -100
            
        else:
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
        
        record = {
            "input_ids": input_ids,
            "target_ids": target_ids,
            # "start_token_idxs": self.records[i]["start_token_idx"],
            # "end_token_idxs": self.records[i]["end_token_idx"],
            # "answer_start_idx": self.records[i]["answer_start_idx"]
        }
        return record
        
        
class TestAugLMDataset(Dataset):
    
    def __init__(self, records, tokenizer, data_args, template):
        self.records = records
        self.tokenizer = tokenizer
        self.ignore_index = -100
        self.template = template
        
        self.records = [
            {
                "text": r["text"],
                "question": re.search(r"Q: (.*?)\n", r["text"]).group(1),
                "answer": re.search(r"A: (.*)", r["text"]).group(1),
            } for r in self.records
        ]
        
    def __len__(self,):
        return len(self.records)
    
    def __getitem__(self, index):
        print('Question: ', self.records[index]['question'])
        print('Answer: ', self.records[index]['answer'])
        
        input_text = re.sub(r'\[QUESTION\]', self.records[index]['question'], self.template)
        record = {
            "text": self.records[index]['text'],
            "input_text": input_text,
            "input_ids": torch.tensor(self.tokenizer.encode(input_text)),
            "answer": self.records[index]['answer']
        }
        return record