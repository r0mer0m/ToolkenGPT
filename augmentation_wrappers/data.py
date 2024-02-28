import json
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

TEST_LEN = {
    "gsm8k-xl": 1000,
    "funcqa": 39,
    "vh": 47,
    "kamel": 1000
}

class AugLMDataset(Dataset):
    
    def __init__(self, records, tokenizer):
        self.records = records
        self.tokenizer = tokenizer
        self.ignore_index = -100
        
    def __len__(self,):
        return len(self.records)
    
    def __getitem__(self, i):
        text = self.records[i]['text']
        
        input_ids = torch.tensor(self.tokenizer.encode(text, bos=True, eos=True))[:]
        target_ids = input_ids.clone()
        
        # ignore indices before first API call. 
        idxs = (target_ids == self.tokenizer.boc_id).nonzero()
        idx = idxs[0] if len(idxs.shape) == 1 else idxs[0][0]
        target_ids[:idx] = self.ignore_index
        
        input_ids = input_ids[:-1].expand(1, -1).to("cuda")
        target_ids = target_ids[1:].expand(1, -1).to("cuda")
        
        record = {
            "input_ids": input_ids,
            "target_ids": target_ids
        }
        
        return record
        

def collate_fn(batch):
    pass


class PLDataModule(LightningDataModule):
    def __init__(self, tokenizer, data_args):
        super().__init__()
        
        self.tokenizer = tokenizer
        self.args = data_args
        if not hasattr(data_args, 'test_len'):
            self.args.test_len = TEST_LEN[self.args.dataset_name]
    
    def setup(self, stage=None):
        if self.input_file.endswith(".json"):
            with open(self.input_file, "r") as f:
                prompts = json.load(f)
        
        else:
            with open(self.args.input_file, "r") as f:
                prompts = f.readlines()
            prompts = [prompt.strip().replace("\\n", "\n") for prompt in prompts if len(prompt) > 1]

        train_data = prompts[:-self.args.test_len]
        test_data = prompts[-self.args.test_len:]
        if self.args.rank == 0: print(f"Total data:\n\tTraining: {len(train_data)}\n\tTesting: {len(test_data)}")
        
        self.train_dataset = AugLMDataset(train_data, self.tokenizer)
        self.test_dataset = AugLMDataset(test_data, self.tokenizer)
            
    def train_dataloader(self):
        sampler = DistributedSampler(self.train_dataset, rank=self.args.rank, num_replicas=self.args.world_size, shuffle=True)
        train_loader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers, pin_memory=self.args.pin_memory, sampler=sampler)
        return train_loader
    
    def test_dataloader(self):
        sampler = DistributedSampler(self.test_dataset, rank=self.args.rank, num_replicas=self.args.world_size)
        test_loader = DataLoader(self.test_dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers, pin_memory=self.args.pin_memory, sampler=sampler)
        return test_loader
