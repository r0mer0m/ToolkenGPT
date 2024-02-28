import json
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


class PLDataModule(LightningDataModule):
    def __init__(self, input_file:str, dataset_name:str, rank, world_size, batch_size=4, num_workers=2, pin_memory=True, shuffle=True):
        super().__init__()
        
        self.input_file = input_file
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.rank = rank
        self.world_size = world_size
    
    def setup(self, stage=None):
        if self.input_file.endswith(".json"):
            with open(self.input_file, "r") as f:
                prompts = json.load(f)
        
        else:
            with open(self.input_file, "r") as f:
                prompts = f.readlines()
            prompts = [prompt.strip().replace("\\n", "\n") for prompt in prompts if len(prompt) > 1]

        if self.dataset_name == "gsm8k-xl":
            # the last 1000 prompts are the testset
            test_len = 1000
        elif self.dataset_name == "funcqa":
            # the last 39 prompts are the testset
            test_len = 39
        elif self.dataset_name == "vh":
            test_len = 47
        elif self.dataset_name == "kamel":
            test_len = 1000
        
        print("Total data: ")
        self.train_dataset = prompts[:-test_len]
        self.test_dataset = prompts[-test_len:]
            
    def train_dataloader(self):
        sampler = DistributedSampler(self.train_dataset, rank=self.rank, num_replicas=self.world_size, shuffle=True)
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory, sampler=sampler)
        return train_loader
    
    def test_dataloader(self):
        sampler = DistributedSampler(self.test_dataset, rank=self.rank, num_replicas=self.world_size)
        test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory, sampler=sampler)
        return test_loader
