import json
from os import path as osp
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from torch.utils.data.distributed import DistributedSampler
from fuse.base.data import AugLMDataset, TestAugLMDataset

class PLDataModule(LightningDataModule):
    def __init__(self, tokenizer, data_args, rank, world_size):
        super().__init__()
        self.tokenizer = tokenizer
        self.rank = rank
        self.world_size = world_size
        self.args = data_args
    
    def _get_dev_data(self, input_dir):
        if self.args.augmented_data:
            # input_filepath = osp.join(input_dir, "augmented_train.json")
            # input_filepath = osp.join(input_dir, "augmented_train_call_only.json") # loss with call only
            input_filepath = osp.join(input_dir, "augmented_train_call_only_equal_after_call.json")
        else:
            input_filepath = osp.join(input_dir, "train.json")
        if input_filepath.endswith(".json"):
            with open(input_filepath, "r") as f:
                prompts = json.load(f)
        else:
            with open(input_filepath, "r") as f:
                prompts = f.readlines()
            prompts = [prompt.strip().replace("\\n", "\n") for prompt in prompts if len(prompt) > 1]
        
        train_data = prompts[:-self.args.test_len]
        test_data = prompts[-self.args.test_len:]
        
        return train_data, test_data
    
    def _get_test_data(self, input_dir):
        template_path = osp.join(input_dir, "template.txt")
        with open(template_path) as f:
            template = f.read()
        
        test_path = osp.join(input_dir, "test.json")
        with open(test_path) as f:
            data = f.read()
        samples = [r['question'] for r in data]
        
        return template, samples
    
    def setup(self, stage=None):
        if stage == "fit":
            train_data, val_data = self._get_dev_data(self.args.input_dir)
            self.train_dataset = AugLMDataset(train_data, self.tokenizer, self.args)
            self.val_dataset = AugLMDataset(val_data, self.tokenizer, self.args)
            if self.rank == 0: 
                print(f"Total data:\n\tTraining: {len(train_data)}\n\tTesting: {len(val_data)}")
        elif stage == "test":
            template, test_data = self._get_test_data(self.args.input_dir)
            self.test_dataset = TestAugLMDataset(test_data, self.tokenizer, template)
            print(f"Total data:\n\tTest: {len(test_data)}")
            
    def train_dataloader(self):
        sampler = DistributedSampler(self.train_dataset, shuffle=True)
        train_loader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers, pin_memory=self.args.pin_memory, sampler=sampler)
        return train_loader
    
    def val_dataloader(self):
        sampler = DistributedSampler(self.val_dataset, shuffle=False)
        val_loader = DataLoader(self.val_dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers, pin_memory=self.args.pin_memory, sampler=sampler)
        return val_loader
    
    def test_dataloader(self):
        sampler = DistributedSampler(self.test_dataset, shuffle=False)
        test_loader = DataLoader(self.test_dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers, pin_memory=self.args.pin_memory, sampler=sampler)
        return test_loader
 