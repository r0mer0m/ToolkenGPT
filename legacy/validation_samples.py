import torch
import hydra
from fuse.utils import load
from fuse import PLDataModule
from torch.utils.data import DataLoader


@torch.no_grad()
@hydra.main(version_base=None, config_path="./configs/runs")
def generate_call(config):

    tokenizer, _, model = load(config.model, rank=0, 
                                training_config=config.training, 
                                augmentation_config=config.augmentation,
                                checkpoint_filepath=config.best_checkpoint_filepath
                                )
    model.eval()
    data_module = PLDataModule(tokenizer=tokenizer, data_args=config.data,
                               rank=0, world_size=1)
    
    data_module.setup('fit')
    model = model.to('cuda:0')
    val_loader = DataLoader(data_module.val_dataset, batch_size=1, num_workers=1, pin_memory=False)
    for batch in val_loader:
        print("#"*50)
        input_ids = batch['input_ids'].to('cuda:0')
        expected, predicted = model.generate_call(input_ids,
                            start_token_idxs = batch['start_token_idxs'],
                            end_token_idxs = batch['end_token_idxs'])
        print("Expected: ", tokenizer.decode(expected))
        print("Predicted: ", tokenizer.decode(predicted))
        print("\n")
        
@torch.no_grad()
@hydra.main(version_base=None, config_path="./configs/runs")
def generate_answer(config):

    tokenizer, _, model = load(config.model, rank=0, 
                                training_config=config.training, 
                                augmentation_config=config.augmentation,
                                checkpoint_filepath=config.best_checkpoint_filepath
                                )
    data_module = PLDataModule(tokenizer=tokenizer, data_args=config.data,
                               rank=0, world_size=1)
    
    # model = model.to('cuda:0')
    data_module.setup('test')
    model = model.to('cuda:0')
    val_loader = DataLoader(data_module.test_dataset, batch_size=1, num_workers=1, pin_memory=False)
    for batch in val_loader:
        print("#"*50)
        input_ids = batch['input_ids'].to('cuda:0')
        # answer_start_idx = batch['answer_start_idx']
        predicted = model.generate_answer(input_ids)
        print("\n\nInput: ", tokenizer.decode(batch['input_ids'][0]))
        print("\nExpected: ", batch['answer'][0])
        print("\nPredicted: ", tokenizer.decode(predicted[batch['input_ids'].shape[1]:]))
        print("\n" + "-"*100)
        

if __name__ == "__main__":
    # generate_call()
    generate_answer()