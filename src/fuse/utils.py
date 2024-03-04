import torch
from . import AugmentedLM, AugmentedConfig, AugmentedTokenizer
                  
def load(model_config, augmentation_config, rank: int, checkpoint_filepath: str = '') -> AugmentedLM:
    
    if rank == 0: print("Loading tokenizer")
    tokenizer = AugmentedTokenizer.from_pretrained(
        model_config.base_model_id,
        augmentation_config=augmentation_config,
        )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    if rank == 0: print("Loading config")
    augmented_model_config = AugmentedConfig.from_pretrained(
        model_config.base_model_id,
        augment=True,
        aug_vocab_size = tokenizer.aug_vocab_size # tokenizer.n_aug_words,
    )
    
    if rank == 0: print("Loading model")
    model = AugmentedLM.from_pretrained(
        model_config.base_model_id,
        load_in_8bit=False,
        torch_dtype=torch.bfloat16
        )
    
    if rank == 0: print("Augmenting model")
    model.augment(augmented_model_config)
    
    if checkpoint_filepath:
        if rank == 0: print("Loading checkpoint")
        model.load_state_dict(torch.load(checkpoint_filepath, map_location='cpu'))
    
    return tokenizer, model
