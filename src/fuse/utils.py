import torch
from . import AugmentedLM, AugmentedConfig, AugmentedTokenizer, PLModel
                  
def load(model_config, rank: int, training_config=None, augmentation_config=None, checkpoint_filepath: str = '') -> AugmentedLM:
    
    if rank == 0: print("Loading tokenizer")
    tokenizer = AugmentedTokenizer.from_pretrained(
        model_config.base_model_id,
        device_map="auto",
        augmentation_config=augmentation_config,
        )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    if rank == 0: print("Loading config")
    augmented_model_config = AugmentedConfig.from_pretrained(
        model_config.base_model_id,
        embedding_augmentation_type=augmentation_config.embedding_augmentation_type,
        aug_vocab_size = tokenizer.aug_vocab_size, # tokenizer.n_aug_words,
        aug_token_bias = augmentation_config.aug_token_bias
    )
    
    
    if checkpoint_filepath:
        _model = AugmentedLM.from_pretrained(
                model_config.base_model_id,
                load_in_8bit=False,
                torch_dtype=torch.bfloat16
                )
        _model.augment(tokenizer, augmented_model_config)
        model = PLModel.load_from_checkpoint(checkpoint_filepath, tokenizer=tokenizer, model=_model)
    else:
        if rank == 0: print("Loading model")
        _model = AugmentedLM.from_pretrained(
                model_config.base_model_id,
                load_in_8bit=False,
                torch_dtype=torch.bfloat16
                )
        _model.augment(tokenizer, augmented_model_config)
        model = PLModel(model=_model, 
                        tokenizer=tokenizer,
                        rank=rank,
                        config=training_config)
    
    return tokenizer, model, _model
