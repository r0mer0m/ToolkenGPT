import torch
from . import AugmentedLM, AugmentedConfig, AugmentedTokenizer, PLModel
                  
def load(model_config, rank: int, training_config=None, augmentation_config=None, checkpoint_filepath: str = '') -> AugmentedLM:
    
    if rank == 0: print("Loading tokenizer")
    tokenizer = AugmentedTokenizer.from_pretrained(
        model_config.base_model_id,
        augmentation_config=augmentation_config,
        )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    if rank == 0: print("Loading config")
    augmented_model_config = AugmentedConfig.from_pretrained(
        model_config.base_model_id,
        augment_embeddings=augmentation_config.augment_embeddings,
        aug_vocab_size = tokenizer.aug_vocab_size # tokenizer.n_aug_words,
    )
    
    if checkpoint_filepath:
        if rank == 0: print("Loading checkpoint")
        model = model.load_state_dict(checkpoint_filepath)
    
    elif training_config:
        if rank == 0: print("Loading model")
        _model = AugmentedLM.from_pretrained(
            model_config.base_model_id,
            load_in_8bit=False,
            torch_dtype=torch.bfloat16
            )
        if rank == 0: print("Augmenting model")
        _model.augment(augmented_model_config)
        model = PLModel(model=_model, tokenizer=tokenizer, rank=rank, config=training_config)
    
    return tokenizer, model, _model
