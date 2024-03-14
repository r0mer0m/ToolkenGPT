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
        embedding_augmentation_type=augmentation_config.embedding_augmentation_type,
        aug_vocab_size = tokenizer.aug_vocab_size # tokenizer.n_aug_words,
    )
    
    if rank == 0: print("Loading model")
    _model = AugmentedLM.from_pretrained(
            model_config.base_model_id,
            load_in_8bit=False,
            torch_dtype=torch.bfloat16
            )
    _model.augment(augmented_model_config)
    
    if checkpoint_filepath:
        if rank == 0: print("Loading checkpoint")
        pl_state_dict = torch.load(checkpoint_filepath)
        # print("#"*30 + " State dict keys " + "#"*30)
        state_dict = {k[6:]:v for k,v in pl_state_dict['state_dict'].items()}
        # print(state_dict['model.layers.0.self_attn.q_proj.weight'])
        # print(state_dict.keys())
        # print("#"*30 + " Loading state dict " + "#"*30)
        _model.load_state_dict(state_dict)
        model = None
        
    # elif training_config:
    model = PLModel(model=_model, tokenizer=tokenizer, rank=rank, config=training_config)
    
    return tokenizer, model, _model
